import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import test from "node:test"
import { dirname, join } from "node:path"
import { fileURLToPath } from "node:url"

import { importDshHostModule } from "../host.mjs"
import {
  applyToolPolicy,
  filterTools,
  ALLOWED_TOOL_NAMES,
  TOOL_PROMPT_SECTION,
  WORKSPACE_PROMPT_SECTION,
  WORKSPACE_PROMPT_SECTION_V2,
  WORKSPACE_PROMPT_SECTION_V3,
} from "../tool-policy.mjs"
import {
  projectQwenMessages,
  projectQwenTools,
  QwenOutputParser,
} from "../qwen-protocol.mjs"

let hostModules
try {
  const adapter = await import("../qwen-adapter.mjs")
  const runner = await import("../runner.mjs")
  const llm = await importDshHostModule("@deepseek-ai/dsh-llm")
  hostModules = { ...adapter, ...llm, RunnerConfig: runner.Config }
} catch (error) {
  const missingHost = ["MODULE_NOT_FOUND", "ERR_MODULE_NOT_FOUND"].includes(error?.code)
    && String(error?.message).includes("@deepseek-ai/")
  if (!missingHost) throw error
}

const BlockAssembler = hostModules?.BlockAssembler
const EMPTY_RESPONSE_CODE = hostModules?.EMPTY_RESPONSE_CODE
const LOCAL_QWEN_MODEL = hostModules?.LOCAL_QWEN_MODEL
  ?? "Qwen/Qwen2.5-7B-Instruct"
const LOCAL_QWEN_PROVIDER = hostModules?.LOCAL_QWEN_PROVIDER ?? "local-qwen"
const LocalQwenAdapter = hostModules?.LocalQwenAdapter
const RunnerConfig = hostModules?.RunnerConfig

function hostBackedTest(name, body) {
  if (hostModules) return test(name, body)
  return test(name, { skip: "DSH host modules are unavailable" }, body)
}

const here = dirname(fileURLToPath(import.meta.url))

const metadata = {
  revision: "qwen-test-revision",
  precision: "nf4",
  device: "cuda:0",
  model_class: "Qwen2Model",
  is_quantized: true,
  quant_type: "nf4",
  double_quant: true,
  compute_dtype: "torch.bfloat16",
  tie_word_embeddings: false,
  linear4bit_modules: 196,
  lm_head_dtype: "torch.bfloat16",
  repetition_penalty: 1.05,
  eos_token_ids: [151645, 151643],
}

function adapterOptions(overrides = {}) {
  return {
    provider: LOCAL_QWEN_PROVIDER,
    model: LOCAL_QWEN_MODEL,
    reasoningEffort: "off",
    system: "Follow the tool contract.",
    messages: [
      {
        role: "user",
        content: [{ type: "text", text: "Complete the task." }],
        source: { kind: "user" },
      },
    ],
    tools: [
      {
        name: "pwsh",
        description: "Run one PowerShell command.",
        parameters: {
          type: "object",
          properties: { command: { type: "string" } },
          required: ["command"],
        },
      },
    ],
    temperature: 0,
    maxTokens: 64,
    stop: [],
    ...overrides,
  }
}

function fakeWorker(events) {
  return {
    metadata,
    payload: undefined,
    payloads: [],
    async *stream(payload) {
      this.payload = payload
      this.payloads.push(payload)
      for (const event of events) {
        if (event.type === "throw") throw event.error
        yield event.type === "result" ? { ...event, metadata } : event
      }
    },
    async dispose() {},
  }
}

function adapterWith(events, config = {}) {
  const adapter = new LocalQwenAdapter({
    python: "unused-python",
    worker: "unused-worker",
    snapshot: "unused-snapshot",
    ...config,
  })
  const worker = fakeWorker(events)
  adapter.worker = worker
  return { adapter, worker }
}

async function collect(iterable) {
  const chunks = []
  for await (const chunk of iterable) chunks.push(chunk)
  return chunks
}

async function collectFailure(iterable) {
  const chunks = []
  try {
    for await (const chunk of iterable) chunks.push(chunk)
  } catch (error) {
    return { chunks, error }
  }
  assert.fail("expected the adapter stream to fail")
}

test("Qwen projection preserves ordered text, tool calls, results, and schemas", () => {
  const options = {
    system: "system text",
    messages: [
      {
        role: "user",
        content: [
          { type: "text", text: "user " },
          { type: "reasoning", text: "message" },
        ],
      },
      {
        role: "assistant",
        content: [
          { type: "text", text: "checking" },
          {
            type: "tool-call",
            id: "call-1",
            name: "pwsh",
            arguments: '{"command":"Write-Output ok"}',
          },
        ],
      },
      {
        role: "user",
        content: [
          {
            type: "tool-result",
            toolCallId: "call-1",
            name: "pwsh",
            isError: false,
            content: [{ type: "text", text: "ok\n" }],
          },
        ],
      },
    ],
  }

  assert.deepEqual(projectQwenMessages(options), [
    { role: "system", content: "system text" },
    { role: "user", content: "user message" },
    {
      role: "assistant",
      content: "checking",
      tool_calls: [
        {
          type: "function",
          function: { name: "pwsh", arguments: { command: "Write-Output ok" } },
        },
      ],
    },
    { role: "tool", content: "ok\n" },
  ])

  assert.deepEqual(
    projectQwenTools([
      {
        name: "pwsh",
        description: "Run PowerShell.",
        parameters: { type: "object", required: ["command"] },
      },
    ]),
    [
      {
        type: "function",
        function: {
          name: "pwsh",
          description: "Run PowerShell.",
          parameters: { type: "object", required: ["command"] },
        },
      },
    ],
  )
})

test("Qwen projection rejects images at the message boundary and inside tool results", () => {
  const image = { type: "image", mediaType: "image/png", data: "AA==" }
  for (const content of [
    [image],
    [{
      type: "tool-result",
      toolCallId: "call-image",
      name: "read_image",
      isError: false,
      content: [image],
    }],
  ]) {
    assert.throws(
      () => projectQwenMessages({ messages: [{ role: "user", content }] }),
      /accepts text-only messages/,
    )
  }
})

test("Qwen parser handles tags and JSON split across arbitrary deltas", () => {
  const parser = new QwenOutputParser()
  const visible = [
    ...parser.push("I will check. <too"),
    ...parser.push('l_call>{"na'),
    ...parser.push('me":"pwsh","arguments":{"command":"Write-'),
    ...parser.push('Output ok"}}</tool_'),
    ...parser.push("call>"),
  ]
  const parsed = parser.finish()

  assert.equal(visible.join(""), "I will check. ")
  assert.equal(parsed.text, "I will check. ")
  assert.deepEqual(parsed.tools, [
    { name: "pwsh", arguments: '{"command":"Write-Output ok"}' },
  ])
})

test("Qwen parser rejects a truncated second tool after one complete tool", () => {
  const parser = new QwenOutputParser()
  parser.push(
    '<tool_call>{"name":"pwsh","arguments":{"command":"one"}}</tool_call>'
      + '<tool_call>{"name":"pwsh","arguments":',
  )
  assert.throws(() => parser.finish(), /incomplete tool call/)
})

test("Qwen parser ignores a close tag inside a JSON string", () => {
  const parser = new QwenOutputParser()
  parser.push(
    '<tool_call>{"name":"pwsh","arguments":{"command":"Write-Output \\\"</tool_call>\\\""}}</tool_call>',
  )
  assert.deepEqual(parser.finish().tools, [
    { name: "pwsh", arguments: '{"command":"Write-Output \\\"</tool_call>\\\""}' },
  ])
})

test("Qwen parser surfaces non-object tool arguments as visible text instead of aborting", () => {
  const parser = new QwenOutputParser()
  parser.push('<tool_call>{"name":"pwsh","arguments":[]}</tool_call>')
  const finished = parser.finish()
  assert.deepEqual(finished.tools, [])
  assert.equal(finished.text, '<tool_call>{"name":"pwsh","arguments":[]}</tool_call>')
})

test("Qwen parser preserves the model's exact JSON argument bytes", () => {
  const parser = new QwenOutputParser()
  const argumentsJson = '{ "integer": 9007199254740993, "exponent": 1e+30 }'
  parser.push(`<tool_call>{"name":"pwsh","arguments":${argumentsJson}}</tool_call>`)
  assert.deepEqual(parser.finish().tools, [
    { name: "pwsh", arguments: argumentsJson },
  ])
})

test("DSH tool policy states the workspace root only for the local Qwen route", () => {
  const assembly = { sections: [{ name: "harness:identity", text: "Harness identity." }], tools: [] }

  assert.deepEqual(applyToolPolicy(assembly, "local-qwen").sections, [
    { name: "harness:identity", text: "Harness identity." },
    { ...WORKSPACE_PROMPT_SECTION },
    { ...TOOL_PROMPT_SECTION },
  ])
  assert.deepEqual(applyToolPolicy(assembly, "openai-codex").sections, [
    { name: "harness:identity", text: "Harness identity." },
    { ...TOOL_PROMPT_SECTION },
  ])
})

test("new local-Qwen prompt policy resolves unsupported ls and missing paths", () => {
  const assembly = { sections: [], tools: [] }
  assert.deepEqual(
    applyToolPolicy(assembly, "local-qwen", "swebench-source-patch-submit-v5").sections,
    [{ ...WORKSPACE_PROMPT_SECTION_V2 }, { ...TOOL_PROMPT_SECTION }],
  )
  assert.match(WORKSPACE_PROMPT_SECTION_V2.text, /There is no `ls` tool/)
  assert.match(WORKSPACE_PROMPT_SECTION_V2.text, /do not retry that path/)
  assert.deepEqual(
    applyToolPolicy(assembly, "local-qwen", "swebench-source-patch-submit-v4").sections,
    [{ ...WORKSPACE_PROMPT_SECTION }, { ...TOOL_PROMPT_SECTION }],
  )
})

test("v6 local-Qwen prompt uses first-request pwsh file discovery", () => {
  const assembly = { sections: [], tools: [{ name: "pwsh" }, { name: "read" }] }
  assert.deepEqual(
    applyToolPolicy(assembly, "local-qwen", "swebench-source-patch-submit-v6").sections,
    [{ ...WORKSPACE_PROMPT_SECTION_V3 }, { ...TOOL_PROMPT_SECTION }],
  )
  assert.match(WORKSPACE_PROMPT_SECTION_V3.text, /rg --files/)
  assert.match(WORKSPACE_PROMPT_SECTION_V3.text, /Get-ChildItem -Name/)
  assert.match(WORKSPACE_PROMPT_SECTION_V3.text, /Before the first source `read`/)
  assert.match(WORKSPACE_PROMPT_SECTION_V3.text, /Never retry or guess a path/)
})

test("Qwen parser passes an unregistered tool name through to the harness", () => {
  const parser = new QwenOutputParser()
  parser.push('Let me check. <tool_call>{"name":"pwd","arguments":{}}</tool_call>')
  assert.deepEqual(parser.finish().tools, [{ name: "pwd", arguments: "{}" }])
})

test("Qwen parser surfaces a malformed tool call as visible text instead of aborting", () => {
  for (const payload of [
    '{"name":"pwsh","name":"pwsh","arguments":{}}',
    '{"name":"pwsh","arguments":{},"arguments":{}}',
  ]) {
    const parser = new QwenOutputParser()
    parser.push(`<tool_call>${payload}</tool_call>`)
    const finished = parser.finish()
    assert.deepEqual(finished.tools, [])
    assert.equal(finished.text, `<tool_call>${payload}</tool_call>`)
  }
})

test("Qwen parser drops text emitted after a tool call that already parsed", () => {
  const parser = new QwenOutputParser()
  const visible = parser.push('<tool_call>{"name":"pwsh","arguments":{}}</tool_call>and then I ran it')
  assert.deepEqual(visible, [])
  const finished = parser.finish()
  assert.deepEqual(finished.tools, [{ name: "pwsh", arguments: "{}" }])
  assert.equal(finished.text, "")
})

test("Qwen parser keeps a later, well-formed tool call after an earlier malformed one", () => {
  const parser = new QwenOutputParser()
  parser.push('<tool_call>{"name":"pwsh"}</tool_call>retrying: <tool_call>{"name":"pwsh","arguments":{}}</tool_call>')
  const finished = parser.finish()
  assert.deepEqual(finished.tools, [{ name: "pwsh", arguments: "{}" }])
  assert.ok(finished.text.startsWith("<tool_call>"))
})

test("DSH tool policy removes delegation, continuation, interaction, and web tools", () => {
  const routed = [
    ...ALLOWED_TOOL_NAMES.map((name) => ({ name })),
    ...[
      "ask_user_question",
      "create_goal",
      "dev_mode_subagent",
      "dev_router_mode",
      "dev_router_status",
      "get_goal",
      "interrupt_agent",
      "list_agents",
      "ralph",
      "read_image",
      "send_message",
      "subagent",
      "subagent_fork",
      "update_goal",
      "web_search",
      "workflow",
    ].map((name) => ({ name })),
  ]
  assert.deepEqual(
    filterTools(routed).map((tool) => tool.name),
    [...ALLOWED_TOOL_NAMES],
  )
})

test("DSH tool policy keeps guidance only for tools in the Request", () => {
  const assembly = {
    sections: [
      { name: "harness:identity", text: "Harness identity." },
      { name: "tool:pwsh", text: "PowerShell guidance." },
      { name: "tool:jobs", text: "Job guidance." },
      { name: "tool:web_search", text: "Web guidance." },
      { name: "deployment:persona", text: "Persona." },
    ],
    contexts: [{ name: "runtime", text: "context" }],
    tools: [{ name: "pwsh" }, { name: "job_list" }, { name: "web_search" }],
    variables: { cwd: "workspace" },
  }

  assert.deepEqual(applyToolPolicy(assembly), {
    ...assembly,
    sections: [
      { name: "harness:identity", text: "Harness identity." },
      { name: "tool:pwsh", text: "PowerShell guidance." },
      { name: "tool:jobs", text: "Job guidance." },
      { name: "deployment:persona", text: "Persona." },
      { ...TOOL_PROMPT_SECTION },
    ],
    tools: [{ name: "pwsh" }, { name: "job_list" }],
  })
})

hostBackedTest("local Qwen adapter emits a Host-valid online text stream", async () => {
  const { adapter, worker } = adapterWith([
    { type: "start", input_tokens: 11 },
    { type: "delta", text: "hel", output_tokens: 1 },
    { type: "delta", text: "lo", output_tokens: 2 },
    { type: "flush", output_tokens: 2 },
    {
      type: "result",
      output: "hello",
      input_tokens: 11,
      output_tokens: 2,
      acknowledged_deltas: 2,
      finish_reason: "stop",
    },
  ])

  const chunks = await collect(adapter.stream(adapterOptions()))
  assert.deepEqual(
    chunks.map((chunk) => chunk.type),
    ["block-start", "text-delta", "text-delta", "block-end", "usage", "finish"],
  )
  assert.equal(chunks.at(-1).reason.kind, "stop")
  assert.deepEqual(chunks.at(-2).usage, { inputTokens: 11, outputTokens: 2 })
  assert.deepEqual(worker.payload, {
    messages: projectQwenMessages(adapterOptions()),
    tools: projectQwenTools(adapterOptions().tools),
    temperature: 0,
    max_tokens: 64,
    stop: [],
  })

  const assembler = new BlockAssembler()
  for (const chunk of chunks) assembler.push(chunk)
  assert.deepEqual(assembler.blocks(), [{ type: "text", text: "hello" }])
  assert.deepEqual(assembler.usage, { inputTokens: 11, outputTokens: 2 })
  assert.deepEqual(assembler.finish, { kind: "stop" })
  assert.equal(assembler.replayState.response.acknowledgedDeltaCount, 2)
})

hostBackedTest("local Qwen adapter derives one Probe sidecar per DSH session", async () => {
  const probeDir = join(here, "probe-output")
  const { adapter, worker } = adapterWith(
    [
      { type: "start", input_tokens: 3 },
      { type: "delta", text: "ok", output_tokens: 1 },
      { type: "flush", output_tokens: 1 },
      {
        type: "result",
        output: "ok",
        input_tokens: 3,
        output_tokens: 1,
        acknowledged_deltas: 1,
        finish_reason: "stop",
      },
    ],
    {
      generationProbeDir: probeDir,
      generationProbeCaptureRevisionPrefix: "probe-capture-v3:run-001",
    },
  )

  await collect(adapter.stream(adapterOptions({ sessionId: "session-a" })))
  await collect(adapter.stream(adapterOptions({ sessionId: "session-b" })))

  assert.deepEqual(
    worker.payloads.map((payload) => payload.generation_probe),
    [
      {
        path: join(probeDir, "session-a.jsonl"),
        capture_revision: "probe-capture-v3:run-001:session-a",
        session_id: "session-a",
        model_id: LOCAL_QWEN_MODEL,
      },
      {
        path: join(probeDir, "session-b.jsonl"),
        capture_revision: "probe-capture-v3:run-001:session-b",
        session_id: "session-b",
        model_id: LOCAL_QWEN_MODEL,
      },
    ],
  )
})

hostBackedTest("Generation Probe directory and revision prefix are a paired config", async () => {
  for (const config of [
    { generationProbeDir: join(here, "probe-output") },
    { generationProbeCaptureRevisionPrefix: "probe-capture-v3:run-001" },
  ]) {
    const { adapter } = adapterWith([], config)
    await assert.rejects(
      () => collect(adapter.stream(adapterOptions({ sessionId: "session-a" }))),
      (error) => {
        assert.equal(error.code, "LOCAL_QWEN_PROTOCOL")
        assert.match(error.message, /requires directory, capture revision prefix/)
        return true
      },
    )
  }
})

hostBackedTest("worker failure after partial deltas does not invent terminal usage", async () => {
  const { adapter } = adapterWith([
    { type: "start", input_tokens: 11 },
    { type: "delta", text: "partial", output_tokens: 1 },
    { type: "throw", error: new Error("worker failed before terminal accounting") },
  ])

  const { chunks, error } = await collectFailure(adapter.stream(adapterOptions()))
  assert.deepEqual(
    chunks.map((chunk) => chunk.type),
    ["block-start", "text-delta", "block-end"],
  )
  assert.equal(chunks.some((chunk) => chunk.type === "usage"), false)
  assert.equal(error.code, "LOCAL_QWEN_INFERENCE")
})

hostBackedTest("max-token truncation never exposes a generated tool call", async () => {
  const output =
    '<tool_call>{"name":"pwsh","arguments":{"command":"Write-Output ok"}}</tool_call>'
  const { adapter } = adapterWith([
    { type: "start", input_tokens: 8 },
    { type: "delta", text: output, output_tokens: 12 },
    {
      type: "result",
      output,
      input_tokens: 8,
      output_tokens: 12,
      acknowledged_deltas: 1,
      finish_reason: "max-tokens",
    },
  ])

  const chunks = await collect(adapter.stream(adapterOptions()))
  assert.deepEqual(chunks.map((chunk) => chunk.type), ["usage", "finish"])
  assert.equal(chunks.at(-1).reason.kind, "max-tokens")
  assert.equal(chunks.some((chunk) => chunk.type.includes("tool-call")), false)
})

hostBackedTest("max-token truncation cannot republish a rejected tool close tag", async () => {
  const output = "hello</tool_call>"
  const { adapter } = adapterWith([
    { type: "start", input_tokens: 8 },
    { type: "delta", text: output, output_tokens: 2 },
    {
      type: "result",
      output,
      input_tokens: 8,
      output_tokens: 2,
      acknowledged_deltas: 1,
      finish_reason: "max-tokens",
    },
  ])

  const chunks = await collect(adapter.stream(adapterOptions()))
  assert.deepEqual(chunks.map((chunk) => chunk.type), ["usage", "finish"])
  assert.equal(JSON.stringify(chunks).includes("</tool_call>"), false)
})

hostBackedTest("stream and terminal output mismatch fails at the max-token boundary", async () => {
  const { adapter } = adapterWith([
    { type: "start", input_tokens: 8 },
    { type: "delta", text: "A", output_tokens: 1 },
    {
      type: "result",
      output: "B",
      input_tokens: 8,
      output_tokens: 1,
      acknowledged_deltas: 1,
      finish_reason: "max-tokens",
    },
  ])

  const { chunks, error } = await collectFailure(adapter.stream(adapterOptions()))
  assert.equal(chunks.at(-1).type, "usage")
  assert.deepEqual(chunks.at(-1).usage, { inputTokens: 8, outputTokens: 1 })
  assert.equal(error.code, "LOCAL_QWEN_PROTOCOL")
  assert.match(error.message, /stream and terminal output disagree/)
})

hostBackedTest("usage precedes a tool-parse failure", async () => {
  const output = '<tool_call>{"name":"pwsh","arguments":'
  const { adapter } = adapterWith([
    { type: "start", input_tokens: 9 },
    { type: "delta", text: output, output_tokens: 4 },
    {
      type: "result",
      output,
      input_tokens: 9,
      output_tokens: 4,
      acknowledged_deltas: 1,
      finish_reason: "stop",
    },
  ])

  const { chunks, error } = await collectFailure(adapter.stream(adapterOptions()))
  assert.deepEqual(chunks, [
    { type: "usage", usage: { inputTokens: 9, outputTokens: 4 } },
  ])
  assert.equal(error.code, "LOCAL_QWEN_TOOL_PARSE")
})

hostBackedTest("usage precedes an empty-response failure", async () => {
  const { adapter } = adapterWith([
    { type: "start", input_tokens: 7 },
    {
      type: "result",
      output: "",
      input_tokens: 7,
      output_tokens: 0,
      acknowledged_deltas: 0,
      finish_reason: "stop",
    },
  ])

  const { chunks, error } = await collectFailure(adapter.stream(adapterOptions()))
  assert.deepEqual(chunks, [
    { type: "usage", usage: { inputTokens: 7, outputTokens: 0 } },
  ])
  assert.equal(error.code, EMPTY_RESPONSE_CODE)
})

test("package, Cordis patch, and runner expose the local-only Qwen route", () => {
  const packageJson = JSON.parse(readFileSync(join(here, "../package.json"), "utf8"))
  assert.equal(packageJson.exports["./qwen-adapter"], "./qwen-adapter.mjs")
  assert.deepEqual(
    [
      "tool-policy.mjs",
      "qwen-adapter.mjs",
      "qwen-protocol.mjs",
      "qwen_probe.py",
      "qwen-worker.py",
    ].filter(
      (path) => !packageJson.files.includes(path),
    ),
    [],
  )

  const patch = readFileSync(join(here, "../cordis.patch.yml"), "utf8")
  assert.match(patch, /- id: session-title-llm\r?\n  disabled: true/)
  for (const id of [
    "session-telemetry-otel",
    "web",
    "web-search-deepseek",
    "tool-web",
  ]) {
    assert.match(
      patch,
      new RegExp(`- id: ${id}\\r?\\n  disabled: true`),
    )
  }
  assert.match(
    patch,
    /- id: llm-pi-ai\r?\n  disabled: !!js process\.env\.DSH_PROVIDER !== 'qwen-compatible'/,
  )
  assert.match(
    patch,
    /- id: llm-deepseek\r?\n  disabled: !!js process\.env\.DSH_LOCAL_QWEN === '1' \|\| process\.env\.DSH_FIXTURE === '1'/,
  )
  assert.match(
    patch,
    /id: dsh-agent-presets[\s\S]*?DSH_PRESET_ROOT[\s\S]*?trust: system[\s\S]*?includeUserRoot: false/,
  )
  assert.match(
    patch,
    /id: dsh-local-qwen-adapter[\s\S]*?name: '@token-prediction\/dsh-headless\/qwen-adapter'[\s\S]*?DSH_QWEN_PYTHON[\s\S]*?DSH_QWEN_WORKER[\s\S]*?DSH_QWEN_SNAPSHOT/,
  )
  assert.match(
    patch,
    /exclusiveProvider: !!js process\.env\.DSH_EXCLUSIVE_PROVIDER/,
  )
  assert.match(patch, /generationProbeDir: !!js process\.env\.DSH_GENERATION_PROBE_DIR/)
  assert.match(
    patch,
    /generationProbeCaptureRevisionPrefix: !!js process\.env\.DSH_GENERATION_PROBE_CAPTURE_REVISION_PREFIX/,
  )

  if (RunnerConfig) {
    const runnerConfig = RunnerConfig({
      task: "test task",
      provider: LOCAL_QWEN_PROVIDER,
      model: LOCAL_QWEN_MODEL,
      agentPreset: "router-standard",
      reasoningEffort: "off",
      temperature: 0,
      maxTokens: 64,
      stop: [],
      maxLogicalCalls: 12,
      maxProviderAccountedTokens: 30000,
      exclusiveProvider: LOCAL_QWEN_PROVIDER,
    })
    assert.equal(runnerConfig.exclusiveProvider, LOCAL_QWEN_PROVIDER)
  }

  const runner = readFileSync(join(here, "../runner.mjs"), "utf8")
  assert.match(runner, /return applyToolPolicy\(assembled, config\.provider, job\.system_prompt_policy_id\)/)
  assert.match(runner, /agentCtx\.tools\.restrict\(\{ allow: \[\.\.\.ALLOWED_TOOL_NAMES\] \}\)/)
  assert.ok(
    runner.indexOf('agentCtx.on("system-prompt/assemble"')
      < runner.indexOf("ctx.agentPresets.mount(agentCtx, config.agentPreset)"),
  )
  assert.match(runner, /ctx\.llm\.listProviders\(\)\.map\(\(provider\) => provider\.id\)/)
  assert.match(runner, /providers\.length !== 1 \|\| providers\[0\] !== config\.exclusiveProvider/)
})
