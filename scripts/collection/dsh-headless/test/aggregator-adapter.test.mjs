import assert from "node:assert/strict"
import test from "node:test"
import { findPackageJSON } from "node:module"
import { dirname, join } from "node:path"
import { pathToFileURL } from "node:url"

import { apply, repairChunk, repairRequestBody, repairResponse, repairSseLine } from "../aggregator-adapter.mjs"
import { resolveDshHostModuleUrl } from "../host.mjs"

const FIRST = {
  choices: [
    {
      delta: {
        tool_calls: [
          { index: 0, type: "function", id: "call_3B2q", function: { name: "run_shell" } },
        ],
      },
    },
  ],
}
const LATER = {
  choices: [
    {
      delta: {
        tool_calls: [
          { index: 0, type: "function", id: "", function: { name: "", arguments: '{"' } },
        ],
      },
    },
  ],
}

test("the first delta keeps its id and name", () => {
  const call = repairChunk(FIRST).choices[0].delta.tool_calls[0]
  assert.equal(call.id, "call_3B2q")
  assert.equal(call.function.name, "run_shell")
})

test("a later delta no longer clears the id and name", () => {
  const call = repairChunk(LATER).choices[0].delta.tool_calls[0]
  assert.ok(!("id" in call), "blank id must be dropped, not passed through")
  assert.ok(!("name" in call.function), "blank name must be dropped")
})

test("argument fragments survive the repair", () => {
  const call = repairChunk(LATER).choices[0].delta.tool_calls[0]
  assert.equal(call.function.arguments, '{"')
})

test("merging the repaired sequence recovers the whole call", () => {
  const merged = {}
  for (const chunk of [FIRST, LATER].map(repairChunk)) {
    const call = chunk.choices[0].delta.tool_calls[0]
    if (call.id !== undefined) merged.id = call.id
    if (call.function?.name !== undefined) merged.name = call.function.name
    if (call.function?.arguments !== undefined) {
      merged.arguments = (merged.arguments ?? "") + call.function.arguments
    }
  }
  assert.deepEqual(merged, { id: "call_3B2q", name: "run_shell", arguments: '{"' })
})

test("a chunk without tool calls is returned unchanged", () => {
  const plain = { choices: [{ delta: { content: "hello" } }] }
  assert.equal(repairChunk(plain), plain)
})

test("a malformed chunk does not throw", () => {
  assert.equal(repairChunk(null), null)
  assert.deepEqual(repairChunk({}), {})
})

test("an SSE line with blank fields is rewritten", () => {
  const line =
    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"","function":{"name":"","arguments":"{\\""}}]}}]}'
  const out = repairSseLine(line)
  const call = JSON.parse(out.slice(5)).choices[0].delta.tool_calls[0]
  assert.ok(!("id" in call))
  assert.ok(!("name" in call.function))
  assert.equal(call.function.arguments, '{"')
})

test("lines that are not data frames pass through untouched", () => {
  assert.equal(repairSseLine("data: [DONE]"), "data: [DONE]")
  assert.equal(repairSseLine(""), "")
  assert.equal(repairSseLine("event: ping"), "event: ping")
  assert.equal(repairSseLine("data: not json"), "data: not json")
})

test("a stream is repaired across chunk boundaries", async () => {
  const body = new ReadableStream({
    start(controller) {
      const encoder = new TextEncoder()
      controller.enqueue(
        encoder.encode(
          'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_x","fun',
        ),
      )
      controller.enqueue(encoder.encode('ction":{"name":"run_shell"}}]}}]}\n'))
      controller.enqueue(
        encoder.encode(
          'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"","function":{"name":"","arguments":"{}"}}]}}]}\ndata: [DONE]\n',
        ),
      )
      controller.close()
    },
  })
  const response = new Response(body, {
    headers: { "content-type": "text/event-stream" },
  })
  const text = await repairResponse(response).text()
  const frames = text
    .split("\n")
    .filter((line) => line.startsWith("data:") && !line.includes("[DONE]"))
    .map((line) => JSON.parse(line.slice(5)).choices[0].delta.tool_calls[0])
  assert.equal(frames[0].id, "call_x")
  assert.equal(frames[0].function.name, "run_shell")
  assert.ok(!("id" in frames[1]), "second frame must not carry a blank id")
  assert.equal(frames[1].function.arguments, "{}")
})

function geminiCall(id, signature, index = 0) {
  return {
    index, id, type: "function", function: { name: "lookup", arguments: "{}" },
    extra_content: { google: { thought_signature: signature } },
  }
}

function responseOf(chunks) {
  return new Response(
    chunks.map((chunk) => `data: ${JSON.stringify(chunk)}\n\n`).join("") + "data: [DONE]\n\n",
    { headers: { "content-type": "text/event-stream" } },
  )
}

async function repairedFrames(chunks) {
  return (await repairResponse(responseOf(chunks)).text()).split("\n")
    .filter((line) => line.startsWith("data:") && !line.includes("[DONE]"))
    .map((line) => JSON.parse(line.slice(5)))
}

test("Gemini signatures cross the replay bridge without replacing existing metadata", () => {
  const prior = { type: "reasoning.encrypted", id: "call_a", data: "original-encrypted", format: "provider-format" }
  const incoming = { choices: [{ delta: {
    tool_calls: [geminiCall("call_a", "opaque+/=signature")], reasoning_details: [prior],
  } }] }
  const original = structuredClone(incoming)
  const bridged = repairChunk(incoming).choices[0].delta
  assert.deepEqual(incoming, original)
  assert.deepEqual(bridged.reasoning_details[0], prior)
  const carrier = bridged.reasoning_details.at(-1)
  assert.equal(carrier.id, "call_a")
  assert.equal(carrier.data, "opaque+/=signature")
  assert.equal(carrier.type, "reasoning.encrypted")
  const request = { model: "google/gemini-3.5-flash", messages: [{
    role: "assistant", content: null,
    tool_calls: [{ id: "call_a", function: { name: "lookup", arguments: "{}" },
      extra_content: { provider_note: "retain", google: { other: "retain-too" } } }],
    reasoning_details: [carrier],
  }] }
  const repaired = repairRequestBody(request)
  assert.equal(repaired.messages[0].tool_calls[0].extra_content.google.thought_signature, "opaque+/=signature")
  assert.equal(repaired.messages[0].tool_calls[0].extra_content.google.other, "retain-too")
  assert.equal(repaired.messages[0].tool_calls[0].extra_content.provider_note, "retain")
  assert.deepEqual(repaired.messages[0].reasoning_details, [prior])
  assert.equal(request.messages[0].tool_calls[0].extra_content.google.thought_signature, undefined)
})

test("parallel tool calls and choices retain separate signatures when IDs arrive later", async () => {
  const frames = await repairedFrames([
    { choices: [
      { index: 0, delta: { tool_calls: [geminiCall("", "signature-zero", 0), geminiCall("call_b", "signature-one", 1)] } },
      { index: 1, delta: { tool_calls: [geminiCall("call_c", "signature-other-choice", 0)] } },
    ] },
    { choices: [{ index: 0, delta: { tool_calls: [{ index: 0, id: "call_a" }] } }] },
    { choices: [{ index: 0, delta: { tool_calls: [{ index: 0, function: { arguments: " " } }] } }] },
  ])
  assert.equal(frames[0].choices[0].delta.reasoning_details[0].id, "call_b")
  assert.equal(frames[0].choices[1].delta.reasoning_details[0].data, "signature-other-choice")
  assert.equal(frames[1].choices[0].delta.reasoning_details[0].id, "call_a")
  assert.equal(frames[1].choices[0].delta.reasoning_details[0].data, "signature-zero")
  assert.equal(frames[2].choices[0].delta.reasoning_details, undefined,
    "argument-only chunks should not repeatedly duplicate the opaque signature")
})

test("a later encrypted provider detail preserves the previously received Gemini signature", async () => {
  const prior = { type: "reasoning.encrypted", id: "call_a", data: "provider-detail" }
  const frames = await repairedFrames([
    { choices: [{ delta: { tool_calls: [geminiCall("call_a", "native-signature")] } }] },
    { choices: [{ delta: { reasoning_details: [prior] } }] },
  ])
  const latest = frames[1].choices[0].delta.reasoning_details.at(-1)
  assert.equal(latest.data, "native-signature")
  assert.deepEqual(latest.original_reasoning_details, [prior])
})

test("unsigned tools and unrelated provider reasoning remain unchanged", () => {
  const body = { messages: [{ role: "assistant", tool_calls: [{ id: "a" }],
    reasoning_details: [{ type: "reasoning.encrypted", id: "a", data: "provider-owned" }] }] }
  assert.equal(repairRequestBody(body), body)
  const signed = repairChunk({ choices: [{ delta: { tool_calls: [geminiCall("call_a", "signature")] } }] })
  const carrier = signed.choices[0].delta.reasoning_details[0]
  const restored = repairRequestBody({ messages: [{ role: "assistant", tool_calls: [{ id: "call_a" }],
    reasoning_details: [carrier] }] })
  assert.equal(restored.messages[0].reasoning_details, undefined)
  assert.equal(restored.messages[0].tool_calls[0].extra_content.google.thought_signature, "signature")
})

test("conflicting or orphaned replay signatures fail without exposing opaque values", () => {
  const carrier = repairChunk({ choices: [{ delta: {
    tool_calls: [geminiCall("call_a", "opaque-secret-a")],
  } }] }).choices[0].delta.reasoning_details[0]
  assert.throws(() => repairRequestBody({ messages: [{ role: "assistant",
    tool_calls: [geminiCall("call_a", "opaque-secret-b")], reasoning_details: [carrier] }] }),
  /^Error: Conflicting Gemini tool-call signatures in replay$/)
  assert.throws(() => repairRequestBody({ messages: [{ role: "assistant", tool_calls: [],
    reasoning_details: [carrier] }] }), /^Error: Gemini signature replay has no matching tool call$/)
})

test("fetch restores Request bodies and leaves unrelated endpoints untouched", async (t) => {
  const carrier = repairChunk({ choices: [{ delta: {
    tool_calls: [geminiCall("call_a", "request-signature")],
  } }] }).choices[0].delta.reasoning_details[0]
  const body = JSON.stringify({ messages: [{ role: "assistant", tool_calls: [{ id: "call_a" }],
    reasoning_details: [carrier] }] })
  const observed = []
  t.mock.method(globalThis, "fetch", async (input, init) => {
    observed.push({ input, init, body: init?.body ?? await input.clone().text() })
    return new Response("{}", { headers: { "content-type": "application/json" } })
  })
  let dispose
  apply({ effect: (start) => { dispose = start() } }, { host: "https://gateway.example.test/v1" })
  try {
    const request = new Request("https://gateway.example.test/v1/chat/completions", {
      method: "POST", body, headers: { "content-type": "application/json", "content-length": String(body.length) },
    })
    await fetch(request)
    const other = new Request("https://unrelated.example.test/v1/chat/completions", { method: "POST", body })
    await fetch(other)
    assert.equal(JSON.parse(observed[0].body).messages[0].tool_calls[0].extra_content.google.thought_signature, "request-signature")
    assert.equal(observed[0].input.headers.has("content-length"), false)
    assert.equal(observed[1].input, other)
    assert.equal(observed[1].body, body)
  } finally {
    dispose()
  }
})

test("installed pi-ai preserves the Gemini signature through two mocked streaming calls", async (t) => {
  let stream
  try {
    const adapterUrl = resolveDshHostModuleUrl("@deepseek-ai/dsh-llm-pi-ai")
    const piManifest = findPackageJSON("@earendil-works/pi-ai", adapterUrl)
    if (piManifest === undefined) {
      t.skip("The installed pi-ai dependency is unavailable")
      return
    }
    const installed = await import(pathToFileURL(join(dirname(piManifest), "dist/api/openai-completions.js")).href)
    stream = installed.stream
  } catch (error) {
    if (!["MODULE_NOT_FOUND", "ERR_MODULE_NOT_FOUND"].includes(error.code)) throw error
    t.skip("The installed pi-ai dependency is unavailable")
    return
  }
  const payloads = []
  t.mock.method(globalThis, "fetch", async (input, init) => {
    assert.match(String(input), /^https:\/\/gateway\.example\.test\/v1\/chat\/completions/)
    payloads.push(JSON.parse(init.body))
    if (payloads.length === 1) {
      return responseOf([
        { id: "fixture-response", choices: [{ index: 0, delta: {
          tool_calls: [{ index: 0, id: "call_a", type: "function", function: { name: "lookup", arguments: "{}" } }],
        }, finish_reason: null }] },
        { id: "fixture-response", choices: [{ index: 0, delta: { tool_calls: [{ index: 0,
          extra_content: { google: { thought_signature: "opaque+/=roundtrip" } },
        }] }, finish_reason: "tool_calls" }], usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 } },
      ])
    }
    assert.equal(payloads.length, 2, "the fixture permits exactly two intercepted requests")
    return responseOf([{ id: "fixture-final", choices: [{ index: 0, delta: { content: "done" }, finish_reason: "stop" }],
      usage: { prompt_tokens: 6, completion_tokens: 1, total_tokens: 7 } }])
  })
  let dispose
  apply({ effect: (start) => { dispose = start() } }, { host: "https://gateway.example.test/v1" })
  try {
    const model = {
      id: "google/gemini-3.5-flash", name: "fixture", api: "openai-completions", provider: "qwen-compatible",
      baseUrl: "https://gateway.example.test/v1", reasoning: true, input: ["text"], contextWindow: 10000, maxTokens: 64,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      compat: { thinkingFormat: "openai", supportsDeveloperRole: false, supportsStore: false, maxTokensField: "max_tokens" },
    }
    const user = { role: "user", content: "look up fixture", timestamp: 1 }
    const tools = [{ name: "lookup", description: "fixture", parameters: { type: "object", properties: {} } }]
    const options = { apiKey: "fake-key", maxTokens: 64, maxRetries: 0, env: {} }
    const first = await stream(model, { messages: [user], tools }, options).result()
    assert.equal(first.stopReason, "toolUse")
    const call = first.content.find((block) => block.type === "toolCall")
    assert.equal(JSON.parse(call.thoughtSignature).data, "opaque+/=roundtrip")
    const tool = { role: "toolResult", toolCallId: call.id, toolName: call.name,
      content: [{ type: "text", text: "fixture result" }], isError: false, timestamp: 2 }
    const second = await stream(model, { messages: [user, first, tool], tools }, options).result()
    assert.equal(second.stopReason, "stop")
    const assistant = payloads[1].messages.find((message) => message.role === "assistant")
    assert.equal(assistant.tool_calls[0].extra_content.google.thought_signature, "opaque+/=roundtrip")
    assert.equal(assistant.reasoning_details, undefined)
    assert.equal(first.usage.totalTokens, 5)
    assert.equal(second.usage.totalTokens, 7)
  } finally {
    dispose()
  }
})
