import { readFileSync } from "node:fs";
import { mkdir, mkdtemp, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

import {
  DshStreamObserver,
  createStreamMiddleware,
} from "../observer.mjs";

const outputDir = resolve(
  process.argv[2] ?? join(tmpdir(), "dsh-capture"),
);
const dshRoot = resolve(
  process.argv[3]
    ?? join(process.env.APPDATA ?? "", "npm", "node_modules", "@deepseek-ai", "dsh"),
);
const packageRoot = join(dshRoot, "node_modules", "@deepseek-ai");
const runtimeRoot = await mkdtemp(join(tmpdir(), "token-prediction-dsh-fixture-"));
const projectRoot = join(runtimeRoot, "fixture-project");

const importDshPackage = (name) =>
  import(pathToFileURL(join(packageRoot, name, "lib", "index.js")));
const [
  cordis,
  sessionPackage,
  agentPackage,
  systemPromptPackage,
  toolPackage,
  llmPackage,
  agentLoopPackage,
  persistencePackage,
] = await Promise.all([
  importDshPackage("cordis"),
  importDshPackage("dsh-session"),
  importDshPackage("dsh-agent"),
  importDshPackage("dsh-system-prompt"),
  importDshPackage("dsh-tools"),
  importDshPackage("dsh-llm"),
  importDshPackage("dsh-agent-loop"),
  importDshPackage("dsh-session-persistence-jsonl"),
]);
const { Context } = cordis;
const { default: SessionStore, SessionId, packChunkRuns } = sessionPackage;
const { default: AgentRegistry } = agentPackage;
const { default: SystemPrompt } = systemPromptPackage;
const { default: ToolRuntime } = toolPackage;
const {
  default: LlmRuntime,
  CallId,
  LlmAdapter,
  LlmError,
  createUserMessage,
  isAgentLoopRequest,
} = llmPackage;
const { default: AgentLoop } = agentLoopPackage;
const { default: JsonlSessionPersistence } = persistencePackage;

const baseTime = 1_780_000_000_000;
let clock = baseTime;
const now = () => clock++;
const sessionId = SessionId("session-token-prediction-fixture-v1");
const provider = "fixture-provider";
const model = "fixture-model";
const toolCallId = CallId("tool-call-fixture-0");
const taskText = "Repair the deterministic fixture and verify it.";

class DeterministicFixtureAdapter extends LlmAdapter {
  callCount = 0;

  resolveModel(route, modelId) {
    return Promise.resolve({
      provider: route,
      id: modelId,
      name: modelId,
      context: { contextWindow: 4096 },
      reasoning: {
        efforts: [{ id: "high", name: "High" }],
      },
    });
  }

  async *stream(options) {
    options.signal?.throwIfAborted();
    this.callCount += 1;

    if (this.callCount === 1) {
      yield {
        type: "usage",
        usage: {
          inputTokens: 10,
          outputTokens: 2,
          cacheReadTokens: 3,
          cacheWriteTokens: 1,
          reasoningTokens: 1,
        },
      };
      throw new LlmError("deterministic transient failure", "FIXTURE_RETRY");
    }

    if (this.callCount === 2) {
      const text = "fix is ready";
      yield { type: "block-start", index: 0, blockType: "text" };
      yield { type: "text-delta", index: 0, text: "fix " };
      yield { type: "text-delta", index: 0, text: "is " };
      yield { type: "text-delta", index: 0, text: "ready" };
      yield { type: "block-end", index: 0, block: { type: "text", text } };
      yield { type: "block-start", index: 1, blockType: "tool-call" };
      yield {
        type: "tool-call-delta",
        index: 1,
        id: toolCallId,
        name: "shell",
        argumentsDelta: '{"command":"verify"}',
      };
      yield {
        type: "block-end",
        index: 1,
        block: {
          type: "tool-call",
          id: toolCallId,
          name: "shell",
          arguments: '{"command":"verify"}',
        },
      };
      yield {
        type: "usage",
        usage: {
          inputTokens: 20,
          outputTokens: 8,
          cacheReadTokens: 5,
          cacheWriteTokens: 2,
          reasoningTokens: 2,
        },
      };
      yield { type: "finish", reason: { kind: "tool-calls" } };
      return;
    }

    if (this.callCount === 3) {
      yield { type: "block-start", index: 0, blockType: "text" };
      yield { type: "text-delta", index: 0, text: "all" };
      yield { type: "text-delta", index: 0, text: " " };
      yield { type: "text-delta", index: 0, text: "done" };
      yield {
        type: "block-end",
        index: 0,
        block: { type: "text", text: "all done" },
      };
      yield {
        type: "usage",
        usage: {
          inputTokens: 30,
          outputTokens: 4,
          cacheReadTokens: 10,
          cacheWriteTokens: 0,
          reasoningTokens: 0,
        },
      };
      yield { type: "finish", reason: { kind: "stop" } };
      return;
    }

    throw new LlmError(`unexpected fixture adapter call ${this.callCount}`, "FIXTURE_CALL_OVERFLOW");
  }
}

function activeStep(ctx, requestedSessionId) {
  const session = ctx.sessions.get(SessionId(requestedSessionId));
  if (session === undefined) return undefined;
  for (let index = session.events.length - 1; index >= 0; index -= 1) {
    const event = session.events[index];
    if (event.type === "step/start") {
      return { turn: event.data.turn, step: event.data.step };
    }
  }
  return undefined;
}

function mapRuntimeMessageIds(events) {
  const cloned = structuredClone(events);
  const replacements = [
    "message-user-fixture-0",
    "message-assistant-fixture-0",
    "message-tool-fixture-0",
    "message-assistant-fixture-1",
  ];
  const idMap = new Map();
  const uuid = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

  function visit(value) {
    if (Array.isArray(value)) {
      for (const item of value) visit(item);
      return;
    }
    if (value === null || typeof value !== "object") return;
    for (const [key, child] of Object.entries(value)) {
      if (key === "id" && typeof child === "string" && uuid.test(child)) {
        if (!idMap.has(child)) {
          const replacement = replacements[idMap.size];
          if (replacement === undefined) throw new Error("unexpected extra runtime message id");
          idMap.set(child, replacement);
        }
        value[key] = idMap.get(child);
      } else {
        visit(child);
      }
    }
  }

  visit(cloned);
  if (idMap.size !== replacements.length) {
    throw new Error(`expected ${replacements.length} runtime message ids, got ${idMap.size}`);
  }
  return { events: cloned, idMap };
}

async function findFile(root, name) {
  for (const entry of await readdir(root, { withFileTypes: true })) {
    const path = join(root, entry.name);
    if (entry.isDirectory()) {
      const found = await findFile(path, name);
      if (found !== undefined) return found;
    } else if (entry.name === name) {
      return path;
    }
  }
  return undefined;
}

function packageVersion(name) {
  return JSON.parse(readFileSync(join(packageRoot, name, "package.json"), "utf8")).version;
}

await rm(outputDir, { recursive: true, force: true });
await mkdir(outputDir, { recursive: true });
await mkdir(projectRoot, { recursive: true });

const originalNow = Date.now;
Date.now = now;
const ctx = new Context();
let handle;
let summary;

try {
  await ctx.plugin(SessionStore);
  await ctx.plugin(AgentRegistry);
  await ctx.plugin(SystemPrompt, {
    includeHarnessIdentity: false,
    includeRuntimeContext: false,
    persona: "You are a deterministic fixture Agent.",
  });
  await ctx.plugin(ToolRuntime, { mode: "native" });
  await ctx.plugin(LlmRuntime);
  await ctx.plugin(JsonlSessionPersistence, {
    root: join(runtimeRoot, "sessions"),
    compression: "none",
    packChunks: false,
    writeBatchMaxDelayMs: 1,
  });
  await ctx.plugin(AgentLoop, { agents: [], maxParallelToolCalls: 1 });

  const adapter = new DeterministicFixtureAdapter();
  ctx.llm.registerAdapter([provider], adapter);
  ctx.tools.register({
    name: "shell",
    description: "Run a deterministic verification command.",
    parameters: {
      type: "object",
      properties: { command: { type: "string" } },
    },
    output: {
      schema: { type: "string" },
      render(_args, value) {
        return [{ type: "text", text: value }];
      },
    },
    async execute() {
      return "verification passed";
    },
  });

  const observerRecords = [];
  const missingSteps = [];
  const observer = new DshStreamObserver({
    checkpointIntervalBytes: 4,
    now,
    writeRecord: (_id, record) => observerRecords.push(record),
  });
  const observerMiddleware = createStreamMiddleware({
    observer,
    isAgentLoopRequest,
    activeStep: (id) => activeStep(ctx, id),
    onMissingStep: (id) => missingSteps.push(id),
  });
  ctx.on("llm/stream", observerMiddleware, { global: true });

  let retryCount = 0;
  handle = await ctx.agents.create({
    sessionId,
    meta: { cwd: projectRoot, agentPreset: "router-standard" },
    agentOptions: { provider, model, maxTokens: 64 },
    setup(agentCtx) {
      agentCtx.on("agent/request", async (_payload, next) => ({
        ...(await next()),
        reasoningEffort: "high",
        temperature: 0,
      }));
      agentCtx.on("agent/request-error", (payload) => {
        if (payload.failure.code !== "FIXTURE_RETRY" || retryCount !== 0) return undefined;
        retryCount += 1;
        return { kind: "retry" };
      });
    },
  });

  handle.agent.followup(
    createUserMessage({
      content: [{ type: "text", text: taskText }],
      source: { kind: "user" },
    }),
  );
  await handle.agent.whenIdle();
  const session = handle.agent.session;
  await ctx.sessions.flush(session);

  if (adapter.callCount !== 3 || retryCount !== 1 || missingSteps.length !== 0) {
    throw new Error(
      `capture topology mismatch: calls=${adapter.callCount}, retries=${retryCount}, missingSteps=${missingSteps.length}`,
    );
  }
  const starts = observerRecords.filter((record) => record.record_type === "attempt/start");
  const ends = observerRecords.filter((record) => record.record_type === "attempt/end");
  if (starts.length !== 3 || ends.length !== 3) {
    throw new Error("observer did not capture exactly three complete Attempts");
  }

  const sanitized = mapRuntimeMessageIds(session.events);
  for (const event of sanitized.events) event.time = baseTime + 1 + event.seq;
  const observerTimeOffsets = [1, 7, 8, 9, 11, 13, 19, 24, 27, 28, 31];
  if (observerRecords.length !== observerTimeOffsets.length) {
    throw new Error("fixture observer record count changed");
  }
  for (let index = 0; index < observerRecords.length; index += 1) {
    observerRecords[index] = structuredClone(observerRecords[index]);
    observerRecords[index].time = baseTime + observerTimeOffsets[index];
  }
  const manifest = {
    schema_version: 1,
    capture_id: "dsh-observer-fixture-v1",
    session_id: sessionId,
    task: {
      task_id: "fixture:repo-repair:0",
      run_id: "fixture:repo-repair:0:run:0",
      prediction_context_id: "fixture:repo-repair:0:launch",
      condition_id: "condition:dsh-fixture-v1",
      task_text: taskText,
      task_type: "repo_repair",
    },
    condition: {
      agent_id: "deepseek-harness",
      agent_version: packageVersion("dsh-agent-loop"),
      configured_provider: provider,
      configured_model: model,
      reasoning_effort: "high",
      router_mode: "standard",
      router_version: "0.3.0",
      run_policy_id: "deterministic-adapter-v1",
    },
    runtime: {
      dsh_cli_version: JSON.parse(readFileSync(join(dshRoot, "package.json"), "utf8")).version,
      batch_protocol_version: 2,
      dsh_components: {
        "@deepseek-ai/dsh-agent": packageVersion("dsh-agent"),
        "@deepseek-ai/dsh-agent-presets": packageVersion("dsh-agent-presets"),
        "@deepseek-ai/dsh-agent-loop": packageVersion("dsh-agent-loop"),
        "@deepseek-ai/dsh-headless": packageVersion("dsh-headless"),
        "@deepseek-ai/dsh-llm": packageVersion("dsh-llm"),
        "@deepseek-ai/dsh-llm-deepseek": packageVersion("dsh-llm-deepseek"),
        "@deepseek-ai/dsh-llm-retry": packageVersion("dsh-llm-retry"),
        "@deepseek-ai/dsh-session": packageVersion("dsh-session"),
        "@deepseek-ai/dsh-session-checkpoint-policy": packageVersion(
          "dsh-session-checkpoint-policy",
        ),
        "@deepseek-ai/dsh-session-persistence": packageVersion("dsh-session-persistence"),
        "@deepseek-ai/dsh-session-persistence-jsonl": packageVersion(
          "dsh-session-persistence-jsonl",
        ),
        "@deepseek-ai/dsh-token-meter": packageVersion("dsh-token-meter"),
        "@deepseek-ai/dsh-tools": packageVersion("dsh-tools"),
      },
      attempt_boundary_contract: "llm-stream-one-provider-attempt-v1",
      single_attempt_boundary: true,
    },
    terminal: {
      event_type: "task_finished",
      outcome: "fixture_success",
      reason: "deterministic_runner_completed",
      occurred_at: new Date(baseTime + 1_000).toISOString(),
      attempt_ledger_complete: true,
      opaque_retry_free: true,
    },
  };

  const headerLine = {
    type: "session",
    version: session.header.version,
    id: session.header.id,
    createdAt: baseTime,
    cwd: "C:\\fixture\\token-prediction",
    delegationDepth: session.header.delegationDepth ?? 0,
    agentPreset: session.header.agentPreset,
  };
  const sessionRows = packChunkRuns(sanitized.events);
  await writeFile(
    join(outputDir, "session.jsonl"),
    `${[headerLine, ...sessionRows].map((row) => JSON.stringify(row)).join("\n")}\n`,
    "utf8",
  );
  await writeFile(
    join(outputDir, "observer.jsonl"),
    `${observerRecords.map((record) => JSON.stringify(record)).join("\n")}\n`,
    "utf8",
  );
  await writeFile(
    join(outputDir, "capture.json"),
    `${JSON.stringify(manifest, null, 2)}\n`,
    "utf8",
  );

  const persistedPath = await findFile(join(runtimeRoot, "sessions"), "session.jsonl");
  if (persistedPath === undefined) throw new Error("real JSONL persistence artifact is missing");
  const persistedLines = readFileSync(persistedPath, "utf8").trimEnd().split("\n").length;
  summary = {
    outputDir,
    persistedPath,
    persistedLines,
    sessionRows: sessionRows.length,
    liveEvents: session.events.length,
    adapterCalls: adapter.callCount,
    retryCount,
    attempts: observerRecords
      .filter((record) => record.record_type === "attempt/end")
      .map((record) => ({
        turn: record.turn,
        step: record.step,
        attemptOrdinal: record.attempt_ordinal,
        finishReason: record.finish_reason,
        usage: record.usage,
        visibleBytes: record.observer_visible_output_tokens,
      })),
  };
} finally {
  if (handle !== undefined) await handle.dispose();
  await ctx.fiber.dispose();
  Date.now = originalNow;
}

console.log(JSON.stringify(summary, null, 2));
await rm(runtimeRoot, { recursive: true, force: true });
