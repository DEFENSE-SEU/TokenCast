import { randomUUID } from "node:crypto"
import { spawn } from "node:child_process"
import { once } from "node:events"
import { join } from "node:path"
import { createInterface } from "node:readline"

import { importDshHostModule } from "./host.mjs"
import {
  projectQwenMessages,
  projectQwenTools,
  QwenOutputParser,
} from "./qwen-protocol.mjs"

const llmPackage = await importDshHostModule("@deepseek-ai/dsh-llm")
const schemastery = await importDshHostModule("@deepseek-ai/schemastery")
const {
  CONTEXT_WINDOW_EXCEEDED_CODE,
  EMPTY_RESPONSE_CODE,
  LlmAdapter,
  LlmError,
  resolveRetryPolicy,
} = llmPackage
const z = schemastery.default ?? schemastery

export const LOCAL_QWEN_PROVIDER = "local-qwen"
export const LOCAL_QWEN_MODEL = "Qwen/Qwen2.5-7B-Instruct"
export const LOCAL_QWEN_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
export const GENERATION_PROBE_BRIDGE = Symbol.for("token-prediction.dsh-generation-probe.v1")

function generationProbeBridgeRegistry() {
  globalThis[GENERATION_PROBE_BRIDGE] ??= new WeakMap()
  return globalThis[GENERATION_PROBE_BRIDGE]
}

function attachGenerationProbeBridge(chunk, event) {
  const bridge = event[GENERATION_PROBE_BRIDGE]
  if (bridge) generationProbeBridgeRegistry().set(chunk, bridge)
  return chunk
}

class AsyncQueue {
  constructor() {
    this.values = []
    this.waiters = []
    this.closed = false
    this.error = undefined
    this.terminal = false
  }

  push(value) {
    const waiter = this.waiters.shift()
    if (waiter) waiter.resolve({ value, done: false })
    else this.values.push(value)
  }

  close() {
    this.closed = true
    for (const waiter of this.waiters.splice(0)) waiter.resolve({ done: true })
  }

  fail(error) {
    this.error = error
    for (const waiter of this.waiters.splice(0)) waiter.reject(error)
  }

  next() {
    if (this.values.length > 0) return Promise.resolve({ value: this.values.shift(), done: false })
    if (this.error) return Promise.reject(this.error)
    if (this.closed) return Promise.resolve({ done: true })
    return new Promise((resolve, reject) => this.waiters.push({ resolve, reject }))
  }

  [Symbol.asyncIterator]() {
    return this
  }
}

class WorkerResponseError extends Error {
  constructor(message, code) {
    super(message)
    this.code = code
  }
}

export class QwenWorker {
  constructor({ python, worker, snapshot }) {
    this.python = python
    this.worker = worker
    this.snapshot = snapshot
    this.process = undefined
    this.ready = undefined
    this.pending = new Map()
    this.metadata = undefined
    this.disposing = undefined
  }

  async start() {
    if (this.disposing) await this.disposing
    if (this.ready) return this.ready
    const child = spawn(this.python, [this.worker, "--snapshot", this.snapshot], {
      stdio: ["pipe", "pipe", "inherit"],
      windowsHide: true,
      env: {
        ...process.env,
        HF_HUB_OFFLINE: "1",
        TRANSFORMERS_OFFLINE: "1",
        PYTHONUTF8: "1",
        PYTHONIOENCODING: "utf-8",
      },
    })
    this.process = child
    this.ready = new Promise((resolve, reject) => {
      const fail = (error) => {
        reject(error)
        for (const queue of this.pending.values()) queue.fail(error)
        this.pending.clear()
        if (this.process === child) {
          this.process = undefined
          this.ready = undefined
          this.metadata = undefined
        }
      }
      const lines = createInterface({ input: child.stdout })
      lines.on("line", (line) => {
        let message
        try {
          message = JSON.parse(line)
        } catch (error) {
          void this.dispose(error)
          return
        }
        if (message.type === "ready") {
          this.metadata = message
          resolve(message)
          return
        }
        const queue = this.pending.get(message.id)
        if (!queue) return
        if (message.type === "error") {
          queue.terminal = true
          this.pending.delete(message.id)
          queue.fail(new WorkerResponseError(message.message, message.code))
          return
        }
        if (message.type === "result") {
          queue.terminal = true
          queue.push({ ...message, metadata: this.metadata })
          this.pending.delete(message.id)
          queue.close()
          return
        }
        queue.push(message)
      })
      child.once("error", fail)
      child.stdin.on("error", (error) => void this.dispose(error))
      child.once("exit", (code) => fail(new Error(`local Qwen worker exited with code ${code}`)))
    })
    return this.ready
  }

  async waitUntilReady(signal) {
    signal?.throwIfAborted()
    const ready = this.start()
    if (!signal) return ready
    let abort
    const aborted = new Promise((resolve, reject) => {
      abort = () => {
        const reason = signal.reason ?? new Error("local Qwen Request aborted")
        void this.dispose(reason).then(
          () => reject(reason),
          () => reject(reason),
        )
      }
      signal.addEventListener("abort", abort, { once: true })
    })
    try {
      return await Promise.race([ready, aborted])
    } finally {
      signal.removeEventListener("abort", abort)
    }
  }

  async *stream(payload, signal) {
    await this.waitUntilReady(signal)
    if (this.pending.size > 0) throw new Error("local Qwen worker already has an active Request")
    const child = this.process
    if (!child) throw new Error("local Qwen worker disappeared before its Request")
    const id = randomUUID()
    const queue = new AsyncQueue()
    this.pending.set(id, queue)
    const abort = () => {
      queue.fail(signal.reason ?? new Error("local Qwen Request aborted"))
      void this.dispose(signal.reason)
    }
    signal?.addEventListener("abort", abort, { once: true })
    child.stdin.write(`${JSON.stringify({ ...payload, id })}\n`)
    try {
      for await (const message of queue) {
        let bridge
        if (message.type === "delta" || message.type === "flush") {
          bridge = { checkpoints: [] }
          Object.defineProperty(message, GENERATION_PROBE_BRIDGE, { value: bridge })
        }
        yield message
        if (message.type === "delta" || message.type === "flush") {
          child.stdin.write(
            `${JSON.stringify({
              type: "ack",
              id,
              output_tokens: message.output_tokens,
              generation_probe_checkpoints: bridge.checkpoints,
            })}\n`,
          )
        }
      }
    } finally {
      signal?.removeEventListener("abort", abort)
      this.pending.delete(id)
      if (!queue.terminal) {
        await this.dispose(new Error("local Qwen Request consumer ended before terminal"))
      }
    }
  }

  async dispose(reason = new Error("local Qwen worker disposed")) {
    if (this.disposing) return this.disposing
    const operation = this.stop(reason)
    this.disposing = operation
    try {
      await operation
    } finally {
      if (this.disposing === operation) this.disposing = undefined
    }
  }

  async stop(reason) {
    const child = this.process
    this.process = undefined
    this.ready = undefined
    this.metadata = undefined
    for (const queue of this.pending.values()) queue.fail(reason)
    this.pending.clear()
    if (!child || child.exitCode !== null) return
    const exited = once(child, "exit")
    child.kill()
    await exited
  }
}

export class LocalQwenAdapter extends LlmAdapter {
  constructor(config, worker = new QwenWorker(config)) {
    super()
    this.worker = worker
    this.generationProbeDir = config.generationProbeDir ?? ""
    this.generationProbeCaptureRevisionPrefix =
      config.generationProbeCaptureRevisionPrefix ?? ""
  }

  providerInfo(provider) {
    return { id: provider, name: "Local Qwen" }
  }

  providerRetryPolicy() {
    return resolveRetryPolicy({ mode: "normal", maxRetries: 0 }, "local Qwen retry policy")
  }

  listModels(provider) {
    return Promise.resolve([{ provider, id: LOCAL_QWEN_MODEL, name: "Qwen2.5 7B Instruct NF4" }])
  }

  resolveModel(provider, model) {
    if (provider !== LOCAL_QWEN_PROVIDER || model !== LOCAL_QWEN_MODEL) {
      throw new LlmError(`unexpected local Qwen route ${provider}/${model}`, "LOCAL_QWEN_ROUTE")
    }
    return Promise.resolve({
      provider,
      id: model,
      name: "Qwen2.5 7B Instruct NF4",
      context: { contextWindow: 32768 },
      defaultMaxTokens: 1024,
      reasoning: { efforts: [{ id: "off", name: "Off" }], defaultEffort: "off" },
    })
  }

  async *stream(options) {
    options.signal?.throwIfAborted()
    if (options.provider !== LOCAL_QWEN_PROVIDER || options.model !== LOCAL_QWEN_MODEL) {
      throw new LlmError("local Qwen Request route drift", "LOCAL_QWEN_ROUTE")
    }
    if (options.reasoningEffort !== "off" || (options.temperature ?? 0) !== 0) {
      throw new LlmError(
        "local Qwen DSH requires reasoning effort off and temperature 0",
        "LOCAL_QWEN_CONDITION",
      )
    }
    if (
      (this.generationProbeDir || this.generationProbeCaptureRevisionPrefix)
      && (
        !this.generationProbeDir
        || !this.generationProbeCaptureRevisionPrefix
        || typeof options.sessionId !== "string"
        || options.sessionId.length === 0
      )
    ) {
      throw new LlmError(
        "Generation Probe requires directory, capture revision prefix, and DSH session id",
        "LOCAL_QWEN_PROTOCOL",
      )
    }
    const tools = projectQwenTools(options.tools)
    const parser = new QwenOutputParser()
    let result
    let workerError
    let parseError
    let rawOutput = ""
    let visibleText = ""
    let textStarted = false
    let parsed = { tools: [], visible: [] }
    let parserFinished = false
    try {
      for await (const event of this.worker.stream(
        {
          messages: projectQwenMessages(options),
          tools,
          temperature: 0,
          max_tokens: options.maxTokens ?? 1024,
          stop: options.stop ?? [],
          ...(this.generationProbeDir
            ? {
                generation_probe: {
                  path: join(this.generationProbeDir, `${options.sessionId}.jsonl`),
                  capture_revision:
                    `${this.generationProbeCaptureRevisionPrefix}:${options.sessionId}`,
                  session_id: options.sessionId,
                  model_id: options.model,
                },
              }
            : {}),
        },
        options.signal,
      )) {
        if (event.type === "delta") {
          rawOutput += event.text
          if (!parseError) {
            try {
              for (const text of parser.push(event.text)) {
                if (!textStarted) {
                  yield { type: "block-start", index: 0, blockType: "text" }
                  textStarted = true
                }
                visibleText += text
                yield attachGenerationProbeBridge(
                  { type: "text-delta", index: 0, text },
                  event,
                )
              }
            } catch (error) {
              parseError = error
            }
          }
        }
        if (event.type === "flush" && !parseError) {
          try {
            parsed = parser.finish()
            parserFinished = true
            for (const text of parsed.visible) {
              if (!textStarted) {
                yield { type: "block-start", index: 0, blockType: "text" }
                textStarted = true
              }
              visibleText += text
              yield attachGenerationProbeBridge(
                { type: "text-delta", index: 0, text },
                event,
              )
            }
          } catch (error) {
            parseError = error
          }
        }
        if (event.type === "result") result = event
      }
    } catch (error) {
      workerError = error
    }
    if (result && !parseError && !parserFinished) {
      try {
        parsed = parser.finish()
        for (const text of parsed.visible) {
          if (!textStarted) {
            yield { type: "block-start", index: 0, blockType: "text" }
            textStarted = true
          }
          visibleText += text
          yield { type: "text-delta", index: 0, text }
        }
      } catch (error) {
        parseError ??= error
      }
    }
    if (textStarted) {
      yield { type: "block-end", index: 0, block: { type: "text", text: visibleText } }
    }
    const outputMismatch = result && rawOutput !== result.output
    const usage = result
      ? {
          inputTokens: result.input_tokens,
          outputTokens: result.output_tokens,
        }
      : undefined
    if (workerError) {
      if (usage) yield { type: "usage", usage }
      if (options.signal?.aborted) throw options.signal.reason
      const code =
        workerError instanceof WorkerResponseError
          ? workerError.code
          : "LOCAL_QWEN_INFERENCE"
      throw new LlmError(
        workerError instanceof Error ? workerError.message : String(workerError),
        code === CONTEXT_WINDOW_EXCEEDED_CODE ? CONTEXT_WINDOW_EXCEEDED_CODE : code,
      )
    }
    if (!result) throw new LlmError("local Qwen worker returned no terminal", "LOCAL_QWEN_PROTOCOL")
    if (outputMismatch) {
      yield { type: "usage", usage }
      throw new LlmError(
        "local Qwen stream and terminal output disagree",
        "LOCAL_QWEN_PROTOCOL",
      )
    }
    if (parseError && result.finish_reason !== "max-tokens") {
      yield { type: "usage", usage }
      throw new LlmError(
        parseError instanceof Error ? parseError.message : String(parseError),
        "LOCAL_QWEN_TOOL_PARSE",
      )
    }
    let index = textStarted ? 1 : 0
    if (result.finish_reason !== "max-tokens") {
      for (const call of parsed.tools) {
        const id = `call-local-qwen-${randomUUID()}`
        yield { type: "block-start", index, blockType: "tool-call" }
        yield {
          type: "tool-call-delta",
          index,
          id,
          name: call.name,
          argumentsDelta: call.arguments,
        }
        yield {
          type: "block-end",
          index,
          block: { type: "tool-call", id, name: call.name, arguments: call.arguments },
        }
        index += 1
      }
    }
    if (!visibleText.trim() && parsed.tools.length === 0 && result.finish_reason !== "max-tokens") {
      yield { type: "usage", usage }
      throw new LlmError("local Qwen returned an empty response", EMPTY_RESPONSE_CODE)
    }
    const metadata = result.metadata
    if (!metadata) {
      yield { type: "usage", usage }
      throw new LlmError("local Qwen terminal omitted runtime metadata", "LOCAL_QWEN_PROTOCOL")
    }
    yield { type: "usage", usage }
    yield {
      type: "finish",
      reason:
        result.finish_reason === "max-tokens"
          ? { kind: "max-tokens" }
          : parsed.tools.length > 0
            ? { kind: "tool-calls" }
            : { kind: "stop" },
      replayState: {
        response: {
          revision: metadata.revision,
          precision: metadata.precision,
          device: metadata.device,
          modelClass: metadata.model_class,
          isQuantized: metadata.is_quantized,
          quantType: metadata.quant_type,
          doubleQuant: metadata.double_quant,
          computeDtype: metadata.compute_dtype,
          tieWordEmbeddings: metadata.tie_word_embeddings,
          linear4bitModules: metadata.linear4bit_modules,
          lmHeadDtype: metadata.lm_head_dtype,
          repetitionPenalty: metadata.repetition_penalty,
          eosTokenIds: metadata.eos_token_ids,
          tokenizerInputTokens: result.input_tokens,
          tokenizerOutputTokens: result.output_tokens,
          acknowledgedDeltaCount: result.acknowledged_deltas,
        },
      },
    }
  }

  dispose() {
    return this.worker.dispose()
  }
}

export const name = "local-qwen-adapter"
export const inject = ["llm"]
export const Config = z.object({
  python: z.string().default("python"),
  worker: z.string().required(),
  snapshot: z.string().required(),
  generationProbeDir: z.string().default(""),
  generationProbeCaptureRevisionPrefix: z.string().default(""),
})

export function apply(ctx, config) {
  const adapter = new LocalQwenAdapter(config)
  ctx.effect(() => {
    const unregister = ctx.llm.registerAdapter([LOCAL_QWEN_PROVIDER], adapter)
    return async () => {
      unregister()
      await adapter.dispose()
    }
  }, "local-qwen-adapter")
}
