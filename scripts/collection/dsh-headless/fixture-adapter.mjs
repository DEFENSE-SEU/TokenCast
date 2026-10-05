import { importDshHostModule } from "./host.mjs"

const llmPackage = await importDshHostModule("@deepseek-ai/dsh-llm")
const schemastery = await importDshHostModule("@deepseek-ai/schemastery")
const {
  LlmAdapter,
  LlmError,
  isAgentLoopRequest,
  resolveRetryPolicy,
} = llmPackage
const z = schemastery.default ?? schemastery

export const FIXTURE_PROVIDER = "dsh-fixture"
export const FIXTURE_MODEL = "dsh-fixture-model"
export const FIXTURE_RETRY_CODE = "DSH_FIXTURE_RETRY"

export class FixtureAdapter extends LlmAdapter {
  constructor(provider = FIXTURE_PROVIDER, model = FIXTURE_MODEL) {
    super()
    this.provider = provider
    this.model = model
    this.agentAttemptCountBySession = new Map()
  }

  providerRetryPolicy() {
    return resolveRetryPolicy(
      {
        mode: "normal",
        maxRetries: 1,
        retryableCodes: [FIXTURE_RETRY_CODE],
        backoff: { initialDelayMs: 1, maxDelayMs: 1, jitterRatio: 0 },
      },
      "DSH fixture retry policy",
    )
  }

  resolveModel(provider, model) {
    if (provider !== this.provider || model !== this.model) {
      throw new LlmError(`unexpected fixture route ${provider}/${model}`, "DSH_FIXTURE_ROUTE")
    }
    return Promise.resolve({
      provider,
      id: model,
      name: model,
      context: { contextWindow: 4096 },
      reasoning: { efforts: [{ id: "off", name: "Off" }, { id: "high", name: "High" }] },
    })
  }

  async *stream(options) {
    options.signal?.throwIfAborted()
    if (!isAgentLoopRequest(options)) {
      yield { type: "block-start", index: 0, blockType: "text" }
      yield { type: "text-delta", index: 0, text: "DSH fixture" }
      yield {
        type: "block-end",
        index: 0,
        block: { type: "text", text: "DSH fixture" },
      }
      yield { type: "usage", usage: { inputTokens: 1, outputTokens: 2 } }
      yield { type: "finish", reason: { kind: "stop" } }
      return
    }

    const expectedCondition =
      options.provider === this.provider
      && options.model === this.model
      && options.reasoningEffort === "off"
      && options.temperature === 0
      && options.maxTokens === 64
      && Array.isArray(options.stop)
      && options.stop.length === 0
    if (!expectedCondition) {
      throw new LlmError("DSH fixture Request condition drift", "DSH_FIXTURE_CONDITION")
    }

    const sessionId = options.sessionId
    if (typeof sessionId !== "string" || sessionId.length === 0) {
      throw new LlmError("DSH fixture Request session is missing", "DSH_FIXTURE_CONDITION")
    }
    const attemptCount = (this.agentAttemptCountBySession.get(sessionId) ?? 0) + 1
    this.agentAttemptCountBySession.set(sessionId, attemptCount)
    if (attemptCount === 1) {
      yield {
        type: "usage",
        usage: {
          inputTokens: 10,
          outputTokens: 2,
          cacheReadTokens: 3,
          cacheWriteTokens: 1,
          reasoningTokens: 0,
        },
      }
      throw new LlmError("deterministic DSH retry", FIXTURE_RETRY_CODE)
    }
    if (attemptCount !== 2) {
      throw new LlmError(
        `unexpected DSH fixture Attempt ${attemptCount}`,
        "DSH_FIXTURE_OVERFLOW",
      )
    }

    const text = "fixture complete"
    yield { type: "block-start", index: 0, blockType: "text" }
    yield { type: "text-delta", index: 0, text: "fixture" }
    yield { type: "text-delta", index: 0, text: " " }
    yield { type: "text-delta", index: 0, text: "complete" }
    yield { type: "block-end", index: 0, block: { type: "text", text } }
    yield {
      type: "usage",
      usage: {
        inputTokens: 20,
        outputTokens: 4,
        cacheReadTokens: 5,
        cacheWriteTokens: 2,
        reasoningTokens: 0,
      },
    }
    yield { type: "finish", reason: { kind: "stop" } }
  }
}

export const name = "dsh-fixture-adapter"
export const inject = ["llm"]
export const Config = z.object({
  provider: z.string().default(FIXTURE_PROVIDER),
  model: z.string().default(FIXTURE_MODEL),
})

export function apply(ctx, config = {}) {
  const provider = config.provider ?? FIXTURE_PROVIDER
  const model = config.model ?? FIXTURE_MODEL
  ctx.llm.registerAdapter([provider], new FixtureAdapter(provider, model))
}
