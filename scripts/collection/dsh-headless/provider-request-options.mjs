export const SUBSCRIPTION_PROVIDER = "openai-codex"
export const QWEN_PROVIDER = "qwen-compatible"

export function compatibleModelFamily(model = "") {
  if (/^(?:google\/)?gemini-/i.test(model)) return "gemini"
  if (/^(?:moonshotai\/)?kimi-/i.test(model)) return "kimi"
  if (/^(?:deepseek\/)?deepseek-v4-/i.test(model)) return "deepseek"
  if (/^(?:anthropic\/)?claude-/i.test(model)) return "claude"
  return undefined
}

function usesNativeReasoning(provider, model, effort) {
  const family = compatibleModelFamily(model)
  return provider === QWEN_PROVIDER && (
    (family !== undefined && family !== "claude")
    || (/^(?:openai\/)?gpt-5\.4$/i.test(model) && effort === "low")
  )
}

function isGpt54Low(provider, model, effort) {
  return provider === QWEN_PROVIDER
    && /^(?:openai\/)?gpt-5\.4$/i.test(model)
    && effort === "low"
}

export function validateCompatibleReasoning(model, effort) {
  const family = compatibleModelFamily(model)
  if (family === "gemini") {
    const canDisable = /^(?:google\/)?gemini-2\.5-flash(?:-lite)?$/i.test(model)
    if (!(canDisable && effort === "off") && !["minimal", "low", "medium", "high"].includes(effort)) {
      throw new Error(`Gemini model ${model} requires a supported reasoning effort; use low, medium, or high (off is limited to Gemini 2.5 Flash/Flash-Lite)`)
    }
  }
  if (family === "kimi" && (!/^(?:moonshotai\/)?kimi-k2\.5$/i.test(model) || !["off", "high"].includes(effort))) {
    throw new Error("Kimi compatibility supports kimi-k2.5 with off (instant) or high (thinking enabled)")
  }
  if (family === "deepseek" && !["off", "high"].includes(effort)) {
    throw new Error("DeepSeek V4 compatibility supports off or high reasoning")
  }
  if (family === "claude" && effort !== "off") {
    throw new Error("Claude Chat Completions compatibility currently supports off reasoning only")
  }
}


export function unsupportedRequestOptions(provider, model, effort) {
  if (provider === SUBSCRIPTION_PROVIDER) return ["stop", "temperature"]
  if (isGpt54Low(provider, model, effort)) return ["stop", "temperature"]
  if (provider === QWEN_PROVIDER) return usesNativeReasoning(provider, model, effort) ? ["stop"] : ["reasoningEffort", "stop"]
  return []
}

export function buildModelSelection(provider, model, reasoningEffort) {
  if (provider === QWEN_PROVIDER) validateCompatibleReasoning(model, reasoningEffort)
  return {
    provider,
    model,
    ...(provider === QWEN_PROVIDER && !usesNativeReasoning(provider, model, reasoningEffort) ? {} : { reasoningEffort }),
  }
}

export function buildProviderRequestOptions(provider, decoding, model) {
  if (provider === QWEN_PROVIDER) validateCompatibleReasoning(model, decoding.reasoning_effort)
  const gpt54Low = isGpt54Low(provider, model, decoding.reasoning_effort)
  if (gpt54Low && decoding.temperature !== 0) {
    throw new Error("GPT-5.4 low does not support an explicit temperature")
  }
  const options = {
    ...(provider === QWEN_PROVIDER && !usesNativeReasoning(provider, model, decoding.reasoning_effort)
      ? {}
      : { reasoningEffort: decoding.reasoning_effort }),
    ...(!gpt54Low ? { temperature: decoding.temperature } : {}),
    maxTokens: decoding.max_tokens,
  }
  if (provider === SUBSCRIPTION_PROVIDER) {
    if (decoding.stop.length > 0) {
      throw new Error("the subscription route does not support stop strings")
    }
    if (decoding.temperature !== 0) {
      throw new Error("the subscription route samples at its own fixed temperature")
    }
    const { temperature: _omitted, ...supported } = options
    return supported
  }
  return { ...options, stop: [...decoding.stop] }
}
