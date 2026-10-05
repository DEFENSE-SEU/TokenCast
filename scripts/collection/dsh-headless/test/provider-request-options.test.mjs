import assert from "node:assert/strict"
import test from "node:test"

import {
  QWEN_PROVIDER,
  SUBSCRIPTION_PROVIDER,
  buildModelSelection,
  buildProviderRequestOptions,
  unsupportedRequestOptions,
  validateCompatibleReasoning,
} from "../provider-request-options.mjs"

const decoding = {
  reasoning_effort: "high",
  temperature: 0,
  max_tokens: 64,
  stop: [],
}

test("subscription requests omit both stop and temperature", () => {
  assert.deepEqual(buildProviderRequestOptions(SUBSCRIPTION_PROVIDER, decoding), {
    reasoningEffort: "high",
    maxTokens: 64,
  })
})

test("a subscription request refuses a temperature it cannot honour", () => {
  assert.throws(
    () => buildProviderRequestOptions(
      SUBSCRIPTION_PROVIDER,
      { ...decoding, temperature: 0.7 },
    ),
    /fixed temperature/,
  )
})

test("subscription requests reject configured stop strings", () => {
  assert.throws(
    () => buildProviderRequestOptions(
      SUBSCRIPTION_PROVIDER,
      { ...decoding, stop: ["done"] },
    ),
    /does not support stop strings/,
  )
})

test("other providers retain their stop option", () => {
  assert.deepEqual(buildProviderRequestOptions("dsh-fixture", decoding), {
    reasoningEffort: "high",
    temperature: 0,
    maxTokens: 64,
    stop: [],
  })
})

test("the Qwen route leaves thinking control to its wire compatibility", () => {
  assert.deepEqual(
    buildModelSelection(QWEN_PROVIDER, "qwen-model", "off"),
    { provider: QWEN_PROVIDER, model: "qwen-model" },
  )
  assert.deepEqual(buildProviderRequestOptions(QWEN_PROVIDER, decoding), {
    temperature: 0,
    maxTokens: 64,
    stop: [],
  })
  assert.deepEqual(unsupportedRequestOptions(QWEN_PROVIDER), [
    "reasoningEffort",
    "stop",
  ])
})

test("the subscription route names the keys a merge has to delete", () => {
  assert.deepEqual(unsupportedRequestOptions(SUBSCRIPTION_PROVIDER), [
    "stop",
    "temperature",
  ])
  assert.deepEqual(unsupportedRequestOptions("dsh-fixture"), [])
})

for (const [model, effort] of [
  ["google/gemini-3.5-flash", "low"],
  ["moonshotai/kimi-k2.5", "off"],
  ["kimi-k2.5", "high"],
  ["gemini-2.5-flash", "off"],
  ["deepseek/deepseek-v4-flash", "off"],
  ["deepseek-v4-pro", "high"],
]) {
  test(`${model} retains its declared reasoning in selection and request`, () => {
    assert.deepEqual(buildModelSelection(QWEN_PROVIDER, model, effort), {
      provider: QWEN_PROVIDER, model, reasoningEffort: effort,
    })
    const merged = {
      stop: [], reasoningEffort: "inherited",
      ...buildProviderRequestOptions(QWEN_PROVIDER, { ...decoding, reasoning_effort: effort }, model),
    }
    for (const key of unsupportedRequestOptions(QWEN_PROVIDER, model)) delete merged[key]
    assert.deepEqual(merged, { reasoningEffort: effort, temperature: 0, maxTokens: 64 })
  })
}

for (const [model, effort] of [
  ["google/gemini-3.5-flash", "off"],
  ["gemini-2.5-pro", "off"],
  ["gemini-3.1-pro-preview", "max"],
  ["moonshotai/kimi-k2.5", "low"],
  ["moonshotai/kimi-k2-thinking", "off"],
  ["deepseek/deepseek-v4-flash", "low"],
  ["anthropic/claude-4.5-haiku", "high"],
]) {
  test(`${model} rejects unsupported ${effort} before making a request`, () => {
    assert.throws(() => validateCompatibleReasoning(model, effort))
    assert.throws(() => buildModelSelection(QWEN_PROVIDER, model, effort))
    assert.throws(() => buildProviderRequestOptions(QWEN_PROVIDER, { ...decoding, reasoning_effort: effort }, model))
  })
}

test("existing GPT gateway conditions retain their recorded option contract", () => {
  assert.deepEqual(buildModelSelection(QWEN_PROVIDER, "openai/gpt-5.4", "off"), {
    provider: QWEN_PROVIDER, model: "openai/gpt-5.4",
  })
  assert.deepEqual(unsupportedRequestOptions(QWEN_PROVIDER, "openai/gpt-5.4", "off"), ["reasoningEffort", "stop"])
})

test("GPT-5.4 low retains reasoning effort for the gateway wire request", () => {
  const model = "openai/gpt-5.4"
  assert.deepEqual(buildModelSelection(QWEN_PROVIDER, model, "low"), {
    provider: QWEN_PROVIDER, model, reasoningEffort: "low",
  })
  assert.deepEqual(buildProviderRequestOptions(QWEN_PROVIDER,
    { ...decoding, reasoning_effort: "low" }, model), {
    reasoningEffort: "low", maxTokens: 64, stop: [],
  })
  assert.deepEqual(unsupportedRequestOptions(QWEN_PROVIDER, model, "low"), ["stop", "temperature"])
  assert.throws(() => buildProviderRequestOptions(QWEN_PROVIDER,
    { ...decoding, reasoning_effort: "low", temperature: 0.7 }, model),
  /does not support an explicit temperature/)
})

test("Claude non-reasoning profile omits the unsupported DSH off selection", () => {
  const model = "claude-4.5-haiku"
  assert.deepEqual(buildModelSelection(QWEN_PROVIDER, model, "off"), {
    provider: QWEN_PROVIDER, model,
  })
  assert.deepEqual(unsupportedRequestOptions(QWEN_PROVIDER, model), ["reasoningEffort", "stop"])
  assert.deepEqual(buildProviderRequestOptions(QWEN_PROVIDER,
    { ...decoding, reasoning_effort: "off" }, model), { temperature: 0, maxTokens: 64, stop: [] })
})
