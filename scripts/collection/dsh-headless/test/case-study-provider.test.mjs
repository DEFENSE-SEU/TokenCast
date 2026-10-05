import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import { findPackageJSON } from "node:module"
import { dirname, join } from "node:path"
import test from "node:test"
import { pathToFileURL } from "node:url"

import { importDshHostModule, resolveDshHostModuleUrl } from "../host.mjs"
import { buildModelSelection, buildProviderRequestOptions, unsupportedRequestOptions } from "../provider-request-options.mjs"

function profileField(name, model) {
  const config = readFileSync(new URL("../cordis.patch.yml", import.meta.url), "utf8")
  const expression = config.match(new RegExp(`^\\s+${name}: !!js (".*")$`, "m"))
  assert.ok(expression, `missing profile expression ${name}`)
  return Function("process", `return (${JSON.parse(expression[1])})`)({ env: { DSH_MODEL: model } })
}

test("Claude selection passes installed DSH non-reasoning capability validation", async (t) => {
  let LlmRuntime
  try {
    ;({ LlmRuntime } = await importDshHostModule("@deepseek-ai/dsh-llm"))
  } catch (error) {
    if (!["MODULE_NOT_FOUND", "ERR_MODULE_NOT_FOUND"].includes(error.code)) throw error
    return t.skip("installed DSH LLM runtime unavailable")
  }
  const provider = "qwen-compatible"
  const model = "claude-4.5-haiku"
  assert.equal(profileField("reasoningEfforts", model), false)
  const info = { provider, id: model, name: model, defaultMaxTokens: 1536 }
  const resolve = (options) => LlmRuntime.prototype.resolveCallWithInfo.call(null, options, info)
  assert.throws(() => resolve({ provider, model, reasoningEffort: "off" }),
    { code: "UNSUPPORTED_REASONING_EFFORT" })
  const selection = buildModelSelection(provider, model, "off")
  assert.equal(resolve(selection).config.reasoningEffort, undefined)
  const options = { reasoningEffort: "off", ...selection,
    ...buildProviderRequestOptions(provider,
      { reasoning_effort: "off", temperature: 0, max_tokens: 1536, stop: [] }, model) }
  for (const key of unsupportedRequestOptions(provider, model)) delete options[key]
  const resolved = resolve(options)
  assert.equal(resolved.config.reasoningEffort, undefined)
  assert.equal(resolved.config.maxTokens, 1536)
})

for (const id of ["claude-4.5-haiku", "deepseek/deepseek-v4-flash"]) {
  test(`${id}: installed pi-ai emits the native off request from profile settings`, async (t) => {
    let stream
    try {
      const adapterUrl = resolveDshHostModuleUrl("@deepseek-ai/dsh-llm-pi-ai")
      const manifest = findPackageJSON("@earendil-works/pi-ai", adapterUrl)
      if (!manifest) return t.skip("installed pi-ai unavailable")
      ;({ stream } = await import(pathToFileURL(join(dirname(manifest), "dist/api/openai-completions.js")).href))
    } catch (error) {
      if (!["MODULE_NOT_FOUND", "ERR_MODULE_NOT_FOUND"].includes(error.code)) throw error
      return t.skip("installed pi-ai unavailable")
    }
    const requests = []
    t.mock.method(globalThis, "fetch", async (url, init) => {
      assert.match(String(url), /^https:\/\/offline\.example\.test\/v1\/chat\/completions/)
      requests.push(JSON.parse(init.body))
      return new Response(`data: ${JSON.stringify({ id: "fake", choices: [{ index: 0,
        delta: { content: "ok" }, finish_reason: "stop" }],
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 } })}\n\ndata: [DONE]\n\n`,
      { headers: { "content-type": "text/event-stream" } })
    })
    const efforts = profileField("reasoningEfforts", id)
    const model = {
      id, name: id, api: "openai-completions", provider: "qwen-compatible",
      baseUrl: "https://offline.example.test/v1", reasoning: efforts !== false,
      ...(efforts === false ? {} : { thinkingLevelMap: efforts }),
      input: ["text"], contextWindow: 200000, maxTokens: 1536,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      compat: { supportsDeveloperRole: false, supportsStore: false, maxTokensField: "max_tokens",
        ...Object.fromEntries(["thinkingFormat", "supportsReasoningEffort",
          "requiresReasoningContentOnAssistantMessages"].map((key) => [key, profileField(key, id)])) },
    }
    const result = await stream(model, { messages: [{ role: "user", content: "offline fixture", timestamp: 1 }] },
      { apiKey: "fake-only", maxTokens: 1536, temperature: 0, maxRetries: 0, env: {} }).result()
    assert.equal(result.stopReason, "stop")
    assert.equal(requests.length, 1)
    assert.equal(requests[0].max_tokens, 1536)
    assert.equal(requests[0].enable_thinking, undefined)
    assert.equal(requests[0].reasoning_effort, undefined)
    assert.deepEqual(requests[0].thinking, id.startsWith("deepseek/") ? { type: "disabled" } : undefined)
  })
}
