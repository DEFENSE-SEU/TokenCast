import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import { createRequire } from "node:module"
import test from "node:test"

import { importDshHostModule, resolveDshHostModuleUrl } from "../host.mjs"

let host
try {
  const includeUrl = resolveDshHostModuleUrl("@deepseek-ai/cordis-plugin-include")
  const { entryListSchema } = await import(includeUrl)
  const { interpolate, isJsExpr } = await importDshHostModule("@deepseek-ai/cordis-plugin-loader")
  const yaml = createRequire(includeUrl)("js-yaml")
  host = { yaml, entryListSchema, interpolate, isJsExpr }
} catch (error) {
  if (!["MODULE_NOT_FOUND", "ERR_MODULE_NOT_FOUND"].includes(error?.code)) throw error
}

test("retention config parses and evaluates with the installed DSH YAML dialect", {
  skip: !host && "installed DSH host modules are unavailable",
}, () => {
  const patch = host.yaml.load(readFileSync(new URL("../cordis.patch.yml", import.meta.url), "utf8"), {
    schema: host.entryListSchema,
  })
  const entries = []
  function visit(value) {
    if (Array.isArray(value)) return value.forEach(visit)
    if (value && typeof value === "object") {
      if (value.id === "dsh-headless-runner") entries.push(value)
      Object.values(value).forEach(visit)
    }
  }
  visit(patch)
  assert.equal(entries.length, 1)
  for (const [key, variable] of [
    ["toolResultMaxUtf8Bytes", "DSH_TOOL_RESULT_MAX_UTF8_BYTES"],
    ["toolResultHistoryKeepRecent", "DSH_TOOL_RESULT_HISTORY_KEEP_RECENT"],
  ]) {
    const expression = entries[0].config[key]
    assert.equal(host.isJsExpr(expression), true, `${key} must parse as one scalar expression`)
    const previous = process.env[variable]
    try {
      for (const [input, expected] of [[undefined, undefined], ["", undefined], ["0", 0], ["2", 2]]) {
        if (input === undefined) delete process.env[variable]
        else process.env[variable] = input
        assert.equal(host.interpolate({}, expression), expected, `${key}: ${input}`)
      }
    } finally {
      if (previous === undefined) delete process.env[variable]
      else process.env[variable] = previous
    }
  }
})
