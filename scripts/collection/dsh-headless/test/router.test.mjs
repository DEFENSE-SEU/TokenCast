import assert from "node:assert/strict"
import test from "node:test"

import { QWEN_PREFLIGHT_TASK } from "../qwen-preflight-contract.mjs"
import {
  bandOf,
  classifyTask,
  coreFor,
  extractText,
  personaFor,
} from "../preset-root/router-standard/router-core.mjs"

test("router classifies build, repair, and ambiguous tasks", () => {
  assert.equal(classifyTask("Build a small parser"), 1)
  assert.equal(classifyTask("Repair the broken parser"), 0)
  assert.equal(classifyTask(QWEN_PREFLIGHT_TASK), 1)
  assert.equal(classifyTask("Explain this repository"), "weak")
})

test("router keeps deterministic first-request bands and nested message text", () => {
  assert.equal(bandOf(0), "spec")
  assert.equal(bandOf(1), "react")
  assert.deepEqual(coreFor(0), ["read", "edit", "glob", "grep"])
  assert.deepEqual(coreFor(1), ["read", "write", "edit"])
  assert.equal(
    extractText({ message: { content: [{ type: "text", text: "nested" }] } }),
    "nested",
  )
  assert.match(personaFor("weak", "deepseek-v4-pro"), /carry out the task/i)
})
