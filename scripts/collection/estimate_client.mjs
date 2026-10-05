import { readFile, writeFile } from "node:fs/promises"
import { findPackageJSON } from "node:module"
import { dirname, join } from "node:path"
import { performance } from "node:perf_hooks"
import { pathToFileURL } from "node:url"

import { resolveDshHostModuleUrl } from "./dsh-headless/host.mjs"
import {
  createReadOnlyCredentialStore,
  resolveSubscriptionAuthFile,
  selectSubscriptionProvider,
} from "./dsh-headless/subscription-adapter.mjs"

function argument(name, fallback) {
  const index = process.argv.indexOf(name)
  return index >= 0 ? process.argv[index + 1] : fallback
}

const systemPath = argument("--system")
const userPath = argument("--user")
const outPath = argument("--out")
const modelId = argument("--model", "gpt-5.4")
const reasoning = argument("--reasoning", "low")
if (!systemPath || !userPath || !outPath) {
  console.error("usage: --system <file> --user <file> --out <json> [--model id] [--reasoning level]")
  process.exit(2)
}

const piManifest = findPackageJSON(
  "@earendil-works/pi-ai",
  resolveDshHostModuleUrl("@deepseek-ai/dsh-llm-pi-ai"),
)
if (piManifest === undefined) throw new Error("pi-ai is missing from the DSH profile")
const { createModels } = await import(
  pathToFileURL(join(dirname(piManifest), "dist", "models.js")).href
)
const { openaiCodexProvider } = await import(
  pathToFileURL(join(dirname(piManifest), "dist", "providers", "openai-codex.js")).href
)
if (process.argv.includes("--list-models")) {
  for (const entry of openaiCodexProvider().getModels()) console.log(entry.id)
  process.exit(0)
}
const authFile = resolveSubscriptionAuthFile()
const credentials = createReadOnlyCredentialStore({ provider: "openai-codex", authFile })
const models = createModels({
  credentials,
  authContext: { env: async () => undefined, fileExists: async () => false },
})
const provider =
  modelId === "gpt-5.4"
    ? selectSubscriptionProvider("openai-codex", modelId)
    : { ...openaiCodexProvider(), getModels: () => openaiCodexProvider().getModels().filter((m) => m.id === modelId) }
models.setProvider(provider)
const model = models.getModel("openai-codex", modelId)
if (model === undefined) throw new Error(`model ${modelId} is not installed`)

const systemPrompt = await readFile(systemPath, "utf8")
const userText = await readFile(userPath, "utf8")
const started = performance.now()
const message = await models.completeSimple(
  model,
  { systemPrompt, messages: [{ role: "user", content: userText, timestamp: Date.now() }] },
  { reasoning },
)
const elapsedMs = performance.now() - started
const text = message.content
  .filter((part) => part.type === "text")
  .map((part) => part.text)
  .join("")
await writeFile(
  outPath,
  JSON.stringify(
    {
      model: modelId,
      reasoning,
      stop_reason: message.stopReason,
      error: message.errorMessage ?? null,
      text,
      usage: message.usage ?? null,
      elapsed_ms: elapsedMs,
    },
    null,
    2,
  ),
  "utf8",
)
if (message.stopReason === "error") {
  console.error(message.errorMessage)
  process.exit(1)
}
process.stdout.write(text.slice(0, 400) + "\n")
