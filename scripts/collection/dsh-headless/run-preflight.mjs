import { readFileSync } from "node:fs"
import { mkdir, readdir, stat } from "node:fs/promises"
import { createRequire } from "node:module"
import { homedir } from "node:os"
import { dirname, join, resolve } from "node:path"
import { spawnSync } from "node:child_process"
import { fileURLToPath, pathToFileURL } from "node:url"

import { parseObserverJsonl, verifyPreflight } from "./preflight-contract.mjs"
import { verifyInstalledRuntime } from "./host.mjs"

const here = dirname(fileURLToPath(import.meta.url))
const outputRoot = resolve(process.argv[2] ?? join(here, "preflight-output"))
const dshHome = process.env.DSH_HOME || join(homedir(), ".dsh")
const dshRoot = resolve(
  process.env.DSH_PACKAGE_ROOT
    ?? join(process.env.APPDATA ?? "", "npm", "node_modules", "@deepseek-ai", "dsh"),
)
const components = verifyInstalledRuntime({ dshRoot, dshHome, profile: "dsh-headless" })
const dshRequire = createRequire(join(dshRoot, "package.json"))
const { decodeStorageRecord } = await import(
  pathToFileURL(dshRequire.resolve("@deepseek-ai/dsh-session")).href
)
const observerDir = join(outputRoot, "observer")
const sessionDir = join(outputRoot, "sessions")
const workspaceDir = join(outputRoot, "workspace")
const sessionExportPath = join(outputRoot, "session.jsonl")

async function exists(path) {
  try {
    await stat(path)
    return true
  } catch (error) {
    if (error?.code === "ENOENT") return false
    throw error
  }
}

async function findFiles(root, name) {
  const found = []
  for (const entry of await readdir(root, { withFileTypes: true })) {
    const path = join(root, entry.name)
    if (entry.isDirectory()) found.push(...(await findFiles(path, name)))
    else if (entry.name === name) found.push(path)
  }
  return found
}

if (await exists(outputRoot)) throw new Error(`preflight output already exists: ${outputRoot}`)
await mkdir(observerDir, { recursive: true })
await mkdir(sessionDir, { recursive: true })
await mkdir(workspaceDir, { recursive: true })

const result = spawnSync(
  process.execPath,
  [join(dshRoot, "lib", "bin.js"), "--profile", "dsh-headless", "repair the fixture"],
  {
    cwd: workspaceDir,
    env: {
      ...process.env,
      DSH_HOME: dshHome,
      DSH_BATCH: "0",
      DSH_LOCAL_QWEN: "0",
      DSH_FIXTURE: "1",
      DSH_OBSERVER_DIR: observerDir,
      DSH_SESSION_DIR: sessionDir,
      DSH_SESSION_EXPORT_PATH: sessionExportPath,
      DSH_CHECKPOINT_INTERVAL_BYTES: "4",
      DSH_PROVIDER: "dsh-fixture",
      DSH_MODEL: "dsh-fixture-model",
      DSH_AGENT_PRESET: "router-standard",
      DSH_PRESET_ROOT: join(here, "preset-root"),
      DSH_REASONING_EFFORT: "off",
      DSH_TEMPERATURE: "0",
      DSH_MAX_TOKENS: "64",
      DSH_EXCLUSIVE_PROVIDER: "dsh-fixture",
      DSH_PERMISSION_MODE: "danger-full-access",
    },
    encoding: "utf8",
    timeout: 60_000,
  },
)
if (result.error) throw result.error
if (result.status !== 0) throw new Error(result.stderr || `headless exited ${result.status}`)

const sidecars = (await readdir(observerDir)).filter((name) => name.endsWith(".jsonl"))
if (sidecars.length !== 1) throw new Error(`expected one observer sidecar, got ${sidecars.length}`)
const observerRows = parseObserverJsonl(readFileSync(join(observerDir, sidecars[0]), "utf8"))
const sessions = await findFiles(sessionDir, "session.jsonl.zstd")
if (sessions.length !== 1) throw new Error(`expected one compressed session log, got ${sessions.length}`)
if (!(await exists(sessionExportPath))) throw new Error("decoded session export is missing")
const sessionText = readFileSync(sessionExportPath, "utf8")
const receipt = verifyPreflight({
  observerRows,
  sessionText,
  stdout: result.stdout,
  decodeStorageRecord,
  requirePackedSession: true,
  expectedWorkspace: workspaceDir,
})

console.log(
  JSON.stringify(
    {
      status: "passed",
      profile: "dsh-headless",
      components,
      externalProviderRequestCount: 0,
      outputRoot,
      sessionPath: sessions[0],
      sessionExportPath,
      sidecarPath: join(observerDir, sidecars[0]),
      ...receipt,
    },
    null,
    2,
  ),
)
