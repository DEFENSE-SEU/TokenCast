import { readFileSync } from "node:fs"
import { mkdir, readdir, stat } from "node:fs/promises"
import { createRequire } from "node:module"
import { homedir } from "node:os"
import { dirname, join, resolve } from "node:path"
import { spawnSync } from "node:child_process"
import { fileURLToPath, pathToFileURL } from "node:url"

import {
  QWEN_PREFLIGHT_CHECKPOINT_INTERVAL_BYTES,
  QWEN_PREFLIGHT_TASK,
  verifyGenerationProbePreflight,
  verifyQwenPreflight,
} from "./qwen-preflight-contract.mjs"
import { verifyInstalledRuntime } from "./host.mjs"

const here = dirname(fileURLToPath(import.meta.url))
const verifyExisting = process.argv[2] === "--verify-existing"
const outputArgument = process.argv[verifyExisting ? 3 : 2]
const checkpointIntervalBytes = Number(
  process.argv[verifyExisting ? 4 : 3] ?? QWEN_PREFLIGHT_CHECKPOINT_INTERVAL_BYTES,
)
if (!Number.isSafeInteger(checkpointIntervalBytes) || checkpointIntervalBytes < 1) {
  throw new Error("checkpoint interval must be a positive integer")
}
const outputRoot = resolve(outputArgument ?? join(here, "qwen-preflight-output"))
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
const generationProbeDir = join(outputRoot, "probe")

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

if (!verifyExisting) {
  if (await exists(outputRoot)) throw new Error(`Qwen preflight output already exists: ${outputRoot}`)
  const python = process.env.DSH_QWEN_PYTHON
  const snapshot = process.env.DSH_QWEN_SNAPSHOT
  if (!python || !snapshot) {
    throw new Error("DSH_QWEN_PYTHON and DSH_QWEN_SNAPSHOT are required")
  }
  await mkdir(observerDir, { recursive: true })
  await mkdir(sessionDir, { recursive: true })
  await mkdir(generationProbeDir, { recursive: true })
  await mkdir(workspaceDir, { recursive: true })
  const result = spawnSync(
    process.execPath,
    [join(dshRoot, "lib", "bin.js"), "--profile", "dsh-headless", QWEN_PREFLIGHT_TASK],
    {
      cwd: workspaceDir,
      env: {
        ...process.env,
        DSH_HOME: dshHome,
        DSH_BATCH: "0",
        DSH_LOCAL_QWEN: "1",
        DSH_FIXTURE: "0",
        DSH_OBSERVER_DIR: observerDir,
        DSH_SESSION_DIR: sessionDir,
        DSH_SESSION_EXPORT_PATH: sessionExportPath,
        DSH_CHECKPOINT_INTERVAL_BYTES: String(checkpointIntervalBytes),
        DSH_PROVIDER: "local-qwen",
        DSH_MODEL: "Qwen/Qwen2.5-7B-Instruct",
        DSH_AGENT_PRESET: "router-standard",
        DSH_PRESET_ROOT: join(here, "preset-root"),
        DSH_REASONING_EFFORT: "off",
        DSH_TEMPERATURE: "0",
        DSH_MAX_TOKENS: "192",
        DSH_EXCLUSIVE_PROVIDER: "local-qwen",
        DSH_QWEN_PYTHON: python,
        DSH_QWEN_WORKER: process.env.DSH_QWEN_WORKER || join(here, "qwen-worker.py"),
        DSH_QWEN_SNAPSHOT: snapshot,
        DSH_GENERATION_PROBE_DIR: generationProbeDir,
        DSH_GENERATION_PROBE_CAPTURE_REVISION_PREFIX:
          "local-qwen-generation-probe-v2:preflight",
        DSH_PERMISSION_MODE: "danger-full-access",
        HF_HUB_OFFLINE: "1",
        TRANSFORMERS_OFFLINE: "1",
      },
      encoding: "utf8",
      timeout: 600_000,
      maxBuffer: 4 * 1024 * 1024,
    },
  )
  if (result.error) throw result.error
  if (result.status !== 0) throw new Error(result.stderr || `headless exited ${result.status}`)
}

const sidecars = (await readdir(observerDir)).filter((name) => name.endsWith(".jsonl"))
if (sidecars.length !== 1) throw new Error(`expected one observer sidecar, got ${sidecars.length}`)
const sessions = await findFiles(sessionDir, "session.jsonl.zstd")
if (sessions.length !== 1) throw new Error(`expected one compressed session log, got ${sessions.length}`)
if (!(await exists(sessionExportPath))) throw new Error("decoded session export is missing")
const observerText = readFileSync(join(observerDir, sidecars[0]), "utf8")
const observerSession = JSON.parse(observerText.slice(0, observerText.indexOf("\n")))
const generationProbePath = join(generationProbeDir, `${observerSession.session_id}.jsonl`)
if (!(await exists(generationProbePath))) throw new Error("Generation Probe sidecar is missing")
const generationProbeText = readFileSync(generationProbePath, "utf8")
const receipt = verifyQwenPreflight({
  observerText,
  sessionText: readFileSync(sessionExportPath, "utf8"),
  checkpointIntervalBytes,
  decodeStorageRecord,
  expectedWorkspace: workspaceDir,
})
const probeEvidence = verifyGenerationProbePreflight({
  observerText,
  probeText: generationProbeText,
})

console.log(
  JSON.stringify(
    {
      status: "passed",
      profile: "dsh-headless",
      components,
      ...receipt,
      generation_probe: {
        capture_revision_prefix: probeEvidence.captureRevisionPrefix,
        capture_revision_policy: probeEvidence.captureRevisionPolicy,
        checkpoint_count: probeEvidence.checkpointCount,
        projection_policy_id: probeEvidence.projectionPolicyId,
        projection_dimension: probeEvidence.projectionDimension,
        causal_alignment: probeEvidence.causalAlignment,
        source_output_tokens: probeEvidence.sourceOutputTokens,
      },
    },
    null,
    2,
  ),
)
