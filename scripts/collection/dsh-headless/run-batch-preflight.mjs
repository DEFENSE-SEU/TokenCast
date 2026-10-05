import { existsSync, readFileSync, statSync } from "node:fs"
import { mkdir, readdir, stat, writeFile } from "node:fs/promises"
import { createRequire } from "node:module"
import { homedir } from "node:os"
import { dirname, isAbsolute, join, relative, resolve } from "node:path"
import { createInterface } from "node:readline"
import { spawn } from "node:child_process"
import { once } from "node:events"
import { fileURLToPath, pathToFileURL } from "node:url"

import {
  BATCH_PREFLIGHT_LIFECYCLE_BASIS,
  BATCH_PREFLIGHT_TRANSCRIPT_FILENAME,
  canonicalBatchPreflightJobs,
  createBatchPreflightJobs,
  hostedCredentialPresent,
  normalizeBatchTranscript,
  parseBatchTranscript,
  sendFirstBatchJobAfterReady,
  verifyBatchPreflight,
  withoutHostedCredential,
} from "./batch-preflight-contract.mjs"
import { parseObserverJsonl, verifyPreflight } from "./preflight-contract.mjs"
import { verifyInstalledRuntime } from "./host.mjs"

const here = dirname(fileURLToPath(import.meta.url))
const verifyExisting = process.argv[2] === "--verify-existing"
const outputArgument = process.argv[verifyExisting ? 3 : 2]
const outputRoot = resolve(outputArgument ?? join(here, "batch-preflight-output"))
const dshHome = process.env.DSH_HOME || join(homedir(), ".dsh")
const dshRoot = resolve(
  process.env.DSH_PACKAGE_ROOT
    ?? join(process.env.APPDATA ?? "", "npm", "node_modules", "@deepseek-ai", "dsh"),
)
const components = verifyInstalledRuntime({ dshRoot, dshHome, profile: "dsh-headless" })
const observerDir = join(outputRoot, "observer")
const sessionDir = join(outputRoot, "sessions")
const workspaceDirs = [join(outputRoot, "workspace-1"), join(outputRoot, "workspace-2")]
const transportPath = join(outputRoot, "transport.jsonl")
const normalizedTranscriptPath = join(outputRoot, BATCH_PREFLIGHT_TRANSCRIPT_FILENAME)
const jobs = createBatchPreflightJobs(workspaceDirs)

const dshRequire = createRequire(join(dshRoot, "package.json"))
const sessionPackage = await import(
  pathToFileURL(dshRequire.resolve("@deepseek-ai/dsh-session")).href
)
const persistencePackage = await import(
  pathToFileURL(dshRequire.resolve("@deepseek-ai/dsh-session-persistence-jsonl")).href
)
const cordisPackage = await import(pathToFileURL(dshRequire.resolve("@deepseek-ai/cordis")).href)
const { decodeStorageRecord, SessionId } = sessionPackage
const { JsonlSessionPersistence } = persistencePackage
const { Context } = cordisPackage
const readerContext = new Context()
readerContext.sessions = {
  get() {
    return undefined
  },
  list() {
    return []
  },
}
const sessionReader = new JsonlSessionPersistence(readerContext, {
  root: sessionDir,
  compression: "zstd",
  packChunks: true,
  writeBatchMaxDelayMs: 200,
})

function localIsoDate(date = new Date()) {
  const year = date.getFullYear()
  const month = String(date.getMonth() + 1).padStart(2, "0")
  const day = String(date.getDate()).padStart(2, "0")
  return `${year}-${month}-${day}`
}

async function exists(path) {
  try {
    await stat(path)
    return true
  } catch (error) {
    if (error?.code === "ENOENT") return false
    throw error
  }
}

function within(root, path) {
  const child = relative(resolve(root), resolve(path))
  return child !== "" && !child.startsWith("..") && !isAbsolute(child)
}

function pathKey(path) {
  const absolute = resolve(path)
  return process.platform === "win32" ? absolute.toUpperCase() : absolute
}

function requireSamePathSet(actual, expected, label) {
  const actualKeys = actual.map(pathKey).sort()
  const expectedKeys = expected.map(pathKey).sort()
  if (JSON.stringify(actualKeys) !== JSON.stringify(expectedKeys)) {
    throw new Error(`${label} path inventory drifted`)
  }
}

async function recursiveFiles(root) {
  const files = []
  for (const entry of await readdir(root, { withFileTypes: true })) {
    const path = join(root, entry.name)
    if (entry.isDirectory()) files.push(...(await recursiveFiles(path)))
    else if (entry.isFile()) files.push(path)
    else throw new Error(`unsupported artifact inventory entry: ${path}`)
  }
  return files.sort((left, right) => pathKey(left).localeCompare(pathKey(right)))
}

function requireOrderedMarkers(source, markers, label) {
  let cursor = -1
  for (const marker of markers) {
    cursor = source.indexOf(marker, cursor + 1)
    if (cursor < 0) throw new Error(`${label} lifecycle marker drifted`)
  }
}

function verifyInstalledLifecycle() {
  const installedIntegrationRoot = join(
    dshHome,
    "profiles",
    "dsh-headless",
    "node_modules",
    "@token-prediction",
    "dsh-headless",
  )
  requireOrderedMarkers(
    readFileSync(join(installedIntegrationRoot, "runner.mjs"), "utf8"),
    [
      "const participated = await ctx.sessions.flush(agent.session)",
      "await handle.dispose()",
      "await ctx.sessionPersistence.load(agent.session.id)",
    ],
    "installed headless runner",
  )
  requireOrderedMarkers(
    readFileSync(join(installedIntegrationRoot, "batch-runner.mjs"), "utf8"),
    ["const outcome = await runHeadlessJob", "type: \"job/done\""],
    "installed batch runner",
  )
  requireOrderedMarkers(
    readFileSync(dshRequire.resolve("@deepseek-ai/dsh-session-persistence"), "utf8"),
    ["async load(id)", "await this.waitForRetirement(id)"],
    "installed session persistence",
  )
  return true
}

async function inspectArtifacts(expectedJob, startedFrame, doneFrame) {
  const observerPath = join(observerDir, `${encodeURIComponent(expectedJob.session_id)}.jsonl`)
  const sessionPath = startedFrame.session_path
  if (!within(sessionDir, sessionPath)) {
    throw new Error("batch session artifact escaped its configured root")
  }
  if (!existsSync(observerPath) || statSync(observerPath).size === 0) {
    throw new Error("batch observer sidecar is missing")
  }
  if (!existsSync(sessionPath) || statSync(sessionPath).size === 0) {
    throw new Error("batch session artifact is missing")
  }
  const observerRows = parseObserverJsonl(readFileSync(observerPath, "utf8"))
  const rawSession = await sessionReader.readRaw(SessionId(expectedJob.session_id))
  if (rawSession === undefined) throw new Error("batch session raw artifact is missing")
  const sessionFacts = verifyPreflight({
    observerRows,
    sessionText: rawSession.content,
    stdout: `${doneFrame.text}\n`,
    decodeStorageRecord,
    requirePackedSession: true,
    expectedWorkspace: expectedJob.cwd,
  })
  const entry = {
    direction: "artifact",
    job_id: expectedJob.job_id,
    session_id: expectedJob.session_id,
    observer_readable_after_job_done: true,
    session_readable_after_job_done: true,
    session_semantics_verified: true,
    job_done_barrier_basis: BATCH_PREFLIGHT_LIFECYCLE_BASIS,
    logical_call_count: sessionFacts.logicalCallCount,
    attempt_count: sessionFacts.attemptCount,
    retry_count: sessionFacts.retryCount,
    native_request_header_count: sessionFacts.nativeRequestHeaderCount,
    packed_row_count: sessionFacts.packedRowCount,
    observer_provider: sessionFacts.effectiveCondition.provider,
  }
  return { entry, observerPath, sessionPath }
}

async function inventoryArtifacts(artifacts) {
  const [observerFiles, sessionFiles] = await Promise.all([
    recursiveFiles(observerDir),
    recursiveFiles(sessionDir),
  ])
  const expectedObserverPaths = artifacts.map((artifact) => artifact.observerPath)
  const expectedSessionPaths = artifacts.map((artifact) => artifact.sessionPath)
  if (observerFiles.length !== jobs.length || sessionFiles.length !== jobs.length) {
    throw new Error("batch artifact inventory must contain exactly two observer and session files")
  }
  requireSamePathSet(observerFiles, expectedObserverPaths, "batch observer")
  requireSamePathSet(sessionFiles, expectedSessionPaths, "batch session")
  const sessionIds = artifacts.map((artifact) => artifact.entry.session_id).sort()
  const expectedIds = jobs.map((job) => job.session_id).sort()
  if (JSON.stringify(sessionIds) !== JSON.stringify(expectedIds)) {
    throw new Error("batch artifact inventory identities drifted")
  }
  return {
    direction: "launcher",
    event: "artifact/inventory",
    inventory_verified: true,
    observer_file_count: observerFiles.length,
    session_file_count: sessionFiles.length,
    observer_session_ids: [...sessionIds],
    session_ids: [...sessionIds],
    observer_paths_verified: true,
    started_session_paths_verified: true,
  }
}

function withTimeout(promise, label, timeoutMs = 60_000) {
  let timer
  return Promise.race([
    promise,
    new Promise((_, reject) => {
      timer = setTimeout(() => reject(new Error(`${label} timed out`)), timeoutMs)
    }),
  ]).finally(() => clearTimeout(timer))
}

function childEnvironment() {
  return withoutHostedCredential({
    ...process.env,
    DSH_HOME: dshHome,
    DSH_BATCH: "1",
    DSH_LOCAL_QWEN: "0",
    DSH_FIXTURE: "1",
    DSH_OBSERVER_DIR: observerDir,
    DSH_SESSION_DIR: sessionDir,
    DSH_SESSION_EXPORT_PATH: "",
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
    HF_HUB_OFFLINE: "1",
    TRANSFORMERS_OFFLINE: "1",
  })
}

async function runInstalledBatch() {
  if (await exists(outputRoot)) {
    throw new Error(`batch preflight output already exists: ${outputRoot}`)
  }
  await Promise.all([
    mkdir(observerDir, { recursive: true }),
    mkdir(sessionDir, { recursive: true }),
    ...workspaceDirs.map((path) => mkdir(path, { recursive: true })),
  ])

  const environment = childEnvironment()
  const environmentEvidence = {
    direction: "launcher",
    event: "child/environment",
    credential_present: hostedCredentialPresent(environment),
    exclusive_provider: environment.DSH_EXCLUSIVE_PROVIDER,
    fixture_only: environment.DSH_FIXTURE === "1",
    installed_lifecycle_verified: verifyInstalledLifecycle(),
  }
  const child = spawn(
    process.execPath,
    [join(dshRoot, "lib", "bin.js"), "--profile", "dsh-headless", "batch"],
    {
      cwd: workspaceDirs[0],
      env: environment,
      stdio: ["pipe", "pipe", "pipe"],
      windowsHide: true,
    },
  )
  child.stdout.setEncoding("utf8")
  child.stderr.setEncoding("utf8")
  let stderr = ""
  child.stderr.on("data", (chunk) => {
    stderr += chunk
    if (stderr.length > 1024 * 1024) stderr = stderr.slice(-1024 * 1024)
  })
  const exitPromise = new Promise((resolveExit, rejectExit) => {
    child.once("error", rejectExit)
    child.once("exit", (code, signal) => resolveExit({ code, signal }))
  })
  const lines = createInterface({ input: child.stdout, crlfDelay: Infinity })
  const iterator = lines[Symbol.asyncIterator]()
  const entries = []
  const artifacts = []

  async function receive(label) {
    const item = await withTimeout(iterator.next(), label)
    if (item.done) throw new Error(`${label} reached EOF: ${stderr.trim()}`)
    let frame
    try {
      frame = JSON.parse(item.value)
    } catch (error) {
      throw new Error(`${label} emitted invalid JSON: ${error.message}`)
    }
    entries.push({ direction: "from_dsh", frame })
    return frame
  }

  async function send(frame) {
    const payload = `${JSON.stringify(frame)}\n`
    if (!child.stdin.write(payload)) await once(child.stdin, "drain")
    entries.push({ direction: "to_dsh", frame })
  }

  try {
    const ready = await receive("batch ready")
    await sendFirstBatchJobAfterReady({
      readyFrame: ready,
      expectedPid: child.pid,
      job: jobs[0],
      recordEnvironment() {
        entries.push(environmentEvidence)
      },
      send,
    })
    for (const [index, expectedJob] of jobs.entries()) {
      if (index > 0) await send(expectedJob)
      const started = await receive(`${expectedJob.job_id} started`)
      if (started.type !== "job/started") {
        throw new Error(`${expectedJob.job_id} did not start`)
      }
      const done = await receive(`${expectedJob.job_id} done`)
      if (done.type !== "job/done") {
        throw new Error(`${expectedJob.job_id} did not complete: ${JSON.stringify(done)}`)
      }
      const artifact = await inspectArtifacts(expectedJob, started, done)
      artifacts.push(artifact)
      entries.push(artifact.entry)
    }
    entries.push(await inventoryArtifacts(artifacts))
    await send({ type: "shutdown" })
    const bye = await receive("batch bye")
    if (bye.type !== "bye") throw new Error("batch shutdown did not return bye")
    child.stdin.end()
    const exited = await withTimeout(exitPromise, "batch process exit")
    entries.push({ direction: "process", exit_code: exited.code, signal: exited.signal })
    const trailing = await withTimeout(iterator.next(), "batch stdout close")
    if (!trailing.done) throw new Error("batch process emitted a trailing frame")
  } catch (error) {
    child.kill()
    throw error
  } finally {
    lines.close()
  }
  return entries
}

async function verifyStoredBatch() {
  if (!existsSync(transportPath)) throw new Error("batch transport transcript is missing")
  if (!existsSync(normalizedTranscriptPath)) {
    throw new Error("normalized batch transport transcript is missing")
  }
  const entries = parseBatchTranscript(readFileSync(transportPath, "utf8"))
  verifyBatchPreflight({ entries, expectedJobs: jobs })
  const artifacts = []
  for (const [index, expectedJob] of jobs.entries()) {
    const started = entries[index === 0 ? 3 : 7]?.frame
    const done = entries[index === 0 ? 4 : 8]?.frame
    const observed = await inspectArtifacts(expectedJob, started, done)
    const stored = entries[index === 0 ? 5 : 9]
    if (JSON.stringify(observed.entry) !== JSON.stringify(stored)) {
      throw new Error(`stored batch artifact evidence ${index + 1} drifted`)
    }
    artifacts.push(observed)
  }
  const inventory = await inventoryArtifacts(artifacts)
  if (JSON.stringify(inventory) !== JSON.stringify(entries[10])) {
    throw new Error("stored batch artifact inventory drifted")
  }
  const normalized = normalizeBatchTranscript(entries, jobs)
  const storedNormalized = parseBatchTranscript(readFileSync(normalizedTranscriptPath, "utf8"))
  if (JSON.stringify(normalized) !== JSON.stringify(storedNormalized)) {
    throw new Error("stored normalized batch transport transcript drifted")
  }
  return entries
}

if (verifyExisting && !(await exists(outputRoot))) {
  throw new Error(`batch preflight output is missing: ${outputRoot}`)
}
const entries = verifyExisting ? await verifyStoredBatch() : await runInstalledBatch()
const facts = verifyBatchPreflight({ entries, expectedJobs: jobs })
const normalizedEntries = normalizeBatchTranscript(entries, jobs)
verifyBatchPreflight({ entries: normalizedEntries, expectedJobs: canonicalBatchPreflightJobs() })
if (!verifyExisting) {
  const serialize = (rows) => `${rows.map((entry) => JSON.stringify(entry)).join("\n")}\n`
  await Promise.all([
    writeFile(transportPath, serialize(entries), { encoding: "utf8", flag: "wx" }),
    writeFile(normalizedTranscriptPath, serialize(normalizedEntries), {
      encoding: "utf8",
      flag: "wx",
    }),
  ])
}
const receipt = {
  schema_version: 1,
  status: "passed",
  observed_on: localIsoDate(),
  platform: process.platform,
  profile: "dsh-headless",
  agent_preset: "router-standard",
  transcript_filename: BATCH_PREFLIGHT_TRANSCRIPT_FILENAME,
  components,
  runtime: {
    installed_profile_verified: true,
    headless_integration_version: "0.5.0",
    observer_plugin_version: "0.4.0",
    observer_durability_contract_id: "observer-jsonl-fsync-on-session-flush-v1",
    batch_protocol_version: facts.batch_protocol_version,
  },
  ...facts,
}
process.stdout.write(`${JSON.stringify(receipt, null, 2)}\n`)
