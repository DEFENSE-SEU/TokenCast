const EXACT_JOB_KEYS = ["caps", "cwd", "decoding", "job_id", "session_id", "task", "type"]
const JOB_OFFSETS = [2, 6]
const HOSTED_CREDENTIAL_KEY = "DEEPSEEK_API_KEY"

export const BATCH_PREFLIGHT_PROTOCOL_VERSION = 2
export const BATCH_PREFLIGHT_TRANSCRIPT_FILENAME = "batch-preflight-transcript.jsonl"
export const BATCH_PREFLIGHT_LIFECYCLE_BASIS =
  "session-flush-handle-dispose-persistence-retirement-before-job-done"
export const BATCH_PREFLIGHT_NORMALIZED_ROOT = "<output-root>"
export const BATCH_PREFLIGHT_SESSIONS = Object.freeze([
  "batch-preflight-session-1",
  "batch-preflight-session-2",
])

function exactKeys(value, expected, label) {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${label} must be an object`)
  }
  const actual = Object.keys(value).sort()
  const wanted = [...expected].sort()
  if (JSON.stringify(actual) !== JSON.stringify(wanted)) {
    throw new Error(`${label} fields drifted`)
  }
}

function sameJson(actual, expected, label) {
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    throw new Error(`${label} drifted`)
  }
}

export function createBatchPreflightJobs(workspaceDirs) {
  if (!Array.isArray(workspaceDirs) || workspaceDirs.length !== 2) {
    throw new Error("batch preflight requires two workspaces")
  }
  return workspaceDirs.map((cwd, index) => ({
    type: "job",
    job_id: `batch-preflight-job-${index + 1}`,
    session_id: BATCH_PREFLIGHT_SESSIONS[index],
    cwd,
    task: `repair deterministic fixture ${index + 1}`,
    decoding: {
      reasoning_effort: "off",
      temperature: 0,
      max_tokens: 64,
      stop: [],
    },
    caps: {
      max_logical_calls: 4,
      max_provider_accounted_tokens: 1_000,
    },
  }))
}

export function canonicalBatchPreflightJobs() {
  return createBatchPreflightJobs([
    `${BATCH_PREFLIGHT_NORMALIZED_ROOT}/workspace-1`,
    `${BATCH_PREFLIGHT_NORMALIZED_ROOT}/workspace-2`,
  ])
}

export function hostedCredentialPresent(environment) {
  return Object.keys(environment).some((key) => key.toUpperCase() === HOSTED_CREDENTIAL_KEY)
}

export function withoutHostedCredential(environment) {
  const childEnvironment = { ...environment }
  for (const key of Object.keys(childEnvironment)) {
    if (key.toUpperCase() === HOSTED_CREDENTIAL_KEY) delete childEnvironment[key]
  }
  return childEnvironment
}

export function verifyBatchReadyFrame(frame, expectedPid) {
  exactKeys(frame, ["pid", "protocol_version", "type"], "batch ready frame")
  if (
    frame.type !== "ready"
    || frame.protocol_version !== BATCH_PREFLIGHT_PROTOCOL_VERSION
    || !Number.isSafeInteger(frame.pid)
    || frame.pid < 1
    || (expectedPid !== undefined && frame.pid !== expectedPid)
  ) {
    throw new Error("batch ready frame drifted")
  }
  return frame
}

export async function sendFirstBatchJobAfterReady({
  readyFrame,
  expectedPid,
  job,
  recordEnvironment,
  send,
}) {
  verifyBatchReadyFrame(readyFrame, expectedPid)
  await recordEnvironment()
  await send(job)
}

export function parseBatchTranscript(text) {
  if (!text.endsWith("\n")) throw new Error("batch transcript must end with LF")
  const lines = text.slice(0, -1).split("\n")
  if (lines.some((line) => line.length === 0)) {
    throw new Error("batch transcript contains a blank row")
  }
  return lines.map((line) => JSON.parse(line))
}

export function normalizeBatchTranscript(entries, actualJobs) {
  if (!Array.isArray(entries) || !Array.isArray(actualJobs) || actualJobs.length !== 2) {
    throw new Error("batch transcript normalization requires two jobs")
  }
  const normalized = structuredClone(entries)
  const canonicalJobs = canonicalBatchPreflightJobs()
  for (const [index, offset] of JOB_OFFSETS.entries()) {
    sameJson(normalized[offset]?.frame, actualJobs[index], `batch job ${index + 1}`)
    normalized[offset].frame = canonicalJobs[index]
    const started = normalized[offset + 1]?.frame
    if (started?.type !== "job/started") {
      throw new Error(`batch job ${index + 1} started frame is missing`)
    }
    started.session_path = `${BATCH_PREFLIGHT_NORMALIZED_ROOT}/sessions/${actualJobs[index].session_id}/session.jsonl.zstd`
  }
  return normalized
}

function verifyJobFrame(actual, expected, index) {
  exactKeys(actual, EXACT_JOB_KEYS, `batch job ${index + 1}`)
  sameJson(actual, expected, `batch job ${index + 1}`)
}

function verifyArtifactEntry(entry, job, index) {
  exactKeys(
    entry,
    [
      "attempt_count",
      "direction",
      "job_done_barrier_basis",
      "job_id",
      "logical_call_count",
      "native_request_header_count",
      "observer_provider",
      "observer_readable_after_job_done",
      "packed_row_count",
      "retry_count",
      "session_id",
      "session_readable_after_job_done",
      "session_semantics_verified",
    ],
    `batch artifact ${index + 1}`,
  )
  if (
    entry.direction !== "artifact"
    || entry.job_id !== job.job_id
    || entry.session_id !== job.session_id
    || entry.observer_readable_after_job_done !== true
    || entry.session_readable_after_job_done !== true
    || entry.session_semantics_verified !== true
    || entry.job_done_barrier_basis !== BATCH_PREFLIGHT_LIFECYCLE_BASIS
    || entry.logical_call_count !== 1
    || entry.attempt_count !== 2
    || entry.retry_count !== 1
    || entry.native_request_header_count !== 1
    || entry.observer_provider !== "dsh-fixture"
    || !Number.isSafeInteger(entry.packed_row_count)
    || entry.packed_row_count < 1
  ) {
    throw new Error(`batch artifact ${index + 1} did not cross its readable durability barrier`)
  }
  return entry
}

function verifyInventoryEntry(entry, expectedJobs) {
  exactKeys(
    entry,
    [
      "direction",
      "event",
      "inventory_verified",
      "observer_file_count",
      "observer_paths_verified",
      "observer_session_ids",
      "session_file_count",
      "session_ids",
      "started_session_paths_verified",
    ],
    "batch artifact inventory",
  )
  const expectedIds = expectedJobs.map((job) => job.session_id).sort()
  if (
    entry.direction !== "launcher"
    || entry.event !== "artifact/inventory"
    || entry.inventory_verified !== true
    || entry.observer_file_count !== 2
    || entry.session_file_count !== 2
    || entry.observer_paths_verified !== true
    || entry.started_session_paths_verified !== true
  ) {
    throw new Error("batch artifact inventory drifted")
  }
  sameJson(entry.observer_session_ids, expectedIds, "batch observer inventory identities")
  sameJson(entry.session_ids, expectedIds, "batch session inventory identities")
  return entry
}

export function verifyBatchPreflight({ entries, expectedJobs }) {
  if (!Array.isArray(entries) || !Array.isArray(expectedJobs) || expectedJobs.length !== 2) {
    throw new Error("batch preflight requires two jobs")
  }
  if (entries.length !== 14) throw new Error("batch transcript length drifted")

  const ready = entries[0]
  exactKeys(ready, ["direction", "frame"], "batch ready entry")
  if (ready.direction !== "from_dsh") throw new Error("batch ready direction drifted")
  verifyBatchReadyFrame(ready.frame)

  const environment = entries[1]
  exactKeys(
    environment,
    [
      "credential_present",
      "direction",
      "event",
      "exclusive_provider",
      "fixture_only",
      "installed_lifecycle_verified",
    ],
    "batch child environment entry",
  )
  if (
    environment.direction !== "launcher"
    || environment.event !== "child/environment"
    || environment.credential_present !== false
    || environment.exclusive_provider !== "dsh-fixture"
    || environment.fixture_only !== true
    || environment.installed_lifecycle_verified !== true
  ) {
    throw new Error("batch child environment drifted")
  }

  const artifacts = []
  for (const [index, job] of expectedJobs.entries()) {
    const offset = JOB_OFFSETS[index]
    const sent = entries[offset]
    exactKeys(sent, ["direction", "frame"], `batch job ${index + 1} send entry`)
    if (sent.direction !== "to_dsh") {
      throw new Error(`batch job ${index + 1} direction drifted`)
    }
    verifyJobFrame(sent.frame, job, index)

    const started = entries[offset + 1]
    exactKeys(started, ["direction", "frame"], `batch job ${index + 1} started entry`)
    exactKeys(
      started.frame,
      ["job_id", "session_id", "session_path", "type"],
      `batch job ${index + 1} started frame`,
    )
    if (
      started.direction !== "from_dsh"
      || started.frame.type !== "job/started"
      || started.frame.job_id !== job.job_id
      || started.frame.session_id !== job.session_id
      || typeof started.frame.session_path !== "string"
      || started.frame.session_path.length === 0
    ) {
      throw new Error(`batch job ${index + 1} started frame drifted`)
    }

    const done = entries[offset + 2]
    exactKeys(done, ["direction", "frame"], `batch job ${index + 1} done entry`)
    exactKeys(
      done.frame,
      ["job_id", "outcome", "session_id", "text", "type"],
      `batch job ${index + 1} done frame`,
    )
    if (
      done.direction !== "from_dsh"
      || done.frame.type !== "job/done"
      || done.frame.job_id !== job.job_id
      || done.frame.session_id !== job.session_id
      || done.frame.outcome !== "completed"
      || done.frame.text !== "fixture complete"
    ) {
      throw new Error(`batch job ${index + 1} terminal drifted`)
    }
    artifacts.push(verifyArtifactEntry(entries[offset + 3], job, index))
  }

  const inventory = verifyInventoryEntry(entries[10], expectedJobs)
  const shutdown = entries[11]
  exactKeys(shutdown, ["direction", "frame"], "batch shutdown entry")
  sameJson(shutdown, { direction: "to_dsh", frame: { type: "shutdown" } }, "batch shutdown")
  const bye = entries[12]
  exactKeys(bye, ["direction", "frame"], "batch bye entry")
  sameJson(bye, { direction: "from_dsh", frame: { type: "bye" } }, "batch bye")
  const processExit = entries[13]
  exactKeys(processExit, ["direction", "exit_code", "signal"], "batch process exit")
  sameJson(
    processExit,
    { direction: "process", exit_code: 0, signal: null },
    "batch process exit",
  )

  const fixtureRouteVerified = artifacts.every((entry) => entry.observer_provider === "dsh-fixture")
  const exactArtifactInventory = inventory.inventory_verified === true
  return {
    batch_protocol_version: BATCH_PREFLIGHT_PROTOCOL_VERSION,
    fixture_only: environment.fixture_only,
    credential_present: environment.credential_present,
    exclusive_provider: environment.exclusive_provider,
    external_provider_request_count: 0,
    external_provider_evidence: {
      fixture_route_verified: fixtureRouteVerified,
      credential_absent: environment.credential_present === false,
      exclusive_provider_verified: environment.exclusive_provider === "dsh-fixture",
      exact_artifact_inventory: exactArtifactInventory,
    },
    job_count: expectedJobs.length,
    observer_file_count: inventory.observer_file_count,
    session_count: inventory.session_file_count,
    distinct_workspace_count: new Set(expectedJobs.map((job) => job.cwd)).size,
    logical_call_count: artifacts.map((entry) => entry.logical_call_count),
    attempt_count: artifacts.map((entry) => entry.attempt_count),
    retry_count: artifacts.map((entry) => entry.retry_count),
    native_request_header_count: artifacts.map((entry) => entry.native_request_header_count),
    packed_row_count: artifacts.map((entry) => entry.packed_row_count),
    observer_provider: artifacts.map((entry) => entry.observer_provider),
    observer_readable_after_job_done: artifacts.every(
      (entry) => entry.observer_readable_after_job_done,
    ),
    session_readable_after_job_done: artifacts.every(
      (entry) => entry.session_readable_after_job_done,
    ),
    session_semantics_verified: artifacts.every((entry) => entry.session_semantics_verified),
    job_done_barrier_basis: BATCH_PREFLIGHT_LIFECYCLE_BASIS,
    artifact_barrier_before_next_job: true,
    inventory_verified: inventory.inventory_verified,
    shutdown_handshake: ["shutdown", "bye", "exit-0"],
  }
}
