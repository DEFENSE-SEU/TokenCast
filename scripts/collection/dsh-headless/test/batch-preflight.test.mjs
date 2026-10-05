import assert from "node:assert/strict"
import { existsSync, readFileSync } from "node:fs"
import { join } from "node:path"
import test from "node:test"

import {
  BATCH_PREFLIGHT_TRANSCRIPT_FILENAME,
  canonicalBatchPreflightJobs,
  hostedCredentialPresent,
  parseBatchTranscript,
  sendFirstBatchJobAfterReady,
  verifyBatchPreflight,
  withoutHostedCredential,
} from "../batch-preflight-contract.mjs"

const packageRoot = join(import.meta.dirname, "..")
const receiptPath = join(packageRoot, "batch-preflight-receipt.json")

function fixtureEntries() {
  const jobs = canonicalBatchPreflightJobs()
  const sessionIds = jobs.map((job) => job.session_id).sort()
  return [
    { direction: "from_dsh", frame: { type: "ready", protocol_version: 2, pid: 101 } },
    {
      direction: "launcher", event: "child/environment", credential_present: false,
      exclusive_provider: "dsh-fixture", fixture_only: true, installed_lifecycle_verified: true,
    },
    ...jobs.flatMap((job) => [
      { direction: "to_dsh", frame: job },
      {
        direction: "from_dsh", frame: {
          type: "job/started", job_id: job.job_id, session_id: job.session_id,
          session_path: `<output-root>/sessions/${job.session_id}/session.jsonl.zstd`,
        },
      },
      {
        direction: "from_dsh", frame: {
          type: "job/done", job_id: job.job_id, session_id: job.session_id,
          outcome: "completed", text: "fixture complete",
        },
      },
      {
        direction: "artifact", job_id: job.job_id, session_id: job.session_id,
        observer_readable_after_job_done: true, session_readable_after_job_done: true,
        session_semantics_verified: true,
        job_done_barrier_basis: "session-flush-handle-dispose-persistence-retirement-before-job-done",
        logical_call_count: 1, attempt_count: 2, retry_count: 1,
        native_request_header_count: 1, observer_provider: "dsh-fixture", packed_row_count: 1,
      },
    ]),
    {
      direction: "launcher", event: "artifact/inventory", inventory_verified: true,
      observer_file_count: 2, session_file_count: 2, observer_paths_verified: true,
      started_session_paths_verified: true, observer_session_ids: sessionIds, session_ids: sessionIds,
    },
    { direction: "to_dsh", frame: { type: "shutdown" } },
    { direction: "from_dsh", frame: { type: "bye" } },
    { direction: "process", exit_code: 0, signal: null },
  ]
}

function checkedEvidence() {
  const receipt = JSON.parse(readFileSync(receiptPath, "utf8"))
  const transcriptPath = join(packageRoot, receipt.transcript_filename)
  const transcriptText = readFileSync(transcriptPath, "utf8")
  return {
    receipt,
    transcriptText,
    entries: parseBatchTranscript(transcriptText),
  }
}

test("accepts canonical normalized batch-v2 transcript", () => {
  const entries = fixtureEntries()
  const facts = verifyBatchPreflight({ entries, expectedJobs: canonicalBatchPreflightJobs() })
  assert.equal(facts.batch_protocol_version, 2)
  assert.equal(facts.job_count, 2)
  assert.equal(facts.distinct_workspace_count, 2)
  assert.deepEqual(facts.attempt_count, [2, 2])
  assert.deepEqual(facts.retry_count, [1, 1])
  assert.equal(facts.artifact_barrier_before_next_job, true)
  assert.equal(facts.inventory_verified, true)
  assert.deepEqual(facts.shutdown_handshake, ["shutdown", "bye", "exit-0"])
})

test("archived installed-DSH batch receipt matches its transcript", (t) => {
  if (!existsSync(receiptPath) || !existsSync(join(packageRoot, BATCH_PREFLIGHT_TRANSCRIPT_FILENAME))) {
    t.skip("Archived batch preflight evidence is excluded from the source release")
    return
  }
  const { receipt, transcriptText, entries } = checkedEvidence()
  assert.equal(receipt.transcript_filename, BATCH_PREFLIGHT_TRANSCRIPT_FILENAME)
  assert.doesNotMatch(transcriptText, /[A-Za-z]:[\\/]/)
  const facts = verifyBatchPreflight({
    entries,
    expectedJobs: canonicalBatchPreflightJobs(),
  })
  const {
    schema_version: schemaVersion,
    status,
    observed_on: observedOn,
    platform,
    profile,
    agent_preset: agentPreset,
    transcript_filename: transcriptFilename,
    components,
    runtime,
    ...recordedFacts
  } = receipt
  assert.equal(schemaVersion, 1)
  assert.equal(status, "passed")
  assert.equal(observedOn, "2026-08-25")
  assert.equal(platform, "win32")
  assert.equal(profile, "dsh-headless")
  assert.equal(agentPreset, "router-standard")
  assert.equal(transcriptFilename, BATCH_PREFLIGHT_TRANSCRIPT_FILENAME)
  assert.equal(runtime.installed_profile_verified, true)
  assert.equal(runtime.batch_protocol_version, 2)
  assert.equal(runtime.headless_integration_version, "0.5.0")
  assert.equal(runtime.observer_plugin_version, "0.4.0")
  assert.equal(
    runtime.observer_durability_contract_id,
    "observer-jsonl-fsync-on-session-flush-v1",
  )
  assert.equal(components.token_prediction_observer, "0.4.0")
  assert.equal(components.token_prediction_dsh_headless, "0.5.0")
  assert.deepEqual(recordedFacts, facts)
})

test("canonical batch-v2 transcript rejects protocol and barrier drift", () => {
  const entries = fixtureEntries()
  const expectedJobs = canonicalBatchPreflightJobs()
  const protocolDrift = structuredClone(entries)
  protocolDrift[0].frame.protocol_version = 1
  assert.throws(
    () => verifyBatchPreflight({ entries: protocolDrift, expectedJobs }),
    /ready frame drifted/,
  )
  const barrierDrift = structuredClone(entries)
  barrierDrift[5].session_semantics_verified = false
  assert.throws(
    () => verifyBatchPreflight({ entries: barrierDrift, expectedJobs }),
    /readable durability barrier/,
  )
})

test("child environment removes every hosted credential key casing", () => {
  const source = {
    Path: "C:/bin",
    DeepSeek_Api_Key: "first",
    dEePsEeK_aPi_KeY: "second",
  }
  assert.equal(hostedCredentialPresent(source), true)
  const child = withoutHostedCredential(source)
  assert.equal(hostedCredentialPresent(child), false)
  assert.deepEqual(child, { Path: "C:/bin" })
  assert.equal(Object.keys(source).length, 3)
})

test("launcher rejects ready v1 before sending the first job", async () => {
  let environmentRecords = 0
  let sentJobs = 0
  await assert.rejects(
    sendFirstBatchJobAfterReady({
      readyFrame: { type: "ready", protocol_version: 1, pid: 2718 },
      expectedPid: 2718,
      job: canonicalBatchPreflightJobs()[0],
      recordEnvironment() {
        environmentRecords += 1
      },
      send() {
        sentJobs += 1
      },
    }),
    /ready frame drifted/,
  )
  assert.equal(environmentRecords, 0)
  assert.equal(sentJobs, 0)
})

test("batch transcript parser requires strict JSONL", () => {
  const entries = fixtureEntries()
  const transcriptText = entries.map((entry) => JSON.stringify(entry)).join("\n") + "\n"
  assert.equal(parseBatchTranscript(transcriptText).length, 14)
  assert.deepEqual(parseBatchTranscript(transcriptText), entries)
  assert.throws(() => parseBatchTranscript(transcriptText.slice(0, -1)), /end with LF/)
})
