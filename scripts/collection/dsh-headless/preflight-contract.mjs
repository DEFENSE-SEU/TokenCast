export function parseObserverJsonl(text) {
  if (!text.endsWith("\n")) throw new Error("observer JSONL must end with LF")
  const lines = text.slice(0, -1).split("\n")
  if (lines.some((line) => line.length === 0)) {
    throw new Error("observer JSONL contains a blank row")
  }
  return lines.map((line) => JSON.parse(line))
}

export function verifyPreflight({
  observerRows,
  sessionText,
  stdout,
  decodeStorageRecord,
  requirePackedSession = false,
  expectedWorkspace,
}) {
  if (observerRows.some((row) => row.schema_version !== 2)) {
    throw new Error("observer rows must use schema version 2")
  }
  if (observerRows[0]?.record_type !== "observer/session") {
    throw new Error("observer session header must be the first row")
  }
  const observerSessions = observerRows.filter((row) => row.record_type === "observer/session")
  if (observerSessions.length !== 1) throw new Error("expected one observer session header")
  const observerSession = observerSessions[0]
  if (
    observerSession.plugin_version !== "0.4.0"
    || observerSession.dsh_llm_compatibility !== ">=0.1.1-rc.2 <0.2.0"
    || observerSession.observer_durability_contract_id !== "observer-jsonl-fsync-on-session-flush-v1"
  ) {
    throw new Error("observer plugin compatibility drifted")
  }
  const firstLf = sessionText.indexOf("\n")
  if (firstLf < 0) throw new Error("session log header is incomplete")
  if (!sessionText.endsWith("\n")) throw new Error("session JSONL must end with LF")
  const sessionHeader = JSON.parse(sessionText.slice(0, firstLf))
  if (
    sessionHeader.type !== "session"
    || sessionHeader.version !== 0
    || sessionHeader.agentPreset !== "router-standard"
    || typeof sessionHeader.id !== "string"
    || typeof sessionHeader.cwd !== "string"
  ) {
    throw new Error("DSH session header drifted")
  }
  if (
    sessionHeader.id !== observerSession.session_id
    || observerRows.some((row) => row.session_id !== sessionHeader.id)
  ) {
    throw new Error("observer and DSH session ids differ")
  }
  if (expectedWorkspace !== undefined && sessionHeader.cwd !== expectedWorkspace) {
    throw new Error("DSH session workspace differs from the requested workspace")
  }
  const storedRows = sessionText.slice(firstLf + 1, -1).split("\n").map((line) => JSON.parse(line))
  const packedTypes = new Set(["text-chunks", "reasoning-chunks", "tool-call-chunks"])
  const packedRowCount = storedRows.filter((row) => packedTypes.has(row?.type)).length
  if (requirePackedSession && packedRowCount === 0) {
    throw new Error("session contains no packed chunk row")
  }
  if (packedRowCount > 0 && typeof decodeStorageRecord !== "function") {
    throw new Error("packed session requires the DSH storage decoder")
  }
  let events
  try {
    events = storedRows.flatMap((row) => (
      typeof decodeStorageRecord === "function" ? decodeStorageRecord(row) : [row]
    ))
  } catch (error) {
    throw new Error(`session storage row is invalid: ${error.message}`)
  }
  for (const [index, event] of events.entries()) {
    if (event?.seq !== index) throw new Error("decoded session seq must be continuous")
  }
  const permissionEvents = events.filter((event) => (
    event.type === "permission/preset"
    || event.type === "sandbox/mode"
    || event.type === "approval/policy"
  ))
  if (JSON.stringify(permissionEvents.map((event) => [event.type, event.data])) !== JSON.stringify([
    ["permission/preset", { preset: "unrestricted" }],
    ["sandbox/mode", { mode: "danger-full-access" }],
    ["approval/policy", { policy: "never" }],
  ])) {
    throw new Error("session permission facts differ from the frozen actor policy")
  }
  const starts = observerRows.filter((row) => row.record_type === "attempt/start")
  const ends = observerRows.filter((row) => row.record_type === "attempt/end")
  if (stdout.trim() !== "fixture complete") throw new Error("unexpected headless output")
  if (!events.some((row) => row.type === "llm/retry")) {
    throw new Error("DSH retry event is missing")
  }
  const nativeRequestHeaderCount = events.filter((row) => row.type === "request/header").length
  if (nativeRequestHeaderCount !== 1) {
    throw new Error(`expected one native request/header, got ${nativeRequestHeaderCount}`)
  }
  if (starts.length !== 2 || ends.length !== 2) {
    throw new Error(`expected two Attempts, got ${starts.length}/${ends.length}`)
  }
  if (starts[0].logical_call_id !== starts[1].logical_call_id) {
    throw new Error("retry crossed a logical Call boundary")
  }
  if (starts[0].request_id !== starts[1].request_id) {
    throw new Error("retry changed the effective Request")
  }
  if (typeof starts[0].request_id !== "string" || starts[0].request_id.length === 0) {
    throw new Error("Request identity is missing")
  }
  for (const start of starts) {
    if (
      start.provider !== "dsh-fixture"
      || start.model !== "dsh-fixture-model"
      || start.reasoning_effort !== "off"
    ) {
      throw new Error("observer recorded an unexpected effective condition")
    }
  }
  if (starts[0].attempt_ordinal !== 1 || starts[1].attempt_ordinal !== 2) {
    throw new Error("Attempt ordinals are not 1,2")
  }
  if (starts[1].retry_of_attempt_id !== starts[0].attempt_id) {
    throw new Error("retry linkage is missing")
  }
  if (ends[0].finish_reason !== "error" || ends[0].failure_code !== "DSH_FIXTURE_RETRY") {
    throw new Error("first Attempt did not end with the fixture retry")
  }
  if (ends[1].finish_reason !== "stop" || ends[1].observer_finalized !== true) {
    throw new Error("second Attempt did not reach a finalized stop")
  }
  if (ends[1].usage?.inputTokens !== 20 || ends[1].usage?.outputTokens !== 4) {
    throw new Error("terminal provider usage drifted")
  }
  if (starts[0].request_shape?.tool_count !== 5) {
    throw new Error("router-standard first-Request tool count drifted")
  }
  return {
    logicalCallCount: 1,
    attemptCount: 2,
    retryCount: 1,
    agentPreset: "router-standard",
    firstRequestToolCount: 5,
    workspaceIdentityVerified: expectedWorkspace !== undefined,
    permissionPreset: "unrestricted",
    sandboxMode: "danger-full-access",
    approvalPolicy: "never",
    packedRowCount,
    nativeRequestHeaderCount,
    effectiveCondition: {
      provider: "dsh-fixture",
      model: "dsh-fixture-model",
      reasoningEffort: "off",
      temperature: 0,
      maxTokens: 64,
      stop: [],
    },
    terminalUsage: ends[1].usage,
  }
}
