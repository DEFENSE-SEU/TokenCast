import { createRequire } from "node:module"
import { homedir } from "node:os"
import { join } from "node:path"
import { pathToFileURL } from "node:url"

import { createStreamMiddleware, DshStreamObserver } from "./observer.mjs"
import { ObserverSidecarWriter } from "./sidecar-writer.mjs"
import { RequestContentLedger } from "./request-content-ledger.mjs"

function dshHomePath() {
  return process.env.DSH_HOME || join(homedir(), ".dsh")
}

export function resolveDshHostModuleUrl(specifier) {
  const profileRequire = createRequire(join(dshHomePath(), "profiles", "package.json"))
  return pathToFileURL(profileRequire.resolve(specifier)).href
}

const dshLlm = await import(resolveDshHostModuleUrl("@deepseek-ai/dsh-llm"))
const schemastery = await import(resolveDshHostModuleUrl("@deepseek-ai/schemastery"))
const { isAgentLoopRequest } = dshLlm
const z = schemastery.default ?? schemastery

export const name = "token-prediction-dsh-observer"
export const inject = ["llm", "sessions"]
export const Config = z.object({
  outputDir: z.string().default(""),
  checkpointIntervalBytes: z.number().step(1).min(1).default(128),
  countToolArguments: z.boolean().default(false),
  requestContentLedger: z.boolean().default(false),
  requestContentOutputDir: z.string().default(""),
})

function defaultOutputDir() {
  return join(dshHomePath(), "token-prediction", "sidecars")
}

export function installObserver(ctx, config = {}, sidecars = undefined) {
  const outputDir = config.outputDir || defaultOutputDir()
  const activeSteps = new Map()
  const warnedSessions = new Set()
  const writer = sidecars ?? new ObserverSidecarWriter({ outputDir })
  const contentWriter = config.requestContentLedger
    ? new ObserverSidecarWriter({ outputDir: config.requestContentOutputDir || join(outputDir, "request-content") })
    : null
  const ledger = contentWriter === null ? null : new RequestContentLedger({
    writeRecord: (sessionId, record) => contentWriter.writeRecord(sessionId, record),
  })

  const observer = new DshStreamObserver({
    checkpointIntervalBytes: config.checkpointIntervalBytes ?? 128,
    countToolArguments: config.countToolArguments ?? false,
    onAttempt: ledger === null ? null : (request, facts) => ledger.capture(request, facts),
    writeRecord(sessionId, record) {
      writer.writeRecord(sessionId, record)
    },
  })

  ctx.on(
    "session/event",
    (session, event) => {
      const sessionId = String(session.id)
      ledger?.observeNativeEvent(sessionId, event)
      if (event.type === "step/start") {
        activeSteps.set(sessionId, { turn: event.data.turn, step: event.data.step })
      } else if (event.type === "step/end") {
        activeSteps.delete(sessionId)
      }
    },
    { global: true },
  )
  ctx.on(
    "session/flush",
    (session) => {
      writer.flushSession(String(session.id))
      contentWriter?.flushSession(String(session.id))
    },
    { global: true },
  )
  ctx.on(
    "session/disposed",
    (session) => {
      const sessionId = String(session.id)
      try {
        try { writer.closeSession(sessionId) } finally { contentWriter?.closeSession(sessionId) }
      } finally {
        activeSteps.delete(sessionId)
        warnedSessions.delete(sessionId)
        observer.disposeSession(sessionId)
        ledger?.disposeSession(sessionId)
      }
    },
    { global: true },
  )
  ctx.on(
    "llm/stream",
    createStreamMiddleware({
      observer,
      isAgentLoopRequest,
      activeStep: (sessionId) => activeSteps.get(sessionId),
      onMissingStep(sessionId) {
        if (!warnedSessions.has(sessionId)) {
          warnedSessions.add(sessionId)
          ctx.logger?.warn?.(
            "[token-prediction-observer] skipped a request without an active DSH step: %s",
            sessionId,
          )
        }
      },
    }),
    { global: true },
  )
  ctx.effect(() => () => {
    try { writer.closeAll() } finally { contentWriter?.closeAll() }
  }, "observer sidecar writer")
  return { observer, writer, ledger, contentWriter }
}

export function apply(ctx, config = {}) {
  installObserver(ctx, config)
}
