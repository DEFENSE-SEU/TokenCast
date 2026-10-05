import { randomUUID } from "node:crypto"
import { writeFile } from "node:fs/promises"

import { importDshHostModule } from "./host.mjs"
import { applyToolPolicy, ALLOWED_TOOL_NAMES } from "./tool-policy.mjs"
import { providerAccounting } from "./provider-accounting.mjs"
import {
  buildModelSelection,
  buildProviderRequestOptions,
  unsupportedRequestOptions,
} from "./provider-request-options.mjs"

const agentPackage = await importDshHostModule("@deepseek-ai/dsh-agent")
const llmPackage = await importDshHostModule("@deepseek-ai/dsh-llm")
const sessionPackage = await importDshHostModule("@deepseek-ai/dsh-session")
const schemastery = await importDshHostModule("@deepseek-ai/schemastery")
const { installModelSelection } = agentPackage
const { createUserMessage, freezeMessage, isAgentLoopRequest } = llmPackage
const { SessionId } = sessionPackage
const z = schemastery.default ?? schemastery

export { ALLOWED_TOOL_NAMES }
export { providerAccounting }

export const name = "dsh-headless-runner"
export const inject = ["agentPresets", "agents", "llm", "sessionPersistence", "sessions"]
export const Config = z.object({
  task: z.string().required(),
  provider: z.string().required(),
  model: z.string().required(),
  agentPreset: z.string().required(),
  reasoningEffort: z.string().required(),
  temperature: z.number().required(),
  maxTokens: z.number().step(1).min(1).required(),
  stop: z.array(z.string()).default([]),
  maxLogicalCalls: z.number().step(1).min(1),
  maxProviderAccountedTokens: z.number().step(1).min(1),
  exclusiveProvider: z.string(),
  sessionExportPath: z.string().default(""),
  dryRun: z.boolean().default(false),
  toolResultMaxUtf8Bytes: z.number().step(1).min(0),
  toolResultHistoryKeepRecent: z.number().step(1).min(0),
})

function summarize(events, firstSeq) {
  let started = false
  let text = ""
  let reason
  for (const event of events) {
    if (event.seq < firstSeq) continue
    if (event.type === "turn/start") {
      started = true
      continue
    }
    if (!started) continue
    if (event.type === "assistant/message") {
      const current = event.data.message.content
        .filter((block) => block.type === "text")
        .map((block) => block.text)
        .join("")
      if (current) text = current
    }
    if (event.type === "turn/end") reason = event.data.reason
  }
  return { text, reason }
}

export async function prepareHeadlessRunner(ctx, config) {
  await ctx.get("loader")?.await()
  if (config.exclusiveProvider) {
    const providers = ctx.llm.listProviders().map((provider) => provider.id)
    if (providers.length !== 1 || providers[0] !== config.exclusiveProvider) {
      throw new Error(`DSH headless provider topology drift: ${providers.join(",")}`)
    }
  }
  const resolved = await ctx.llm.resolveModelInfo(config.provider, config.model)
  if (resolved.provider !== config.provider || resolved.id !== config.model) {
    throw new Error(
      `DSH headless model resolution drift: ${resolved.provider}/${resolved.id}`,
    )
  }
}

export async function runHeadlessJob(ctx, config, job, hooks = {}) {
  const toolResultMaxUtf8Bytes = job.tool_result_max_utf8_bytes === undefined
    ? config.toolResultMaxUtf8Bytes : job.tool_result_max_utf8_bytes
  const toolResultHistoryKeepRecent = job.tool_result_history_keep_recent === undefined
    ? config.toolResultHistoryKeepRecent : job.tool_result_history_keep_recent
  const contentPolicy = toolResultMaxUtf8Bytes == null && toolResultHistoryKeepRecent == null ? null
    : await import("../dsh-token-observer/request-content-ledger.mjs")
  const selection = buildModelSelection(
    config.provider,
    config.model,
    job.decoding.reasoning_effort,
  )
  const sessionId = SessionId(job.session_id)
  let agent
  let handle
  let firstSeq = 0
  let logicalCalls = 0
  const logicalCallKeys = new Set()
  let outcome
  let jobError
  let sessionPath
  let started = false
  let activeRequestStep = null
  let budgetShadow = null
  try {
    if (job.budget_shadow !== undefined) {
      budgetShadow = (await import("./budget-shadow.mjs")).createBudgetShadow({
        path: job.budget_shadow.output_path,
        sourceId: job.budget_shadow.source_id,
        taskFamily: job.budget_shadow.task_family,
        runId: job.budget_shadow.run_id,
        sessionId: String(sessionId),
        provider: config.provider,
        model: config.model,
        budgetTokens: job.budget_shadow.budget_tokens,
        marginTokens: job.budget_shadow.margin_tokens,
        measure() {
          const meter = ctx.get("tokenMeter")
          if (meter === undefined || agent === undefined) {
            throw new Error("Budget shadow requires the DSH tokenMeter and active session")
          }
          const header = agent.session.requestHeader()
          if (header === undefined) throw new Error("Budget shadow: current request header missing")
          return meter.measure(agent.session, header)
        },
        accountedBefore() {
          if (agent === undefined) throw new Error("Budget shadow requires an active session")
          const accounting = providerAccounting(agent.session.events.slice(firstSeq))
          return accounting.terminalWithoutUsage ? null : accounting.tokens
        },
        currentStep: () => activeRequestStep,
      })
    }
    handle = await ctx.agents.create({
      sessionId,
      meta: { cwd: job.cwd, agentPreset: config.agentPreset },
      agentOptions: {
        provider: config.provider,
        model: config.model,
        maxTokens: job.decoding.max_tokens,
      },
      async setup(agentCtx) {
        if (contentPolicy !== null) {
          agentCtx.effect(() => () => contentPolicy.disposeRequestContentSources(String(sessionId)), "request content policy")
        }
        if (toolResultMaxUtf8Bytes != null) {
          contentPolicy.registerRequestContentPolicy(String(sessionId), toolResultMaxUtf8Bytes)
          agentCtx.on("tools/post-execute", contentPolicy.createToolResultContentPolicy({
            maxUtf8Bytes: toolResultMaxUtf8Bytes,
          }))
        }
        if (toolResultHistoryKeepRecent != null) {
          contentPolicy.registerRequestHistoryPolicy(String(sessionId), toolResultHistoryKeepRecent)
          agentCtx.on("agent/pre-step", async (_payload, next) => {
            const decision = await next()
            if (decision.kind === "enter") {
              const meter = ctx.get("tokenMeter")
              if (meter === undefined) throw new Error("tool result history retention requires DSH tokenMeter")
              contentPolicy.pruneToolResultHistory(agent.session, {
                keepRecent: toolResultHistoryKeepRecent, freezeMessage,
                estimateMessage: (message) => meter.estimateMessage(message),
              })
            }
            return decision
          })
        }
        agentCtx.on("system-prompt/assemble", async (_assembly, _context, next) => {
          const assembled = await next()
          return applyToolPolicy(assembled, config.provider, job.system_prompt_policy_id)
        })
        const preset = await ctx.agentPresets.mount(agentCtx, config.agentPreset)
        if (preset.id !== config.agentPreset) {
          throw new Error(`DSH headless preset drift: ${preset.id}`)
        }
        agentCtx.tools.restrict({ allow: [...ALLOWED_TOOL_NAMES] })
        installModelSelection(agentCtx, { current: selection, assembled: undefined })
        if (budgetShadow !== null) {
          agentCtx.on("llm/stream", (options, next) =>
            isAgentLoopRequest(options) ? budgetShadow.middleware(options, next) : next())
        }
        agentCtx.on("agent/request", async (payload, next) => {
          activeRequestStep = { turn: payload.turn, step: payload.step }
          const callKey = `${payload.turn}:${payload.step}`
          if (!logicalCallKeys.has(callKey)) {
            if (job.caps.max_logical_calls != null && logicalCalls >= job.caps.max_logical_calls) {
              throw new Error("DSH logical Call cap reached")
            }
            const accounting = providerAccounting(agent.session.events.slice(firstSeq))
            if (accounting.terminalWithoutUsage) {
              throw new Error("DSH Attempt usage is missing before the next logical Call")
            }
            if (job.caps.max_provider_accounted_tokens != null && accounting.tokens >= job.caps.max_provider_accounted_tokens) {
              throw new Error("DSH provider-accounted Token cap reached")
            }
            logicalCallKeys.add(callKey)
            logicalCalls += 1
          }
          await hooks.requestBoundaryCrossed?.()
          const options = {
            ...(await next()),
            ...buildProviderRequestOptions(config.provider, job.decoding, config.model),
          }
          for (const key of unsupportedRequestOptions(
            config.provider, config.model, job.decoding.reasoning_effort,
          )) {
            delete options[key]
          }
          return options
        })
      },
    })
    agent = handle.agent
    const location = ctx.sessionPersistence.locate(agent.session.header)
    if (location === undefined) {
      throw new Error("DSH session persistence has no physical artifact location")
    }
    sessionPath = location.path
    await hooks.started?.({ sessionId: String(agent.session.id), sessionPath })
    started = true
    await agent.whenIdle()
    firstSeq = agent.session.seq
    agent.followup(
      createUserMessage({
        content: [{ type: "text", text: job.task }],
        source: { kind: "user" },
      }),
    )
    await agent.whenIdle()
    outcome = summarize(agent.session.events, firstSeq)
    if (outcome.reason === undefined) throw new Error("DSH job ended without a turn terminal")
  } catch (error) {
    if (!started) throw error
    jobError = error
  } finally {
    try {
      if (handle !== undefined) {
        try {
          if (agent !== undefined) {
            const participated = await ctx.sessions.flush(agent.session)
            if (!participated) {
              throw new Error("DSH session flush had no durability listener")
            }
          }
        } finally {
          await handle.dispose()
        }
        if (agent !== undefined) {
          await ctx.sessionPersistence.load(agent.session.id)
          if (jobError === undefined && job.session_export_path) {
            const artifact = await ctx.sessionPersistence.readRaw(agent.session.id)
            if (artifact === undefined) {
              throw new Error("DSH persisted session artifact is missing")
            }
            await writeFile(job.session_export_path, artifact.content, {
              encoding: "utf8",
              flag: "wx",
            })
          }
        }
      }
    } finally {
      budgetShadow?.close()
    }
  }
  return {
    ...outcome,
    error: jobError,
    sessionId: String(sessionId),
    sessionPath,
  }
}

async function run(ctx, config, io) {
  await prepareHeadlessRunner(ctx, config)
  if (config.dryRun) {
    io.stdout.write(`${JSON.stringify({
      status: "ready",
      provider: config.provider,
      model: config.model,
    })}\n`)
    io.exit(0)
    return
  }
  const outcome = await runHeadlessJob(ctx, config, {
    session_id: `session-${randomUUID()}`,
    cwd: process.cwd(),
    task: config.task,
    decoding: {
      reasoning_effort: config.reasoningEffort,
      temperature: config.temperature,
      max_tokens: config.maxTokens,
      stop: [...config.stop],
    },
    caps: {
      max_logical_calls: config.maxLogicalCalls,
      max_provider_accounted_tokens: config.maxProviderAccountedTokens,
    },
    session_export_path: config.sessionExportPath,
  })
  if (outcome.error !== undefined) throw outcome.error
  io.stdout.write(`${outcome.text}\n`)
  if (outcome.reason?.kind === "error") {
    io.stderr.write(`dsh: ${outcome.reason.error.code}: ${outcome.reason.error.message}\n`)
  }
  io.exit(outcome.reason?.kind === "completed" ? 0 : 1)
}

export function apply(ctx, config) {
  const exit = ctx.get("appExit")
  if (exit === undefined) {
    throw new Error("dsh-headless-runner requires the DSH appExit service")
  }
  const io = { stdout: process.stdout, stderr: process.stderr, exit }
  run(ctx, config, io).catch((error) => {
    io.stderr.write(`dsh: ${error instanceof Error ? error.message : String(error)}\n`)
    io.exit(1)
  })
}
