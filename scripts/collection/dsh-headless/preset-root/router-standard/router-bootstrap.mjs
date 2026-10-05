import {
  applyPersona,
  bandOf,
  classifyTask,
  coreFor,
  extractText,
  isComplexTask,
  personaFor,
  sessionMode,
} from './router-core.mjs'

export const name = 'router-bootstrap'
export const inject = ['systemPrompt', 'tools']

export function apply(ctx, config) {
  const firstUserText = new Map()

  function currentMode(session) {
    const text = firstUserText.get(session.id)
    return text === undefined ? sessionMode(session) : classifyTask(text)
  }

  ctx.on('agent/inbox/claimed', ({ agent, message }) => {
    if (message?.source?.kind !== 'user') return
    const text = extractText(message).trim()
    const session = agent?.session
    if (session !== undefined && text && !firstUserText.has(session.id)) {
      firstUserText.set(session.id, text)
    }
  })

  ctx.on('system-prompt/assemble', async (_assembly, context, next) => {
    const assembled = await next()
    const agent = context.agent
    if (agent === undefined || agent.session?.header?.parentSession !== undefined) {
      return assembled
    }

    const session = agent.session
    const mode = currentMode(session)
    const modelId = assembled.variables?.model ?? agent.options?.model
    const sections = applyPersona(assembled.sections, personaFor(mode, modelId))

    if (session.events.some((event) => event.type === 'tool/call')) {
      return { ...assembled, sections, contexts: [] }
    }

    const core = new Set(coreFor(mode))
    const available = new Set(assembled.tools.map((tool) => tool.name))
    if (!available.has('pwsh')) {
      throw new Error(`${name}: pwsh is absent from the DSH tool catalog`)
    }
    core.add('pwsh')
    return {
      ...assembled,
      sections,
      contexts: [],
      tools: assembled.tools.filter((tool) => core.has(tool.name)),
    }
  })

  const GUIDE_WEAK =
    '\nRouter: decide the task type internally, choose the matching working style, '
    + 'and carry out the task now. Do not stop after announcing the classification.'
  const GUIDE_DEEP =
    '\nRouter: decide the task type internally, then carry out the task now. Consider '
    + 'architecture, edge cases, and integration points before producing the result.'

  ctx.on('session/event', (session, event) => {
    if (event.type !== 'user/message' || event.data?.source?.kind !== 'user') return
    const text = extractText(event.data).trim()
    if (text && !firstUserText.has(session.id)) firstUserText.set(session.id, text)
  })

  ctx.on('agent/pre-step', async (payload, next) => {
    const decision = await next()
    if (decision.kind !== 'enter') return decision
    const session = payload.agent?.session
    if (session === undefined || bandOf(currentMode(session)) !== 'weak') return decision
    const messages = decision.messages ?? []
    if (messages.length === 0) return decision

    const guides = []
    for (let index = (payload.messages ?? []).length - 1; index >= 0; index--) {
      const message = payload.messages[index]
      if (message?.source?.kind !== 'user') continue
      const text = extractText(message).trim()
      if (!text) continue
      const id = `router-guide-${message.id}`
      if (session.events.some(
        (event) => event.type === 'user/message' && event.data?.id === id,
      )) continue
      guides.push({
        id,
        role: 'user',
        source: { kind: 'plugin', plugin: 'router-bootstrap' },
        content: [{ type: 'text', text: isComplexTask(text) ? GUIDE_DEEP : GUIDE_WEAK }],
      })
    }

    const output = [...messages]
    for (const guide of guides) {
      const sourceId = guide.id.slice('router-guide-'.length)
      const anchor = output.findIndex((message) => message.id === sourceId)
      if (anchor !== -1) output.splice(anchor + 1, 0, guide)
    }
    return guides.length === 0 ? decision : { ...decision, messages: output }
  })
}
