const SPEC_PERSONA = 'You are a helpful software engineer assistant.'

const MIXED_PERSONA =
  'You are a helpful software engineer assistant.\n'
  + 'Work directly: prefer writing or editing code over describing plans. '
  + 'Verify your changes by reading and running them.'

const REACT_PERSONA =
  'You are a hands-on software engineer who delivers working output fast.\n'
  + 'Carry out the requested work, verify it, and finish with a usable deliverable '
  + 'and a short summary.'

const WEAK_PRO =
  'You are a helpful software engineer assistant.\n'
  + 'Decide the task type internally, choose an appropriate working style, and carry '
  + 'out the task in this response. Do not stop after stating the classification.'

const WEAK_FLASH =
  'You are a helpful assistant.\n'
  + 'Decide the task type internally, choose an appropriate working style, and carry '
  + 'out the task in this response. Continue from completed work and avoid redundant '
  + 'environment checks or exhaustive scans.'

const COMPLEX_RE =
  /(\u91cd\u6784|\u67b6\u6784|\u5168\u9762|\u8be6\u7ec6|\u8bbe\u8ba1|\u7cfb\u7edf|\u4f18\u5316|\u5206\u6790|survey|overview|architecture|refactor|comprehensive|detailed|design|system|optimize|analyze)/i
const REACT_RE =
  /(\u5f00\u53d1|\u521b\u5efa|\u5199\u4e00\u4e2a|\u751f\u6210|\u4ece\u96f6|\u505a\u4e00\u4e2a|\u6e38\u620f|\u7f51\u9875|\u7f51\u7ad9|\u6784\u5efa|\u65b0\u9879\u76ee|\u642d\u5efa|\u5b9e\u73b0|\u505a\u51fa|\u4e0a\u7ebf|\u843d\u5730|\u811a\u672c|\u5de5\u5177|\u5e94\u7528|\u8fd0\u884c|\u8c03\u7528|\u6267\u884c|build|create|develop|generate|implement|make a|new project|run|call|invoke|execute)/gi
const SPEC_RE =
  /(\u4fee\u590d|\u4fee\u4e00\u4e0b|\u8c03\u8bd5|\u91cd\u6784|\u7ef4\u62a4|\u6392\u67e5|\u62a5\u9519|\u51fa\u9519|\u5d29\u6e83|\u4f18\u5316|\u5ba1\u67e5|review|fix|debug|refactor|maintain|repair|broken|break|\u4e3a\u4ec0\u4e48|\u5f02\u5e38|\u6545\u969c|\u8fc1\u79fb|\u5347\u7ea7|\u517c\u5bb9)/gi

export function isComplexTask(text) {
  return typeof text === 'string' && (text.length > 120 || COMPLEX_RE.test(text))
}

function isFlashModel(modelId) {
  return typeof modelId === 'string' && /flash/i.test(modelId)
}

function clamp01(value) {
  return Math.min(1, Math.max(0, Number(value) || 0))
}

export function bandOf(mode) {
  if (mode === 'weak') return 'weak'
  const value = clamp01(mode)
  if (value < 0.2) return 'spec'
  if (value < 0.5) return 'transition'
  return 'react'
}

export function personaFor(mode, modelId) {
  switch (bandOf(mode)) {
    case 'spec': return SPEC_PERSONA
    case 'transition': return MIXED_PERSONA
    case 'weak': return isFlashModel(modelId) ? WEAK_FLASH : WEAK_PRO
    default: return REACT_PERSONA
  }
}

export function coreFor(mode) {
  switch (bandOf(mode)) {
    case 'spec': return ['read', 'edit', 'glob', 'grep']
    case 'transition': return ['read', 'edit', 'write', 'glob', 'grep']
    default: return ['read', 'write', 'edit']
  }
}

function countHits(regex, text) {
  return [...text.matchAll(regex)].length
}

export function classifyTask(text) {
  const react = countHits(REACT_RE, text)
  const spec = countHits(SPEC_RE, text)
  if (react > spec) return 1
  if (spec > react) return 0
  return 'weak'
}

export function extractText(data) {
  if (!data) return ''
  const payload = typeof data.message === 'object' && data.message !== null
    ? data.message
    : data
  const content = Array.isArray(payload.content) ? payload.content : []
  return content.map((item) => (typeof item === 'string' ? item : (item.text ?? ''))).join(' ')
}

export function sessionMode(session) {
  const userMessage = session.events.find(
    (event) => event.type === 'user/message'
      && (event.data?.source?.kind === 'user' || event.data?.source?.kind === undefined),
  ) ?? session.events.find((event) => event.type === 'user/message')
  return classifyTask(extractText(userMessage?.data))
}

export function applyPersona(sections, personaText) {
  const remaining = (sections || []).filter(
    (section) => section.name !== 'persona' && !/persona/i.test(section.name),
  )
  return [...remaining, { name: 'router-persona', text: personaText, order: 0 }]
}
