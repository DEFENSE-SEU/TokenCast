function textFromBlocks(blocks) {
  return blocks
    .filter((block) => block.type === "text" || block.type === "reasoning")
    .map((block) => block.text)
    .join("")
}

function toolResultText(block) {
  return block.content
    .filter((item) => item.type === "text" || item.type === "reasoning")
    .map((item) => item.text)
    .join("")
}

function containsImage(block) {
  if (block?.type === "image") return true
  if (block?.type !== "tool-result" || !Array.isArray(block.content)) return false
  return block.content.some(containsImage)
}

export function projectQwenMessages(options) {
  const messages = []
  if (options.system) messages.push({ role: "system", content: options.system })
  for (const message of options.messages) {
    const unsupported = message.content.find(containsImage)
    if (unsupported) throw new Error("local Qwen DSH adapter accepts text-only messages")
    const toolResults = message.content.filter((block) => block.type === "tool-result")
    if (toolResults.length > 0) {
      for (const block of toolResults) {
        messages.push({ role: "tool", content: toolResultText(block) })
      }
      continue
    }
    if (message.role === "assistant") {
      const toolCalls = message.content
        .filter((block) => block.type === "tool-call")
        .map((block) => ({
          type: "function",
          function: {
            name: block.name,
            arguments: JSON.parse(block.arguments),
          },
        }))
      messages.push({
        role: "assistant",
        content: textFromBlocks(message.content),
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      })
      continue
    }
    messages.push({ role: message.role, content: textFromBlocks(message.content) })
  }
  return messages
}

export function projectQwenTools(tools = []) {
  return tools.map((tool) => ({
    type: "function",
    function: {
      name: tool.name,
      description: tool.description,
      parameters: tool.parameters,
    },
  }))
}

const TOOL_OPEN = "<tool_call>"
const TOOL_CLOSE = "</tool_call>"
const JSON_WHITESPACE = new Set([" ", "\t", "\r", "\n"])

function skipJsonWhitespace(value, start) {
  let index = start
  while (index < value.length && JSON_WHITESPACE.has(value[index])) index += 1
  return index
}

function jsonStringEnd(value, start) {
  let escaped = false
  for (let index = start + 1; index < value.length; index += 1) {
    const character = value[index]
    if (escaped) escaped = false
    else if (character === "\\") escaped = true
    else if (character === '"') return index + 1
  }
  throw new Error("local Qwen emitted an invalid tool call")
}

function jsonValueEnd(value, start) {
  if (value[start] === '"') return jsonStringEnd(value, start)
  if (value[start] === "{" || value[start] === "[") {
    let depth = 0
    for (let index = start; index < value.length; index += 1) {
      const character = value[index]
      if (character === '"') {
        index = jsonStringEnd(value, index) - 1
        continue
      }
      if (character === "{" || character === "[") depth += 1
      else if (character === "}" || character === "]") {
        depth -= 1
        if (depth === 0) return index + 1
      }
    }
    throw new Error("local Qwen emitted an invalid tool call")
  }
  let index = start
  while (
    index < value.length
    && !JSON_WHITESPACE.has(value[index])
    && value[index] !== ","
    && value[index] !== "}"
    && value[index] !== "]"
  ) index += 1
  return index
}

function topLevelJsonMembers(payload) {
  JSON.parse(payload)
  let index = skipJsonWhitespace(payload, 0)
  if (payload[index] !== "{") return []
  index = skipJsonWhitespace(payload, index + 1)
  const members = []
  while (payload[index] !== "}") {
    if (payload[index] !== '"') throw new Error("local Qwen emitted an invalid tool call")
    const keyEnd = jsonStringEnd(payload, index)
    const key = JSON.parse(payload.slice(index, keyEnd))
    index = skipJsonWhitespace(payload, keyEnd)
    if (payload[index] !== ":") throw new Error("local Qwen emitted an invalid tool call")
    const valueStart = skipJsonWhitespace(payload, index + 1)
    const valueEnd = jsonValueEnd(payload, valueStart)
    members.push({ key, rawValue: payload.slice(valueStart, valueEnd) })
    index = skipJsonWhitespace(payload, valueEnd)
    if (payload[index] === ",") index = skipJsonWhitespace(payload, index + 1)
    else if (payload[index] !== "}") throw new Error("local Qwen emitted an invalid tool call")
  }
  return members
}

function toolCloseIndex(value) {
  let depth = 0
  let escaped = false
  let inString = false
  for (let index = 0; index < value.length; index += 1) {
    const character = value[index]
    if (inString) {
      if (escaped) escaped = false
      else if (character === "\\") escaped = true
      else if (character === '"') inString = false
      continue
    }
    if (character === '"') inString = true
    else if (character === "{" || character === "[") depth += 1
    else if (character === "}" || character === "]") depth -= 1
    else if (depth === 0 && value.startsWith(TOOL_CLOSE, index)) return index
  }
  return -1
}

function heldPrefixLength(value) {
  const limit = Math.min(value.length, TOOL_OPEN.length - 1)
  for (let length = limit; length > 0; length -= 1) {
    if (TOOL_OPEN.startsWith(value.slice(-length))) return length
  }
  return 0
}

export class QwenOutputParser {
  constructor() {
    this.buffer = ""
    this.state = "text"
    this.sawTool = false
    this.tools = []
    this.text = ""
  }

  push(delta) {
    this.buffer += delta
    const visible = []
    while (this.buffer) {
      if (this.state === "tool") {
        const close = toolCloseIndex(this.buffer)
        if (close < 0) break
        const payload = this.buffer.slice(0, close).trim()
        this.buffer = this.buffer.slice(close + TOOL_CLOSE.length)
        this.state = "text"
        try {
          this.addTool(payload)
        } catch (error) {
          // Keep malformed tool calls visible so the model can correct them on a later step.
          this.addText(`${TOOL_OPEN}${payload}${TOOL_CLOSE}`, visible)
        }
        continue
      }
      const open = this.buffer.indexOf(TOOL_OPEN)
      if (open >= 0) {
        this.addText(this.buffer.slice(0, open), visible)
        this.buffer = this.buffer.slice(open + TOOL_OPEN.length)
        this.state = "tool"
        continue
      }
      if (this.buffer.includes(TOOL_CLOSE)) {
        throw new Error("local Qwen emitted an unmatched tool-call close tag")
      }
      // Keep a partial opening tag until the next stream chunk arrives.
      const held = heldPrefixLength(this.buffer)
      const boundary = this.buffer.length - held
      this.addText(this.buffer.slice(0, boundary), visible)
      this.buffer = this.buffer.slice(boundary)
      break
    }
    return visible
  }

  addText(value, visible) {
    if (!value) return
    if (this.sawTool) {
      return
    }
    this.text += value
    visible.push(value)
  }

  addTool(payload) {
    const members = topLevelJsonMembers(payload)
    const names = members.filter((member) => member.key === "name")
    const argumentsMembers = members.filter((member) => member.key === "arguments")
    if (names.length !== 1 || argumentsMembers.length !== 1) {
      throw new Error("local Qwen emitted an invalid tool call")
    }
    const name = JSON.parse(names[0].rawValue)
    if (typeof name !== "string") throw new Error("local Qwen emitted an invalid tool call")
    const rawArguments = argumentsMembers[0].rawValue
    const parsedArguments = JSON.parse(rawArguments)
    if (
      typeof parsedArguments !== "object"
      || parsedArguments === null
      || Array.isArray(parsedArguments)
    ) {
      throw new Error("local Qwen emitted invalid tool arguments")
    }
    this.tools.push({ name, arguments: rawArguments })
    this.sawTool = true
  }

  finish() {
    if (this.state === "tool" || this.buffer.includes("<tool")) {
      throw new Error("local Qwen emitted an incomplete tool call")
    }
    const visible = []
    this.addText(this.buffer, visible)
    this.buffer = ""
    return { text: this.text, tools: [...this.tools], visible }
  }
}

export function parseQwenOutput(output) {
  const parser = new QwenOutputParser()
  parser.push(output)
  const parsed = parser.finish()
  return { text: parsed.text.trim(), tools: parsed.tools }
}

export function splitVisibleText(value, width = 24) {
  const characters = [...value]
  const chunks = []
  for (let index = 0; index < characters.length; index += width) {
    chunks.push(characters.slice(index, index + width).join(""))
  }
  return chunks
}
