import { createServer } from "node:http"
import { randomUUID } from "node:crypto"
import { fileURLToPath } from "node:url"
import { dirname, join, resolve } from "node:path"

import { QwenWorker, LOCAL_QWEN_MODEL } from "./qwen-adapter.mjs"
import { QwenOutputParser } from "./qwen-protocol.mjs"

const here = dirname(fileURLToPath(import.meta.url))
const snapshot = resolve(process.env.DSH_QWEN_SNAPSHOT || "")
const port = Number(process.env.DSH_QWEN_HTTP_PORT || 8765)
const host = "127.0.0.1"
if (!process.env.DSH_QWEN_SNAPSHOT) throw new Error("DSH_QWEN_SNAPSHOT is required")
if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error("invalid port")

const worker = new QwenWorker({
  python: process.env.DSH_QWEN_PYTHON || "python",
  worker: join(here, "qwen-worker.py"),
  snapshot,
})
const metadata = await worker.start()
let previous = Promise.resolve()

function json(res, status, body) {
  const data = Buffer.from(JSON.stringify(body))
  res.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "content-length": data.length,
  })
  res.end(data)
}

function error(res, status, message, code = "invalid_request_error") {
  if (!res.headersSent) json(res, status, { error: { message, type: code, code } })
  else res.end()
}

async function bodyJson(req) {
  const pieces = []
  let length = 0
  for await (const chunk of req) {
    length += chunk.length
    if (length > 8 * 1024 * 1024) throw new Error("request body exceeds 8 MiB")
    pieces.push(chunk)
  }
  return JSON.parse(Buffer.concat(pieces).toString("utf8"))
}

function normalizeMessages(messages) {
  if (!Array.isArray(messages) || messages.length === 0) throw new Error("messages are required")
  return messages.map((message) => {
    if (!message || typeof message.role !== "string") throw new Error("invalid message")
    const content = typeof message.content === "string" ? message.content : message.content == null ? "" :
      Array.isArray(message.content) && message.content.every((x) => x.type === "text") ?
        message.content.map((x) => x.text).join("") : null
    if (content === null) throw new Error("local Qwen accepts text-only messages")
    const result = { role: message.role, content }
    if (message.role === "assistant" && Array.isArray(message.tool_calls) && message.tool_calls.length) {
      result.tool_calls = message.tool_calls.map((call) => ({
        type: "function",
        function: {
          name: call.function.name,
          arguments: typeof call.function.arguments === "string" ?
            JSON.parse(call.function.arguments) : call.function.arguments,
        },
      }))
    }
    return result
  })
}

function validate(body) {
  if (!body || typeof body !== "object") throw new Error("request must be a JSON object")
  if (body.model !== LOCAL_QWEN_MODEL && body.model !== `openai/${LOCAL_QWEN_MODEL}`) {
    throw new Error(`unsupported model: ${body.model}`)
  }
  if (body.temperature != null && Number(body.temperature) !== 0) {
    throw new Error("local Qwen requires temperature=0")
  }
  if (body.top_p != null && Number(body.top_p) !== 1) {
    throw new Error("local Qwen requires top_p=1")
  }
  if (body.n != null && Number(body.n) !== 1) throw new Error("local Qwen requires n=1")
  const maxTokens = Number(body.max_completion_tokens ?? body.max_tokens ?? 1024)
  if (!Number.isInteger(maxTokens) || maxTokens < 1 || maxTokens > 32768) {
    throw new Error("max_tokens must be an integer in [1, 32768]")
  }
  const stop = body.stop == null ? [] : typeof body.stop === "string" ? [body.stop] : body.stop
  if (!Array.isArray(stop) || stop.some((x) => typeof x !== "string" || !x)) {
    throw new Error("stop must be a string or an array of nonempty strings")
  }
  const tools = body.tool_choice === "none" ? [] : body.tools ?? []
  if (!Array.isArray(tools)) throw new Error("tools must be an array")
  for (const tool of tools) {
    if (tool?.type !== "function" || typeof tool.function?.name !== "string") {
      throw new Error("only function tools are supported")
    }
  }
  return {
    messages: normalizeMessages(body.messages),
    tools,
    temperature: 0,
    max_tokens: maxTokens,
    stop,
  }
}

async function completion(req, res, body) {
  let payload
  try { payload = validate(body) } catch (cause) {
    error(res, 400, cause.message)
    return
  }
  const id = `chatcmpl-local-${randomUUID()}`
  const created = Math.floor(Date.now() / 1000)
  const stream = body.stream === true
  const signal = new AbortController()
  res.on("close", () => { if (!res.writableEnded) signal.abort(new Error("client disconnected")) })
  const send = (delta, finish_reason = null, usage = undefined) => {
    const chunk = {
      id, object: "chat.completion.chunk", created, model: LOCAL_QWEN_MODEL,
      choices: delta === null ? [] : [{ index: 0, delta, finish_reason }],
      ...(usage ? { usage } : {}),
    }
    res.write(`data: ${JSON.stringify(chunk)}\n\n`)
  }
  if (stream) {
    res.writeHead(200, {
      "content-type": "text/event-stream; charset=utf-8",
      "cache-control": "no-cache",
      connection: "keep-alive",
    })
    send({ role: "assistant", content: "" })
  }
  const parser = new QwenOutputParser()
  let terminal
  let content = ""
  let output = ""
  let parsed
  const started = Date.now()
  try {
    for await (const event of worker.stream(payload, signal.signal)) {
      if (event.type === "delta") {
        output += event.text
        for (const visible of parser.push(event.text)) {
          content += visible
          if (stream) send({ content: visible })
        }
      } else if (event.type === "flush") {
        parsed = parser.finish()
        for (const visible of parsed.visible) {
          content += visible
          if (stream) send({ content: visible })
        }
      } else if (event.type === "result") terminal = event
    }
    if (!terminal || output !== terminal.output) throw new Error("worker output mismatch")
    parsed ??= parser.finish()
    const toolCalls = terminal.finish_reason === "max-tokens" ? [] : parsed.tools.map((tool, index) => ({
      id: `call-local-qwen-${randomUUID()}`, index, type: "function",
      function: { name: tool.name, arguments: tool.arguments },
    }))
    const finishReason = terminal.finish_reason === "max-tokens" ? "length" :
      toolCalls.length ? "tool_calls" : "stop"
    const usage = {
      prompt_tokens: terminal.input_tokens,
      completion_tokens: terminal.output_tokens,
      total_tokens: terminal.input_tokens + terminal.output_tokens,
    }
    if (stream) {
      for (const tool of toolCalls) send({ tool_calls: [tool] })
      send({}, finishReason)
      send(null, null, usage)
      res.end("data: [DONE]\n\n")
    } else {
      json(res, 200, {
        id, object: "chat.completion", created, model: LOCAL_QWEN_MODEL,
        choices: [{ index: 0, message: {
          role: "assistant", content: content || null,
          ...(toolCalls.length ? { tool_calls: toolCalls.map(({ index, ...call }) => call) } : {}),
        }, finish_reason: finishReason }],
        usage,
      })
    }
    console.log(JSON.stringify({
      event: "completion", model: LOCAL_QWEN_MODEL, stream,
      ...usage, finish_reason: finishReason, tool_calls: toolCalls.length,
      seconds: Math.round((Date.now() - started) / 100) / 10,
    }))
  } catch (cause) {
    console.error(JSON.stringify({ event: "completion_error", message: String(cause) }))
    error(res, String(cause).includes("CONTEXT_WINDOW_EXCEEDED") ? 400 : 500, String(cause))
  }
}

const server = createServer((req, res) => {
  const path = req.url?.split("?", 1)[0]
  if (req.method === "GET" && path === "/healthz") {
    json(res, 200, { status: "ready", model: LOCAL_QWEN_MODEL, revision: metadata.revision,
      precision: metadata.precision })
    return
  }
  if (req.method === "GET" && path === "/v1/models") {
    json(res, 200, { object: "list", data: [{ id: LOCAL_QWEN_MODEL, object: "model" }] })
    return
  }
  if (req.method !== "POST" || path !== "/v1/chat/completions") {
    error(res, 404, "unknown route")
    return
  }
  bodyJson(req).then((body) => {
    const task = previous.then(() => completion(req, res, body))
    previous = task.catch(() => {})
  }).catch((cause) => error(res, 400, String(cause)))
})
server.listen(port, host, () => console.log(JSON.stringify({ event: "ready", host, port,
  model: LOCAL_QWEN_MODEL, revision: metadata.revision, precision: metadata.precision,
  worker_pid: worker.process?.pid })))

for (const name of ["SIGINT", "SIGTERM"]) {
  process.on(name, () => { server.close(); void worker.dispose().then(() => process.exit(0)) })
}
