import { appendFileSync } from "node:fs"
import { createInterface } from "node:readline"

const snapshotIndex = process.argv.indexOf("--snapshot")
const logPath = process.argv[snapshotIndex + 1]
const readyDelayMs = logPath.includes("ready-delay") ? 10_000 : 0

function record(event) {
  appendFileSync(logPath, `${JSON.stringify({ ...event, pid: process.pid })}\n`)
}

function emit(event) {
  process.stdout.write(`${JSON.stringify(event)}\n`)
}

record({
  type: "spawn",
  pythonUtf8: process.env.PYTHONUTF8,
  pythonIoEncoding: process.env.PYTHONIOENCODING,
})
setTimeout(() => {
  emit({
    type: "ready",
    revision: "fake-revision",
    precision: "nf4",
    device: "cuda:0",
    model_class: "FakeQwenModel",
    is_quantized: true,
    quant_type: "nf4",
    double_quant: true,
    compute_dtype: "torch.bfloat16",
    tie_word_embeddings: false,
    linear4bit_modules: 196,
    lm_head_dtype: "torch.bfloat16",
    repetition_penalty: 1.05,
    eos_token_ids: [151645, 151643],
  })
  record({ type: "ready" })
}, readyDelayMs)

let awaitingAck
let output = ""
const lines = createInterface({ input: process.stdin })
lines.on("line", (line) => {
  const message = JSON.parse(line)
  if (message.type === "ack" && message.id === awaitingAck) {
    record({ type: "ack", id: message.id })
    emit({
      type: "result",
      id: message.id,
      output,
      input_tokens: 3,
      output_tokens: 1,
      acknowledged_deltas: 1,
      finish_reason: "stop",
    })
    awaitingAck = undefined
    output = ""
    return
  }
  if (awaitingAck !== undefined) {
    record({ type: "request-while-awaiting-ack", id: message.id })
    return
  }
  const toolResult = message.messages.findLast((item) => item.role === "tool")
  const firstUser = message.messages.find((item) => item.role === "user")
  output = toolResult?.content ?? firstUser?.content ?? message.messages.at(-1).content
  record({
    type: "request",
    id: message.id,
    output,
    generationProbe: message.generation_probe,
  })
  awaitingAck = message.id
  emit({ type: "start", id: message.id, input_tokens: 3 })
  emit({ type: "delta", id: message.id, text: output, output_tokens: 1 })
})
