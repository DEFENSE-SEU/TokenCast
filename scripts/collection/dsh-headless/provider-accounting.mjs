export function providerAccounting(events) {
  let tokens = 0
  let usageSinceTerminal = false
  let terminalWithoutUsage = false
  for (const event of events) {
    if (event.type !== "assistant/chunk") continue
    const chunk = event.data?.chunk
    if (chunk?.type === "usage") {
      const usage = chunk.usage
      tokens +=
        usage.inputTokens
        + usage.outputTokens
        + (usage.cacheReadTokens ?? 0)
        + (usage.cacheWriteTokens ?? 0)
      usageSinceTerminal = true
    } else if (chunk?.type === "finish") {
      // One terminal without usage leaves the session's total accounting incomplete.
      if (!usageSinceTerminal) terminalWithoutUsage = true
      usageSinceTerminal = false
    }
  }
  return { tokens, terminalWithoutUsage }
}
