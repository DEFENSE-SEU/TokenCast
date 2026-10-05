import { subscriptionReadiness } from "./subscription-adapter.mjs"

const authFile = process.argv[2]

try {
  const result = await subscriptionReadiness(
    authFile === undefined ? undefined : { authFile },
  )
  process.stdout.write(`${JSON.stringify(result)}\n`)
} catch (error) {
  process.stderr.write(
    `subscription preflight: ${error instanceof Error ? error.message : String(error)}\n`,
  )
  process.exitCode = 1
}
