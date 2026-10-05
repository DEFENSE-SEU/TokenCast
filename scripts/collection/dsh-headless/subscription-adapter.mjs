import { readFile } from "node:fs/promises"
import { findPackageJSON } from "node:module"
import { homedir } from "node:os"
import { dirname, isAbsolute, join } from "node:path"
import { pathToFileURL } from "node:url"

import { importDshHostModule, resolveDshHostModuleUrl } from "./host.mjs"

const EXPECTED_PI_ADAPTER_VERSION = "0.1.1-rc.2"
const EXPECTED_PI_VERSION = "0.82.1"
const AUTH_NAMESPACE = "https://api.openai.com/auth"
const DEFAULT_PROVIDER = "openai-codex"
const DEFAULT_MODEL = "gpt-5.4"

const piAdapterModuleUrl = resolveDshHostModuleUrl("@deepseek-ai/dsh-llm-pi-ai")
const piAdapterManifest = findPackageJSON("@deepseek-ai/dsh-llm-pi-ai", piAdapterModuleUrl)
const piManifest = findPackageJSON("@earendil-works/pi-ai", piAdapterModuleUrl)
if (piAdapterManifest === undefined || piManifest === undefined) {
  throw new Error("subscription adapter dependencies are missing from the DSH profile")
}

const piAdapterPackage = await importDshHostModule("@deepseek-ai/dsh-llm-pi-ai")
const llmPackage = await importDshHostModule("@deepseek-ai/dsh-llm")
const schemastery = await importDshHostModule("@deepseek-ai/schemastery")
const providerPackage = await import(
  pathToFileURL(join(dirname(piManifest), "dist", "providers", "openai-codex.js")).href
)

const { PiAiAdapter } = piAdapterPackage
const { resolveRetryPolicy } = llmPackage
const { openaiCodexProvider } = providerPackage
const z = schemastery.default ?? schemastery

export const name = "dsh-subscription-adapter"
export const inject = ["llm"]
export const Config = z.object({
  authFile: z.string().default(""),
  provider: z.string().default(DEFAULT_PROVIDER),
  model: z.string().default(DEFAULT_MODEL),
  streamIdleTimeoutMs: z.number().min(1).default(300000),
  maxRetries: z.number().min(0).default(1),
})

export class SubscriptionAuthError extends Error {
  constructor(message) {
    super(message)
    this.name = "SubscriptionAuthError"
  }
}

function requireObject(value, label) {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new SubscriptionAuthError(`${label} must be a JSON object`)
  }
  return value
}

function requireText(value, label) {
  if (typeof value !== "string" || value.length === 0) {
    throw new SubscriptionAuthError(`${label} must be a non-empty string`)
  }
  return value
}

function accessClaims(accessToken) {
  const parts = accessToken.split(".")
  if (parts.length !== 3) {
    throw new SubscriptionAuthError("auth.json access token is not a JWT")
  }
  let value
  try {
    value = JSON.parse(Buffer.from(parts[1], "base64url").toString("utf8"))
  } catch {
    throw new SubscriptionAuthError("auth.json access token has an invalid JWT payload")
  }
  return requireObject(value, "auth.json access token payload")
}

export function parseSubscriptionAuthDocument(value) {
  const document = requireObject(value, "auth.json")
  if (document.auth_mode !== "chatgpt") {
    throw new SubscriptionAuthError("auth.json auth_mode must be chatgpt")
  }
  const tokens = requireObject(document.tokens, "auth.json tokens")
  requireText(tokens.id_token, "auth.json tokens.id_token")
  const access = requireText(tokens.access_token, "auth.json tokens.access_token")
  const refresh = requireText(tokens.refresh_token, "auth.json tokens.refresh_token")
  const accountId = requireText(tokens.account_id, "auth.json tokens.account_id")
  const lastRefresh = requireText(document.last_refresh, "auth.json last_refresh")
  if (!Number.isFinite(Date.parse(lastRefresh))) {
    throw new SubscriptionAuthError("auth.json last_refresh must be an ISO timestamp")
  }

  const claims = accessClaims(access)
  const expiresSeconds = claims.exp
  if (!Number.isSafeInteger(expiresSeconds) || expiresSeconds <= 0) {
    throw new SubscriptionAuthError("auth.json access token JWT has no valid exp claim")
  }
  const authClaims = requireObject(claims[AUTH_NAMESPACE], "auth.json access token auth claim")
  const claimedAccountId = requireText(
    authClaims.chatgpt_account_id,
    "auth.json access token account claim",
  )
  if (claimedAccountId !== accountId) {
    throw new SubscriptionAuthError("auth.json account id differs from its access token claim")
  }

  return {
    type: "oauth",
    access,
    refresh,
    expires: expiresSeconds * 1000,
    accountId,
  }
}

export function resolveSubscriptionAuthFile(
  configured = "",
  environment = process.env,
  userHome = homedir(),
) {
  const explicit = configured.trim()
  if (explicit.length > 0) {
    if (!isAbsolute(explicit)) {
      throw new SubscriptionAuthError("subscription auth file path must be absolute")
    }
    return explicit
  }
  const configuredHome = environment.CODEX_HOME?.trim()
  if (configuredHome) {
    if (!isAbsolute(configuredHome)) {
      throw new SubscriptionAuthError("CODEX_HOME must be absolute")
    }
    return join(configuredHome, "auth.json")
  }
  return join(userHome, ".codex", "auth.json")
}

export async function readSubscriptionCredential(authFile) {
  let text
  try {
    text = await readFile(authFile, "utf8")
  } catch (error) {
    if (error?.code === "ENOENT") {
      throw new SubscriptionAuthError("subscription auth.json does not exist")
    }
    throw new SubscriptionAuthError("subscription auth.json cannot be read")
  }
  let document
  try {
    document = JSON.parse(text)
  } catch {
    throw new SubscriptionAuthError("subscription auth.json is not valid JSON")
  }
  return parseSubscriptionAuthDocument(document)
}

export function createReadOnlyCredentialStore({
  provider = DEFAULT_PROVIDER,
  authFile,
  readCredential = readSubscriptionCredential,
  now = () => Date.now(),
}) {
  return {
    async read(providerId) {
      return providerId === provider ? readCredential(authFile) : undefined
    },
    async list() {
      return [{ providerId: provider, type: "oauth" }]
    },
    async modify(providerId, mutate) {
      if (providerId !== provider) return undefined
      const current = await readCredential(authFile)
      if (now() >= current.expires) {
        throw new SubscriptionAuthError(
          "subscription auth.json access token is expired; refresh the Codex sign-in first",
        )
      }
      const next = await mutate(current)
      if (next !== undefined) {
        throw new SubscriptionAuthError(
          "subscription auth.json is read-only; Codex retains OAuth refresh ownership",
        )
      }
      return current
    },
    async delete(providerId) {
      if (providerId === provider) {
        throw new SubscriptionAuthError(
          "subscription auth.json is read-only; use Codex to sign out",
        )
      }
    },
  }
}

async function assertRuntimeVersion(manifestPath, expected, label) {
  let document
  try {
    document = JSON.parse(await readFile(manifestPath, "utf8"))
  } catch {
    throw new Error(`cannot read the installed ${label} package manifest`)
  }
  if (document.version !== expected) {
    throw new Error(`installed ${label} version differs from the audited subscription runtime`)
  }
}

export async function verifySubscriptionRuntime() {
  await Promise.all([
    assertRuntimeVersion(piAdapterManifest, EXPECTED_PI_ADAPTER_VERSION, "DSH pi-ai adapter"),
    assertRuntimeVersion(piManifest, EXPECTED_PI_VERSION, "pi-ai"),
  ])
  return {
    dsh_pi_adapter: EXPECTED_PI_ADAPTER_VERSION,
    pi_ai: EXPECTED_PI_VERSION,
  }
}

export function selectSubscriptionProvider(providerId, modelId) {
  if (providerId !== DEFAULT_PROVIDER) {
    throw new Error(`subscription adapter does not own provider "${providerId}"`)
  }
  const provider = openaiCodexProvider()
  const selected = provider.getModels().find((model) => model.id === modelId)
  if (selected === undefined) {
    throw new Error(`subscription provider has no installed model "${modelId}"`)
  }
  return {
    ...provider,
    getModels: () => [selected],
  }
}

export function createSubscriptionAdapter({
  authFile,
  provider = DEFAULT_PROVIDER,
  model = DEFAULT_MODEL,
  streamIdleTimeoutMs = 300000,
  maxRetries = 1,
  resolveAttachments,
  readCredential,
  now,
}) {
  const piProvider = selectSubscriptionProvider(provider, model)
  const retryPolicy = resolveRetryPolicy(
    {
      mode: "normal",
      maxRetries,
      retryableCodes: [
        "EMPTY_RESPONSE",
        "RATE_LIMIT",
        "SERVER",
        "TIMEOUT",
        "TRANSPORT",
      ],
      backoff: {
        initialDelayMs: 500,
        maxDelayMs: 10000,
        jitterRatio: 0,
      },
    },
    "subscription adapter retry policy",
  )
  const profile = {
    provider,
    displayName: "OpenAI subscription",
    piProvider,
    configuredMaxTokens: new Map(),
    retryPolicy,
    streamIdleTimeoutMs,
    maxRequestImageBytes: 20 * 1024 * 1024,
    requestImagePixelBudget: 2048 * 2048,
    requestImageMaxBytes: 1024 * 1024,
    transport: "sse",
  }
  const credentials = createReadOnlyCredentialStore({
    provider,
    authFile,
    ...(readCredential === undefined ? {} : { readCredential }),
    ...(now === undefined ? {} : { now }),
  })
  const profiles = new Map([[provider, profile]])
  return new PiAiAdapter({
    profiles: () => profiles,
    resolveApiKey: async () => undefined,
    auth: {
      credentials,
      authContext: {
        env: async () => undefined,
        fileExists: async () => false,
      },
    },
    resolveAttachments,
  })
}

export async function subscriptionReadiness({
  authFile = resolveSubscriptionAuthFile(),
  provider = DEFAULT_PROVIDER,
  model = DEFAULT_MODEL,
  now = Date.now(),
} = {}) {
  const versions = await verifySubscriptionRuntime()
  const credential = await readSubscriptionCredential(authFile)
  if (now >= credential.expires) {
    throw new SubscriptionAuthError(
      "subscription auth.json access token is expired; refresh the Codex sign-in first",
    )
  }
  const selected = selectSubscriptionProvider(provider, model).getModels()[0]
  return {
    provider,
    model: selected.id,
    api: selected.api,
    local_auth: "ready",
    provider_access: "unverified",
    network_request_count: 0,
    refresh_owner: "codex-auth-file",
    access_expires_at: new Date(credential.expires).toISOString(),
    versions,
  }
}

export async function apply(ctx, config) {
  const authFile = resolveSubscriptionAuthFile(config.authFile)
  await subscriptionReadiness({
    authFile,
    provider: config.provider,
    model: config.model,
  })
  const adapter = createSubscriptionAdapter({
    authFile,
    provider: config.provider,
    model: config.model,
    streamIdleTimeoutMs: config.streamIdleTimeoutMs,
    maxRetries: config.maxRetries,
    resolveAttachments: () => ctx.get("attachments"),
  })
  ctx.llm.registerAdapter([config.provider], adapter)
}
