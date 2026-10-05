import assert from "node:assert/strict"
import { mkdtemp, rm, writeFile } from "node:fs/promises"
import test from "node:test"
import { tmpdir } from "node:os"
import { join } from "node:path"

import {
  SubscriptionAuthError,
  createReadOnlyCredentialStore,
  createSubscriptionAdapter,
  parseSubscriptionAuthDocument,
  readSubscriptionCredential,
  resolveSubscriptionAuthFile,
  selectSubscriptionProvider,
  subscriptionReadiness,
  verifySubscriptionRuntime,
} from "../subscription-adapter.mjs"

function jwt(payload) {
  const header = Buffer.from(JSON.stringify({ alg: "none", typ: "JWT" })).toString("base64url")
  const body = Buffer.from(JSON.stringify(payload)).toString("base64url")
  return `${header}.${body}.signature`
}

function authDocument({
  accountId = "account-fixture",
  claimedAccountId = accountId,
  expires = 4102444800,
  access = undefined,
} = {}) {
  return {
    auth_mode: "chatgpt",
    OPENAI_API_KEY: null,
    tokens: {
      id_token: "fixture-id-token",
      access_token: access ?? jwt({
        exp: expires,
        "https://api.openai.com/auth": {
          chatgpt_account_id: claimedAccountId,
        },
      }),
      refresh_token: "fixture-refresh-token",
      account_id: accountId,
    },
    last_refresh: "2026-08-28T00:00:00.000Z",
  }
}

test("auth.json maps to the pi-ai OAuth credential without exposing id_token", () => {
  const credential = parseSubscriptionAuthDocument(authDocument())
  assert.deepEqual(credential, {
    type: "oauth",
    access: authDocument().tokens.access_token,
    refresh: "fixture-refresh-token",
    expires: 4102444800 * 1000,
    accountId: "account-fixture",
  })
  assert.equal("idToken" in credential, false)
})

test("auth parser fails closed on account mismatch without echoing secrets", () => {
  const document = authDocument({
    accountId: "stored-account-secret",
    claimedAccountId: "claim-account-secret",
  })
  assert.throws(
    () => parseSubscriptionAuthDocument(document),
    (error) => {
      assert.ok(error instanceof SubscriptionAuthError)
      assert.match(error.message, /account id differs/)
      assert.doesNotMatch(error.message, /stored-account-secret|claim-account-secret/)
      assert.doesNotMatch(error.message, /fixture-refresh-token/)
      return true
    },
  )
})

test("auth reader rejects malformed JSON without including file contents", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "subscription-auth-"))
  t.after(() => rm(root, { force: true, recursive: true }))
  const path = join(root, "auth.json")
  await writeFile(path, '{"tokens":"secret-value"', "utf8")
  await assert.rejects(
    readSubscriptionCredential(path),
    (error) => {
      assert.match(error.message, /not valid JSON/)
      assert.doesNotMatch(error.message, /secret-value/)
      return true
    },
  )
})

test("credential reads observe an externally refreshed auth file", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "subscription-auth-"))
  t.after(() => rm(root, { force: true, recursive: true }))
  const path = join(root, "auth.json")
  await writeFile(path, JSON.stringify(authDocument({ expires: 4102444800 })), "utf8")
  const first = await readSubscriptionCredential(path)
  await writeFile(path, JSON.stringify(authDocument({ expires: 4102444900 })), "utf8")
  const second = await readSubscriptionCredential(path)
  assert.equal(first.expires, 4102444800 * 1000)
  assert.equal(second.expires, 4102444900 * 1000)
})

test("expired credentials stop before the refresh callback", async () => {
  let refreshCalls = 0
  const credential = parseSubscriptionAuthDocument(authDocument({ expires: 100 }))
  const store = createReadOnlyCredentialStore({
    provider: "openai-codex",
    authFile: "unused",
    readCredential: async () => credential,
    now: () => 101000,
  })
  await assert.rejects(
    store.modify("openai-codex", async () => {
      refreshCalls += 1
      return undefined
    }),
    /access token is expired/,
  )
  assert.equal(refreshCalls, 0)
})

test("valid credential recheck permits pi-ai's no-op modify", async () => {
  const credential = parseSubscriptionAuthDocument(authDocument())
  const store = createReadOnlyCredentialStore({
    provider: "openai-codex",
    authFile: "unused",
    readCredential: async () => credential,
    now: () => 1,
  })
  const result = await store.modify("openai-codex", async (current) => {
    assert.equal(current, credential)
    return undefined
  })
  assert.equal(result, credential)
})

test("credential enumeration returns metadata without reading token material", async () => {
  let reads = 0
  const store = createReadOnlyCredentialStore({
    provider: "openai-codex",
    authFile: "unused",
    readCredential: async () => {
      reads += 1
      return parseSubscriptionAuthDocument(authDocument())
    },
  })
  assert.deepEqual(await store.list(), [
    { providerId: "openai-codex", type: "oauth" },
  ])
  assert.equal(reads, 0)
})

test("offline readiness distinguishes local auth from provider access", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "subscription-auth-"))
  t.after(() => rm(root, { force: true, recursive: true }))
  const path = join(root, "auth.json")
  await writeFile(path, JSON.stringify(authDocument()), "utf8")
  const readiness = await subscriptionReadiness({ authFile: path, now: 1 })
  assert.equal(readiness.local_auth, "ready")
  assert.equal(readiness.provider_access, "unverified")
  assert.equal(readiness.network_request_count, 0)
})

test("configured provider exposes only the selected GPT-5.4 model", async () => {
  const provider = selectSubscriptionProvider("openai-codex", "gpt-5.4")
  assert.deepEqual(provider.getModels().map((model) => model.id), ["gpt-5.4"])

  const adapter = createSubscriptionAdapter({
    authFile: "unused",
    provider: "openai-codex",
    model: "gpt-5.4",
    readCredential: async () => parseSubscriptionAuthDocument(authDocument()),
  })
  const model = await adapter.resolveModel("openai-codex", "gpt-5.4")
  assert.equal(model.id, "gpt-5.4")
  await assert.rejects(
    adapter.resolveModel("openai-codex", "gpt-5.3"),
    /has no configured model/,
  )
  assert.throws(
    () => selectSubscriptionProvider("openai-codex", "gpt-5.3"),
    /has no installed model "gpt-5\.3"/,
  )
  assert.deepEqual(
    selectSubscriptionProvider("openai-codex", "gpt-5.5").getModels().map((model) => model.id),
    ["gpt-5.5"],
  )
})

test("runtime and auth path resolution are explicit and deterministic", async () => {
  assert.deepEqual(await verifySubscriptionRuntime(), {
    dsh_pi_adapter: "0.1.1-rc.2",
    pi_ai: "0.82.1",
  })
  assert.equal(
    resolveSubscriptionAuthFile("", { CODEX_HOME: "C:\\runtime-auth" }, "C:\\user"),
    "C:\\runtime-auth\\auth.json",
  )
  assert.equal(
    resolveSubscriptionAuthFile("", {}, "C:\\user"),
    "C:\\user\\.codex\\auth.json",
  )
  assert.throws(
    () => resolveSubscriptionAuthFile("relative/auth.json", {}, "C:\\user"),
    /must be absolute/,
  )
})
