import { createRequire } from "node:module"
import { existsSync, readFileSync } from "node:fs"
import { homedir } from "node:os"
import { join } from "node:path"
import { pathToFileURL } from "node:url"

function dshHomePath() {
  return process.env.DSH_HOME || join(homedir(), ".dsh")
}

export function resolveDshHostModuleUrl(specifier) {
  const profileRequire = createRequire(join(dshHomePath(), "profiles", "package.json"))
  return pathToFileURL(profileRequire.resolve(specifier)).href
}

export function importDshHostModule(specifier) {
  return import(resolveDshHostModuleUrl(specifier))
}

const FROZEN_PROFILE_BUNDLES = [
  "@deepseek-ai/dsh-base",
  "@deepseek-ai/dsh-headless",
  "@token-prediction/dsh-token-observer",
  "@token-prediction/dsh-headless",
]

function requireEmptyProfileLayer(path, label) {
  const effectiveLines = readFileSync(path, "utf8")
    .split(/\r?\n/u)
    .map((line) => line.trim())
    .filter((line) => line && !line.startsWith("#"))
  if (effectiveLines.length !== 1 || effectiveLines[0] !== "[]") {
    throw new Error(`${label} must remain an empty list`)
  }
}

export function verifyInstalledRuntime({ dshRoot, dshHome, profile }) {
  if (profile !== "dsh-headless") {
    throw new Error("DSH collection requires the frozen dsh-headless profile")
  }
  if (existsSync(join(dshHome, "cordis.patch.yml"))) {
    throw new Error("DSH home cordis.patch.yml overlay is forbidden for DSH runs")
  }
  const profileRoot = join(dshHome, "profiles", profile)
  requireEmptyProfileLayer(join(profileRoot, "cordis.yml"), "DSH profile cordis.yml")
  requireEmptyProfileLayer(
    join(profileRoot, "cordis.patch.yml"),
    "DSH profile cordis.patch.yml",
  )
  const profileManifest = JSON.parse(readFileSync(join(profileRoot, "package.json"), "utf8"))
  const profileBundles = profileManifest?.dsh?.profile?.bundles
  if (
    !Array.isArray(profileBundles)
    || profileBundles.length !== FROZEN_PROFILE_BUNDLES.length
    || profileBundles.some((value, index) => value !== FROZEN_PROFILE_BUNDLES[index])
  ) {
    throw new Error("installed DSH profile bundle order differs from the frozen runtime")
  }
  const core = [
    ["dsh_cli", join(dshRoot, "package.json"), "@deepseek-ai/dsh", "0.1.1-rc.2"],
    ["dsh_agent", "@deepseek-ai/dsh-agent", null, "0.1.1-rc.2"],
    ["dsh_agent_presets", "@deepseek-ai/dsh-agent-presets", null, "0.1.1-rc.2"],
    ["dsh_agent_loop", "@deepseek-ai/dsh-agent-loop", null, "0.1.1-rc.2"],
    ["dsh_headless", "@deepseek-ai/dsh-headless", null, "0.1.1-rc.2"],
    ["dsh_llm", "@deepseek-ai/dsh-llm", null, "0.1.1-rc.2"],
    ["dsh_llm_deepseek", "@deepseek-ai/dsh-llm-deepseek", null, "0.1.1-rc.2"],
    ["dsh_llm_retry", "@deepseek-ai/dsh-llm-retry", null, "0.1.1-rc.2"],
    ["dsh_session", "@deepseek-ai/dsh-session", null, "0.1.1-rc.2"],
    [
      "dsh_session_checkpoint_policy",
      "@deepseek-ai/dsh-session-checkpoint-policy",
      null,
      "0.1.1-rc.2",
    ],
    [
      "dsh_session_persistence",
      "@deepseek-ai/dsh-session-persistence",
      null,
      "0.1.1-rc.2",
    ],
    [
      "dsh_session_persistence_jsonl",
      "@deepseek-ai/dsh-session-persistence-jsonl",
      null,
      "0.1.1-rc.2",
    ],
    ["dsh_token_meter", "@deepseek-ai/dsh-token-meter", null, "0.1.1-rc.2"],
    ["dsh_tools", "@deepseek-ai/dsh-tools", null, "0.1.1-rc.2"],
  ]
  const packages = core.map(([key, location, explicitName, version]) => {
    const packageName = explicitName ?? location
    const path = explicitName
      ? location
      : join(dshRoot, "node_modules", ...packageName.split("/"), "package.json")
    return [key, path, packageName, version]
  })
  packages.push(
    [
      "token_prediction_observer",
      join(
        dshHome,
        "profiles",
        profile,
        "node_modules",
        "@token-prediction",
        "dsh-token-observer",
        "package.json",
      ),
      "@token-prediction/dsh-token-observer",
      "0.4.0",
    ],
    [
      "token_prediction_dsh_headless",
      join(
        dshHome,
        "profiles",
        profile,
        "node_modules",
        "@token-prediction",
        "dsh-headless",
        "package.json",
      ),
      "@token-prediction/dsh-headless",
      "0.5.0",
    ],
  )
  const observed = {}
  for (const [key, path, name, version] of packages) {
    const document = JSON.parse(readFileSync(path, "utf8"))
    if (document.name !== name || document.version !== version) {
      throw new Error(`installed DSH package ${name} differs from the frozen runtime`)
    }
    observed[key] = document.version
  }
  observed.router_standard = "0.3.0"
  return observed
}
