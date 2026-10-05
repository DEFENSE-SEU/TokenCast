import {
  closeSync,
  fsyncSync,
  mkdirSync,
  openSync,
  writeFileSync,
} from "node:fs"
import { join } from "node:path"

export function observerSidecarPath(outputDir, sessionId) {
  return join(outputDir, `${encodeURIComponent(sessionId)}.jsonl`)
}

export class ObserverSidecarWriter {
  constructor({
    outputDir,
    openFile = openSync,
    writeFile = writeFileSync,
    syncFile = fsyncSync,
    closeFile = closeSync,
  } = {}) {
    if (typeof outputDir !== "string" || outputDir.length === 0) {
      throw new TypeError("observer sidecar outputDir is required")
    }
    this.outputDir = outputDir
    this.openFile = openFile
    this.writeFile = writeFile
    this.syncFile = syncFile
    this.closeFile = closeFile
    this.sessions = new Map()
    mkdirSync(outputDir, { recursive: true })
  }

  stateFor(sessionId) {
    let state = this.sessions.get(sessionId)
    if (state === undefined) {
      const path = observerSidecarPath(this.outputDir, sessionId)
      state = { fd: this.openFile(path, "ax"), path, dirty: false, failure: undefined }
      this.sessions.set(sessionId, state)
    }
    return state
  }

  writeRecord(sessionId, record) {
    const state = this.stateFor(sessionId)
    if (state.failure !== undefined) throw state.failure
    state.dirty = true
    try {
      this.writeFile(state.fd, `${JSON.stringify(record)}\n`)
    } catch (error) {
      state.failure = error
      throw error
    }
  }

  flushSession(sessionId) {
    const state = this.sessions.get(sessionId)
    if (state === undefined) return false
    let participated = false
    let syncFailure
    if (state.dirty) {
      try {
        // Keep the dirty flag if fsync fails so a later flush can retry.
        this.syncFile(state.fd)
        state.dirty = false
        participated = true
      } catch (error) {
        syncFailure = error
      }
    }
    if (state.failure !== undefined) throw state.failure
    if (syncFailure !== undefined) throw syncFailure
    return participated
  }

  closeSession(sessionId) {
    const state = this.sessions.get(sessionId)
    if (state === undefined) return false
    let failure
    try {
      this.flushSession(sessionId)
    } catch (error) {
      failure = error
    }
    let closeFailure
    try {
      this.closeFile(state.fd)
    } catch (error) {
      closeFailure = error
      failure ??= error
    }
    if (closeFailure === undefined) this.sessions.delete(sessionId)
    if (failure !== undefined) throw failure
    return true
  }

  closeAll() {
    let failure
    for (const sessionId of [...this.sessions.keys()]) {
      try {
        this.closeSession(sessionId)
      } catch (error) {
        failure ??= error
      }
    }
    if (failure !== undefined) throw failure
  }
}
