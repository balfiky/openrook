import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";

const context = vm.createContext({});
vm.runInContext(readFileSync(
  new URL("../interface/static/lib/storage.js", import.meta.url), "utf8",
).replace(/^export /gm, ""), context);
const migrate = context.migrateBrowserStorage;

function storage(initial = {}) {
  const values = new Map(Object.entries(initial));
  return {
    getItem: key => values.has(key) ? values.get(key) : null,
    setItem: (key, value) => values.set(key, String(value)),
    removeItem: key => values.delete(key),
    snapshot: () => Object.fromEntries(values),
  };
}

const local = storage({ nur_user_id: "existing-user" });
const session = storage({
  nur_chat_id: "existing-chat", nur_api_token: "existing-token", nur_wizard_dismissed: "1",
});
migrate(local, session);
assert.deepEqual(local.snapshot(), { openrook_user_id: "existing-user" });
assert.deepEqual(session.snapshot(), {
  openrook_chat_id: "existing-chat",
  openrook_api_token: "existing-token",
  openrook_wizard_dismissed: "1",
});
// Reloading does not replace an identity, token, or dismissal.
migrate(local, session);
assert.equal(local.getItem("openrook_user_id"), "existing-user");
assert.equal(session.getItem("openrook_api_token"), "existing-token");

const conflictingLocal = storage({ nur_user_id: "old-user", openrook_user_id: "new-user" });
const conflictingSession = storage({ nur_api_token: "old-token", openrook_api_token: "" });
migrate(conflictingLocal, conflictingSession);
assert.deepEqual(conflictingLocal.snapshot(), { openrook_user_id: "new-user" });
assert.deepEqual(conflictingSession.snapshot(), { openrook_api_token: "" });

const freshLocal = storage();
const freshSession = storage();
migrate(freshLocal, freshSession);
assert.deepEqual(freshLocal.snapshot(), {});
assert.deepEqual(freshSession.snapshot(), {});

const fullLocal = storage({ nur_user_id: "recoverable-user" });
fullLocal.setItem = () => { throw new Error("Storage quota exceeded"); };
assert.throws(() => migrate(fullLocal, storage()), /quota/);
assert.equal(fullLocal.getItem("nur_user_id"), "recoverable-user");
