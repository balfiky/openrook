// Compatibility for browser sessions created before the product rename.
// Copy before removing a legacy key; a failed write must leave it recoverable.
export function migrateBrowserStorage(local, session) {
  for (const [store, legacyKey, currentKey] of [
    [local, "nur_user_id", "openrook_user_id"],
    [session, "nur_chat_id", "openrook_chat_id"],
    [session, "nur_api_token", "openrook_api_token"],
    [session, "nur_wizard_dismissed", "openrook_wizard_dismissed"],
  ]) {
    const previous = store.getItem(legacyKey);
    if (previous === null) continue;
    if (store.getItem(currentKey) === null) store.setItem(currentKey, previous);
    store.removeItem(legacyKey);
  }
}
