// Keep transport failures readable even when a proxy/server returns plain text.
export class ServerResponseError extends Error {}

export async function readJsonResponse(response) {
  const body = await response.text();
  try {
    return JSON.parse(body);
  } catch (_) {
    const message = body.trim();
    // HTML error pages and tracebacks are not useful UI messages.
    const plainMessage = message && message.length <= 500
      && !/[<>]/.test(message) && !/traceback/i.test(message);
    if (!response.ok && plainMessage) throw new ServerResponseError(message);
    throw new ServerResponseError(response.ok
      ? `The server returned an unreadable response (HTTP ${response.status}). Please retry.`
      : `Request failed (HTTP ${response.status}). Please retry.`);
  }
}
