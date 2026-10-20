import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";

const context = vm.createContext({});
vm.runInContext(readFileSync(
  new URL("../interface/static/lib/http.js", import.meta.url), "utf8",
).replace(/^export /gm, ""), context);
const read = context.readJsonResponse;

const json = {detail: "No model is selected.", settings_url: "/settings"};
assert.equal(JSON.stringify(await read(new Response(JSON.stringify(json), {status: 503}))),
  JSON.stringify(json), "Structured diagnostics reach the caller intact");
assert.equal(JSON.stringify(await read(new Response('{"count":0,"tools":[]}'))),
  '{"count":0,"tools":[]}');
await assert.rejects(read(new Response("Internal Server Error", {status: 500})),
  {message: "Internal Server Error"});
await assert.rejects(read(new Response("Maintenance in progress", {status: 503})),
  {message: "Maintenance in progress"});
await assert.rejects(read(new Response("<html><body>Proxy failed</body></html>", {status: 502})),
  {message: "Request failed (HTTP 502). Please retry."});
await assert.rejects(read(new Response("Traceback: server internals", {status: 500})),
  {message: "Request failed (HTTP 500). Please retry."});
await assert.rejects(read(new Response("")),
  {message: "The server returned an unreadable response (HTTP 200). Please retry."});
