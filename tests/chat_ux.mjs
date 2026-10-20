// Run the real chat module against a small DOM and controllable network/clock.
// Markdown rendering itself is covered elsewhere; connection feedback must use
// text nodes even when the backend's error contains HTML.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";

const staticRoot = new URL("../interface/static/", import.meta.url);
const escapeHtml = value => String(value).replaceAll("&", "&amp;")
  .replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;");

class Element {
  constructor(tagName = "div") {
    this.tagName = tagName;
    this.children = [];
    this.attributes = {};
    this.dataset = {};
    this.value = "";
    this.disabled = false;
    this.hidden = false;
    this.focused = false;
    this.text = "";
    this.html = null;
    this.htmlWrites = 0;
    this.isConnected = true;
    this.listeners = {};
    const properties = new Map();
    this.style = {
      setProperty: (key, value) => properties.set(key, value),
      getPropertyValue: key => properties.get(key) ?? "",
    };
    const classes = new Set();
    this.classList = {
      add: value => classes.add(value),
      remove: value => classes.delete(value),
      contains: value => classes.has(value),
      toggle: (value, enabled) => enabled ? classes.add(value) : classes.delete(value),
    };
  }
  set textContent(value) {
    this.text = String(value);
    this.html = null;
    this.children = [];
  }
  get textContent() {
    return this.text + this.children.map(child => child.textContent).join("");
  }
  set innerHTML(value) {
    this.html = String(value);
    this.text = "";
    this.children = [];
    this.htmlWrites++;
  }
  get innerHTML() { return this.html ?? escapeHtml(this.text); }
  appendChild(child) { this.children.push(child); return child; }
  append(...children) { children.forEach(child => this.appendChild(child)); }
  querySelector(selector) {
    this.matches ||= new Map();
    if (!this.matches.has(selector)) {
      const child = new Element();
      child.className = selector.slice(1);
      this.matches.set(selector, child);
      this.appendChild(child);
    }
    return this.matches.get(selector);
  }
  setAttribute(name, value) { this.attributes[name] = value; }
  addEventListener(name, handler) { this.listeners[name] = handler; }
  dispatchEvent(event) { return this.listeners[event.type]?.(event); }
  replaceChildren() { this.children = []; this.value = ""; }
  before(element) { this.row = element; }
  after(element) { this.hint = element; }
  closest() { return this; }
  remove() { this.removed = true; }
  focus() { this.focused = true; }
}

const storage = () => {
  const values = new Map();
  return {
    getItem: key => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, value),
    removeItem: key => values.delete(key),
  };
};
const response = (data, status = 200) => ({
  ok: status >= 200 && status < 300, status, text: async () => JSON.stringify(data),
});
const deferred = () => {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
};

function setup() {
  const elements = new Map();
  const element = id => {
    if (!elements.has(id)) {
      const el = new Element();
      el.id = id;
      elements.set(id, el);
    }
    return elements.get(id);
  };
  const actions = [element("end"), element("rest")];
  const calls = [];
  const timers = new Map();
  let now = 0;
  let timerId = 0;
  let respond = async () => response({});
  let initialSettings = true;
  let initialHistory = true;
  const settingsRequest = deferred();
  const document = {
    getElementById: id => id === "chatBackendStatus" && element("emptyState").removed ? null : element(id),
    createElement: tag => new Element(tag),
    querySelectorAll: selector => {
      if (selector.includes("end-session")) return actions;
      if (selector === ".message-row.assistant .msg-name") {
        return element("messages").children
          .filter(child => child.className === "message-row assistant")
          .map(child => child.querySelector(".msg-name"));
      }
      return [];
    },
    addEventListener() {},
    documentElement: new Element("html"),
  };
  const context = vm.createContext({
    document, console, URLSearchParams, Event,
    localStorage: storage(), sessionStorage: storage(),
    crypto: { randomUUID: () => "test-id" },
    window: { prompt: () => null },
    MutationObserver: class { observe() {} },
    requestAnimationFrame: callback => callback(),
    setTimeout(callback, delay) {
      timers.set(++timerId, { callback, at: now + delay });
      return timerId;
    },
    clearTimeout: id => timers.delete(id),
    fetch(url, options) {
      if (url.startsWith("/session/history?") && initialHistory) {
        initialHistory = false;
        return Promise.resolve(response({ history: [] }));
      }
      // Initial settings hydration is outside these action tests. Keep it
      // pending, so it cannot race the state transitions under test.
      if (url === "/admin/config" && initialSettings) {
        initialSettings = false;
        return settingsRequest.promise;
      }
      calls.push({ url, options });
      return respond(url, options);
    },
    marked: { setOptions() {}, parse: text => escapeHtml(text) },
    DOMPurify: { sanitize: text => text },
    AdminOverlayState: {}, DebugState: {}, PersonaPanelState: {}, WizardState: {},
  });
  for (const name of ["lib/http.js", "lib/shared.js", "lib/llm-status.js", "lib/llm-model-picker.js", "chat.js"]) {
    const source = readFileSync(new URL(name, staticRoot), "utf8")
      .replace(/^import .+;\r?\n/gm, "").replace(/^export /gm, "");
    vm.runInContext(source, context, { filename: name });
  }
  vm.runInContext(`
    globalThis.subject = {
      sendMessage, endSession, applyRest, connectionSummary, renderConnectionResult,
      addMessage, showWhyResponse, restoreConversation, historyReady, settingsReady, applyAgentName,
      updatePersonaPanel, updateRelationshipState,
      backendLabel, healthLabel, refreshMoodContrast,
      wizardSelectPreset, wizardSaveLLM, wizardSaveRuntimeChannels, wizardRenderSummary,
      modelPickerFor,
      stubSettingsHydration() { applySettings = () => {}; },
      selectBackend(name) { currentBackend = name; },
      seedExplanation() {
        lastExplanation = { summary: "Previous reply" };
        addMessage("Previous reply", "assistant", "neutral", lastExplanation);
        document.getElementById('whyBtn').disabled = false;
      },
      explanation() { return lastExplanation; },
    };
  `, context);
  return {
    ...context.subject, element, actions, calls, timers, root: document.documentElement,
    respondWith(handler) { respond = handler; },
    promptWith(value) { context.window.prompt = () => value; },
    failSettings(error) { settingsRequest.reject(error); },
    advance(milliseconds) {
      now += milliseconds;
      for (const [id, timer] of timers) {
        if (timer.at <= now) {
          timers.delete(id);
          timer.callback();
        }
      }
    },
    notice() {
      return element("messages").children.findLast(child => child.className === "session-notice");
    },
  };
}

const scenario = process.argv[2];
const app = setup();
await app.historyReady;

if (scenario.startsWith("session_")) {
  const savedCounts = {
    session_saved: 2,
    session_saved_one: 1,
    session_no_new_memory: 0,
  };
  const expectedNotices = {
    session_saved: "Saved 2 things from this conversation · new conversation started",
    session_saved_one: "Saved 1 thing from this conversation · new conversation started",
    session_no_new_memory: "Nothing saved from this conversation · new conversation started",
  };
  app.seedExplanation();
  app.element("msgInput").value = "Next message";
  const pending = deferred();
  app.respondWith(() => pending.promise);
  const finished = app.endSession();
  assert.match(app.notice().textContent, /Saving this conversation/);
  assert.equal(app.element("sendBtn").disabled, true);
  if (scenario === "session_network_failure") {
    pending.reject(new Error("Network unavailable"));
  } else if (scenario === "session_failure") {
    pending.resolve(response({ detail: "Session could not be saved" }, 503));
  } else {
    pending.resolve(response({ memories_written: savedCounts[scenario] }));
  }
  await finished;
  const notice = app.notice();
  assert.equal(notice.attributes.role, "status");
  if (Object.hasOwn(expectedNotices, scenario)) {
    assert.equal(notice.textContent, expectedNotices[scenario]);
    assert.equal(notice.classList.contains("session-divider"), true);
    assert.match(notice.title, /Stored memories and preferences remain/);
    assert.match(notice.title, /only the current conversation is restored/);
    const transcript = app.element("messages").children;
    assert.match(transcript[0].innerHTML, /Previous reply/);
    const next = app.addMessage("Fresh conversation", "user");
    assert.ok(transcript.indexOf(next) > transcript.indexOf(notice));
    assert.equal(app.explanation(), null);
    assert.equal(app.element("whyBtn").disabled, true);
  } else {
    assert.match(notice.textContent, /Could not end the session/);
    assert.doesNotMatch(notice.textContent, /fresh session has started/);
    assert.ok(app.explanation());
    assert.equal(app.element("whyBtn").disabled, false);
  }
  assert.equal(app.element("sendBtn").disabled, false);
  assert.ok(app.actions.every(button => !button.disabled));
} else if (scenario === "explanation_card") {
  const first = app.addMessage("First reply", "assistant", "neutral", {
    memory: "Recalled 2 things from saved memory: You prefer concise replies; You work in Python.",
    life_history: "Life History contributed 1 belief, 1 saved priority.",
    strategy: "Offer practical help.",
  });
  const second = app.addMessage("Second reply", "assistant", "neutral", {
    memory: "No saved details about you were used.",
    life_history: "Only the starting character settings applied.",
  });
  const before = app.element("messages").children.length;
  first.explanationButton.dispatchEvent({ type: "click" });
  const card = first.explanationCard;
  assert.equal(card.tagName, "aside");
  assert.equal(card.attributes.role, "note");
  assert.equal(card.attributes["aria-label"], "Explanation of this reply");
  assert.equal(card.id, first.explanationButton.attributes["aria-controls"]);
  assert.ok(first.querySelector(".msg-content").children.includes(card));
  const firstBody = card.children[1].innerHTML;
  assert.match(firstBody, /<strong>Memory<\/strong><div class="markdown-copy">Recalled 2 things from saved memory: You prefer concise replies; You work in Python\.<\/div><\/div>/);
  assert.match(firstBody, /<strong>Life History<\/strong><div class="markdown-copy">Life History contributed 1 belief, 1 saved priority\.<\/div><\/div>/);
  assert.equal(firstBody.match(/Life History contributed/g).length, 1);
  assert.doesNotMatch(firstBody, /No saved details about you/);
  assert.equal(first.explanationButton.attributes["aria-expanded"], "true");
  const dismiss = card.children[0].children[1];
  dismiss.dispatchEvent({ type: "click" });
  assert.equal(card.hidden, true);
  assert.equal(first.explanationButton.attributes["aria-expanded"], "false");
  assert.equal(first.explanationButton.focused, true);
  app.showWhyResponse(first);
  assert.equal(card.hidden, false);
  assert.equal(first.querySelector(".msg-content").children.filter(child => child === card).length, 1);
  app.showWhyResponse();
  const secondBody = second.explanationCard.children[1].innerHTML;
  assert.match(secondBody, /<strong>Memory<\/strong><div class="markdown-copy">No saved details about you were used\.<\/div><\/div>/);
  assert.match(secondBody, /<strong>Life History<\/strong><div class="markdown-copy">Only the starting character settings applied\.<\/div><\/div>/);
  assert.doesNotMatch(secondBody, /Life History contributed|saved priorities|Recalled/);
  assert.equal(app.element("messages").children.length, before, "Explanations are not chat messages");
  assert.equal(app.calls.length, 0, "Explaining a reply never sends or stores a conversation turn");
} else if (scenario === "plain_strategy_labels") {
  const relationship = {
    strategy: "practical_help", strategy_reason: "actionable_request_practical_help",
    strategy_label: "Practical help",
    strategy_reason_label: "You asked for help with an actionable task.",
  };
  app.updatePersonaPanel({ relationship });
  app.updateRelationshipState({
    response_strategy: relationship.strategy,
    strategy_trace: { matched_rule: relationship.strategy_reason },
    persona_view: { relationship },
  }, relationship);
  for (const id of ["personaPanel", "relationshipState"]) {
    const html = app.element(id).innerHTML;
    assert.match(html, /Practical help/);
    assert.match(html, /You asked for help with an actionable task/);
    assert.doesNotMatch(html, /practical_help|actionable_request/);
  }
} else if (scenario === "history_auth_restore") {
  app.respondWith(() => response({ detail: "Sign in required" }, 401));
  assert.equal(await app.restoreConversation(), false);
  app.promptWith("history-token");
  app.element("msgInput").value = "Continue";
  app.respondWith((url, options) => {
    if (url.startsWith("/session/history?")) {
      if (!options.headers.Authorization) return response({ detail: "Sign in required" }, 401);
      assert.equal(options.headers.Authorization, "Bearer history-token");
      return response({ history: [
        { role: "user", content: "Earlier message" },
        { role: "assistant", content: "Earlier reply" },
      ] });
    }
    assert.equal(url, "/chat");
    assert.equal(options.headers.Authorization, "Bearer history-token");
    const visibleRows = app.element("messages").children.filter(child => !child.removed);
    assert.match(visibleRows[0].innerHTML, /Earlier message/);
    assert.match(visibleRows[1].innerHTML, /Earlier reply/);
    assert.match(visibleRows[2].innerHTML, /Continue/);
    return response({ response: "New reply" });
  });
  await app.sendMessage();
  assert.equal(app.calls.filter(call => call.url === "/chat").length, 1);
  assert.equal(app.element("messages").children.filter(child => !child.removed).length, 4);
} else if (scenario === "history_settings_race") {
  app.respondWith(() => response({ history: [{ role: "assistant", content: "Earlier reply" }] }));
  await app.restoreConversation();
  assert.equal(app.element("emptyState").removed, true);
  app.failSettings(new Error("Settings unavailable"));
  await app.settingsReady;
  app.applyAgentName("Changed Name");
  const reply = app.element("messages").children[0];
  assert.equal(reply.querySelector(".msg-name").textContent, "Changed Name");
} else if (["history_restore", "history_restore_failure"].includes(scenario)) {
  app.element("msgInput").value = "Keep draft";
  const pending = deferred();
  app.respondWith(() => pending.promise);
  const restored = app.restoreConversation();
  assert.equal(app.element("sendBtn").disabled, true);
  assert.ok(app.actions.every(button => button.disabled));
  await app.sendMessage();
  await app.endSession();
  await app.applyRest();
  assert.equal(app.calls.length, 1, "History must load before a new turn or session action");
  assert.match(app.calls[0].url, /^\/session\/history\?user_id=test-id&chat_id=test-id$/);
  pending.resolve(scenario === "history_restore_failure"
    ? response({ detail: "Unavailable" }, 503)
    : response({ history: [
      { role: "user", content: "I prefer bullet points <script>" },
      { role: "assistant", content: "I will remember." },
      { role: "system", content: "Never display this" },
    ] }));
  await restored;
  assert.equal(app.element("sendBtn").disabled, false);
  assert.equal(app.element("msgInput").value, "Keep draft");
  if (scenario === "history_restore") {
    const rows = app.element("messages").children;
    assert.equal(rows.length, 2);
    assert.equal(rows[0].className, "message-row user");
    assert.match(rows[0].innerHTML, /bullet points &lt;script&gt;/);
    assert.equal(rows[1].className, "message-row assistant");
    assert.match(rows[1].innerHTML, /I will remember/);
    assert.equal(rows[1].explanation, null);
    assert.equal(app.element("whyBtn").disabled, true);
    assert.equal(app.element("emptyState").removed, true);
  } else {
    assert.match(app.notice().textContent, /Could not restore earlier messages/);
  }
} else if (["chat_pending", "end_pending", "rest_pending"].includes(scenario)) {
  app.element("msgInput").value = "First message";
  const pending = deferred();
  app.respondWith(url => url.startsWith("/debug") ? response({}) : pending.promise);
  const operation = {
    chat_pending: app.sendMessage, end_pending: app.endSession, rest_pending: app.applyRest,
  }[scenario];
  const finished = operation();
  assert.equal(app.calls.length, 1);
  assert.ok(app.actions.every(button => button.disabled));
  assert.equal(app.element("sendBtn").disabled, true);
  app.element("msgInput").value = "Keep this draft";
  await app.sendMessage();
  await app.endSession();
  await app.applyRest();
  assert.equal(app.calls.length, 1, "No overlapping chat/session/rest request is allowed");
  assert.equal(app.element("msgInput").value, "Keep this draft");
  pending.resolve(response({
    response: "Hello", memories_written: 1, energy_before: 0.4, energy_after: 0.8,
  }));
  await finished;
  assert.ok(app.actions.every(button => !button.disabled));
  assert.equal(app.element("sendBtn").disabled, false);
  await app.sendMessage();
  assert.equal(app.calls.filter(call => call.url === "/chat").length,
    scenario === "chat_pending" ? 2 : 1, "A subsequent message must still work");
} else if (["slow_reply", "fast_reply", "backend_failure", "network_failure", "plaintext_failure"].includes(scenario)) {
  app.selectBackend("Codex CLI");
  app.element("msgInput").value = "Hello";
  const pending = deferred();
  app.respondWith(() => pending.promise);
  const finished = app.sendMessage();
  assert.equal(app.element("typingRow").classList.contains("visible"), true);
  assert.equal(app.element("typingLabel").textContent, "Thinking…");
  app.advance(7999);
  assert.equal(app.element("typingLabel").textContent, "Thinking…");
  if (scenario !== "fast_reply") {
    app.advance(1);
    assert.match(app.element("typingLabel").textContent, /Still thinking — Codex CLI/);
  }
  if (scenario === "network_failure") {
    pending.reject(new Error("Network unavailable"));
  } else if (scenario === "plaintext_failure") {
    pending.resolve({ok: false, status: 500, text: async () => "Internal Server Error"});
  } else if (scenario === "backend_failure") {
    pending.resolve(response({ llm_error: {
      message: "Codex is not logged in.", action: "Run codex login.",
    } }, 503));
  } else {
    pending.resolve(response({ response: "Hello back" }));
  }
  await finished;
  assert.equal(app.element("typingRow").classList.contains("visible"), false);
  assert.equal(app.timers.size, 0, "The slow-response timer must be cleared on every outcome");
  const label = app.element("typingLabel").textContent;
  app.advance(60000);
  assert.equal(app.element("typingLabel").textContent, label);
  assert.equal(app.element("msgInput").focused, true);
  assert.ok(app.actions.every(button => !button.disabled));
  if (scenario.endsWith("failure")) {
    assert.match(app.notice().textContent, /Could not get a reply/);
    assert.match(app.notice().textContent, scenario === "backend_failure"
      ? /Codex is not logged in\. Run codex login\./
      : scenario === "plaintext_failure" ? /Internal Server Error/ : /Network unavailable/);
    assert.doesNotMatch(app.notice().textContent, /Unexpected token|JSON/);
    assert.equal(app.notice().children[0].href, "/settings#settings");
  } else {
    assert.match(app.element("messages").children.at(-1).innerHTML, /Hello back/);
  }
} else if (scenario === "connection_result") {
  const output = new Element();
  const unsafe = '<img src=x onerror="throw new Error(1)">';
  app.renderConnectionResult(output, {
    ok: false, diagnostic: { message: unsafe, action: "Run codex login.", detail: unsafe },
  });
  assert.equal(output.hidden, false);
  assert.equal(output.htmlWrites, 0);
  assert.match(output.textContent, /Run codex login/);
  assert.equal(output.children.length, 1);
  const disclosure = output.children[0];
  assert.equal(disclosure.tagName, "details");
  assert.equal(disclosure.children[0].tagName, "summary");
  assert.equal(disclosure.children[0].textContent, "Technical details");
  assert.equal(disclosure.children[1].tagName, "pre");
  assert.equal(disclosure.children[1].textContent, unsafe);
  assert.equal(disclosure.children[1].htmlWrites, 0);
  assert.equal(disclosure.children[1].children.length, 0);
  app.renderConnectionResult(output, { ok: true, live: true });
  assert.match(output.textContent, /Connection working/);
  assert.equal(output.children.length, 0, "A later success must clear stale error details");
} else if (scenario === "connection_summary") {
  assert.match(app.connectionSummary({ ok: true, live: true }), /sample reply received/);
  assert.match(app.connectionSummary({ ok: true, live: false }), /not tested/);
  assert.equal(app.connectionSummary({ error: "Unavailable" }), "Unavailable");
  assert.equal(app.connectionSummary({ detail: "Authentication required" }), "Authentication required");
  assert.equal(app.connectionSummary({ detail: [{ msg: "Set a model" }] }), "Set a model");
  assert.equal(app.connectionSummary({ detail: { message: "Authentication required" } }), "Authentication required");
  assert.match(app.connectionSummary({ detail: { unexpected: true } }), /Settings/);
  assert.match(app.connectionSummary({ warnings: [{ message: "Set a model" }] }), /Set a model/);
  assert.match(app.connectionSummary({}), /Settings/);
  assert.equal(app.backendLabel({ llm_backend: "codex" }), "Codex CLI");
  assert.equal(app.backendLabel({
    llm_backend: "openai_compatible", llm_base_url: "http://localhost:11434/v1",
  }), "Ollama");
  assert.equal(app.backendLabel({
    llm_backend: "provider", llm_base_url: "https://openrouter.ai/api/v1",
  }), "OpenRouter");
  assert.match(app.healthLabel({ status: "error" }), /Last recorded error/);
  assert.match(app.healthLabel({}), /Not tested/);
} else if (scenario === "wizard_model_choice") {
  app.stubSettingsHydration();
  app.respondWith(async (url, options) => {
    if (url === "/admin/llm/models") {
      return response({ok: true, models: [
        {id: "installed-first"}, {id: "installed-second"},
      ]});
    }
    if (url === "/admin/config") return response({config: {}});
    throw new Error("Unexpected request: " + url);
  });
  app.wizardSelectPreset("local");
  await app.modelPickerFor("wiz").refresh();
  const input = app.element("wiz-llm_model");
  assert.equal(input.value, "", "Multiple models must not silently select the first");
  const select = input.row.children[0];
  select.value = "installed-second";
  select.dispatchEvent(new Event("change"));
  await app.wizardSaveLLM();
  let save = app.calls.findLast(call => call.url === "/admin/config" && call.options?.method === "POST");
  assert.equal(JSON.parse(save.options.body).llm_model, "installed-second");
  app.wizardRenderSummary();
  assert.match(app.element("wiz-summary").innerHTML, /installed-second/);
  input.value = "manual-choice";
  await app.wizardSaveLLM();
  save = app.calls.findLast(call => call.url === "/admin/config" && call.options?.method === "POST");
  assert.equal(JSON.parse(save.options.body).llm_model, "manual-choice");
} else if (scenario === "wizard_preserves_independence_setting") {
  app.stubSettingsHydration();
  for (const enabled of [false, true]) {
    app.respondWith(async () => response({config: {character_independence: enabled}}));
    await app.wizardSaveRuntimeChannels();
    const save = app.calls.findLast(call =>
      call.url === "/admin/config" && call.options?.method === "POST");
    assert.equal(JSON.parse(save.options.body).character_independence, enabled);
    app.wizardRenderSummary();
    assert.doesNotMatch(app.element("wiz-summary").innerHTML, /Independence/);
  }
} else if (scenario === "wizard_presets_no_model") {
  app.respondWith(async () => response({ok: true, models: []}));
  for (const preset of ["local", "vllm", "hosted", "openrouter", "codex"]) {
    app.element("wiz-llm_model").value = "previously-chosen";
    app.wizardSelectPreset(preset);
    assert.equal(app.element("wiz-llm_model").value, "", preset);
    assert.ok(!app.element("wiz-llm_model").placeholder.includes("previously-chosen"));
  }
} else if (scenario === "theme_tokens") {
  // With no explicit theme, CSS follows the OS. Mood updates must not override
  // those theme-aware text/surface tokens with inline white on a light page.
  for (const theme of [undefined, "light", "dark"]) {
    if (theme) app.root.dataset.theme = theme;
    else delete app.root.dataset.theme;
    app.refreshMoodContrast();
    for (const name of ["--text-primary", "--text-muted", "--surface-bg"]) {
      assert.equal(app.root.style.getPropertyValue(name), "", `${name} must remain controlled by the theme`);
    }
    assert.ok(app.root.style.getPropertyValue("--mood-hue"), "Mood colors should still update");
  }
} else {
  throw new Error(`Unknown scenario: ${scenario}`);
}
