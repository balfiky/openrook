import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';

class Element {
  constructor(tag = 'input') {
    this.tagName = tag.toUpperCase();
    this.children = [];
    this.listeners = {};
    this.attributes = {};
    this.value = '';
    this.textContent = '';
    this.isConnected = true;
  }
  setAttribute(key, value) { this.attributes[key] = value; }
  appendChild(child) { this.children.push(child); return child; }
  append(...children) { children.forEach(child => this.appendChild(child)); }
  before(element) { this.row = element; }
  after(element) { this.hint = element; }
  replaceChildren() { this.children = []; this.value = ''; }
  addEventListener(name, handler) { this.listeners[name] = handler; }
  dispatchEvent(event) { return this.listeners[event.type]?.(event); }
}

const pending = () => {
  let resolve;
  const promise = new Promise(yes => { resolve = yes; });
  return {promise, resolve};
};
const response = (data, status = 200) => ({
  ok: status < 400, status, text: async () => JSON.stringify(data),
});
function setup() {
  const context = vm.createContext({
    document: {createElement: tag => new Element(tag)}, Event,
  });
  for (const name of ['http.js', 'llm-model-picker.js']) {
    const source = readFileSync(new URL('../interface/static/lib/' + name, import.meta.url), 'utf8');
    vm.runInContext(source.replace(/^import .+;\r?\n/gm, '').replace(/^export /gm, ''), context);
  }
  const input = new Element();
  input.id = 'model';
  let config = {llm_backend: 'openai_compatible', llm_base_url: 'http://localhost:11434/v1', llm_api_key: ''};
  let reply = async () => response({ok: true, models: []});
  const calls = [];
  const picker = context.attachModelPicker(input, {
    fetchModels: (url, options) => { calls.push({url, options}); return reply(); },
    getConfig: () => config,
  });
  return {
    picker, input, calls,
    select: input.row.children[0], button: input.row.children[1],
    reply: callback => { reply = callback; },
    config: value => { config = {...config, ...value}; },
  };
}
const model = id => ({id, display_name: id});
const app = setup();
const scenario = process.argv[2];

if (scenario === 'multiple_choose_manual') {
  app.reply(async () => response({ok: true, models: [model('served-a'), model('served-b')]}));
  await app.picker.refresh();
  assert.equal(app.input.value, '');
  assert.equal(app.select.value, '');
  assert.deepEqual(app.select.children.map(option => option.value), ['', 'served-a', 'served-b']);
  app.select.value = 'served-b';
  app.select.dispatchEvent(new Event('change'));
  assert.equal(app.input.value, 'served-b');
  app.input.value = 'my-manual-choice';
  app.input.dispatchEvent(new Event('input'));
  await app.picker.refresh();
  assert.equal(app.input.value, 'my-manual-choice');
  assert.equal(app.select.value, 'my-manual-choice');
  assert.equal(app.calls[0].url, '/admin/llm/models');
  assert.equal(JSON.parse(app.calls[0].options.body).llm_base_url, 'http://localhost:11434/v1');
} else if (scenario === 'sole_visible') {
  app.reply(async () => response({ok: true, models: [model('only-installed')]}));
  await app.picker.refresh();
  assert.equal(app.input.value, 'only-installed');
  assert.match(app.input.hint.textContent, /Only one model.*preselected/);
} else if (scenario === 'preserve_configured') {
  app.input.value = 'existing-configured-choice';
  app.reply(async () => response({ok: true, models: [model('different-installed')]}));
  await app.picker.refresh();
  assert.equal(app.input.value, 'existing-configured-choice');
  assert.equal(app.select.value, 'existing-configured-choice');
} else if (scenario === 'empty_ollama') {
  const message = 'No models are installed in Ollama yet — install one with `ollama pull <model>` (see https://ollama.com/library), then rerun setup or choose in Settings.';
  app.reply(async () => response({ok: true, models: [], message}));
  await app.picker.refresh();
  assert.equal(app.input.value, '');
  assert.equal(app.input.hint.textContent, message);
  assert.equal(app.select.children.length, 1);
  assert.equal(app.button.disabled, false);
} else if (scenario === 'error_and_retry') {
  app.reply(async () => response({ok: false, error: 'The backend rejected authentication.'}));
  await app.picker.refresh();
  assert.match(app.input.hint.textContent, /authentication/);
  app.input.value = 'typed-after-error';
  app.reply(async () => response({ok: true, models: [model('served-a'), model('served-b')]}));
  await app.button.dispatchEvent(new Event('click'));
  assert.equal(app.input.value, 'typed-after-error');
} else if (scenario === 'plaintext_failure') {
  app.reply(async () => ({ok: false, status: 503, text: async () => 'Backend unavailable'}));
  await app.picker.refresh();
  assert.match(app.input.hint.textContent, /Backend unavailable/);
  assert.doesNotMatch(app.input.hint.textContent, /Unexpected token|JSON/);
  assert.equal(app.button.disabled, false);
} else if (scenario === 'network_failure') {
  app.reply(async () => { throw new Error('secret-in-network-message'); });
  await app.picker.refresh();
  assert.match(app.input.hint.textContent, /Could not discover/);
  assert.doesNotMatch(app.input.hint.textContent, /secret/);
  assert.equal(app.button.disabled, false);
} else if (scenario === 'stale_url_response') {
  const wait = pending();
  app.reply(() => wait.promise);
  const first = app.picker.refresh();
  app.config({llm_base_url: 'http://localhost:1234/v1'});
  app.picker.invalidate({clear: true});
  app.reply(async () => response({ok: true, models: [model('new-model')]}));
  await app.picker.refresh();
  wait.resolve(response({ok: true, models: [model('stale-model')]}));
  await first;
  assert.equal(app.input.value, 'new-model');
  assert.equal(app.select.value, 'new-model');
} else if (scenario === 'manual_while_loading') {
  const wait = pending();
  app.reply(() => wait.promise);
  const refresh = app.picker.refresh();
  app.input.value = 'chosen-during-request';
  app.input.dispatchEvent(new Event('input'));
  wait.resolve(response({ok: true, models: [model('only-served')]}));
  await refresh;
  assert.equal(app.input.value, 'chosen-during-request');
} else if (scenario === 'codex_default') {
  app.config({llm_backend: 'codex', llm_base_url: ''});
  app.reply(async () => response({ok: true, models: [{slug: 'catalog-choice', display_name: 'Catalog choice'}]}));
  await app.picker.refresh();
  assert.equal(app.calls[0].url, '/admin/codex/models');
  assert.equal(app.input.value, '');
  assert.equal(app.select.children[0].textContent, 'Codex default');
  app.select.value = 'catalog-choice';
  app.select.dispatchEvent(new Event('change'));
  assert.equal(app.input.value, 'catalog-choice');
  app.select.value = '';
  app.select.dispatchEvent(new Event('change'));
  assert.equal(app.input.value, '');
} else if (scenario === 'backend_reset') {
  app.input.value = 'previous-choice';
  app.picker.invalidate({clear: true});
  assert.equal(app.input.value, '');
  assert.equal(app.select.value, '');
} else {
  throw new Error('Unknown scenario: ' + scenario);
}
