import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';

const readAsset = name => readFileSync(new URL('../interface/static/' + name, import.meta.url), 'utf8');
const context = vm.createContext({});
vm.runInContext(readAsset('lib/llm-backend-fields.js').replace(/^export /gm, ''), context);

function field(value = '') {
  const card = {hidden: false};
  return {value, checked: false, disabled: false, closest: () => card};
}

function setupBackend(backendName, baseUrlValue) {
  const fields = {
    backend: field(backendName), baseUrl: field(baseUrlValue),
    apiKey: field(), clearApiKey: field(),
  };
  const controller = context.attachBackendFields(fields);
  return {...fields, select(name) { fields.backend.value = name; controller.update(); }};
}

function setupAdmin() {
  const elements = new Map();
  const document = {
    getElementById: id => elements.get(id) || null,
    querySelector: selector => elements.get(selector.match(/="([^"]+)"/)?.[1]) || null,
  };
  const admin = vm.createContext({
    document, CSS: {escape: value => value},
    createSurfaceState: value => value,
    healthLabel: () => '', backendLabel: () => '',
  });
  vm.runInContext(readAsset('lib/http.js').replace(/^export /gm, ''), admin);
  // Load declarations without booting the network-backed page.
  const source = readAsset('admin.js').split('\n  wireEvents();')[0].replace(/^import .*;\n/gm, '');
  vm.runInContext(source, admin);
  return {admin, elements};
}

const scenario = process.argv[2];
if (scenario === 'tool_inventory_error') {
  const {admin} = setupAdmin();
  vm.runInContext(`
    authedFetch = async () => ({
      ok: false, status: 500, text: async () => 'Internal Server Error',
    });
  `, admin);
  await assert.rejects(admin.loadTools(), {message: 'Internal Server Error'});
  vm.runInContext(`
    authedFetch = async () => ({
      ok: false, status: 503,
      text: async () => JSON.stringify({detail: 'No model is selected.'}),
    });
  `, admin);
  await assert.rejects(admin.loadTools(), {message: 'No model is selected.'});
} else if (scenario === 'codex_fields') {
  const app = setupBackend('codex', 'http://localhost:11434/v1');
  for (const input of [app.baseUrl, app.apiKey, app.clearApiKey]) {
    assert.equal(input.disabled, true);
    assert.equal(input.closest('.field-card').hidden, true);
  }
  assert.equal(app.baseUrl.value, '');
  app.select('provider');
  assert.equal(app.baseUrl.value, '');
  assert.equal(app.baseUrl.disabled, false);
  assert.equal(app.apiKey.closest('.field-card').hidden, false);
} else if (scenario === 'backend_defaults_and_drafts') {
  const app = setupBackend('provider', 'https://gateway.example/v1');
  app.apiKey.value = 'unsaved-test-key';
  app.clearApiKey.checked = true;
  app.select('openai_compatible');
  assert.equal(app.baseUrl.value, 'http://localhost:11434/v1');
  assert.equal(app.apiKey.value, '');
  assert.equal(app.clearApiKey.checked, false);
  app.baseUrl.value = 'http://localhost:22444/v1';
  app.select('codex');
  assert.equal(app.baseUrl.value, '');
  assert.equal(app.apiKey.value, '');
  assert.equal(app.clearApiKey.checked, false);
  app.select('openai_compatible');
  assert.equal(app.baseUrl.value, 'http://localhost:22444/v1');
  app.select('provider');
  assert.equal(app.baseUrl.value, 'https://gateway.example/v1');
  assert.equal(app.apiKey.value, 'unsaved-test-key');
  assert.equal(app.clearApiKey.checked, true);
  app.select('auto');
  assert.equal(app.baseUrl.value, '');
} else if (scenario === 'section_help') {
  const {admin, elements} = setupAdmin();
  const form = {closest: () => ({id: 'settings-advanced'})};
  elements.set('characterForm', form);
  vm.runInContext(`
    state.metadata = fieldSections.character.map(([name]) => ({name, session_manager_reload: true}));
    renderConfigForm('characterForm', fieldSections.character);
  `, admin);
  assert.equal(form.innerHTML.match(/Reloads active runtime manager on save\./g).length, 1);
  assert.match(form.innerHTML, /An informational flag; does not change behavior/);
  assert.doesNotMatch(form.innerHTML, /<details/);
} else if (scenario === 'advanced_payload') {
  const {admin, elements} = setupAdmin();
  const values = {
    learning_budget_kind: 'local', learning_max_questions_per_day: 8,
    learning_max_seconds_per_day: 30, metabolism_min_elapsed_days: 0.75,
  };
  for (const [name, value] of Object.entries(values)) {
    elements.set(name, {value: String(value), type: typeof value === 'number' ? 'number' : 'select'});
  }
  const payload = admin.collectConfigPayload();
  for (const [name, value] of Object.entries(values)) assert.equal(payload[name], value);
} else if (scenario === 'advanced_deep_link') {
  const {admin, elements} = setupAdmin();
  const advanced = {tagName: 'DETAILS', open: false, parentElement: null};
  const container = {tagName: 'DIV', parentElement: advanced};
  const section = {
    tagName: 'DETAILS', open: false, parentElement: container,
    scrollIntoView() { assert.equal(advanced.open, true); },
  };
  elements.set('settings-runtime', section);
  admin.openSettingsSection('runtime');
  assert.equal(section.open, true);
  assert.equal(advanced.open, true);
} else if (scenario === 'overview_paths') {
  const {admin, elements} = setupAdmin();
  const cards = {};
  elements.set('overviewCards', cards);
  vm.runInContext(`
    state.status = {tools: {workspace: '/a/long/workspace'}, setup: {state_path: '/a/"quoted"/admin_state.json'}};
    renderOverviewPersona = () => {};
    renderPersonaAdminPage = () => {};
    renderWarnings = () => {};
    setStatus = () => {};
    renderOverview();
  `, admin);
  assert.match(cards.innerHTML, /class="metric-foot metric-path" title="\/a\/long\/workspace"/);
  assert.match(cards.innerHTML, /title="\/a\/&quot;quoted&quot;\/admin_state\.json"/);
} else {
  throw new Error('Unknown scenario: ' + scenario);
}
