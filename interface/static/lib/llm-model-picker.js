import { readJsonResponse, ServerResponseError } from "/assets/lib/http.js";
// Shared by the first-run wizard, chat drawer, and Settings workspace.
// The input remains the submitted value; discovery never chooses among models.
export function attachModelPicker(input, { fetchModels, getConfig }) {
  if (!input) return null;
  if (input.modelPicker) return input.modelPicker;
  const row = document.createElement('div');
  row.className = 'model-picker-row';
  const select = document.createElement('select');
  select.id = input.id + '-discovered';
  select.setAttribute('aria-label', 'Models served by this backend');
  const refreshButton = document.createElement('button');
  refreshButton.type = 'button';
  refreshButton.className = 'secondary';
  refreshButton.textContent = 'Refresh models';
  const hint = document.createElement('div');
  hint.id = input.id + '-discovery-status';
  hint.className = 'settings-note field-help model-picker-status';
  hint.setAttribute('role', 'status');
  input.setAttribute('aria-describedby', hint.id);
  input.placeholder = 'Or type a model name';
  row.append(select, refreshButton);
  input.before(row);
  input.after(hint);
  let generation = 0;
  let models = [];

  function render() {
    const codex = getConfig().llm_backend === 'codex';
    select.replaceChildren();
    const add = (value, label) => {
      const option = document.createElement('option');
      option.value = value;
      option.textContent = label;
      select.appendChild(option);
    };
    add('', codex ? 'Codex default' : 'Choose a discovered model');
    for (const model of models) {
      add(model.id, model.display_name && model.display_name !== model.id
        ? `${model.display_name} (${model.id})` : model.id);
    }
    if (input.value && !models.some(model => model.id === input.value)) {
      add(input.value, input.value + ' (configured or entered manually)');
    }
    select.value = input.value;
    input.placeholder = codex ? 'Optional; blank uses Codex default' : 'Or type a model name';
  }

  function invalidate({ clear = false } = {}) {
    generation++;
    models = [];
    if (clear) input.value = '';
    refreshButton.disabled = false;
    hint.textContent = 'Refresh models after entering the backend URL and any required API key. You can also type a name.';
    render();
  }

  async function refresh() {
    const request = ++generation;
    const config = getConfig();
    const snapshot = JSON.stringify(config);
    const startingValue = input.value;
    if (config.llm_backend !== 'codex' && !config.llm_base_url) {
      invalidate();
      hint.textContent = 'Enter the backend URL, then refresh models or type a model name.';
      return;
    }
    refreshButton.disabled = true;
    hint.textContent = 'Loading models…';
    try {
      const codex = config.llm_backend === 'codex';
      const response = await fetchModels(codex ? '/admin/codex/models' : '/admin/llm/models',
        codex ? undefined : {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(config),
        });
      const data = await readJsonResponse(response);
      // Ignore results for a previous URL/key/backend, or a detached form.
      if (request !== generation || snapshot !== JSON.stringify(getConfig()) || !input.isConnected) return;
      if (!response.ok || !data.ok) {
        models = [];
        hint.textContent = response.status === 401
          ? 'Sign in to OpenRook Settings, then refresh models.'
          : (data.error || data.message || (typeof data.detail === 'string' && data.detail)
            || 'Could not discover models. Check the URL and key, refresh, or type a name manually.');
        render();
        return;
      }
      models = (Array.isArray(data.models) ? data.models : [])
        .map(model => ({id: model.id || model.slug, display_name: model.display_name}))
        .filter(model => typeof model.id === 'string' && model.id);
      let preselected = false;
      if (!codex && models.length === 1 && !input.value && input.value === startingValue) {
        input.value = models[0].id;
        preselected = true;
        input.dispatchEvent(new Event('change', {bubbles: true}));
      }
      render();
      hint.textContent = preselected
        ? 'Only one model is available; it is preselected above. You can change it.'
        : (data.message || (models.length
          ? 'Choose a model from this backend, or type a name manually.'
          : 'No models were returned. Refresh after making a model available, or type a name manually.'));
    } catch (error) {
      if (request !== generation || snapshot !== JSON.stringify(getConfig())) return;
      models = [];
      render();
      hint.textContent = error instanceof ServerResponseError
        ? `${error.message} Refresh models, or type a name manually.`
        : 'Could not discover models. Check the connection, refresh, or type a name manually.';
    } finally {
      if (request === generation) refreshButton.disabled = false;
    }
  }

  select.addEventListener('change', () => {
    input.value = select.value;
    input.dispatchEvent(new Event('change', {bubbles: true}));
  });
  input.addEventListener('input', render);
  refreshButton.addEventListener('click', refresh);
  input.modelPicker = {refresh, invalidate};
  invalidate();
  return input.modelPicker;
}
