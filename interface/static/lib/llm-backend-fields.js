// Keep connection drafts with their backend while the Settings form is open.
// Hosted endpoints are operator supplied; the local default matches setup.
const defaultBaseUrls = {
  auto: '',
  provider: '',
  openai_compatible: 'http://localhost:11434/v1',
  codex: '',
};

export function attachBackendFields({backend, baseUrl, apiKey, clearApiKey}) {
  let previousBackend = backend.value;
  const drafts = new Map();

  function readDraft() {
    return {baseUrl: baseUrl.value, apiKey: apiKey.value, clearApiKey: clearApiKey.checked};
  }

  function update() {
    const selected = backend.value;
    if (selected !== previousBackend) {
      drafts.set(previousBackend, readDraft());
      const draft = drafts.get(selected) || {
        baseUrl: defaultBaseUrls[selected] || '', apiKey: '', clearApiKey: false,
      };
      baseUrl.value = draft.baseUrl;
      apiKey.value = draft.apiKey;
      clearApiKey.checked = draft.clearApiKey;
      previousBackend = selected;
    }
    const codex = selected === 'codex';
    for (const field of [baseUrl, apiKey, clearApiKey]) {
      field.disabled = codex;
      field.closest('.field-card').hidden = codex;
    }
    if (codex) {
      baseUrl.value = '';
      apiKey.value = '';
      clearApiKey.checked = false;
    }
  }

  update();
  return {update};
}
