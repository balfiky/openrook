export function backendLabel(config = {}) {
  if (config.llm_backend === "codex") return "Codex CLI";
  if (config.llm_backend === "openrouter" || (config.llm_base_url || "").includes("openrouter.ai")) return "OpenRouter";
  if ((config.llm_base_url || "").includes(":11434")) return "Ollama";
  return config.llm_backend === "openai_compatible" ? "Model server" : (config.llm_backend || "Model");
}

export function healthLabel(health = {}) {
  if (health.status === "error") return "Last recorded error";
  if (health.status === "ok") return "Last test passed";
  return "Not tested";
}

export function connectionSummary(data) {
  if (data.ok) return data.live ? "Connection working — sample reply received." : "Configuration saved; connection not tested.";
  const diagnostic = data.diagnostic || data.llm_error;
  if (diagnostic) return [diagnostic.message, diagnostic.action].filter(Boolean).join(" ");
  const detail = data.error || data.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) return detail.map(item => item.msg || item.message || "Invalid setting").join(" ");
  if (detail?.message) return detail.message;
  return (data.warnings || []).map(item => item.message).join(" ") || "Connection failed. Check the backend, model, and login in Settings.";
}

export function renderConnectionResult(element, data) {
  element.hidden = false;
  element.textContent = connectionSummary(data);
  const detail = data.error_detail || data.diagnostic?.detail;
  if (detail) {
    const disclosure = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = "Technical details";
    const pre = document.createElement("pre");
    pre.textContent = detail;
    disclosure.append(summary, pre);
    element.appendChild(disclosure);
  }
}
