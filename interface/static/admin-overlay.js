import { createSurfaceState, PURIFY_CONFIG } from "/assets/lib/shared.js";
import DOMPurify from "/assets/lib/dompurify.esm.js";

export const AdminOverlayState = createSurfaceState({
  open: false,
  statusHtml: "Runtime configuration for OpenRook.",
  statusError: false,
  statusOk: false,
}, {
  open(value) {
    const overlay = document.getElementById("settingsOverlay");
    if (!overlay) return;
    overlay.classList.toggle("open", !!value);
    overlay.setAttribute("aria-hidden", value ? "false" : "true");
  },
  statusHtml(value, _previous, state) {
    const el = document.getElementById("settingsStatus");
    if (!el) return;
    el.classList.toggle("error", !!state.statusError);
    el.classList.toggle("ok", !!state.statusOk && !state.statusError);
    // Callers pass trusted markup (chat.js setStatusHtml wraps escaped
    // values in note <div>s), but sanitize at the sink anyway so this
    // never becomes an XSS vector if a future caller forgets to escape.
    el.innerHTML = DOMPurify.sanitize(value, PURIFY_CONFIG);
  },
});
