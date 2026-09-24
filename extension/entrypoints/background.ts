/**
 * Background service worker: relays scoring requests from content scripts to
 * the offscreen document that hosts the model, creating it on first use.
 */
import type { Request } from '@/lib/messages';

const OFFSCREEN_URL = 'offscreen.html';
let creating: Promise<void> | null = null;

async function ensureOffscreen(query = ''): Promise<void> {
  if (await chrome.offscreen.hasDocument()) return;
  creating ??= chrome.offscreen
    .createDocument({
      url: OFFSCREEN_URL + query,
      reasons: [chrome.offscreen.Reason.WORKERS],
      justification: 'Runs the spoiler classifier locally with WebAssembly.',
    })
    .finally(() => {
      creating = null;
    });
  await creating;
}

export default defineBackground(() => {
  chrome.runtime.onMessage.addListener((msg: Request, _sender, sendResponse) => {
    if (msg?.target !== 'background') return false;
    ensureOffscreen()
      .then(() => chrome.runtime.sendMessage({ ...msg, target: 'offscreen' }))
      .then(sendResponse, (err: unknown) => sendResponse({ ok: false, error: String(err) }));
    return true;
  });

  // Exposed for the end-to-end and benchmark harnesses, which drive the worker
  // directly. Restarting with a query string lets the benchmark compare configs.
  Object.assign(globalThis, {
    ssEnsureOffscreen: ensureOffscreen,
    ssRestartOffscreen: async (query: string) => {
      if (await chrome.offscreen.hasDocument()) await chrome.offscreen.closeDocument();
      await ensureOffscreen(query);
    },
  });
});
