/**
 * Content script (Phase 0 spike): split page text into sentences, score them
 * with the on-device model, blur the ones above the threshold.
 *
 * Deliberately simple: it scores every sentence in plain-text paragraphs and
 * list items. The name-matching gate, site adapters and feedback UI arrive in
 * Phase 4.
 */
import type { ScoreResponse } from '@/lib/messages';
import { splitSentences } from '@/lib/segment';

const BATCH = 16;
const DEFAULT_THRESHOLD = 0.5;
const STYLE = `
.ss-sentence.ss-blur { filter: blur(6px); cursor: pointer; transition: filter 120ms ease; }
.ss-sentence.ss-blur:focus-visible { outline: 2px solid #7c5cff; outline-offset: 2px; }
.ss-sentence.ss-revealed { filter: none; }
`;

function candidateElements(root: ParentNode): HTMLElement[] {
  return [...root.querySelectorAll<HTMLElement>('p, li')].filter(
    (el) =>
      !el.dataset.ssDone &&
      el.children.length === 0 && // plain text only; rich markup is handled by site adapters later
      (el.textContent ?? '').trim().length >= 20,
  );
}

function wrapSentences(el: HTMLElement): HTMLSpanElement[] {
  const spans = splitSentences(el.textContent ?? '').map((text) => {
    const span = document.createElement('span');
    span.className = 'ss-sentence';
    span.textContent = text;
    return span;
  });
  el.replaceChildren(...spans.flatMap((s, i) => (i === 0 ? [s] : [document.createTextNode(' '), s])));
  el.dataset.ssDone = '1';
  return spans;
}

function applyScore(span: HTMLSpanElement, score: number, threshold: number): void {
  span.dataset.ssScore = score.toFixed(4);
  if (score < threshold) return;
  span.classList.add('ss-blur');
  span.tabIndex = 0;
  span.setAttribute('role', 'button');
  span.setAttribute('aria-label', 'Possible spoiler, hidden. Activate to reveal.');
  const reveal = () => {
    span.classList.replace('ss-blur', 'ss-revealed');
    span.removeAttribute('role');
    span.removeAttribute('aria-label');
    span.tabIndex = -1;
  };
  span.addEventListener('click', reveal, { once: true });
  span.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      reveal();
    }
  });
}

async function scoreBatch(sentences: string[]): Promise<number[]> {
  const res: ScoreResponse = await chrome.runtime.sendMessage({ target: 'background', type: 'score', sentences });
  if (!res?.ok) throw new Error(res?.error ?? 'no response from background');
  return res.scores;
}

export default defineContentScript({
  matches: ['*://*.reddit.com/*', '*://letterboxd.com/*', '*://*.youtube.com/*'],
  runAt: 'document_idle',
  async main() {
    const root = document.documentElement;
    root.dataset.ssStatus = 'scoring';
    const style = document.createElement('style');
    style.textContent = STYLE;
    document.head.append(style);

    const { threshold } = (await chrome.storage.local.get({ threshold: DEFAULT_THRESHOLD })) as {
      threshold: number;
    };
    const spans = candidateElements(document).flatMap(wrapSentences);
    try {
      for (let i = 0; i < spans.length; i += BATCH) {
        const batch = spans.slice(i, i + BATCH);
        const scores = await scoreBatch(batch.map((s) => s.textContent ?? ''));
        if (scores.length !== batch.length) throw new Error('score count does not match sentence count');
        batch.forEach((span, j) => applyScore(span, scores[j] as number, threshold));
      }
      root.dataset.ssStatus = 'done';
    } catch (err) {
      root.dataset.ssStatus = 'error';
      root.dataset.ssError = String(err);
      console.error('[Spoiler Shield]', err);
    }
  },
});
