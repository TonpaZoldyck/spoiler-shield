/**
 * Phase 0 end-to-end test: the real built extension, in real Chromium, scores
 * every sentence of a fake Reddit thread with the on-device model and blurs
 * the ones at or above the threshold.
 *
 * The spike model has random weights, so the test checks plumbing and
 * behaviour at fixed thresholds, not prediction quality.
 */
import fs from 'node:fs';
import type { BrowserContext, Page } from '@playwright/test';
import { expect, test } from './extension';

const THREAD_URL = 'https://www.reddit.com/r/television/comments/spike/northwind_s2e6/';
const FIXTURE = fs.readFileSync(new URL('./fixtures/reddit-thread.html', import.meta.url), 'utf8');

async function openThread(context: BrowserContext): Promise<Page> {
  // Serve the fixture at a real Reddit URL so the production match patterns apply.
  await context.route('https://www.reddit.com/**', (route) =>
    route.fulfill({ status: 200, contentType: 'text/html', body: FIXTURE }),
  );
  const page = await context.newPage();
  await page.goto(THREAD_URL);
  await page.waitForSelector('html[data-ss-status="done"], html[data-ss-status="error"]', {
    timeout: 90_000,
  });
  const status = await page.getAttribute('html', 'data-ss-status');
  expect(status, (await page.getAttribute('html', 'data-ss-error')) ?? '').toBe('done');
  return page;
}

async function scores(page: Page): Promise<number[]> {
  return page.locator('.ss-sentence').evaluateAll((els) => els.map((el) => Number((el as HTMLElement).dataset.ssScore)));
}

test('scores every sentence on-device and blurs at threshold 0', async ({ context, sw }) => {
  await sw.evaluate(() => chrome.storage.local.set({ threshold: 0 }));

  const page = await openThread(context);
  const all = await scores(page);
  expect(all.length).toBeGreaterThanOrEqual(15);
  for (const s of all) {
    expect(Number.isFinite(s)).toBe(true);
    expect(s).toBeGreaterThanOrEqual(0);
    expect(s).toBeLessThanOrEqual(1);
  }
  await expect(page.locator('.ss-sentence.ss-blur')).toHaveCount(all.length);
  // Rich markup is untouched until site adapters exist.
  await expect(page.locator('#rich .ss-sentence')).toHaveCount(0);
});

test('loads the model and runtime only from the extension package', async ({ context, sw }) => {
  await sw.evaluate(() => chrome.storage.local.set({ threshold: 0.5 }));
  await openThread(context);
  const diag = await sw.evaluate(() => chrome.runtime.sendMessage({ target: 'offscreen', type: 'diagnostics' }));
  const resources: string[] = diag.resources;
  // The first entry is the offscreen page itself. Node's URL gives
  // chrome-extension:// an opaque "null" origin, so compare prefixes.
  const base = resources[0]?.match(/^chrome-extension:\/\/[^/]+\//)?.[0];
  expect(base, resources.join('\n')).toBeTruthy();
  // Sanity: the model and the WASM runtime were actually fetched and observed.
  expect(resources.some((u) => u.endsWith('/onnx/model_quantized.onnx'))).toBe(true);
  expect(resources.some((u) => u.endsWith('.wasm'))).toBe(true);
  expect(resources.filter((u) => !u.startsWith(base as string))).toEqual([]);
});

test('reveals a sentence by click and by keyboard', async ({ context, sw }) => {
  await sw.evaluate(() => chrome.storage.local.set({ threshold: 0 }));
  const page = await openThread(context);

  const first = page.locator('.ss-sentence').nth(0);
  await first.click();
  await expect(first).toHaveClass(/ss-revealed/);

  const second = page.locator('.ss-sentence').nth(1);
  await second.focus();
  await page.keyboard.press('Enter');
  await expect(second).toHaveClass(/ss-revealed/);
  await expect(second).not.toHaveAttribute('role', 'button');
});

test('blurs nothing when the threshold is above every score', async ({ context, sw }) => {
  await sw.evaluate(() => chrome.storage.local.set({ threshold: 1.01 }));
  const page = await openThread(context);
  expect((await scores(page)).length).toBeGreaterThanOrEqual(15);
  await expect(page.locator('.ss-sentence.ss-blur')).toHaveCount(0);
});
