/**
 * Playwright fixtures that launch Chromium with the built extension loaded.
 *
 * Set PW_CHROMIUM_PATH to use a specific Chromium build (for example a
 * pre-installed one); otherwise Playwright's own Chromium is used.
 * Extensions need full Chromium, not the headless shell.
 */
import path from 'node:path';
import { chromium, test as base, type BrowserContext, type Worker } from '@playwright/test';

export const EXTENSION_DIR = path.resolve(import.meta.dirname, '../.output/chrome-mv3');

export const test = base.extend<{ context: BrowserContext; sw: Worker }>({
  // biome-ignore lint/correctness/noEmptyPattern: Playwright fixture signature
  context: async ({}, use) => {
    const executablePath = process.env.PW_CHROMIUM_PATH;
    const context = await chromium.launchPersistentContext('', {
      headless: true,
      // The 'chromium' channel is full Chromium in new headless mode, which
      // supports extensions (the default headless shell does not).
      ...(executablePath ? { executablePath } : { channel: 'chromium' }),
      args: [`--disable-extensions-except=${EXTENSION_DIR}`, `--load-extension=${EXTENSION_DIR}`],
    });
    await use(context);
    await context.close();
  },
  sw: async ({ context }, use) => {
    const sw = context.serviceWorkers()[0] ?? (await context.waitForEvent('serviceworker'));
    await use(sw);
  },
});

export const expect = test.expect;
