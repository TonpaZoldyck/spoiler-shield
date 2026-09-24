import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './e2e',
  timeout: 120_000,
  retries: 0,
  workers: 1, // extensions need a persistent context each; keep runs sequential
  reporter: [['list']],
  projects: [
    { name: 'e2e', testMatch: /.*\.e2e\.ts/ },
    { name: 'bench', testMatch: /.*\.bench\.ts/, timeout: 900_000 },
  ],
});
