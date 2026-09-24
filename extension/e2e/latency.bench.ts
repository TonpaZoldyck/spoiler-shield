/**
 * Phase 0 latency benchmark (task P0.7).
 *
 * Runs the spike model inside the real extension's offscreen document under
 * each runtime configuration and writes the results to docs/results/. The
 * spike has random weights, but latency depends only on the architecture and
 * sequence lengths, so these numbers carry over to a trained MiniLM-L6 student.
 */
import fs from 'node:fs';
import path from 'node:path';
import type { BenchResponse } from '../lib/messages';
import { expect, test } from './extension';

type BenchOk = Extract<BenchResponse, { ok: true }>;

const BUDGET_P95_MS_PER_SENTENCE = 15;
const ITERATIONS = Number(process.env.BENCH_ITERATIONS ?? 60);
const BATCH_SIZES = [1, 8, 16];
const CONFIGS = [
  { label: 'Single thread', query: '?threads=1' },
  { label: 'Multi-threaded (default)', query: '' },
];
const RESULTS_DIR = path.resolve(import.meta.dirname, '../../docs/results');

function table(r: BenchOk): string {
  const rows = r.results.map(
    (s) =>
      `| ${s.batchSize} | ${s.p50MsPerSentence.toFixed(2)} | ${s.p95MsPerSentence.toFixed(2)} | ${s.p50MsPerCall.toFixed(1)} | ${s.p95MsPerCall.toFixed(1)} |`,
  );
  return [
    '| Batch size | p50 ms per sentence | p95 ms per sentence | p50 ms per call | p95 ms per call |',
    '| --- | --- | --- | --- | --- |',
    ...rows,
  ].join('\n');
}

const bestP95 = (r: BenchOk) => Math.min(...r.results.map((s) => s.p95MsPerSentence));

function toMarkdown(runs: { label: string; res: BenchOk }[], model: Record<string, unknown>): string {
  const winner = runs.reduce((a, b) => (bestP95(b.res) < bestP95(a.res) ? b : a));
  const best = bestP95(winner.res);
  const verdict =
    best <= BUDGET_P95_MS_PER_SENTENCE
      ? `**GO.** ${winner.label} reaches a p95 of ${best.toFixed(2)} ms per sentence, within the ${BUDGET_P95_MS_PER_SENTENCE} ms budget.`
      : `**NO-GO as configured.** The best p95 is ${best.toFixed(2)} ms per sentence (${winner.label}), over the ${BUDGET_P95_MS_PER_SENTENCE} ms budget.`;
  const first = runs[0]!.res;
  const sections = runs
    .map(
      ({ label, res }) =>
        `### ${label}\n\n${res.threads} thread(s), cross-origin isolated: ${res.crossOriginIsolated}, model load ${res.loadMs.toFixed(0)} ms.\n\n${table(res)}`,
    )
    .join('\n\n');
  return `# Phase 0 spike: in-browser latency

${verdict}

> The spike model has **random weights**. These numbers measure the speed of the
> MiniLM-L6 architecture only. They say nothing about spoiler detection quality.

## Setup

| Item | Value |
| --- | --- |
| Model | ${model.architecture} |
| Parameters | ${Number(model.parameters).toLocaleString('en')} |
| Model file, int8 | ${model.int8_mb} MB |
| Runtime | transformers.js with ONNX Runtime Web, ${first.backend.toUpperCase()} backend |
| Mean tokens per sentence | ${first.meanTokens.toFixed(1)}, truncated at 64 |
| Iterations per batch size | ${first.results[0]?.iterations}, after 5 warm-up calls |
| Machine | ${first.hardwareConcurrency} logical CPUs, headless Chromium in a shared cloud container |
| User agent | \`${first.userAgent}\` |

## Results

${sections}

## How to reproduce

\`\`\`bash
make bench
\`\`\`

A shared cloud container is noisy, so treat these numbers as indicative. Rerun on a
typical laptop before launch.
`;
}

test('in-browser latency of the spike model', async ({ sw }) => {
  const runs: { label: string; res: BenchOk }[] = [];
  for (const { label, query } of CONFIGS) {
    const res = (await sw.evaluate(
      async ({ iterations, batchSizes, query }) => {
        const g = globalThis as unknown as { ssRestartOffscreen: (q: string) => Promise<void> };
        await g.ssRestartOffscreen(query);
        return chrome.runtime.sendMessage({ target: 'offscreen', type: 'bench', iterations, batchSizes });
      },
      { iterations: ITERATIONS, batchSizes: BATCH_SIZES, query },
    )) as BenchResponse;
    expect(res.ok, JSON.stringify(res)).toBe(true);
    if (res.ok) runs.push({ label, res });
  }

  const model = JSON.parse(
    fs.readFileSync(path.resolve(import.meta.dirname, '../public/models/spike/spike.json'), 'utf8'),
  ) as Record<string, unknown>;
  fs.mkdirSync(RESULTS_DIR, { recursive: true });
  fs.writeFileSync(
    path.join(RESULTS_DIR, 'phase0-spike-latency.json'),
    `${JSON.stringify({ measuredAt: new Date().toISOString(), model, runs }, null, 2)}\n`,
  );
  fs.writeFileSync(path.join(RESULTS_DIR, 'phase0-spike-latency.md'), toMarkdown(runs, model));
  for (const { label, res } of runs) {
    console.log(label, `threads=${res.threads}`, `coi=${res.crossOriginIsolated}`);
    console.table(res.results.map((s) => ({ batch: s.batchSize, p50: s.p50MsPerSentence.toFixed(2), p95: s.p95MsPerSentence.toFixed(2) })));
  }
});
