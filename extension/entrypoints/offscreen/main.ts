/**
 * Offscreen document: hosts the spoiler model.
 *
 * Manifest V3 service workers can't run WebAssembly threads or keep large
 * objects alive, so the model lives here and answers messages relayed by the
 * background worker. Everything runs locally: remote model loading is off and
 * ONNX Runtime's WASM files come from the extension package.
 */
import {
  AutoModelForSequenceClassification,
  AutoTokenizer,
  env,
  type PreTrainedModel,
  type PreTrainedTokenizer,
} from '@huggingface/transformers';
import { BENCH_SENTENCES } from '@/lib/bench-sentences';
import type {
  BenchRequest,
  BenchResponse,
  DiagnosticsResponse,
  LatencyStats,
  Request,
  ScoreResponse,
} from '@/lib/messages';
import { percentile, spoilerProbabilities } from '@/lib/scoring';

// Log every fetch. Together with resource timing (which records any http(s)
// load, including scripts) this lets the locality test prove nothing is
// downloaded from outside the extension package. transformers.js captures
// `fetch` when its module loads, so it also gets the logging fetch via env.
const fetchLog: string[] = [];
const nativeFetch = globalThis.fetch.bind(globalThis);
const loggingFetch = (input: RequestInfo | URL, init?: RequestInit) => {
  fetchLog.push(input instanceof Request ? input.url : String(input));
  return nativeFetch(input, init);
};
globalThis.fetch = loggingFetch;

const MODEL_ID = 'spike';
const MAX_TOKENS = 64;
// Threads need cross-origin isolation (set in the manifest). `?threads=N` on
// the offscreen URL overrides the default, which the benchmark uses to compare.
const MAX_THREADS = 4;
const requestedThreads = Number(new URLSearchParams(location.search).get('threads'));
const THREADS =
  requestedThreads > 0
    ? requestedThreads
    : globalThis.crossOriginIsolated
      ? Math.min(MAX_THREADS, navigator.hardwareConcurrency || 1)
      : 1;

env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = chrome.runtime.getURL('models/');
env.useBrowserCache = false;
env.useWasmCache = false;
env.fetch = loggingFetch;
const wasm = env.backends.onnx.wasm!;
wasm.wasmPaths = {
  mjs: chrome.runtime.getURL('ort/ort-wasm-simd-threaded.asyncify.mjs'),
  wasm: chrome.runtime.getURL('ort/ort-wasm-simd-threaded.asyncify.wasm'),
};
wasm.numThreads = THREADS;

interface Loaded {
  tokenizer: PreTrainedTokenizer;
  model: PreTrainedModel;
  loadMs: number;
}

let loading: Promise<Loaded> | null = null;

function load(): Promise<Loaded> {
  loading ??= (async () => {
    const start = performance.now();
    const [tokenizer, model] = await Promise.all([
      AutoTokenizer.from_pretrained(MODEL_ID),
      AutoModelForSequenceClassification.from_pretrained(MODEL_ID, { dtype: 'q8', device: 'wasm' }),
    ]);
    return { tokenizer, model, loadMs: performance.now() - start };
  })();
  return loading;
}

async function score(sentences: string[]): Promise<number[]> {
  if (sentences.length === 0) return [];
  const { tokenizer, model } = await load();
  const inputs = tokenizer(sentences, { padding: true, truncation: true, max_length: MAX_TOKENS });
  const { logits } = await model(inputs);
  return spoilerProbabilities(logits.data as Float32Array, sentences.length);
}

async function bench(req: BenchRequest): Promise<BenchResponse> {
  const { tokenizer, loadMs } = await load();
  const tokenCounts = BENCH_SENTENCES.map(
    (s) => (tokenizer(s, { truncation: true, max_length: MAX_TOKENS }).input_ids.dims.at(-1) as number),
  );
  const results: LatencyStats[] = [];
  for (const batchSize of req.batchSizes) {
    const calls: number[] = [];
    const warmup = 5;
    for (let i = 0; i < warmup + req.iterations; i++) {
      const offset = (i * batchSize) % BENCH_SENTENCES.length;
      const batch = Array.from(
        { length: batchSize },
        (_, j) => BENCH_SENTENCES[(offset + j) % BENCH_SENTENCES.length] as string,
      );
      const t0 = performance.now();
      await score(batch);
      if (i >= warmup) calls.push(performance.now() - t0);
    }
    const perSentence = calls.map((ms) => ms / batchSize);
    results.push({
      batchSize,
      iterations: req.iterations,
      p50MsPerSentence: percentile(perSentence, 50),
      p95MsPerSentence: percentile(perSentence, 95),
      p50MsPerCall: percentile(calls, 50),
      p95MsPerCall: percentile(calls, 95),
    });
  }
  return {
    ok: true,
    backend: 'wasm',
    threads: THREADS,
    crossOriginIsolated: globalThis.crossOriginIsolated,
    loadMs,
    meanTokens: tokenCounts.reduce((a, b) => a + b, 0) / tokenCounts.length,
    results,
    userAgent: navigator.userAgent,
    hardwareConcurrency: navigator.hardwareConcurrency,
  };
}

chrome.runtime.onMessage.addListener((msg: Request, _sender, sendResponse) => {
  if (msg?.target !== 'offscreen') return false;
  let work: Promise<ScoreResponse | BenchResponse | DiagnosticsResponse>;
  switch (msg.type) {
    case 'score':
      work = (async () => {
        const t0 = performance.now();
        const scores = await score(msg.sentences);
        return { ok: true as const, scores, ms: performance.now() - t0 };
      })();
      break;
    case 'bench':
      work = bench(msg);
      break;
    case 'diagnostics':
      work = Promise.resolve({
        ok: true as const,
        resources: [
          location.href,
          ...performance.getEntriesByType('resource').map((e) => e.name),
          ...fetchLog,
        ],
      });
      break;
  }
  work.then(sendResponse, (err: unknown) => sendResponse({ ok: false, error: String(err) }));
  return true; // keep the channel open for the async response
});
