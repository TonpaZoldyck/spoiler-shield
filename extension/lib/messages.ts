/** Messages exchanged between the content script, background worker and offscreen document. */

export type Target = 'background' | 'offscreen';

export interface ScoreRequest {
  target: Target;
  type: 'score';
  sentences: string[];
}

export interface BenchRequest {
  target: 'offscreen';
  type: 'bench';
  iterations: number;
  batchSizes: number[];
}

/** Lists every resource the offscreen document has loaded, for the locality test. */
export interface DiagnosticsRequest {
  target: 'offscreen';
  type: 'diagnostics';
}

export interface DiagnosticsResponse {
  ok: true;
  resources: string[];
}

export type Request = ScoreRequest | BenchRequest | DiagnosticsRequest;

export type ScoreResponse =
  | { ok: true; scores: number[]; ms: number }
  | { ok: false; error: string };

export interface LatencyStats {
  batchSize: number;
  iterations: number;
  p50MsPerSentence: number;
  p95MsPerSentence: number;
  p50MsPerCall: number;
  p95MsPerCall: number;
}

export type BenchResponse =
  | {
      ok: true;
      backend: string;
      threads: number;
      crossOriginIsolated: boolean;
      loadMs: number;
      meanTokens: number;
      results: LatencyStats[];
      userAgent: string;
      hardwareConcurrency: number;
    }
  | { ok: false; error: string };
