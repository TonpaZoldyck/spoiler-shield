/**
 * Convert classifier logits to spoiler probabilities.
 *
 * `logits` is the flat row-major output of shape [n, 2] where column 0 is
 * "not_spoiler" and column 1 is "spoiler". A two-class softmax reduces to a
 * sigmoid of the logit difference, which is also numerically stable.
 */
export function spoilerProbabilities(logits: ArrayLike<number>, n: number): number[] {
  if (logits.length !== n * 2) {
    throw new Error(`Expected ${n * 2} logits for ${n} rows, got ${logits.length}`);
  }
  const out: number[] = [];
  for (let i = 0; i < n; i++) {
    const diff = (logits[i * 2 + 1] as number) - (logits[i * 2] as number);
    out.push(1 / (1 + Math.exp(-diff)));
  }
  return out;
}

/** Percentile with linear interpolation, `q` in [0, 100]. */
export function percentile(values: number[], q: number): number {
  if (values.length === 0) throw new Error('percentile of empty list');
  const sorted = [...values].sort((a, b) => a - b);
  const pos = (q / 100) * (sorted.length - 1);
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  const low = sorted[lo] as number;
  const high = sorted[hi] as number;
  return low + (high - low) * (pos - lo);
}
