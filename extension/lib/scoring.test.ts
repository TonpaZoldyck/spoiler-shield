import { describe, expect, it } from 'vitest';
import { percentile, spoilerProbabilities } from './scoring';

describe('spoilerProbabilities', () => {
  it('matches a two-class softmax', () => {
    const [p] = spoilerProbabilities([0.3, 1.7], 1);
    const e0 = Math.exp(0.3);
    const e1 = Math.exp(1.7);
    expect(p).toBeCloseTo(e1 / (e0 + e1), 12);
  });

  it('gives 0.5 for equal logits and handles several rows', () => {
    expect(spoilerProbabilities([2, 2, -1, 3, 5, -5], 3)).toEqual([
      0.5,
      expect.closeTo(0.982, 3),
      expect.closeTo(0.0000454, 6),
    ]);
  });

  it('stays finite for extreme logits', () => {
    const [hi, lo] = spoilerProbabilities([-1000, 1000, 1000, -1000], 2);
    expect(hi).toBe(1);
    expect(lo).toBe(0);
  });

  it('rejects a shape mismatch', () => {
    expect(() => spoilerProbabilities([1, 2, 3], 2)).toThrow(/Expected 4 logits/);
  });
});

describe('percentile', () => {
  it('interpolates linearly', () => {
    expect(percentile([1, 2, 3, 4, 5], 50)).toBe(3);
    expect(percentile([10, 20], 95)).toBeCloseTo(19.5);
    expect(percentile([7], 95)).toBe(7);
  });

  it('does not mutate its input', () => {
    const xs = [3, 1, 2];
    percentile(xs, 50);
    expect(xs).toEqual([3, 1, 2]);
  });

  it('rejects an empty list', () => {
    expect(() => percentile([], 50)).toThrow();
  });
});
