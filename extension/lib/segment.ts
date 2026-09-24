/**
 * Sentence splitting for page text.
 *
 * Uses the browser's built-in `Intl.Segmenter`, then repairs its main weakness:
 * ICU breaks after abbreviations such as "Dr." or "ep.", which would cut
 * "Dr. Mara Quinn was the informant." into two pieces and hide the context the
 * model needs. A piece ending in a known abbreviation or a single-letter
 * initial is merged with the piece that follows it.
 */

const ABBREVIATIONS = new Set(
  [
    'mr', 'mrs', 'ms', 'dr', 'prof', 'st', 'jr', 'sr', 'capt', 'lt', 'sgt', 'gen', 'col',
    'vs', 'etc', 'e.g', 'i.e', 'approx', 'no', 'vol', 'ep', 'eps', 'pt', 'ch', 'fig', 'mt',
  ].map((a) => `${a}.`),
);

function endsWithAbbreviation(piece: string): boolean {
  const lastWord = piece.trimEnd().split(/\s+/).pop()?.toLowerCase() ?? '';
  if (ABBREVIATIONS.has(lastWord)) return true;
  // Single-letter initial such as the "J." in "J. Doe".
  return /^[a-z]\.$/.test(lastWord);
}

export function splitSentences(text: string, locale = 'en'): string[] {
  const segmenter = new Intl.Segmenter(locale, { granularity: 'sentence' });
  const merged: string[] = [];
  let carry = '';
  for (const { segment } of segmenter.segment(text)) {
    carry += segment;
    if (!endsWithAbbreviation(carry)) {
      merged.push(carry);
      carry = '';
    }
  }
  if (carry) merged.push(carry);
  return merged.map((s) => s.trim()).filter((s) => s.length > 0);
}
