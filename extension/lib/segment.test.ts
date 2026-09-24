import { describe, expect, it } from 'vitest';
import { splitSentences } from './segment';

describe('splitSentences', () => {
  it('splits ordinary sentences and trims them', () => {
    expect(splitSentences('The ending was great.  Did anyone cry? I did!')).toEqual([
      'The ending was great.',
      'Did anyone cry?',
      'I did!',
    ]);
  });

  it('keeps common abbreviations inside a sentence', () => {
    expect(splitSentences('Dr. Mara Quinn was the informant. Nobody saw it coming.')).toEqual([
      'Dr. Mara Quinn was the informant.',
      'Nobody saw it coming.',
    ]);
  });

  it('merges episode numbers and initials back into their sentence', () => {
    expect(
      splitSentences('Mr. Smith agreed, e.g. in ep. 3. J. Doe did not. The end!'),
    ).toEqual(['Mr. Smith agreed, e.g. in ep. 3.', 'J. Doe did not.', 'The end!']);
  });

  it('keeps a trailing abbreviation at the very end', () => {
    expect(splitSentences('I watched it with Dr.')).toEqual(['I watched it with Dr.']);
  });

  it('returns nothing for blank text', () => {
    expect(splitSentences('   \n  ')).toEqual([]);
  });
});
