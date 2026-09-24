from spoiler_shield.data.segment import split_sentences

# Mirrors extension/lib/segment.test.ts so both splitters stay in step.


def test_splits_ordinary_sentences():
    assert split_sentences("The ending was great.  Did anyone cry? I did!") == [
        "The ending was great.",
        "Did anyone cry?",
        "I did!",
    ]


def test_keeps_abbreviations_inside_a_sentence():
    assert split_sentences("Dr. Mara Quinn was the informant. Nobody saw it coming.") == [
        "Dr. Mara Quinn was the informant.",
        "Nobody saw it coming.",
    ]


def test_merges_episode_numbers_and_initials():
    assert split_sentences("Mr. Smith agreed, e.g. in ep. 3. J. Doe did not. The end!") == [
        "Mr. Smith agreed, e.g. in ep. 3.",
        "J. Doe did not.",
        "The end!",
    ]


def test_trailing_abbreviation_and_blank_text():
    assert split_sentences("I watched it with Dr.") == ["I watched it with Dr."]
    assert split_sentences("   \n ") == []


def test_closing_quotes_stay_with_their_sentence():
    assert split_sentences('She said "run!" Then it cut to black.') == [
        'She said "run!"',
        "Then it cut to black.",
    ]
