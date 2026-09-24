import pytest
from pydantic import ValidationError

from spoiler_shield.data.cleaning import clean_sentence, is_usable, strip_spoiler_markup
from spoiler_shield.data.schema import Example, read_jsonl, write_jsonl


def _ex(**kw) -> Example:
    base = {"example_id": "x:1", "source": "gold", "title_id": "t1", "sentence": "A fine sentence."}
    return Example.model_validate(base | kw)


def test_example_rejects_blank_sentence_and_bad_label():
    with pytest.raises(ValidationError):
        _ex(sentence="   ")
    with pytest.raises(ValidationError):
        _ex(label=2)
    with pytest.raises(ValidationError):
        _ex(title_id="")


def test_example_is_frozen():
    ex = _ex()
    with pytest.raises(ValidationError):
        ex.label = 1  # type: ignore[misc]


def test_jsonl_round_trip(tmp_path):
    rows = [_ex(example_id=f"x:{i}", label=i % 2) for i in range(5)] + [_ex(example_id="u")]
    path = tmp_path / "sub" / "rows.jsonl"
    assert write_jsonl(path, rows) == 6
    assert list(read_jsonl(path)) == rows


def test_read_jsonl_reports_line_number(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(_ex().model_dump_json() + "\n" + '{"example_id": "y"}\n')
    with pytest.raises(ValueError, match=r"bad.jsonl:2"):
        list(read_jsonl(path))


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("(view spoiler)[She dies at the end. (hide spoiler)]", "She dies at the end."),
        (
            "Honestly >!the dog lives!< which is a relief",
            "Honestly the dog lives which is a relief",
        ),
        ("[spoiler]The butler did it[/spoiler]", "The butler did it"),
        ("**SPOILER ALERT**: he wakes up", "he wakes up"),
    ],
)
def test_spoiler_markup_is_removed_to_prevent_label_leaks(raw, expected):
    assert clean_sentence(raw) == expected
    assert "spoiler" not in clean_sentence(raw).lower()


def test_markup_removal_keeps_the_word_spoiler_in_ordinary_prose():
    # Only bracketed, bolded or Reddit-style markers are markup. The word itself
    # in a normal sentence is content the model should see.
    assert strip_spoiler_markup("No spoilers here, promise") == "No spoilers here, promise"
    assert clean_sentence("I hate spoilers in titles") == "I hate spoilers in titles"


def test_normalises_html_and_whitespace():
    assert clean_sentence("Tom &amp; Jerry\n\n  were   great") == "Tom & Jerry were great"


def test_is_usable_drops_fragments():
    assert not is_usable("Wow.")
    assert not is_usable("!!!!!!!!!!")
    assert is_usable("The ending was great.")
