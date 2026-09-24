import gzip
import json

from spoiler_shield.data.sources import goodreads_examples, imdb_examples


def test_goodreads_sentences_labels_context_and_markup(tmp_path):
    path = tmp_path / "goodreads_reviews_spoiler.json.gz"
    reviews = [
        {
            "user_id": "u1",
            "book_id": 42,
            "review_id": "r1",
            "has_spoiler": True,
            "review_sentences": [
                [0, "Loved this book from start to finish."],
                [1, "(view spoiler)[Mara was the informant all along. (hide spoiler)]"],
                [0, "Wow."],  # too short, dropped
                [0, "Can't wait for the sequel next year."],
            ],
        }
    ]
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in reviews:
            f.write(json.dumps(r) + "\n")

    rows = list(goodreads_examples(path, titles={"42": "The Lighthouse"}))
    assert [r.label for r in rows] == [0, 1, 0]
    assert rows[1].sentence == "Mara was the informant all along."
    assert rows[1].context == "Loved this book from start to finish."
    assert rows[2].context == rows[1].sentence  # context skips dropped fragments
    assert {r.title_id for r in rows} == {"goodreads:42"}
    assert rows[0].title == "The Lighthouse"
    assert len({r.example_id for r in rows}) == 3


def test_imdb_review_level_flags_become_unlabelled_or_negative(tmp_path):
    path = tmp_path / "IMDB_reviews.json"
    rows_in = [
        {
            "movie_id": "tt001",
            "user_id": "ur1",
            "is_spoiler": True,
            "review_text": "Great film. The captain faked the distress call. Loved it.",
        },
        {
            "movie_id": "tt001",
            "user_id": "ur2",
            "is_spoiler": False,
            "review_text": "Beautiful score and costumes. Worth a watch in cinemas.",
        },
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows_in) + "\n")

    rows = list(imdb_examples(path))
    flagged = [r for r in rows if r.review_id and r.review_id.startswith("tt001:ur1")]
    clean = [r for r in rows if r.review_id and r.review_id.startswith("tt001:ur2")]
    assert [r.sentence for r in flagged] == [
        "Great film.",
        "The captain faked the distress call.",
        "Loved it.",
    ]
    assert all(r.label is None for r in flagged)  # awaits the Phase 2 labeller
    assert all(r.label == 0 for r in clean)
    assert {r.title_id for r in rows} == {"imdb:tt001"}
