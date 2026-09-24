from spoiler_shield import __version__
from spoiler_shield.cli import main


def test_version_command_prints_version(capsys):
    assert main(["version"]) == 0
    assert capsys.readouterr().out.strip() == __version__


def test_prepare_then_baselines_end_to_end(tmp_path):
    import json

    from tests.conftest import make_synthetic

    # Write synthetic rows in the Goodreads raw format, then run the real CLI.
    raw = tmp_path / "goodreads.json"
    by_title: dict[str, list] = {}
    for ex in make_synthetic(n_titles=40, per_title=25):
        by_title.setdefault(ex.title_id.split(":")[1], []).append([ex.label, ex.sentence])
    raw.write_text(
        "\n".join(
            json.dumps({"book_id": t, "review_id": f"r{t}", "review_sentences": s})
            for t, s in by_title.items()
        )
        + "\n"
    )
    data = tmp_path / "examples.jsonl"
    out = tmp_path / "report.md"
    assert main(["prepare", "--goodreads", str(raw), "--out", str(data)]) == 0
    assert (
        main(["baselines", "--data", str(data), "--out", str(out), "--target-recall", "0.8"]) == 0
    )
    text = out.read_text()
    assert "Phase 1 baselines" in text and "tfidf-logreg" in text
