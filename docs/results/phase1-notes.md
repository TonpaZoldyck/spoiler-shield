# Phase 1 baselines: what the numbers say

Measured 2026-10-01 on a 5% title-grouped sample of the Goodreads spoiler corpus:
801,306 sentences from about 70,000 reviews, split by book. Full tables are in
[phase1-baselines.md](phase1-baselines.md). These are **validation** numbers; the
test split is untouched.

**Bag-of-words can't do this task.** Both baselines fail the product bar.

| Model | PR-AUC | False-blur rate at 90% recall |
| --- | --- | --- |
| Chance (spoiler share) | 0.029 | 100% |
| Keyword rules | 0.033 | 100% |
| TF-IDF with calibrated logistic regression | 0.135 | 53% |

- **Spoilers are rare.** Only 3.1% of sentences are tagged as spoilers, so accuracy
  would be 97% for a model that never blurs anything. This is why the project reports
  PR-AUC and false-blur rate instead.
- **Generic spoiler words barely help.** The keyword cues ("dies", "twist", "turns
  out") fire on 9.5% of spoiler sentences and 3.9% of the rest. Most spoilers are
  specific to their story, as the Goodreads paper also found.
- **TF-IDF learns something, but not enough.** It is about 4.6 times better than
  chance on PR-AUC, yet catching 90% of spoilers means blurring half of all harmless
  sentences. Its probabilities are well calibrated (error 0.005), so the poor result
  is about ranking, not calibration.
- **Reviews without a known title are more spoiler-heavy:** 4.1% against 2.4%. Those
  are reviews of non-default editions, which skew toward popular, much-discussed
  books. Worth watching when the model uses title context.

**What this means for the design:** the gap between 0.135 and a usable model has to
come from context: the title, the previous sentence, and a model that understands
who did what. That is the job of the Phase 2 teacher. These numbers are the floor it
must clear.
