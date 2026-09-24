<!--
DRAFT for you to rewrite in your own voice. Everything here is factual about
the 2023 notebook as reviewed on 2026-09-23. Sections marked TODO need numbers
that can only be measured once the IMDB data is re-downloaded.
-->

# I revisited my 2023 sentiment project and found the bugs I'd now catch in review

In 2023 I built a sentiment classifier for IMDB movie reviews. It pulled data from BigQuery, cleaned it with NLTK, and compared VADER, Random Forest, XGBoost and Naive Bayes. Naive Bayes won with 87% accuracy, and I was happy with it.

Three years later I reread it before starting a new project, and I'd send it back if it arrived as a pull request. None of the problems are exotic. They're the quiet mistakes that make results look better than they are, and that's exactly why they're worth writing about.

## 1. The ROC curves weren't ROC curves

I drew ROC curves from the models' 0/1 predictions instead of their probabilities. A ROC curve sweeps a threshold over scores. Hard labels have no scores to sweep, so each "curve" was two straight lines through one point, and the reported AUC was really just balanced accuracy.

**Fix:** always pass probabilities. My new metrics module only accepts scores, so the mistake can't happen again.

## 2. The test set picked the winner

My model-selection helper compared candidates by their accuracy on the test set. Whichever model scored highest on test data became "the best model", and I then reported its test accuracy. That number is the winner of a contest held on the test set, so it's biased upwards.

**Fix:** choose models and thresholds on a validation split, and touch the test set once, at the end.

## 3. The vectoriser saw the test data

I fitted TF-IDF on every review, then split into train and test. The vectoriser's vocabulary and document frequencies were learned partly from test reviews, a small but real leak.

**Fix:** fit the vectoriser inside a pipeline, on training rows only.

## 4. I tuned VADER on the data I scored it on

VADER's threshold was tuned on the full dataset and then scored on that same dataset. The jump from 67% to 71% accuracy was measured on data the threshold had already seen.

I also fed VADER text with stop words removed. The NLTK stop word list includes "not" and "no", so "not good" became "good". Stripping punctuation removed the exclamation marks VADER uses too. The baseline never had a fair chance.

## 5. I ignored the official split

The BigQuery table is the Stanford IMDB benchmark, which has its own train and test split. I re-split at random, which made my numbers impossible to compare with published results.

## 6. The winner lost to a simpler baseline

Naive Bayes at 87% sounds good, but logistic regression on TF-IDF with word pairs usually beats it on this benchmark. I never tried the obvious strong baseline.

TODO: rerun the notebook with the fixes above, on the official split, and add a before-and-after table here.

## What I'm doing differently now

I'm building [Spoiler Shield](https://github.com/TonpaZoldyck/spoiler-shield), a browser extension that blurs film and TV spoilers with a small model running on your device. Every rule above is built into its evaluation code:

- Splits are by title, so the model can't memorise character names.
- The test set only runs for tagged releases.
- Metrics take probabilities, and thresholds are chosen on validation data.
- A baseline ladder, from keyword rules to TF-IDF to transformers, runs on the same splits.

The first milestone is done. A MiniLM-sized model runs inside Chrome at about 7 ms per sentence, within budget. The next post covers the data.
