# Gold set labelling guideline

The gold set is the test that counts: real sentences from Reddit, Letterboxd and YouTube, labelled by hand. It is never used for training. Use this guideline so that two people labelling the same sentence usually agree.

## The question

For each sentence, you see the **title** and the **previous sentence**, exactly as the model does. Ask:

> Would someone who has started this title, but not finished it, learn a specific plot event they would rather discover themselves?

- **Yes** → label `1` (spoiler).
- **No** → label `0` (not a spoiler).
- **Can't decide in 10 seconds** → pick your best guess and tick `unsure`. Unsure rows are still scored, and they tell us where the task is genuinely ambiguous.

## Label 1: spoiler

| Kind | Example |
| --- | --- |
| A character dies, survives, or is badly hurt | "Elias doesn't make it out of the storm." |
| A hidden identity or allegiance is revealed | "Mara was the informant all along." |
| A twist, or that a twist exists at a specific point | "Episode 6 flips everything you think about the captain." |
| How it ends, or who wins or loses | "They lose the lighthouse in the finale." |
| A plot-significant relationship outcome | "The wedding never happens." |
| A strong hint that implies a specific event | "Don't get attached to the brother." |

## Label 0: not a spoiler

| Kind | Example |
| --- | --- |
| Opinion on quality, acting, music or visuals | "The cello theme is gorgeous." |
| The premise, as given in the official synopsis or first episode | "It's about a crew stranded on an island." |
| Production facts: cast, dates, where it was filmed | "Season 3 drops in March." |
| Vague reactions with no event information | "That ending made me cry." |
| Questions that reveal nothing | "Is it worth finishing?" |
| Meta comments about spoilers | "Please use spoiler tags, some of us are behind." |

## Tie-breakers

1. **Judge only this sentence**, with the previous sentence as context. Don't use knowledge from later sentences in the comment.
2. **Specific beats vague.** "Someone dies" is 0 unless the context makes clear who. "Someone close to her dies in episode 8" is 1.
3. **Premise is fair game** only up to the end of the first episode, or the first act of a film.
4. **Sarcasm and jokes count** if a reader would still learn the event from them.
5. **Other titles.** If the sentence spoils a different work than the one given, label 1 and tick `unsure`.

## Collecting sentences

- Use public threads only, and never collect usernames.
- Store the sentence, the previous sentence, the title and a link to the thread. The published dataset will hold labels and links, not raw text.
- Aim for a mix: about half from discussion threads after new episodes, where spoilers are common, and half from general threads.

## Agreement check

About 200 sentences are labelled independently by a second person. We report Cohen's kappa between the two labellers. Below 0.6 means the guideline is unclear and needs revising before the remaining sentences are labelled.
