# Feedback API (Phase 5)

Opt-in endpoint that receives "spoiler" and "not a spoiler" ratings from the extension.
Planned as a Cloudflare Worker with a D1 database, with rate limits per install. It
stores only the sentence, the title and the rating, never URLs or usernames.
