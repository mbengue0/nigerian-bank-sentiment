# English-filter manual check (22 Nov 2026)

Who checked: Claude (AI assistant), reading random samples. Not yet confirmed by the student.

Rule: keep a review if it has at least one Latin-letter word and at least 50% of its words
are among the 50,000 most frequent English words (wordfreq). See scripts/03_filter_sample_split.py.

## Why langid was not used as the filter
- langid labelled 11,121 candidates non-English. In a random sample of 25 longer ones, most were
  plainly English (emoji-heavy, capitalised, short, or written in styled Unicode letters).
- langid has no Yoruba, Hausa, Igbo or Nigerian Pidgin model, so it cannot recognise the most
  likely non-English content here.

## Sample of 50 reviews DROPPED for low English share (random_state=7)
- About 22/50 were English words, misspelled or stretched (e.g. "Awsome...", "Exxilent", "Gooddd",
  "Satisifactory"). Nearly all were one-word reviews.
- The rest were personal names, Hausa ("Yaki yayimin aiki"), Zulu/Xhosa, French ("parfait") or keyboard noise.
- Implication: of the 2,015 low-share drops, very roughly 40-45% may be misspelled English. That is
  about 0.5% of the pool; the filter is slightly biased against very short, misspelled reviews.

## Sample of 60 reviews KEPT in the final 100k (random_state=11)
- All read as English, including informal Nigerian English ("I love dis application is gud to use").
- Several show text-rating disagreement (e.g. "Not good forget the rating" with a positive label,
  "I love dis application is gud to use" with a negative label). Labels were NOT changed.

## Uncertainty
Two small samples cannot give a precise error rate. Pidgin-flavoured reviews that are mostly
English words are kept by design; reviews mostly in Pidgin or a Nigerian language are dropped.
