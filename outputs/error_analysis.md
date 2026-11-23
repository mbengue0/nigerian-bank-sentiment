# Error analysis

Error rate = share of reviews in that group that the model got wrong. Types come from simple word rules (see scripts/07_error_analysis.py).

## test (10,000 reviews)

| Review type | Reviews | LR error rate | GRU error rate |
|---|---:|---:|---:|
| All reviews | 10,000 | 9.4% | 9.1% |
| negation | 3,735 | 10.3% | 9.4% |
| contrast (but/however...) | 1,468 | 14.6% | 12.8% |
| very short (1-3 words) | 769 | 9.0% | 10.1% |
| long, truncated (>100 tokens) | 107 | 14.0% | 11.2% |
| Pidgin markers | 109 | 7.3% | 9.2% |

- Both wrong: 671; only LR wrong: 271; only GRU wrong: 243; models disagree on 514 reviews.
- McNemar test, LR vs GRU (seed 42): chi-square = 1.42, p = 0.234 (p above 0.05: no evidence that one model makes fewer errors).
- Errors by true label: LR misses 410 negatives and 532 positives; GRU misses 418 negatives and 496 positives.

**GRU right, LR wrong** (random examples, 3-30 words; label 1 = positive):

- label 0, LR p=0.79, GRU p=0.39: "It's always had issues esp with instant money"
- label 0, LR p=0.51, GRU p=0.09: "Just curious. Why do we really need another Access Bank App? Whats wrong with the first App?"
- label 1, LR p=0.36, GRU p=0.72: "I can update apps"
- label 1, LR p=0.32, GRU p=0.73: "I need to reactivate my account"
- label 0, LR p=0.61, GRU p=0.25: "It's not a pleasant one. Network always acting up"
- label 1, LR p=0.36, GRU p=0.91: "Best service but so not happy about the updates that's need to be made every month."
- label 0, LR p=0.62, GRU p=0.22: "This app is too difficult to use"
- label 0, LR p=0.60, GRU p=0.31: "Not good when come to loan, not a good bank to borrow money from gtb"

**LR right, GRU wrong** (random examples, 3-30 words; label 1 = positive):

- label 1, LR p=0.73, GRU p=0.14: "You are trying in updating and programming your Mbile Application."
- label 1, LR p=0.65, GRU p=0.30: "You people should try and help me reactivate my Quick Bulk Account"
- label 1, LR p=0.88, GRU p=0.43: "It's not bad. But network becomes a problem at times."
- label 0, LR p=0.43, GRU p=0.81: "Good update. But my phone system security keeps detecting the app as a threat which I had to uninstall it."
- label 0, LR p=0.28, GRU p=0.67: "I don't like what i see"
- label 0, LR p=0.37, GRU p=0.50: "No service/ network most of the time."
- label 0, LR p=0.48, GRU p=0.61: "Am getting a message that I have reached the maximum of 10 devices.. Please fix this problem"
- label 0, LR p=0.35, GRU p=0.80: "Approximately 72MB for a bank app, Mennn give me a break!"

## Manual reading of 150 random test errors

Each review was read and put in ONE main category (labels in outputs/manual_error_labels.csv, first pass by Claude, AI assistant).

| Category | Both wrong | LR only | GRU only | Total | Share |
|---|---:|---:|---:|---:|---:|
| genuine model error | 20 | 16 | 11 | 47 | 31% |
| mixed opinion (praise and complaint) | 9 | 8 | 1 | 18 | 12% |
| no clear sentiment (question, request, name, noise) | 12 | 7 | 5 | 24 | 16% |
| text contradicts rating (label noise) | 46 | 8 | 7 | 61 | 41% |

## unseen_zenithBank (12,041 reviews)

| Review type | Reviews | LR error rate | GRU error rate |
|---|---:|---:|---:|
| All reviews | 12,041 | 11.3% | 11.1% |
| negation | 5,217 | 11.3% | 10.8% |
| contrast (but/however...) | 2,079 | 14.7% | 13.9% |
| very short (1-3 words) | 859 | 16.3% | 17.2% |
| long, truncated (>100 tokens) | 192 | 7.3% | 6.8% |
| Pidgin markers | 205 | 8.3% | 6.8% |

- Both wrong: 1,007; only LR wrong: 351; only GRU wrong: 330; models disagree on 681 reviews.
- McNemar test, LR vs GRU (seed 42): chi-square = 0.59, p = 0.443 (p above 0.05: no evidence that one model makes fewer errors).
- Errors by true label: LR misses 538 negatives and 820 positives; GRU misses 550 negatives and 787 positives.

