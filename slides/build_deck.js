// Final presentation, CS-4714. Style matches the proposal deck:
// white slides, Times New Roman, navy headings, green = positive, dark red = negative.
const pptxgen = require("pptxgenjs");
const { applyTheme } = require("/root/.claude/skills/synced/1c51b3e1-a5a2-442e-8604-86ec29f2df22_0c906b00-6e4e-4e65-83a2-1f5b4f89387c/pptx/scripts/apply_theme.js");

const THEME = {
  name: "CS4714 Capstone",
  headFontFace: "Times New Roman",
  bodyFontFace: "Times New Roman",
  colors: {
    dk1: "172B3B", lt1: "FFFFFF", dk2: "173B55", lt2: "EEF2F6",
    accent1: "173B55", accent2: "24704F", accent3: "9B3A35", accent4: "546572",
    accent5: "8FA6BF", accent6: "C9D3DE", hlink: "173B55", folHlink: "546572",
  },
};
const HEX = THEME.colors;
const OUT = "presentation-cs4714-Mouhamadou-Mbengue.pptx";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 in, same as the proposal deck
pres.author = "Mouhamadou Mbengue";
pres.title = "Sentiment Classification of Nigerian Banking App Reviews";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

// ------------------------------------------------------------------ layouts
pres.defineSlideMaster({
  title: "TITLE",
  background: { color: "FFFFFF" },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.9, y: 1.7, w: 11.5, h: 1.7,
      fontSize: 40, bold: true, color: C.text2, valign: "bottom", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.9, y: 3.6, w: 11.5, h: 2.6,
      fontSize: 22, color: C.text1, valign: "top", align: "left", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: "FFFFFF" },
  slideNumber: { x: 12.2, y: 6.95, w: 0.6, h: 0.3, fontSize: 12, color: C.accent4, align: "right" },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.75, y: 0.45, w: 11.83, h: 0.8,
      fontSize: 34, bold: true, color: C.text2, valign: "middle", align: "left", margin: 0 }, text: "" } },
  ],
});

// ------------------------------------------------------------------ helpers
let n = 0;
const name = (s) => `${s}-${++n}`;
function text(slide, t, o) {
  slide.addText(t, Object.assign({ isTextBox: true, margin: 0, color: C.text1, fontSize: 18, valign: "top",
    objectName: name("text") }, o));
}
function card(slide, x, y, w, h, fill) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08,
    fill: { color: fill || C.background2 }, line: { color: fill || C.background2 }, objectName: name("card") });
}
function stat(slide, x, y, w, big, label, color, bigSize) {
  text(slide, big, { x, y, w, h: 0.9, fontSize: bigSize || 44, bold: true, color: color || C.text2, valign: "bottom" });
  text(slide, label, { x, y: y + 0.95, w, h: 0.75, fontSize: 16, color: C.accent4 });
}
const chartText = { catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt",
  legendFontFace: "+mn-lt", titleFontFace: "+mn-lt" };
function content(section, title) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  return s;
}

// ================================================================== 1. Title
pres.addSection({ title: "Introduction" });
let s = pres.addSlide({ masterName: "TITLE", sectionTitle: "Introduction" });
s.addText("Sentiment Classification of Nigerian Banking App Reviews", { placeholder: "title" });
s.addText([
  { text: "Comparing Logistic Regression and GRU Models", options: { breakLine: true } },
  { text: " ", options: { breakLine: true, fontSize: 12 } },
  { text: "CS-4714 Natural Language Processing  ·  Final project presentation", options: { fontSize: 16, color: C.accent4, breakLine: true } },
  { text: "Mouhamadou Mbengue  ·  December 2026", options: { fontSize: 16, color: C.accent4 } },
], { placeholder: "body" });
s.addNotes("Good morning. My project classifies Google Play reviews of Nigerian banking apps as positive or negative, and compares a classic model, TF-IDF with logistic regression, against a neural model, a GRU. I will cover the data, the two models, the results on quality and cost, an extra test on a bank the models never saw, and what the errors tell us.");

// ================================================================== 2. Problem
s = content("Introduction", "Problem and research question");
text(s, "Banking app reviews describe transfer failures, login problems and good service, written informally and in large numbers.",
  { x: 0.75, y: 1.5, w: 5.6, h: 1.1, fontSize: 18 });
card(s, 0.75, 2.85, 5.6, 1.15);
text(s, [{ text: "“great bank, I love your services”", options: { italic: true, breakLine: true } },
  { text: "5 stars  →  positive", options: { fontSize: 14, color: C.accent4 } }],
  { x: 1.0, y: 3.0, w: 5.1, h: 0.9, fontSize: 20, color: C.accent2 });
card(s, 0.75, 4.2, 5.6, 1.15);
text(s, [{ text: "“Not too good after the upgrade”", options: { italic: true, breakLine: true } },
  { text: "1 star  →  negative", options: { fontSize: 14, color: C.accent4 } }],
  { x: 1.0, y: 4.35, w: 5.1, h: 0.9, fontSize: 20, color: C.accent3 });
text(s, "Real reviews from the dataset", { x: 0.75, y: 5.5, w: 5.6, h: 0.35, fontSize: 12, color: C.accent4 });
text(s, "Research question", { x: 7.0, y: 1.5, w: 5.6, h: 0.45, fontSize: 22, bold: true, color: C.text2 });
text(s, "How do TF-IDF + logistic regression and a GRU compare in classification quality and computational cost?",
  { x: 7.0, y: 2.0, w: 5.6, h: 1.4, fontSize: 20 });
text(s, "Hypothesis", { x: 7.0, y: 3.75, w: 5.6, h: 0.45, fontSize: 22, bold: true, color: C.text2 });
text(s, "The GRU reads words in order, so it might handle negation and “but” better. Logistic regression may be just as good at much lower cost.",
  { x: 7.0, y: 4.25, w: 5.6, h: 1.6, fontSize: 18 });
s.addNotes("Banks receive thousands of app reviews. Reading them by hand does not scale, so we want to classify them automatically. The question is whether a sequence model like a GRU, which reads words in order, is worth its extra cost compared with a simple linear model on word features. My hypothesis was that the GRU might help with negation and contrast, but logistic regression might be competitive and much cheaper.");

// ================================================================== 3. Data
pres.addSection({ title: "Data" });
s = content("Data", "Data: Google Play reviews of 16 Nigerian banks");
stat(s, 0.75, 1.45, 3.6, "334,415", "public reviews in the dataset");
stat(s, 4.85, 1.45, 3.6, "16", "banks’ mobile apps (UBA, Access, Zenith, GTBank…)");
stat(s, 8.95, 1.45, 3.6, "2012–2024", "review years; most from 2019–2023");
card(s, 0.75, 3.75, 3.6, 1.5); card(s, 4.85, 3.75, 3.6, 1.5); card(s, 8.95, 3.75, 3.6, 1.5);
text(s, [{ text: "1–2 stars", options: { breakLine: true } }, { text: "Negative (0)", options: { fontSize: 18 } }],
  { x: 1.0, y: 3.95, w: 3.1, h: 1.1, fontSize: 26, bold: true, color: C.accent3 });
text(s, [{ text: "3 stars", options: { breakLine: true } }, { text: "Excluded", options: { fontSize: 18 } }],
  { x: 5.1, y: 3.95, w: 3.1, h: 1.1, fontSize: 26, bold: true, color: C.accent4 });
text(s, [{ text: "4–5 stars", options: { breakLine: true } }, { text: "Positive (1)", options: { fontSize: 18 } }],
  { x: 9.2, y: 3.95, w: 3.1, h: 1.1, fontSize: 26, bold: true, color: C.accent2 });
text(s, "Labels come from stars, not from people reading the text. The models see only the review text.",
  { x: 0.75, y: 5.55, w: 11.8, h: 0.45, fontSize: 18, bold: true, color: C.text2 });
text(s, "Source: Federal University Lokoja, “Nigerian Banks: Bank Reviews Dataset Collection”, Hugging Face (Apache-2.0), revision 27d2b6e",
  { x: 0.75, y: 6.85, w: 11.0, h: 0.35, fontSize: 11, color: C.accent4 });
s.addNotes("The data is a public Hugging Face dataset from Federal University Lokoja: about 334 thousand Google Play reviews of the mobile apps of 16 Nigerian banks, one file per bank. Each review has text, a star rating and a year. I turned the stars into labels: one or two stars negative, four or five positive, three stars excluded because they are neutral. Important: these labels are weak. Nobody checked that the text matches the stars. The model only ever sees the text.");

// ================================================================== 4. Cleaning
s = content("Data", "Cleaning: from 334,415 to 100,000 reviews");
s.addChart(pres.charts.BAR, [{
  name: "Reviews remaining",
  labels: ["Raw reviews", "Non-empty, no 3 stars", "Each text kept once", "Conflicting labels removed",
    "English filter", "Near-copies removed", "Zenith Bank held out", "Seeded sample"],
  values: [334415, 312104, 197676, 196317, 193793, 186902, 173749, 100000],
}], Object.assign({
  x: 0.6, y: 1.4, w: 7.4, h: 5.3, barDir: "bar", catAxisOrientation: "maxMin",
  chartColors: [HEX.accent1], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "#,##0",
  dataLabelFontSize: 13, dataLabelColor: HEX.dk1, catAxisLabelFontSize: 14, catAxisLabelColor: HEX.dk1,
  valAxisHidden: true, valGridLine: { style: "none" }, catGridLine: { style: "none" }, showLegend: false,
  valAxisMaxVal: 400000, barGapWidthPct: 45, objectName: "cleaning-chart",
}, chartText));
const notes4 = [
  ["Repeats", "“Great app” appeared hundreds of times; each text is kept once."],
  ["Conflicts", "Texts rated both high and low (e.g. “good”) have unclear labels and were removed."],
  ["English", "Keep a review if at least half its words are common English words."],
  ["Near-copies", "“Great app!!” and “great app” count as one text."],
];
notes4.forEach(([h, b], i) => {
  text(s, h, { x: 8.4, y: 1.55 + i * 1.3, w: 4.2, h: 0.4, fontSize: 18, bold: true, color: C.text2 });
  text(s, b, { x: 8.4, y: 1.95 + i * 1.3, w: 4.2, h: 0.85, fontSize: 15 });
});
s.addNotes("Here is how the raw data became the modelling data. I removed empty reviews and three-star reviews, kept each repeated text only once, and removed texts that got both high and low ratings, because their label is unclear. Then an English filter: a review is kept if at least half its words are common English words. I tested a language-detection tool first but it rejected many English reviews with emoji or capitals. I also merged near-copies like 'Great app!!' and 'great app'. Then I set aside one bank, Zenith, for the extension, and drew a random sample of 100 thousand reviews.");

// ================================================================== 5. Fair comparison
s = content("Data", "A fair comparison: same data, two test sets");
const boxes = [
  ["Train", "80,000", "models learn from these", C.accent1, 0.75],
  ["Validation", "10,000", "choose settings, stop training", C.accent5, 3.85],
  ["Test", "10,000", "final exam, 15 known banks", C.accent4, 6.95],
];
boxes.forEach(([h, v, d, col, x]) => {
  card(s, x, 1.5, 2.85, 1.9, col);
  text(s, h, { x: x + 0.25, y: 1.65, w: 2.4, h: 0.4, fontSize: 18, bold: true, color: C.background1 });
  text(s, v, { x: x + 0.25, y: 2.05, w: 2.4, h: 0.65, fontSize: 32, bold: true, color: C.background1 });
  text(s, d, { x: x + 0.25, y: 2.75, w: 2.4, h: 0.55, fontSize: 14, color: C.background1 });
});
card(s, 10.05, 1.5, 2.55, 1.9, C.accent2);
text(s, "Unseen bank", { x: 10.3, y: 1.65, w: 2.1, h: 0.4, fontSize: 18, bold: true, color: C.background1 });
text(s, "12,041", { x: 10.3, y: 2.05, w: 2.1, h: 0.65, fontSize: 32, bold: true, color: C.background1 });
text(s, "Zenith Bank, never trained on", { x: 10.3, y: 2.75, w: 2.1, h: 0.55, fontSize: 14, color: C.background1 });
text(s, "100,000 reviews from 15 banks", { x: 0.75, y: 3.5, w: 9.05, h: 0.35, fontSize: 14, color: C.accent4, align: "center" });
text(s, "Extension", { x: 10.05, y: 3.5, w: 2.55, h: 0.35, fontSize: 14, color: C.accent4, align: "center" });
const rules = [
  "Split 80 / 10 / 10, keeping each bank’s share and the positive/negative balance (seed 42)",
  "No review or near-copy appears in two sets, and no Zenith text in training (checked in code)",
  "TF-IDF statistics and the GRU vocabulary are learned from the training set only",
  "Both test sets were used once, after the models were frozen",
];
text(s, rules.map((r, i) => ({ text: r, options: { bullet: true, breakLine: i < rules.length - 1, paraSpaceAfter: 8 } })),
  { x: 0.75, y: 4.25, w: 11.8, h: 2.4, fontSize: 18 });
s.addNotes("Both models use exactly the same saved splits: 80 thousand reviews to train, 10 thousand to choose settings, and 10 thousand as the final exam. The test set was used only once, at the very end. As an extension, the reviews of Zenith Bank were removed before anything was trained. They form a second test: can the models handle a bank they have never seen? I also made sure no review, or a near-copy of it, appears in two sets, otherwise the scores would be inflated.");

// ================================================================== 6. Models
pres.addSection({ title: "Models" });
s = content("Models", "Two models for the same task");
card(s, 0.75, 1.5, 5.75, 4.6); card(s, 6.85, 1.5, 5.75, 4.6);
text(s, "Traditional: TF-IDF + logistic regression", { x: 1.05, y: 1.7, w: 5.2, h: 0.5, fontSize: 20, bold: true, color: C.text2 });
const lrItems = [
  "Each review becomes a vector of word and word-pair weights (TF-IDF)",
  "Logistic regression learns one weight per feature",
  "92,416 features: unigrams + bigrams, e.g. “not bad”",
  "Setting chosen from 6 options on validation (C = 4)",
];
text(s, lrItems.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < lrItems.length - 1, paraSpaceAfter: 10 } })),
  { x: 1.05, y: 2.4, w: 5.2, h: 3.4, fontSize: 17 });
text(s, "Neural: embeddings + GRU", { x: 7.15, y: 1.7, w: 5.2, h: 0.5, fontSize: 20, bold: true, color: C.text2 });
const gruItems = [
  "Words → embeddings (100 dimensions, learned from scratch)",
  "One GRU layer (128 units) reads the review in order",
  "Its last state → probability of positive",
  "1,140,049 parameters; Adam, early stopping; 3 runs (seeds 42, 1, 2)",
];
text(s, gruItems.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < gruItems.length - 1, paraSpaceAfter: 10 } })),
  { x: 7.15, y: 2.4, w: 5.2, h: 3.4, fontSize: 17 });
text(s, "Reference: majority class (always predicts the most common label)", { x: 0.75, y: 6.35, w: 11.8, h: 0.4, fontSize: 15, color: C.accent4 });
s.addNotes("The traditional model turns each review into TF-IDF weights for words and word pairs. TF-IDF gives high weight to words that are frequent in a review but rare overall. Logistic regression then learns one weight per feature: positive weights push toward positive. I tried six settings and picked the best on validation. The neural model maps each word to a learned vector, then a GRU reads the vectors in order and keeps a running memory. Its final state goes to a classifier. I trained it three times with different random seeds to see how much results vary. The majority-class model is a sanity check.");

// ================================================================== 7. Quality
pres.addSection({ title: "Results" });
s = content("Results", "Result 1: classification quality is a tie");
s.addChart(pres.charts.BAR, [{ name: "Macro-F1", labels: ["Majority class", "TF-IDF + LR", "GRU (mean of 3)"], values: [0.338, 0.906, 0.908] }],
  Object.assign({
    x: 0.6, y: 1.4, w: 6.8, h: 5.2, barDir: "col", chartColors: [HEX.accent6, HEX.accent4, HEX.accent1],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", dataLabelFontSize: 16, dataLabelColor: HEX.dk1,
    catAxisLabelFontSize: 15, catAxisLabelColor: HEX.dk1, valAxisLabelFontSize: 12, valAxisLabelColor: HEX.accent4,
    valAxisMinVal: 0, valAxisMaxVal: 1, valAxisMajorUnit: 0.2, valGridLine: { color: "E3E8EE", size: 0.5 },
    catGridLine: { style: "none" }, showLegend: false, showTitle: true, title: "Macro-F1, main test set (10,000 reviews)",
    titleFontSize: 15, titleColor: HEX.dk2, barGapWidthPct: 60, objectName: "quality-chart",
  }, chartText));
stat(s, 8.0, 1.5, 4.6, "+0.002", "GRU advantage, about the same as its own run-to-run spread (± 0.002)");
stat(s, 8.0, 3.5, 4.6, "p = 0.23", "McNemar test: no evidence that either model makes fewer errors");
text(s, "Both score evenly on positive and negative reviews (F1 0.905–0.910).", { x: 8.0, y: 5.6, w: 4.6, h: 0.8, fontSize: 16 });
s.addNotes("This is the main result. Macro-F1 is the average of the F1 scores of the two classes, so both classes count equally. The majority class gets 0.34. Logistic regression gets 0.906 and the GRU 0.908 on average. The GRU is ahead by 0.002, but its three runs differ from each other by about the same amount. McNemar's test looks only at reviews where one model is right and the other wrong: 271 versus 243. The p-value is 0.23, so the difference could easily be chance.");

// ================================================================== 8. Cost
s = content("Results", "Result 2: the GRU costs far more");
const cost = [
  ["Training time", "2.6 s", "300 s", "about 117× longer"],
  ["Predicting 10,000 reviews", "0.18 s", "1.11 s", "about 6× longer"],
  ["Peak memory", "351 MB", "~960 MB", "about 2.7× more"],
];
cost.forEach(([h, lr, gru, ratio], i) => {
  const x = 0.75 + i * 4.05;
  card(s, x, 1.5, 3.75, 4.3);
  text(s, h, { x: x + 0.3, y: 1.7, w: 3.2, h: 0.45, fontSize: 19, bold: true, color: C.text2 });
  text(s, "TF-IDF + LR", { x: x + 0.3, y: 2.35, w: 3.2, h: 0.35, fontSize: 14, color: C.accent4 });
  text(s, lr, { x: x + 0.3, y: 2.7, w: 3.2, h: 0.75, fontSize: 38, bold: true, color: C.accent4 });
  text(s, "GRU", { x: x + 0.3, y: 3.6, w: 3.2, h: 0.35, fontSize: 14, color: C.accent4 });
  text(s, gru, { x: x + 0.3, y: 3.95, w: 3.2, h: 0.75, fontSize: 38, bold: true, color: C.text2 });
  text(s, ratio, { x: x + 0.3, y: 4.95, w: 3.2, h: 0.45, fontSize: 18, italic: true, color: C.accent3 });
});
text(s, "Same 2-core CPU, no GPU. GRU training includes a validation check after each epoch; a GPU would shrink its training time.",
  { x: 0.75, y: 6.1, w: 11.8, h: 0.6, fontSize: 15, color: C.accent4 });
s.addNotes("For the same quality, the costs are very different. On the same two-core CPU, logistic regression trains in under 3 seconds; the GRU takes about 5 minutes per run. Predicting 10 thousand reviews takes 0.18 seconds versus about 1.1 seconds, and the GRU uses almost three times the memory. A GPU would make the GRU faster, so these ratios are for CPU training only.");

// ================================================================== 9. Unseen bank
s = content("Results", "Extension: a bank the models never saw");
s.addChart(pres.charts.BAR, [
  { name: "TF-IDF + LR", labels: ["Main test (15 known banks)", "Unseen Zenith Bank"], values: [0.906, 0.883] },
  { name: "GRU (mean of 3)", labels: ["Main test (15 known banks)", "Unseen Zenith Bank"], values: [0.908, 0.884] },
], Object.assign({
  x: 0.6, y: 1.4, w: 6.8, h: 5.2, barDir: "col", barGrouping: "clustered", chartColors: [HEX.accent4, HEX.accent1],
  showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", dataLabelFontSize: 15, dataLabelColor: HEX.dk1,
  catAxisLabelFontSize: 15, catAxisLabelColor: HEX.dk1, valAxisLabelFontSize: 12, valAxisLabelColor: HEX.accent4,
  valAxisMinVal: 0.8, valAxisMaxVal: 0.95, valAxisMajorUnit: 0.05, valGridLine: { color: "E3E8EE", size: 0.5 },
  catGridLine: { style: "none" }, showLegend: true, legendPos: "b", legendFontSize: 14, legendColor: HEX.dk1,
  showTitle: true, title: "Macro-F1 (axis starts at 0.80)", titleFontSize: 15, titleColor: HEX.dk2,
  barGapWidthPct: 70, objectName: "unseen-chart",
}, chartText));
const unseen = [
  ["Both still work", "Macro-F1 drops by only about 0.02 on 12,041 Zenith reviews."],
  ["Positives get harder", "Positive recall falls from about 0.90 to 0.84; negatives are unchanged."],
  ["Still a tie", "McNemar p = 0.44."],
  ["One bank only", "It shows transfer to one new bank, not to every bank."],
];
unseen.forEach(([h, b], i) => {
  text(s, h, { x: 8.0, y: 1.5 + i * 1.25, w: 4.6, h: 0.4, fontSize: 19, bold: true, color: C.text2 });
  text(s, b, { x: 8.0, y: 1.9 + i * 1.25, w: 4.6, h: 0.75, fontSize: 16 });
});
s.addNotes("For the extension, the models were trained without any Zenith Bank reviews, then tested on all 12 thousand of them. Both lose about 0.02 macro-F1, so they mostly learned general sentiment language, not bank-specific words. The drop comes from positive reviews, which become harder to recognise. The two models are still tied. Note the chart's axis starts at 0.8 to make the small drop visible. One held-out bank is evidence for one bank, not proof for all banks.");

// ================================================================== 10. Error types
pres.addSection({ title: "Error analysis" });
s = content("Error analysis", "Where the models struggle");
s.addChart(pres.charts.BAR, [
  { name: "TF-IDF + LR", labels: ["All reviews", "Negation (not, never…)", "Contrast (but, however…)", "Very short (1–3 words)"], values: [9.4, 10.3, 14.6, 9.0] },
  { name: "GRU", labels: ["All reviews", "Negation (not, never…)", "Contrast (but, however…)", "Very short (1–3 words)"], values: [9.1, 9.4, 12.8, 10.1] },
], Object.assign({
  x: 0.6, y: 1.4, w: 7.6, h: 5.2, barDir: "col", barGrouping: "clustered", chartColors: [HEX.accent4, HEX.accent1],
  showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0\"%\"", dataLabelFontSize: 13, dataLabelColor: HEX.dk1,
  catAxisLabelFontSize: 13, catAxisLabelColor: HEX.dk1, valAxisLabelFontSize: 12, valAxisLabelColor: HEX.accent4,
  valAxisMinVal: 0, valAxisMaxVal: 16, valAxisMajorUnit: 4, valAxisLabelFormatCode: "0\"%\"",
  valGridLine: { color: "E3E8EE", size: 0.5 }, catGridLine: { style: "none" }, showLegend: true, legendPos: "b",
  legendFontSize: 14, legendColor: HEX.dk1, showTitle: true, title: "Error rate on the main test set",
  titleFontSize: 15, titleColor: HEX.dk2, barGapWidthPct: 60, objectName: "errors-chart",
}, chartText));
const errs = [
  ["“But” is hardest", "Reviews with contrast words have the highest error rate for both models."],
  ["GRU slightly better", "on contrast and negation, as hypothesised, but the gaps are small."],
  ["LR slightly better", "on very short reviews."],
];
errs.forEach(([h, b], i) => {
  text(s, h, { x: 8.6, y: 1.6 + i * 1.45, w: 4.0, h: 0.4, fontSize: 19, bold: true, color: C.text2 });
  text(s, b, { x: 8.6, y: 2.0 + i * 1.45, w: 4.0, h: 0.9, fontSize: 16 });
});
text(s, "LR learned “not bad” as a positive word pair.", { x: 8.6, y: 6.0, w: 4.0, h: 0.6, fontSize: 15, italic: true, color: C.accent4 });
s.addNotes("I grouped test reviews by simple word rules. Reviews with 'but' or 'however' are the hardest for both models: about 13 to 15 percent wrong versus 9 percent overall. The GRU does a bit better on contrast and negation, which matches the hypothesis, while logistic regression does a bit better on very short reviews. These differences are small and not statistically significant overall, so I treat them as hints. Interesting detail: logistic regression partly handles negation through word pairs, 'not bad' is one of its strongest positive features.");

// ================================================================== 11. Error reading
s = content("Error analysis", "Many “errors” are really label problems");
const cats = [
  ["41%", "Text contradicts the stars", "“This app is useless…” with 5 stars", C.accent3],
  ["31%", "Genuine model error", "Text and stars agree, model wrong", C.text2],
  ["16%", "No clear sentiment", "“How do I register for the mobile?”", C.accent4],
  ["12%", "Mixed opinion", "“Great app but … Fix this!”", C.accent4],
];
cats.forEach(([pct, h, ex, col], i) => {
  const x = 0.75 + i * 3.05;
  card(s, x, 1.5, 2.8, 3.9);
  text(s, pct, { x: x + 0.25, y: 1.7, w: 2.3, h: 1.0, fontSize: 54, bold: true, color: col, valign: "bottom" });
  text(s, h, { x: x + 0.25, y: 2.85, w: 2.3, h: 0.8, fontSize: 18, bold: true, color: C.text2 });
  text(s, ex, { x: x + 0.25, y: 3.75, w: 2.3, h: 1.4, fontSize: 15, italic: true });
});
text(s, "150 random test errors read one by one. About 57% are not real model mistakes: noisy star labels limit the score that any model can reach.",
  { x: 0.75, y: 5.7, w: 11.8, h: 0.75, fontSize: 18, bold: true, color: C.text2 });
text(s, "First-pass categories assigned with AI assistance (see disclosure).", { x: 0.75, y: 6.5, w: 11.0, h: 0.35, fontSize: 12, color: C.accent4 });
s.addNotes("To understand the errors, I read 150 random mistakes. The biggest group, 41 percent, are reviews whose text contradicts the stars, for example 'This app is useless' with five stars. The model is actually reading the text correctly; the label is wrong. Another 16 percent have no sentiment at all, like questions. Only about a third are clear model mistakes. So noisy labels are a major reason scores stop around 91 percent.");

// ================================================================== 12. Conclusion
pres.addSection({ title: "Conclusion" });
s = content("Conclusion", "Conclusion");
const concl = [
  ["Same quality", "Macro-F1 about 0.91 on the main test and 0.88 on an unseen bank, with no significant difference.", C.text2],
  ["Very different cost", "Logistic regression trains about 117× faster on a CPU: the practical choice for this task.", C.accent2],
  ["Labels set the limit", "Many remaining errors come from stars that contradict the text, not from the models.", C.accent3],
];
concl.forEach(([h, b, col], i) => {
  const y = 1.55 + i * 1.6;
  s.addShape(pres.shapes.OVAL, { x: 0.75, y: y + 0.05, w: 0.7, h: 0.7, fill: { color: col }, line: { color: col }, objectName: name("num") });
  text(s, String(i + 1), { x: 0.75, y: y + 0.05, w: 0.7, h: 0.7, fontSize: 24, bold: true, color: C.background1, align: "center", valign: "middle" });
  text(s, h, { x: 1.75, y: y, w: 10.8, h: 0.5, fontSize: 24, bold: true, color: C.text2 });
  text(s, b, { x: 1.75, y: y + 0.55, w: 10.8, h: 0.75, fontSize: 18 });
});
text(s, "Why no GRU advantage? Reviews are short (median 12 words), sentiment sits in a few strong words, and the GRU learns from scratch on 80,000 reviews.",
  { x: 0.75, y: 6.25, w: 11.8, h: 0.6, fontSize: 15, color: C.accent4 });
s.addNotes("To answer the research question: the two pipelines reach the same quality, about 0.91, and both transfer reasonably to a new bank. But logistic regression is more than a hundred times cheaper to train, so for this task it is the better practical choice. And the main limit is the labels, not the models. Why didn't the GRU win? Reviews are short, sentiment is carried by a few strong words that bigrams already capture, and the GRU has no pretrained knowledge.");

// ================================================================== 13. Limitations
s = content("Conclusion", "Limitations and next steps");
text(s, "Limitations", { x: 0.75, y: 1.5, w: 5.6, h: 0.5, fontSize: 22, bold: true, color: C.text2 });
const lims = [
  "Labels come from stars and are noisy",
  "Removing conflicting texts dropped common phrases like “good”",
  "English filter checked on small samples only",
  "Unseen-bank result rests on one bank",
  "Small tuning budget; CPU-only timings",
];
text(s, lims.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < lims.length - 1, paraSpaceAfter: 10 } })),
  { x: 0.75, y: 2.1, w: 5.6, h: 3.8, fontSize: 18 });
card(s, 6.85, 1.5, 5.75, 4.4);
text(s, "Next steps", { x: 7.15, y: 1.7, w: 5.2, h: 0.5, fontSize: 22, bold: true, color: C.accent2 });
const next = [
  "Check labels by hand for a test sample",
  "Try pretrained embeddings or a small pretrained Transformer",
  "Hold out more banks, one at a time",
  "Test on newer reviews (2024 onward)",
];
text(s, next.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < next.length - 1, paraSpaceAfter: 10 } })),
  { x: 7.15, y: 2.3, w: 5.2, h: 3.4, fontSize: 18 });
s.addNotes("The main limitations: star labels are noisy; removing conflicting texts dropped some very common phrases; the English filter was only checked on samples; the unseen-bank test uses one bank; and timings are CPU-only. Next steps would be to check a test sample by hand, try pretrained models, and hold out more banks.");

// ================================================================== 14. Disclosure + references
s = content("Conclusion", "AI disclosure and references");
text(s, "AI assistance", { x: 0.75, y: 1.5, w: 5.6, h: 0.45, fontSize: 20, bold: true, color: C.text2 });
const ai = [
  "ChatGPT: project scoping, proposal and proposal slides",
  "Claude (Anthropic): wrote and ran the data, training and evaluation code; first-pass error categories; drafted the report and these slides",
  "Scope and extension decisions by the author; all numbers from code run on the real data",
];
text(s, ai.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < ai.length - 1, paraSpaceAfter: 8 } })),
  { x: 0.75, y: 2.0, w: 5.6, h: 3.3, fontSize: 15 });
text(s, [{ text: "Code and data:", options: { breakLine: true } }, { text: "github.com/mbengue0/nigerian-bank-sentiment" }], { x: 0.75, y: 5.4, w: 5.6, h: 0.75, fontSize: 15, bold: true, color: C.text2 });
text(s, "References", { x: 6.85, y: 1.5, w: 5.75, h: 0.45, fontSize: 20, bold: true, color: C.text2 });
const refs = [
  "Federal University Lokoja. Nigerian Banks: Bank Reviews Dataset Collection. Hugging Face, rev. 27d2b6e.",
  "Cho et al. (2014). Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation. EMNLP.",
  "Jurafsky & Martin. Speech and Language Processing, 3rd ed. draft.",
  "McNemar (1947). Psychometrika 12(2).",
  "Pedregosa et al. (2011). Scikit-learn. JMLR 12.  Paszke et al. (2019). PyTorch. NeurIPS.",
];
text(s, refs.map((t, i) => ({ text: t, options: { breakLine: i < refs.length - 1, paraSpaceAfter: 8 } })),
  { x: 6.85, y: 2.0, w: 5.75, h: 4.3, fontSize: 13 });
s.addNotes("Finally, as required, the AI disclosure: ChatGPT helped with the proposal stage, and Claude helped write and run the code and draft the report and slides. All decisions on scope were mine, and all numbers come from code run on the real data. The code and data are on GitHub. Thank you, I am happy to take questions.");

// ------------------------------------------------------------------ write
(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
