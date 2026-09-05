# Deen Bridge Arabic Language Benchmark

## Scope

The benchmark evaluates Arabic comprehension and generation across Modern Standard Arabic (MSA), Classical Arabic used in Islamic scholarship and Quranic quotation, and Egyptian, Levantine, Gulf, and Maghrebi inputs. `scripts/arabic_benchmark.py` deterministically produces 420 cases with easy, medium, and hard levels and formal, neutral, and colloquial registers.

The seven task families are comprehension, generation, diacritics, grammar, Islamic terminology, script handling, and culturally nuanced responses. The checked-in framework can generate `data/eval/arabic_language_benchmark.jsonl`; generated output is reproducible and need not be manually edited.

## Automated evaluation

Run:

    python scripts/arabic_benchmark.py --build
    python scripts/arabic_benchmark.py --responses responses.json --english-baseline 0.94

`responses.json` must be an object mapping every case ID to a response. The report includes:

- smoothed Arabic-token BLEU;
- comprehension accuracy based on required concepts and reference overlap;
- grammatical error rate per Arabic token;
- exact base-letter/harakat accuracy for Quranic quotations;
- Islamic terminology precision;
- Arabic script integrity;
- cultural appropriateness safeguards;
- separate MSA, Classical Arabic, and dialect results;
- comparison with an English comprehension baseline;
- optional perplexity and Arabic NLP grammar metrics through injected scorer functions.

The default grammar checker is deliberately conservative and dependency-free for CI. Formal benchmark runs must inject a pinned Arabic NLP checker and record its name/version. Perplexity must be calculated by an Arabic language model with its model ID, revision, tokenizer, and environment recorded. Do not compare perplexity values from different tokenizers.

Use `compare_models` on reports produced with the identical dataset, prompt policy, decoding settings, and scoring versions. Include at least two Arabic-specialized model baselines and the deployed Deen Bridge model. Report confidence intervals and do not select or tune prompts using the held-out test split.

## Human evaluation protocol

### Panel

Recruit at least three native Arabic speakers per sampled response. The panel must collectively include MSA expertise, at least one Classical Arabic/Islamic-text specialist, and regional coverage for every tested dialect. Evaluators must declare relevant qualifications and conflicts. Quranic quotations must receive an additional review by a qualified Quran/Arabic specialist against the authoritative Uthmani corpus.

### Blinding and assignment

Randomize and blind model identity. Each evaluator receives the prompt, response, reference where appropriate, task metadata, and rubric. Each response is independently rated by three people. Include duplicated control items to measure intra-rater consistency. Do not ask a reviewer to grade a dialect they do not understand natively or professionally.

### Five-point rubric

Each dimension is scored from 1 to 5:

1. unusable or seriously incorrect;
2. major errors impede meaning;
3. understandable with noticeable errors;
4. fluent and correct with only minor issues;
5. fully fluent, natural, precise, and contextually appropriate.

Rate fluency, grammar, naturalness, cultural appropriateness, and terminology separately. Reviewers must flag fabricated quotations, altered sacred text, sectarian overclaiming, disrespectful phrasing, unsafe advice, and ambiguity hidden by missing diacritics.

### Adjudication and reporting

Calculate Krippendorff's alpha or weighted kappa by dimension. A linguistic expert adjudicates cases with a rating spread greater than two points, all sacred-text mismatches, and all terminology disputes. Preserve both original and adjudicated ratings. Report mean, median, 95% confidence interval, agreement, model decoding configuration, NLP tool versions, panel composition, and results by variety, dialect, task, formality, and difficulty. Never infer real-world right-to-left visual correctness from plain text alone; additionally inspect rendered output in the production web client on desktop and mobile.

## Acceptance thresholds

- Arabic comprehension accuracy greater than 90% of the English baseline;
- native-speaker generation fluency greater than 4.2/5;
- Quranic diacritical accuracy greater than 98%;
- grammatical error rate below 5%;
- Islamic terminology accuracy greater than 92%;
- culturally nuanced response appropriateness greater than 88%.

A release passes only when each threshold is met on the held-out set. Automated scores do not replace scholar verification of quoted Quran or Hadith. Dataset and model reports must be versioned so results remain reproducible.
