# Eval Dataset

Ground truth dataset for evaluating Deep Researcher's factual accuracy.
25 questions across 5 topic categories, each with a short, unambiguous gold answer.

## How this dataset was created

Questions were hand-authored to satisfy four constraints:

1. **One correct answer** — no opinion, no ambiguity, no "it depends".
2. **Verifiable via web search** — the system must be able to find and cite a source.
3. **Short gold answer** — 1–3 sentences, specific enough to score automatically.
4. **Stable over time** — answers don't change year to year (no "latest version of X").

## Categories

| Category | Count | Description |
|---|---|---|
| `ai_ml` | 5 | AI and machine learning facts — architectures, techniques, libraries |
| `software_engineering` | 5 | CS fundamentals — algorithms, principles, distributed systems |
| `science_facts` | 5 | Verifiable scientific facts — physics, chemistry, biology |
| `history_geography` | 5 | Historical events and geography — dates, capitals, inventors |
| `tech_companies` | 5 | Technology industry facts — languages, companies, tools |

## Format

The CSV has exactly three columns:

```
question,gold_answer,category
```

Fields that contain commas must be double-quoted. Fields that contain
literal double-quote characters must escape them as `""` (RFC 4180).

Example of a correctly formatted row:

```csv
What does SOLID stand for?,"SOLID stands for Single responsibility, Open-closed, Liskov substitution, Interface segregation, and Dependency inversion.",software_engineering
```

## How to add new questions

1. Add a row to `eval_questions.csv` following the format above.
2. Apply the same four constraints listed in "How this dataset was created".
3. Keep `gold_answer` to 1–3 sentences — long answers make automated scoring
   unreliable because semantic similarity is harder to threshold accurately.
4. Use one of the five existing category slugs, or introduce a new one and
   document it in the table above.
5. Validate the file parses correctly:
   ```bash
   python -c "import csv; rows=list(csv.DictReader(open('evals/dataset/eval_questions.csv'))); print(len(rows), 'rows OK')"
   ```

## Why gold answers are kept short

Automated evaluation compares the system's report against the gold answer using
semantic similarity (embeddings) or LLM-as-judge. Both methods work best when
the reference is concise and contains exactly the claim to be verified. A
multi-paragraph gold answer introduces noise — partial matches, paraphrase
variation, and boundary ambiguity all inflate score variance.

Short gold answers also make it easy for a human reviewer to spot-check
scoring disagreements in under a minute per question.

## Baseline integrity rule

> **Never modify gold answers after establishing your baseline.**
> Changing the target invalidates historical comparisons — a score improvement
> may reflect a better model or a softer target, and you cannot tell which.

If a gold answer is genuinely wrong, deprecate the question (remove the row)
and add a corrected replacement with a new question text. Record the change in
git history with a clear commit message explaining why.
