# docs — the story

This folder is the *why* of the repo, written for readers who may not be
machine-learning experts. The README gives the one-paragraph version; these
documents go deeper.

- **`learnings.md`** — the analysis in plain English (with analogies): how
  speculative decoding works, the three benchmark traps that nearly fooled me,
  the two conditions that must hold for the trick to pay off (and why both
  failed on this hardware), what would make it win, and a glossary. **Start here.**
- **`blockers.md`** — every bug I hit, in *symptom → root cause → fix → lesson*
  form, plus my debugging playbook. Proof that "simple" 300-line algorithms
  still need real verification.
- **`results.md`** — the complete 108-config benchmark table and how to read it
  (the raw JSON data lives in `../results/`).

`internal/` holds private development notes (planning, session state) — it is
gitignored and never part of the published story.
