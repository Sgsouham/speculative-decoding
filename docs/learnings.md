# Learnings & Analysis — why speculative decoding lost, in plain English

*The story of this repo is a measurement. I set out to make text generation faster with
a clever, well-known trick. I built it correctly, measured it honestly, and it didn't
work on my hardware. This document explains what I tried, the traps that nearly fooled
me, and exactly why the trick failed — without assuming you know the internals of
language models.*

---

## The idea, explained simply

A language model generates text one word (token) at a time. Each word takes a small
amount of "thinking time" — a forward pass. To generate 100 words, it thinks 100 times.

**Speculative decoding** is a trick to think fewer times:

- You hire a small, fast **draft** model to *guess* the next few words.
- The big, slow **target** model (the one you actually want) just *checks* the guesses
  instead of thinking from scratch.
- If the target agrees with a guess, you keep it — you got a word for the price of a
  check, not a full thought.

**The kitchen analogy:** the draft is an apprentice chopping vegetables ahead of the
head chef. The chef (target) just glances at each chopped vegetable instead of chopping
it herself. If the apprentice guesses the chef's needs correctly most of the time, the
kitchen produces dishes much faster — for free, with the exact same recipes.

The published papers promise **2–3× faster** with *identical* output. The recipe looks
like an afternoon project. That's the promise I started with.

---

## The three traps that nearly fooled me

Before I could trust any number, I had to fight three ways a benchmark can quietly lie.

### Trap 1 — The flattering practice test

My first benchmarks fed the model a **repeated paragraph** (the same text over and
over). The model had effectively memorized what comes next — like practicing one speech
until you can recite it in your sleep. On that setup the draft looked like a genius
(80–94% of its guesses accepted), and I even measured a **1.01× "win"** at the largest
draft length.

When I switched to **real, varied text** (history, science, a story, an algorithm
problem), acceptance collapsed to ~35% — the draft guessed what the target would say
only about 1 time in 3 — and the "win" became a loss.

> **Lesson:** a benchmark prompt must be text the model hasn't effectively memorized.
> Repeated paragraphs flatter the draft. Real text is the honest exam.

### Trap 2 — The padding trap

To test different prompt lengths (32 / 128 / 512 words), I filled long prompts by
**repeating** a paragraph to reach the length. That quietly re-introduced the flattery:
acceptance jumped back toward 100% at length 512. The fix was to fill long prompts by
**concatenating different** texts, never repeating one.

> **Lesson:** even "realistic" prompts can be secretly repetitive. Fill by joining
> different texts; never pad by repetition.

### Trap 3 — The coin-flip tie

Once in a while, the model considers **two next words perfectly tied**. When two words
score exactly the same, which one you record depends on the last tiny rounding error in
the math — like a coin flip that lands differently depending on who tosses it. My two
benchmarking paths occasionally recorded different sides of the same coin. I verified
this was a rounding artifact (not a bug) against an independent reference implementation
and documented the affected rows.

> **Lesson:** exact ties exist in floating-point math. Check the margin before calling a
> discrepancy a bug.

---

## The real reason it lost: two broken bets

The trick only pays off if **two bets** both win. I measured both. Both lost.

### Bet 1 — "The draft is much cheaper." ❌ Lost.

The draft model is **3× smaller** than the target — surely each of its thoughts is ~3×
cheaper? No. **It was only ~13% cheaper** (24 ms vs 28 ms per thought).

**The commute analogy:** imagine every trip to the shop costs a fixed 20 minutes of
driving, no matter what you buy. A bigger shopping list barely adds time — but a tiny
errand costs the same 20-minute drive. A "3× smaller" draft isn't 3× faster, because the
fixed driving time eats the difference. Language models on small consumer GPUs are
dominated by this fixed overhead: starting each thought costs a large, model-size-
independent chunk of time.

So the apprentice was barely faster than the chef — hiring them saved almost nothing.

### Bet 2 — "The draft agrees with the target." ❌ Lost.

The speedup only materializes when the target **accepts** the draft's guesses. On real
text, the target rejected **about 2 out of every 3** guesses. Every rejected guess is
wasted work — the apprentice chopped vegetables the chef threw away.

> **The honest summary:** speculative decoding multiplies the number of thoughts per
> word (draft guesses + a check + a correction), and each thought was barely cheaper,
> while most guesses were wasted. That combination can only lose.

---

## The verdict

> **Speculative decoding is not a free lunch you bolt onto any model. It is a bet that
> your draft is cheap enough and agrees with your target. I built it correctly, verified
> it, and measured both conditions honestly across 108 configurations and 6 model
> pairs — and no configuration won** (speedup 0.22×–0.65×; 1.0× would be break-even).

The right engineering answer on this hardware was therefore to **not use it** — and to
write down exactly why, which is what this repo is.

For the numbers, see [`results.md`](results.md). For every bug I hit along the way, see
[`blockers.md`](blockers.md).

---

## What would actually make it win

The failure points point to the fixes. Three levers, from most to least practical here:

1. **A bigger target.** The trick's economics improve as the target gets more expensive
   to think (the fixed overhead becomes a smaller fraction). My projections say a
   same-family 8B or 16B model (quantized to 8-bit) would be the first configuration to
   win on this class of hardware — the target's size makes the draft relatively much
   cheaper, if its guesses are still accepted. This is the "when I get a bigger GPU"
   direction.
2. **A draft trained to predict the target.** Instead of a general small model, train
   the draft head on the *target's own* thinking patterns (the EAGLE approach). This
   attacks the acceptance bet directly — the draft learns what the target would say.
3. **Eliminate the fixed overhead.** The ~20 ms launch cost can be cut with compilation
   tricks (`torch.compile` roughly halved it in my probe) or CUDA graphs. Compiled
   speculative decoding still lost at real-world acceptance, but plain compiled decoding
   was itself 2.2× faster — sometimes the best optimization is the boring one.

---

## The draft-head chapter — attacking the acceptance bet (in progress)

Lever 2 above ("a draft trained to predict the target") is the next chapter of
this repo — a tiny **draft head** that reads the target's own internal
thoughts and predicts the target's next word (the EAGLE idea; see
[`draft-head/`](../draft-head/README.md)). The chapter so far has produced the
same shape of lesson as the main story, one level deeper — **the wall was
never where it looked**:

- **More data was the first real lever.** 500K → 3.0M tokens lifted agreement
  0.376 → 0.485.
- **More depth was not a lever.** A second decoder layer made the predicted
  features measurably better, yet agreement barely moved.
- **The objective was the hidden lever — and it is now closed.** We trained
  the head on feature *shape* (MSE) but measured whether the target *agrees*
  (argmax) — a mismatch that only shows up at saturation, exactly where we
  ended up. Switching to the paper's actual loss (Smooth L1 + a token
  cross-entropy term) compressed ~100 epochs of learning into 10... and
  landed on the **same ceiling**: best agreement **0.490**, which is **95% of
  the target's own ceiling** (the target itself only commits to a clear
  winner on 51.4% of positions — the other half of the data has no winner
  for any draft to agree on).

**The plain-English moral:** the head was never the problem, the training time
was never the problem, and the loss was only half the problem — the last mile
is that ~half of real text has no single "right" next word for the target
itself. The remaining question is whether a draft that agrees with the target
49 times out of 100 wins in a real decode loop — that is the engine, and it is
the next chapter.

---

## A tiny glossary (for the non-technical reader)

| Term | Meaning |
|---|---|
| **Token** | A piece of a word (roughly "word"). Models generate text token by token. |
| **Draft / target** | The small guessing model / the big model whose quality you actually want. |
| **Forward pass ("a thought")** | One model computation producing the next word's probabilities. |
| **Acceptance rate** | Fraction of draft guesses the target agrees with. ~1.0 = perfect agreement; ~0.35 = what real text gave us. |
| **KV cache** | The model's running memory of what it has said, so it doesn't re-read everything each step. |
| **fp16** | Half-precision arithmetic — faster but with tiny rounding errors (the coin-flip trap). |
| **Greedy vs sampled** | Choosing the single most likely next word vs rolling dice over the probabilities. |
| **Speedup** | How many times faster speculative decoding is than plain decoding. 1.0× = same speed; 0.5× = twice as slow. |
