# Learnings & Analysis — why speculative decoding lost

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

For the numbers, see
[`results/speculative-decoding.md`](../results/speculative-decoding.md). For
every bug I hit along the way, see [`blockers.md`](blockers.md).

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
[`results/eagle-training.md`](../results/eagle-training.md)). The chapter so
far has produced the same shape of lesson as the main story, one level deeper
— **the wall was never where it looked**:

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

## The engine chapter — the trained draft loses at decode (three walls)

Lever 2's final exam: we built the real decode engine around the trained head
(the e26 CE checkpoint, 0.490 teacher-forced agreement) and measured it on
real text with the same three-gate discipline as the main story —
token-identical correctness, acceptance, and honest timing. **All three gates
passed or failed cleanly, and the verdict is decisive: the trained draft loses
at decode — for three independent reasons, only one of which is hardware.**

- **The engine was correct.** Every configuration produced token-identical
  output to plain decoding (a handful of recorded floating-point coin-flip
  near-ties, per Trap 3). The alignment math — the thing most likely to be
  subtly wrong — was right.
- **Acceptance collapsed at decode.** Teacher-forced, the head agreed with the
  target 49% of the time. Chained at decode — the head predicting its own next
  feature instead of reading the target's real one — acceptance dropped to
  **3–9%**. Even the *first* proposal (which was fed the target's true
  features) was accepted only ~26% of the time.
- **Even perfect acceptance would not have won on this hardware.** The
  measured timing math: each draft-chain + verify cycle costs ~90 ms to
  produce ~1.3 tokens, while plain decoding produces one token every ~35 ms.
  Break-even needs ~1.6 accepted tokens per verify; we measured 0.26. Even if
  every first proposal were accepted (τ = 1), the engine would still be
  **0.78× slower** — the fixed per-call overhead (the commute analogy, now
  living *inside* the draft) eats the win.

**The three walls, simply put:**

1. **The target's own ceiling.** The target only commits to a clear winner on
   ~51% of positions in real text. No draft — however well trained — can agree
   with the target more than the target agrees with itself. Hardware cannot fix
   this.
2. **The transfer gap.** The head trained on ~1000-token context windows, then
   at decode it saw a 1–5 token buffer — a different world. Trained on
   WikiText, tested on diverse prose. This collapse is a training-shape
   mismatch, not a GPU problem.
3. **The hardware economics.** Each small head forward pays the same fixed
   launch cost as a full-model call. This is the wall a bigger target model
   fixes.

> **The honest verdict:** the gate did its job — it caught the transfer
> failure before any scaling commitment. The trained draft lost at decode for
> the same reason the off-the-shelf draft lost in the main story: speculative
> decoding only wins when the target is expensive enough that the draft's
> fixed cost is a rounding error.

### What pair *would* have worked (even if it didn't fit our hardware)

Our benchmark-sweep data holds the surprise: **acceptance was never the binding constraint
at our scale.** Every same-family pair accepted ~2 of every 4 drafted tokens
(τ ≈ 2, acceptance 49–65%) — a genuinely good draft — and *still* lost
(x0.30–0.65), because the machinery's fixed cost exceeded the target's own
per-token cost at 1.5–3B scale. So the fix was never a *stronger* draft — it
was a *bigger* target, which makes the fixed overhead a smaller fraction.

The same-family pairs that the economics say would win (needing more than our
12 GB):

- **qwen2.5-0.5b → qwen2.5-7b** (~16 GB fp16) — the 0.5b draft stays cheap,
  the 7b target is ~3–4× slower than our 3b, and measured τ ≈ 2 carries
  over → projected ~1.5–2×. The cheapest configuration that likely wins.
- **qwen2.5-0.5b → qwen2.5-14b** (~30 GB fp16, or ~16 GB at 8-bit) — the
  target forward now dominates comfortably → projected 2×+. The literature's
  canonical example has the same shape: Llama-3.2-1B → Llama-3.1-70B ≈ 4×.
- **EAGLE on qwen2.5-14b** — EAGLE-2's reported 3–4× speedups live at
  13–33B scale. But size alone does **not** fix Wall 2: the transfer collapse
  (0.49 → 0.09) must be addressed separately — the leading fix is feeding the
  head a window of recent *actual* feature/token pairs at decode, restoring
  its training-time context.

> **The moral of the whole repo, in one line:** speculative decoding is a bet
> on economics, not a bet on draft quality — the draft's agreement only
> matters once the target is expensive enough that the draft's fixed cost is
> cheap by comparison. On small consumer hardware, every configuration loses;
> the fix is a bigger target, and the trained-draft (EAGLE) path needs one
> extra fix that size alone cannot buy.

---

## What would show the promise — the pairs (from the open-source catalog)

A natural question after all this losing: *if there were no hardware limit,
which freely available models would actually show the trick working?* The
answer follows directly from everything measured here, and the catalog has
documented winners for both levels of the trick.

First, the two levels pair differently:

- **Vanilla speculative decoding** is a *pair of two models* — a small,
  separate, pretrained **draft** model and a big **target**. (Our main-story
  harness.)
- **EAGLE** is *one model plus a small trained head* — the "draft" is a tiny
  network trained on the target's *own* internal thinking, so there is no
  second pretrained model. The question becomes which **target** to pick, not
  which pair. (Our draft-head chapter.)

### Vanilla pairs that are documented winners

| Draft | Target | Why |
|---|---|---|
| Llama-3.2-1B | Llama-3.1-8B | The vLLM/Hugging Face canonical example; the smallest config where the economics start to work (~1.4–1.6× measured in the wild). |
| Llama-3.2-1B / 3B | Llama-3.1-70B | The community favorite — same tokenizer, huge target; the real 2–3× lives here. Also Hugging Face's official *assisted generation* example. |
| Qwen2.5-0.5B | Qwen2.5-32B / 72B | The direct extrapolation of *our own benchmark pairs* — the same family we already measured accepting at 52–65%, scaled up 10–20×. The safest bet given our data. |
| Qwen2.5-Coder-1.5B | Qwen2.5-Coder-32B | Same-family coder pair that community members report actually winning on consumer hardware. |

### EAGLE targets that are documented winners

| Target | Published result |
|---|---|
| Vicuna-33B | EAGLE-1/2's workhorse — 2.7×–4× (Spec-Bench: 2.4–2.5× across scales) |
| LLaMA2-Chat-70B | EAGLE-1's flagship — 2.7–3.5× |
| Mixtral-8x7B | EAGLE-2 — ~3.5× (proves the trick works on MoE targets too) |
| DeepSeek-67B | EAGLE-3's flagship — ~2.7× |
| Llama-3.1-70B | AWS Neuron ships a full EAGLE tutorial for exactly this target |
| Mistral-Large-3 (67B) | **Mistral itself published a 12B EAGLE draft on Hugging Face** — zero training needed, vendor-blessed |

### Why these work and ours didn't — one rule

Our own experiments *prove* the selection rule once read correctly:

1. **Acceptance was never the binding constraint.** Every same-family benchmark pair
   accepted ~2 of 4 drafts (τ ≈ 2) and still lost at 1.5–3B scale — the fixed
   overhead (~90 ms per cycle) exceeded the target's own 29–36 ms per token.
2. **The rule is pure economics: the target's forward pass must dwarf the
   draft's cost.** A 70B target costs ~10–20× more per token than our 3B. The
   *same* τ ≈ 2 that lost at 3B wins at 70B — the overhead becomes a rounding
   error. Every published winner has a target ≥ 7B; most are 30–70B.
3. **EAGLE's draft is nearly free on big targets.** The head reuses the
   target's early layers and adds a few small ones — so on a 70B target the
   draft costs a fraction of a percent of the target's forward, and even
   modest agreement (τ ≈ 1) wins. That is why EAGLE's published numbers
   (2.7–4×) beat vanilla's (~1.4–2×) at the same scale.

**The kitchen analogy, extended:** the chef (target) was only slightly more
skilled than the apprentice (draft) in our kitchen, so checking cost almost
as much as cooking. In a professional kitchen (a 30–70B model), the chef is
*so* much slower per dish that even an apprentice who is right half the time
doubles output — and an EAGLE apprentice who was trained *by* that chef
(read: on the chef's own habits) does even better.

### The no-hardware-limit plan worth running

1. **Vanilla demo:** Llama-3.2-1B → Llama-3.1-70B — the most-documented pair,
   expect ~2×, verifiable with our three-gate harness.
2. **EAGLE on top:** train the head on Llama-3.1-70B (AWS Neuron proves the
   toolchain), or skip training entirely with Mistral's published 12B EAGLE
   draft for Mistral-Large-3 — the cleanest "see the promise" option.
3. **Direct continuation of our work:** Qwen2.5-0.5B → Qwen2.5-32B vanilla,
   then an EAGLE head on Qwen2.5-32B — same family we already have acceptance
   data for, so the only new variable is the scale the economics need.

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
