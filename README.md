# Tone Engine 7

<p>
  <img alt="Python" src="https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.6-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="Status" src="https://img.shields.io/badge/status-experimental-eda100">
  <img alt="License" src="https://img.shields.io/github/license/SeventhSG/ToneEngine7?color=2a78d6">
  <img alt="Last commit" src="https://img.shields.io/github/last-commit/SeventhSG/ToneEngine7">
</p>

> An open-source neural audio engine for understanding, modeling, and matching guitar tones.

Tone Engine 7 is an experimental machine learning project built around one simple idea:

**Can we teach a computer to actually understand guitar tone?**

Most guitar tone tools rely on databases, predefined presets, or language models that generate plausible settings based on what they know about an artist or piece of gear.

Tone Engine 7 takes a different approach.

Instead of simply asking an AI what settings might sound good, the goal is to build a system that can **analyze audio mathematically, understand its tonal characteristics, generate a tone, compare the result against the original, and improve the settings automatically.**

---

## The Concept

A guitar recording is ultimately a signal.

That signal contains information about frequency content, harmonics, dynamics, distortion, transients, noise, spatial characteristics, and many other properties.

Tone Engine 7 treats that signal as data that a machine learning model can analyze.

The long-term concept looks like this:

```mermaid
flowchart TD
    A[Reference guitar audio] --> B[Audio preprocessing]
    B --> C[Audio encoder]
    C --> D[Tone embedding]
    D --> E[Initial tone parameters]
    E --> F[Virtual signal chain]
    F --> G[Generated audio]
    G --> H[Audio comparison]
    H --> I[Optimizer]
    I --> J[Improved parameters]
    J -.repeat.-> F
```

The system does not need to get the perfect tone on its first attempt. Instead, it can make a prediction, render that prediction, measure how different it is from the reference, adjust the parameters, and try again. This turns guitar tone matching into an optimization problem.

### What Is a Tone?

For Tone Engine 7, a tone is not simply a collection of words such as:
- High gain
- Dark
- British
- Lots of mids

A tone can be represented as measurable audio information. For example:

```text
Gain:       0.72
Bass:       0.41
Mid:        0.68
Treble:     0.57
Presence:   0.52
Cabinet:    ...
Microphone: ...
Pedals:     ...
```

The exact representation will evolve as the project develops. The important concept is that the engine should learn the relationship between audio characteristics and the parameters that create them.

### The Neural "Ear"

Tone Engine 7 uses neural networks as its mathematical equivalent of an ear. The model does not hear music like a human; it receives numerical representations of audio and learns patterns within them. Given enough training data, the model can learn relationships between audio characteristics, distortion, frequency response, harmonic structure, dynamics, and the parameters that produce them. This allows the system to work from the sound itself instead of relying entirely on textual descriptions.

### Prediction Is Only the Beginning

A major principle of Tone Engine 7 is that prediction does not have to be perfect. Suppose the target tone was created using:

| Parameter | Value |
| :--- | :--- |
| **Gain** | 0.80 |
| **Bass** | 0.45 |
| **Mid** | 0.70 |
| **Treble** | 0.60 |

The neural network might initially predict:

| Parameter | Value |
| :--- | :--- |
| **Gain** | 0.74 |
| **Bass** | 0.49 |
| **Mid** | 0.65 |
| **Treble** | 0.57 |

Instead of treating that as the final answer, Tone Engine 7 can render the predicted settings and compare the resulting audio against the target. The optimizer can then search for a better configuration.

```text
Prediction -> Render -> Compare -> Calculate error -> Adjust parameters -> Render again -> Compare again -> Repeat
```

The objective is to minimize the difference between the generated tone and the reference.

---

## Audio Similarity

Comparing two guitar recordings is more complicated than comparing individual samples. Two signals can sound very similar while having different waveforms. Because of this, Tone Engine 7 is designed to experiment with multiple forms of audio comparison, including:

- **Spectral analysis**
- **STFT-based comparison**
- **Multi-resolution spectral loss**
- **Harmonic characteristics**
- **Dynamic characteristics**
- **Frequency response**
- **Learned audio embeddings**
- **Perceptual similarity metrics**

The goal is to find metrics that correlate with how humans actually perceive guitar tone.

---

## Starting Small

Tone Engine 7 is intentionally starting with a controlled environment. The first objective is not to analyze an entire commercial song and recreate its guitar rig. The first objective is much simpler:

> **Can a neural network listen to a guitar tone generated by a known system and recover the parameters that produced it?**

For example:

```text
Virtual Amplifier
       |
       +-- Gain
       +-- Bass
       +-- Mid
       +-- Treble
       +-- Presence
       +-- Master
```

The engine can generate thousands of tones from random parameter combinations. Each example contains:
- `audio.wav`
- `metadata.json`

The metadata contains the parameters that generated the audio. The model then learns:

**Audio → Parameters**

After training, it is tested on tones it has never seen before. If the model can successfully recover those parameters and reproduce the original sound, the foundation is working.

---

## Where This Can Go

Once the core system works, the same concept can be expanded.

```mermaid
flowchart LR
    A[Single amp] --> B[Amp + cabinet]
    B --> C[Amp + cabinet + IR]
    C --> D[Amp + pedals]
    D --> E[Complete signal chain]
    E --> F[Microphone modeling]
    F --> G[Real guitar recordings]
    G --> H[Guitar stem extraction]
    H --> I[Full song analysis]
```

Eventually, the system could work toward something like:

```mermaid
flowchart TD
    A[Song / guitar recording] --> B[Extract guitar]
    B --> C[Analyze tone]
    C --> D[Identify characteristics]
    D --> E[Generate initial settings]
    E --> F[Search available gear]
    F --> G[Optimize signal chain]
    G --> H[Final tone]
```

The goal is not necessarily to identify the exact equipment originally used to record a tone. If the original recording used a specific amplifier that the user does not own, the useful answer may be a completely different signal chain that produces an extremely similar result.

In other words: **The goal is to reproduce the sound, not simply identify the equipment.**

---

## Tone Engine 7 and LLMs

Tone Engine 7 is not intended to replace language models. Instead, an eventual system could combine both technologies.

```mermaid
flowchart TD
    A["LLM<br/>context, knowledge, gear information"] --> B["Tone Engine 7<br/>audio analysis, prediction, optimization"]
    B --> C[Final tone]
```

- **LLM:** Understands requests, artists, songs, equipment, signal chains, and natural language.
- **Tone Engine 7:** Analyzes actual audio and performs the mathematical work required for tone matching.

This separation allows each system to focus on what it does best.

---

## Open Source

Tone Engine 7 is being developed as an open-source research project. The purpose is to experiment openly with:

- Neural audio models
- Guitar tone representations
- Synthetic datasets
- Amp and effects modeling
- Audio similarity
- Parameter prediction
- Optimization algorithms
- Perceptual audio analysis

The project is intentionally starting from the core engine. The user interface, SaaS platform, authentication, subscriptions, and other product features are separate concerns. 

*Build the engine first. Build the product later.*

---

## Project Status

**Status: Experimental / Early Research**

This project is still in its early stages. The architecture, models, datasets, and approaches may change significantly as experiments are performed.

The first major milestone is simple:
> Train a model that can analyze an unseen generated guitar tone, predict the parameters that created it, and reproduce that tone.

**First milestone: proven on a single virtual amp, including the optimization loop.** Stage 2 (signal chains) is covered: Experiment 2 proved an overdrive, amp, and cabinet chain and showed that training on up to a million synthetic examples lifts the CNN's amp identification from 58% to 82%, and Experiment 3 runs the complete Stage 2 rig from INSTRUCTIONS.md (noise gate, compressor, overdrive, amp, EQ, cabinet, microphone) through the same pipeline. Stage 3 (guitar-specific variables), real recordings, and song input are still ahead.

---

## Results: Experiment 1

A CNN was trained on 8,000 synthetic examples from one virtual amp (gain, bass, mid, treble, presence, master) and evaluated on 1,000 held-out examples it never saw during training. Per INSTRUCTIONS.md, the CNN's prediction is only meant to be a starting point: a CMA-ES search then refines it by rendering, scoring against the same multi-resolution STFT distance used throughout the project, and repeating. Every number and chart below comes straight out of `training/experiments/exp1_smoke/` (`evaluation/make_readme_charts.py` renders them; nothing here is hand-tuned).

| Metric | CNN prediction only | After optimization |
| :--- | :---: | :---: |
| Parameter mean absolute error (0-10 knob scale) | 0.80 | **0.39** |
| Audio similarity, reconstructed vs. reference | 0.65 | **0.92** |
| Test examples that improved after optimization | - | **100%** |

8,000 training examples (plus 1,000 for validation), 1,000 held-out test examples, CMA-ES budget of 300 renders per example.

<table>
<tr>
<td width="50%">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/loss_curve_dark.png">
  <img src="docs/assets/loss_curve_light.png" alt="Training and validation loss over 30 epochs, both curves converging smoothly">
</picture>
</td>
<td width="50%">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/param_mae_dark.png">
  <img src="docs/assets/param_mae_light.png" alt="Mean absolute error per amp knob on the held-out test set, all under 1.2 on a 0-10 scale">
</picture>
</td>
</tr>
<tr>
<td width="50%">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/audio_similarity_dark.png">
  <img src="docs/assets/audio_similarity_light.png" alt="Audio similarity between reconstructed and reference tone, tracked during validation across training">
</picture>
</td>
<td width="50%">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/optimization_comparison_dark.png">
  <img src="docs/assets/optimization_comparison_light.png" alt="Bar chart comparing audio similarity and parameter error before and after the optimization loop, across all 1000 test examples">
</picture>
</td>
</tr>
</table>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/reconstruction_example_dark.png">
  <img src="docs/assets/reconstruction_example_light.png" alt="Spectrogram of a reference test example next to the CNN-only reconstruction and the reconstruction after optimization">
</picture>

The example above is not a cherry-picked best case: it is the *median* test example by CNN-only audio similarity, so it represents typical performance rather than the easiest one. Note how much closer the third panel gets after optimization compared to the second.

That answers the milestone's actual question: yes, a network can recover this virtual amp's parameters from unseen audio, and a render-compare-adjust loop on top of its prediction gets substantially closer to the reference tone. What it does not yet cover: multiple amps, cabinets, IRs, pedals, or real recordings. Those are the next stages.

---

## Results: Experiment 2

Stage 2 of INSTRUCTIONS.md asks for multiple amps, cabinets, IRs, mic modeling, overdrive, EQ, compression, and a noise gate all at once. That is too much to prove in one step, so this is **Stage 2, increment 1**: an overdrive pedal (on or off) into one of 3 amp voicings into one of 4 synthetic cabinet IRs. Compression, a noise gate, mic modeling, and a separate EQ pedal are not in this increment yet.

The target is now a mix of continuous knobs (overdrive drive and level, gain, bass, mid, treble, presence, master) and discrete choices (which amp, which cabinet, whether the overdrive pedal is engaged), predicted by one CNN with multiple output heads. 8,000 training examples (plus 1,000 for validation), evaluated on 1,000 held-out test examples.

| Metric | Result |
| :--- | :--- |
| Continuous knob mean absolute error (0-10 scale) | 2.07 |
| Cabinet identification accuracy (4 choices, chance = 25%) | **100%** |
| Amp identification accuracy (3 choices, chance = 33%) | 59% |
| Overdrive on/off accuracy (chance = 50%) | 61% |
| Audio similarity, CNN prediction only | 0.41 |

Cabinet identification is essentially solved, because a cabinet IR leaves a strong, distinctive spectral fingerprint. Amp and overdrive detection are only modestly above chance, honestly a weaker result than Experiment 1's clean single-amp case, likely because a low-drive overdrive pedal barely changes the audio at all (making "on" and "off" genuinely hard to tell apart from a short clip) and the three amp voicings partially overlap in the frequency ranges their EQ knobs cover. The training curve shows why the checkpoint was picked early: validation loss stops improving and gets noisy well before training loss does, i.e. the model overfits past epoch 15-20 on this harder task.

<table>
<tr>
<td width="50%">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/exp2_loss_curve_dark.png">
  <img src="docs/assets/exp2_loss_curve_light.png" alt="Experiment 2 training and validation loss over 40 epochs, validation loss becoming noisy and rising after around epoch 20 while training loss keeps falling">
</picture>
</td>
<td width="50%">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/exp2_classification_accuracy_dark.png">
  <img src="docs/assets/exp2_classification_accuracy_light.png" alt="Bar chart of test-set classification accuracy for overdrive on or off, amp choice, and cabinet choice, each compared against chance level">
</picture>
</td>
</tr>
</table>

The optimization loop from Experiment 1 carries over here too, refining the continuous knobs with the discrete choices (amp, cabinet, overdrive on/off) held fixed at whatever the CNN predicted. Across all 1,000 test examples, with the same budget of 300 renders per example:

| Metric | CNN prediction only | After optimization |
| :--- | :---: | :---: |
| Continuous knob MAE (0-10 scale) | 2.07 | **1.43** |
| Audio similarity | 0.41 | **0.81** |
| Test examples that improved | - | **100%** |

That the continuous knobs can still be pushed this far even when roughly 40% of the amp choices and overdrive flags are wrong is itself informative: a wrong amp can often be partly compensated for by re-tuning the tone stack. Giving the same 300-render loop the *true* amp, cabinet, and overdrive state only reaches 0.83 on the same 1,000 examples, so at this budget the wrong choices cost about 0.02 of similarity.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/exp2_optimization_comparison_dark.png">
  <img src="docs/assets/exp2_optimization_comparison_light.png" alt="Bar chart comparing audio similarity and continuous knob error before and after the optimization loop, across 1000 test examples">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/exp2_reconstruction_example_dark.png">
  <img src="docs/assets/exp2_reconstruction_example_light.png" alt="Spectrogram of a reference test example next to the CNN-only reconstruction and the reconstruction after optimization, for Experiment 2's signal chain">
</picture>

### Letting the optimizer change the amp

The CNN picks the wrong amp about 4 times in 10, and the loop above never questions that choice. So the optimizer can now also try the other amps: it runs a CMA-ES search on each amp with 300 renders, keeps whichever reached the lowest loss, and spends the rest of the budget refining that one (`optimization/optimizer.py`, `optimize_with_choices`). The cabinet and overdrive calls stay as the CNN made them (see below for why).

Compared on the same 200 test examples, all at the same total budget of about 1,800 renders per example, with a run given the *true* amp, cabinet, and overdrive as a ceiling:

| | Audio similarity | Amp identified | Knob MAE (0-10) |
| :--- | :---: | :---: | :---: |
| CNN prediction only | 0.41 | 53.5% | 2.07 |
| Choices fixed, 300 renders | 0.81 | 53.5% | 1.46 |
| Choices fixed, 1,800 renders | 0.89 | 53.5% | 0.96 |
| **Amp search, 1,800 renders** | **0.92** | **74.0%** | **0.69** |
| True choices, 1,800 renders (ceiling) | 0.95 | 100% | 0.42 |

The amp search gains +0.030 similarity over keeping the CNN's choice at equal compute (95% bootstrap interval +0.019 to +0.042), about half the distance to the ceiling, and it raises amp identification from 53.5% to 74.0% using nothing but render-and-compare. The average hides a trade-off, though: on the 93 examples where the CNN's amp was wrong the search gains +0.088, while on the 107 where it was right it loses 0.020, because the winning amp only gets about 1,200 of the 1,800 renders instead of all of them. Example by example, the search beats keeping the CNN's choice on 46% of the test set and is slightly worse on the other 54%. Only searching when the CNN is unsure of its amp call could keep most of the gain without that cost; that has not been tried yet. How much the search is worth also depends on how good the CNN is, which the next section measures.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/exp2_choice_search_dark.png">
  <img src="docs/assets/exp2_choice_search_light.png" alt="Two bar charts on the same 200 test examples. Audio similarity: choices fixed at 300 renders 0.81, choices fixed at 1800 renders 0.89, amp search at 1800 renders 0.92, true choices at 1800 renders 0.95. Amp identified correctly: 53.5%, 53.5%, 74%, 100%, against a chance level of 33%">
</picture>

The schedule was picked on the validation split (100 examples per row), so the test set played no part in choosing it. Most of what was tried did not work:

| Search schedule (validation) | Renders | Audio similarity | Amp identified |
| :--- | :---: | :---: | :---: |
| Choices fixed | 310 | 0.820 | 56% |
| Top 6 amp/cabinet/overdrive combinations, 21 renders each, best 2 refined | 296 | 0.777 | 53% |
| Top 2 combinations, 101 renders each | 292 | 0.797 | 58% |
| Every amp, 61 renders each | 293 | 0.776 | 49% |
| True choices (ceiling) | 310 | 0.854 | 100% |
| Choices fixed | 1,810 | 0.911 | 56% |
| Top 6 combinations, 300 renders each | 1,796 | 0.890 | 85% |
| Every amp, 150 renders each | 1,793 | 0.905 | 54% |
| **Every amp, 300 renders each** | 1,793 | **0.935** | 78% |
| True choices (ceiling) | 1,810 | 0.964 | 100% |

Two lessons came out of this. First, **at a budget of 300 renders, every search schedule tried lost** to trusting the CNN. A short search on each candidate ranks them by how close each one happened to start, not by how close it could get, and whatever it does pick gets less refinement. Screening needs roughly 300 renders per candidate before it reliably separates the amps, so the search only pays off with a larger total budget. Second, **overdrive on/off is only weakly identifiable from the sound**: given the true amp and cabinet and 300 renders each, the true overdrive state reached the lower loss in only 62 of 100 validation examples. A low-drive pedal is mostly a level change, which the amp's gain knob can reproduce. Searching over it spends budget for little return, which is why the final search varies only the amp.

### More training data

Everything above uses a CNN trained on 8,000 examples, and its validation loss bottomed out at epoch 14 of 40, which is the usual sign of too little data. The amp search also showed the audio does carry the amp's identity (render-and-compare found it 74% of the time) while the CNN only managed 54%. So the next question was simply whether more of the same synthetic data fixes the CNN.

`training/generate_shards_exp2.py` builds a train-only pool of up to 1,000,000 examples from the same signal chain, with its own random seed (1000, against 42 for the original dataset). Each shard of 10,000 examples is stored as one array of log-mel features plus labels instead of three files per example, so a million examples take 12 GB and load as fast sequential reads. (The pool was first generated on another disk and later regenerated under `data/generated/exp2_pool`; generation is deterministic per shard and a sampled shard was verified bit-identical, which is why some runs' `config_used.yaml` still names the first location.) Validation and test are still the original splits, so every row below is scored on the same 1,000 test examples as everything else in this README. The only thing that changes between rows is how many pool examples the CNN is trained on (`configs/experiment2_pool*.yaml`); the model, learning rate, batch size, and stopping rule (stop after 8 epochs without a validation improvement) are identical.

| Training examples | Amp identified | Overdrive on/off | Cabinet | Knob MAE (0-10) | Audio similarity, CNN only |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 8,000 | 57.7% | 61.6% | 100% | 2.14 | 0.41 |
| 32,000 | 67.4% | 64.0% | 100% | 1.77 | 0.45 |
| 100,000 | 70.7% | 65.4% | 100% | 1.55 | 0.46 |
| 300,000 | 77.9% | 66.3% | 100% | 1.37 | 0.49 |
| **1,000,000** | **82.3%** | **68.5%** | 100% | **1.26** | **0.52** |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/exp2_data_scaling_dark.png">
  <img src="docs/assets/exp2_data_scaling_light.png" alt="Two line charts against training set size on a log scale from 8k to 1M. Amp identification rises from 57.7% to 82.3% and overdrive on/off from 61.6% to 68.5%. Knob error falls from 2.14 to 1.26.">
</picture>

Amp identification climbs from 57.7% to 82.3% and is still rising at a million examples: about 5.6 points per tripling of the data on average, with single steps ranging from +3.3 to +9.7. Knob error falls steadily too. Overdrive on/off barely moves (61.6% to 68.5%), which fits the earlier finding that it is only weakly identifiable from the sound at all. The 8,000-example row is a control: trained through the new pool pipeline it lands within noise of the original model (57.7% against 58.9% amp, 2.14 against 2.07 knob error), so the gains come from the data and not the pipeline. Each row is a single training run, and validation loss stays spiky even at a million examples (it jumps from about 1.0 to 1.8 and back between epochs), so the checkpoint picked by validation loss adds some noise to each point; the trend across five sizes is the robust part, not any one step.

A learning-rate schedule was tried as well, on 100,000 examples: cosine decay to zero over 30 epochs instead of a constant rate. It helped a little (amp 72.1% against 70.7%, knob error 1.48 against 1.55, similarity 0.49 against 0.46), far less than tripling the data did, and the validation loss stayed noisy even at a near-zero learning rate, so the noise is not coming from the learning rate. The scaling runs all use the constant rate so that data size is the only variable.

A better CNN pays off after optimization too. Same seeds, same examples (all 1,000 for the 300-render column, the same 200 for the others):

| CNN trained on | Choices fixed, 300 renders | Choices fixed, 1,800 renders | Amp search, 1,800 renders | True choices, 1,800 renders |
| :--- | :---: | :---: | :---: | :---: |
| 8,000 (original) | 0.807 | 0.894 | 0.924 | 0.953 |
| 300,000 | 0.847 | 0.932 | 0.942 | 0.957 |
| **1,000,000** | **0.852** | **0.935** | **0.942** | 0.959 |

At the everyday budget of 300 renders, better training data lifts the final similarity from 0.807 to 0.847 (300,000 examples) and 0.852 (1,000,000 examples; +0.045 over the original, 95% bootstrap interval +0.040 to +0.050). Both are already more than handing the original CNN the true choices at that budget (0.829). The amp search matters less as the CNN improves: over keeping the CNN's choice at 1,800 renders it adds +0.030 for the original CNN, +0.010 for the 300,000-example one (interval +0.000 to +0.020), and +0.006 for the 1,000,000-example one, whose interval (-0.001 to +0.014) includes zero. By then the CNN's own amp call (82.0% on these examples) is about as good as what render-and-compare finds (82.5%), so there is little left for the search to fix. The order of priorities is clear: more training data first, render-and-compare search second.

**Reproducibility note.** The `cma` library treats a seed of 0 as "seed from the clock", and the optimizer used to pass 0, so optimization runs were randomly seeded even with `--seed 0`. That is fixed (`optimization/optimizer.py`, `_cma_seed`), and each example now gets its own seed derived from the base seed (`--seed 0` gives example *i* seed *i*), the same in every mode so that fixed / search / true-choice comparisons stay paired. Per-example seeds matter: one shared seed for every example also reproduces exactly, but it shifted the 1,000-example average by +0.004 against the randomly seeded run, because every example then gets the same random draws. All test-set optimization numbers in this README (both experiments) come from per-example-seeded runs (`*_pseed.json` reports for Experiment 2), and two runs with the same seed match to 10 decimal places. The validation sweep table above was run before the fix; its differences between schedules are much larger than that seed effect.

---

## Results: Experiment 3

**Stage 2, increment 2** completes the Stage 2 list in INSTRUCTIONS.md: the rig now has a noise gate, a compressor, the overdrive, one of 3 amps, an EQ pedal in the effects loop, one of 4 cabinets, and a microphone.

```mermaid
flowchart LR
    A[DI guitar + noise floor] --> G[Noise gate] --> C[Compressor] --> O[Overdrive] --> P[Amp] --> E[EQ pedal] --> K[Cabinet IR] --> M[Microphone]
```

Gate, compressor, overdrive and EQ are each on or off. Continuous knobs: gate threshold, compressor sustain and level, overdrive drive and level, the amp's six knobs, five EQ bands (100 Hz to 3.2 kHz), and the microphone's position on the cone and distance from the grille, 18 in total. Categorical choices: amp (3), cabinet (4), and microphone type (3: dynamic, condenser, ribbon). Like the cabinets, the microphones are synthetic filter shapes standing in for each family's character (`audio/rendering/mics.py`), not measurements of real mics.

Two things had to change for this rig to be a fair test:

- **The source now has gaps and a noise floor.** The original synthetic DI strikes every note at once and lets it ring for the whole clip, and a render check showed a noise gate then changes nothing at all, because the signal never drops below any threshold. Experiment 3 uses `synth_phrase` (notes start at random times, about half are damped early) plus `add_noise_floor` (hiss and 50/60 Hz hum at -70 to -40 dBFS). The noise level is part of the source, recorded but never predicted, which follows the brief's split between source, signal chain and recording chain. Experiments 1 and 2 still use the original source, unchanged.
- **The chain is described by a spec** (`audio/rendering/chain_spec.py`). The dataset, the model's output heads, the loss masking (knobs of a pedal that is off do not count) and the metrics all follow from `EXP3_SPEC` in `audio/rendering/signal_chain3.py`, so adding gear later means extending a spec, not rewriting the pipeline.

The data follows the Experiment 2 recipe: a 1,000,000-example training pool plus fixed 1,000-example validation and test sets, each split with its own seed (`training/generate_exp3.py`), and the same CNN trunk with spec-generated heads (`models/tone_predictor/multihead.py`).

### A training bug found along the way: BatchNorm statistics

The first run on 300,000 examples stopped at epoch 13 with its best validation loss at epoch 5, while training loss was still falling steadily, and the validation loss jumped up and down by as much as 16% between epochs. Scoring the same checkpoints with BatchNorm using each batch's own statistics instead of its stored running averages showed the cause: the epoch-13 weights were clearly better than epoch 5's (validation loss 2.62 against 2.81, amp 75% against 71%), but the stored statistics made them look worse (3.34). BatchNorm's running averages follow only the last few dozen training batches of 32 while the weights are still moving fast, so they are stale and noisy, and early stopping was picking checkpoints by that noise.

The fix is to recompute the statistics exactly over a fixed set of 10,000 training examples after every epoch, before validating and saving (`recalibrate_batchnorm` in `training/train_exp3.py`; weights are never touched). Validation loss then falls smoothly from epoch to epoch instead of jumping:

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Validation loss, stored statistics | 3.66 | 3.26 | 3.17 | 3.33 | 2.86 | 3.08 | 3.11 |
| Validation loss, recalibrated | 3.35 | 3.12 | 3.01 | 2.92 | 2.84 | 2.83 | 2.78 |

Retrained on the same 300,000 examples with recalibration, the run trains to epoch 21 (best at 13) instead of stopping at epoch 13 with the best at 5, and the test results move accordingly:

| 300,000 training examples, CNN only | Stored statistics | Recalibrated |
| :--- | :---: | :---: |
| Amp identified (3 choices) | 68.9% | **74.7%** |
| Compressor on/off | 82.4% | 83.8% |
| Microphone type (3 choices) | 88.1% | 89.9% |
| Knob MAE (0-10) | 2.14 | **2.02** |
| Audio similarity | 0.360 | **0.407** |

The other heads (gate, overdrive, EQ, cabinet) moved by a point or less.

This is almost certainly also why Experiment 2's validation loss was so spiky. Recalibrating Experiment 2's chosen checkpoints afterwards changes their test results by less than a point (amp 82.3% to 82.4% for the 1,000,000-example model), so the published Experiment 2 numbers stand; whether those runs also stopped too early would take retraining, which has not been done.

### What the CNN can read from the audio

Trained with recalibration on the full 1,000,000-example pool (27 epochs, best at 19), and on the first 300,000 of it for comparison, scored on the same 1,000 test examples:

| CNN only | 300,000 examples | 1,000,000 examples | Chance |
| :--- | :---: | :---: | :---: |
| Cabinet (4) | 100% | 100% | 25% |
| EQ pedal on/off | 88.8% | **92.8%** | 50% |
| Microphone type (3) | 89.9% | **92.3%** | 33% |
| Compressor on/off | 83.8% | **85.0%** | 50% |
| Amp (3) | 74.7% | **77.8%** | 33% |
| Noise gate on/off | 64.9% | 66.1% | 50% |
| Overdrive on/off | 58.9% | 58.3% | 50% |
| Knob MAE (0-10, only knobs whose pedal is on) | 2.02 | **1.93** | 3.33 |
| Audio similarity | 0.407 | 0.394 | |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/exp3_readout_dark.png">
  <img src="docs/assets/exp3_readout_light.png" alt="Experiment 3 test accuracy: gate on/off 66%, compressor 85%, overdrive 58%, EQ 93%, amp 78%, cabinet 100%, microphone 92%, each against its chance level; and knob error for all 18 knobs, grouped by pedal, mostly between 1.2 and 2.5 on a 0-10 scale against 3.33 for a random guess">
</picture>

The bigger rig is clearly harder to read than Experiment 2's. The cabinet, EQ pedal and microphone are read reliably, the amp call is about where Experiment 2 was at 300,000 examples, and the knobs are much less certain: knob error stays around 1.9 on the 0-10 scale, against 1.26 for Experiment 2 at the same data size, and more data helps them far less. That is expected rather than a failure: with a compressor, an EQ pedal and a microphone in the chain, very different knob settings can produce nearly the same sound (a mid boost on the EQ pedal and one on the amp's tone stack, for instance), which is exactly the "parameter accuracy is not audio accuracy" warning in INSTRUCTIONS.md. The one odd result is that the 1,000,000-example model's own prediction sounds slightly less like the reference (0.394 against 0.407) even though its gear calls and knobs are better on average; after optimization, below, the order reverses (0.737 against 0.730 at 300 renders, +0.006, 95% bootstrap interval +0.003 to +0.010).

Two of the on/off calls deserve a closer look (`evaluation/identifiability_exp3.py`, which re-renders each test example with a pedal switched off to see whether it was audible at all):

- **The noise gate is often physically invisible.** In 52% of the examples where it is on, its threshold sits below the noise floor, so it never closes and the output is identical to having it off. The model rightly hears those as "off", which caps what any model could score at 73% (270 of the 1,000 test examples have a gate that is on but inaudible). When the gate actually acts, the model gets it right 77.8% of the time, and 93.5% when it is off.
- **The overdrive is the part the model genuinely cannot hear.** It changes the audio in 99% of the examples where it is on, yet the model is right only 37.9% of the time then (78.3% when it is off). A low-to-medium drive into an already distorting amp is mostly a level and gain change that the amp's own gain knob can reproduce, the same weakness seen in Experiment 2.

### The optimization loop on the full rig

`optimization/evaluate_optimization_exp3.py` runs the same CMA-ES loop with per-example seeds, with one refinement: it only searches the knobs that matter for the chain being rendered (knobs of a pedal that is off change nothing), which on average is 12 of the 18. Pedal states, amp, cabinet and microphone are either held at the CNN's calls ("fixed") or set to the true ones as a ceiling:

| CNN trained on | CNN only | Fixed, 300 renders | True choices, 300 renders | Fixed, 1,800 renders | True choices, 1,800 renders |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 300,000 examples | 0.407 | 0.730 | 0.737 | 0.856 | 0.886 |
| **1,000,000 examples** | 0.394 | **0.737** | 0.739 | **0.864** | 0.891 |

The 300-render columns cover all 1,000 test examples, the 1,800-render columns the same 200.

- **The loop still works on the full rig**, lifting similarity from 0.39 to 0.74 at 300 renders and to 0.86 at 1,800; with the 1,000,000-example model every one of the 1,000 test examples improved. It needs more renders than Experiment 2 did (0.85 there at 300): a dozen active knobs is a much bigger space than eight.
- **The sound gets matched much better than the knobs.** For the 1,000,000-example model, knob error only falls from 1.93 to 1.86 at 300 renders (and from 1.94 to 1.45 at 1,800 on the 200-example subset), while similarity more than doubles, which is the many-to-one mapping again.
- **The gear calls start to matter once the knob search has enough budget.** Giving the loop the true pedal states and gear adds only +0.002 at 300 renders, where the knob search is the bottleneck, but +0.028 at 1,800 (95% interval +0.017 to +0.039). That is the same pattern as Experiment 2 before its amp search, so searching over pedal states and gear at high budgets is the natural next step for this rig.
- **More data helps less here than in Experiment 2.** After optimization the 1,000,000-example model beats the 300,000-example one by +0.006 at 300 renders (interval +0.003 to +0.010) and +0.008 at 1,800 (interval -0.004 to +0.020, so not distinguishable from zero). In Experiment 2 the same step was also small (+0.005), but the earlier step from 8,000 to 300,000 examples was worth +0.040.

### A discrete-choice search for the full rig

`evaluate_optimization_exp3.py` now has a `search` mode, generalizing Experiment 2's `optimize_with_choices` from a hardcoded overdrive/amp/cabinet tuple to any switch or choice in the `ChainSpec`: rank candidate combinations by the CNN's own confidence, screen each with a short CMA-ES run, keep the best, and spend the rest of the budget refining it.

A validation sweep (100 examples, 1,800-render budget, the 1,000,000-example model) tried the amp-only search first, the axis that paid off in Experiment 2. It did not repeat there: at Experiment 2's winning schedule (3 candidates, 300 screening renders each) it came out slightly behind just trusting the CNN, 0.865 similarity against 0.869 fixed, because screening 3 candidates leaves the winner less refinement depth than fixed spends on its one choice, and amp accuracy here (77.8% on the full test set) is already high enough that the depth lost buys little back. Trimming to 2 candidates broke even (0.870). Adding compressor on/off, the switch with the next-most room (85% accuracy), as a second axis alongside amp, still at 2 candidates, is what found a real edge: 0.876 on validation.

| Search schedule (validation, 1,800 renders) | Audio similarity | Amp identified |
| :--- | :---: | :---: |
| Choices fixed | 0.869 | 81% |
| Amp only, 3 candidates | 0.865 | 57% |
| Amp only, 2 candidates | 0.870 | 61% |
| Amp + compressor on/off, 2 candidates | **0.876** | 74% |
| True choices (ceiling) | ~0.891 | 100% |

On the full 200-example test set, that winning schedule (amp and compressor on/off together, 2 candidates) reached 0.870 similarity against fixed's 0.864, a gain of +0.006 (95% CI -0.006 to +0.018 by paired bootstrap, so not distinguishable from zero at this sample size) and only a small slice of the oracle's +0.027 ceiling. Unlike Experiment 2, where the amp search's gain concentrated on the examples where the CNN had the amp wrong, here the gain is spread about evenly regardless (+0.0064 similarity where the CNN's amp call was already right, +0.0052 where it was wrong), and the search actually lowers amp accuracy, from 80% (the CNN's own call on these 200 examples) to 73.5%, trading away some correct calls for a better-sounding wrong one.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/exp3_choice_search_dark.png">
  <img src="docs/assets/exp3_choice_search_light.png" alt="Two bar charts on the same 200 test examples. Audio similarity: choices fixed at 1800 renders 0.86, amp+comp search at 1800 renders 0.87, true choices at 1800 renders 0.89. Amp identified correctly: 80% fixed, 73.5% after the search, against a chance level of 33%, versus 100% for true choices">
</picture>

The honest read: on a chain this size, with a CNN already this accurate, render-and-compare search recovers only a sliver of the headroom the oracle shows, a single weak axis (amp alone) is not enough to find any of it, and even the two-axis result that did help is not clearly distinguishable from noise at 200 examples.

---

## Experiment 4: source and performance variables (smoke test)

Stage 3 of INSTRUCTIONS.md is guitar-specific SOURCE/PERFORMANCE variables (pickup type, pickup position, guitar response, tuning, strings, playing dynamics, pick attack), explicitly called out as distinct from signal-chain parameters. This first increment covers three of them, the ones with a clear, cheap physical model on top of the existing additive-pluck synth (`audio/synth/guitar_synth.py`): **pickup position** (a comb filter on the harmonics, `abs(sin(k * pi * pos_frac))`, the classic bridge-vs-neck brightness difference), **playing dynamics** (pick force drives both brightness and the attack transient's loudness), and **tuning** (a semitone offset applied before pitch conversion). Pickup type and guitar-body/string response are deferred, the same way Stage 2 deferred compression and the noise gate.

The signal chain is held fixed (Experiment 1's virtual amp at one preset, never varied), so this only tests whether a CNN can read source/performance characteristics from audio, without also re-solving gear identification at the same time. A small dataset (3,000 examples, single note per clip) and a CNN readout only (`models/tone_predictor/model.py`'s `TonePredictor`, reused unchanged with 3 outputs) are deliberately as far as this increment goes: audio-similarity re-rendering and the optimization loop are **not attempted yet**, because a fair re-render would need to reproduce the exact note played, and tuning shifts which note that is.

| Parameter | Test MAE, 0-10 scale (300 examples) | Chance (random guess) |
| :--- | :---: | :---: |
| Pickup position | **0.86** | 3.33 |
| Playing dynamics | **1.14** | 3.33 |
| Tuning | 2.47 | 3.33 |

Pickup position and dynamics are clearly readable from audio, well below chance. Tuning is barely better than guessing, and for a specific, identifiable reason rather than a training shortfall: each clip plays one note chosen uniformly at random across three octaves, then shifted by the tuning offset, so the model only ever hears the *final* pitch. Without a reference for what the "untransposed" note should have been, there is no way to separate "which note was picked" from "how far it was detuned" on a single isolated note, the same kind of genuine unidentifiability the overdrive pedal ran into in Experiments 2 and 3. Recovering tuning would need multiple notes with a known relative interval (a scale or chord) rather than one note in isolation, which is a concrete next step rather than more training.

---

## Repository Layout

```text
audio/
    synth/         synthetic DI guitar source generation (synth_phrase and
                    add_noise_floor for Experiment 3; synth_source_pickup
                    and source_spec.py for Experiment 4's pickup position,
                    dynamics and tuning)
    rendering/     Experiment 1's single virtual amp, Experiment 2's
                    overdrive/amp/cabinet signal chain (amp_models.py,
                    cabinets.py, chain_params.py, signal_chain.py), and
                    Experiment 3's full Stage 2 rig (chain_spec.py, mics.py,
                    signal_chain3.py)
    preprocessing/ wav I/O and normalization
    features/      log-mel spectrogram extraction
    similarity/    multi-resolution STFT audio similarity

models/
    tone_predictor/ audio -> parameters CNN (model.py), the multi-head
                    audio -> signal chain predictor (chain_model.py), and its
                    spec-driven successor (multihead.py)
    checkpoints/    trained weights (not versioned)

training/
    generate_dataset.py, generate_dataset_exp2.py   build a dataset from the renderer
    generate_shards_exp2.py, generate_exp3.py        large datasets as feature shards
    dataset.py, dataset_exp2.py, pool_exp2.py,
    shards.py                                        torch Datasets / shard batches
    train.py, train_exp2.py, train_exp3.py           training loops
    evaluate.py, evaluate_exp2.py, evaluate_exp3.py  held-out test set milestone reports
    exp3_common.py                                   Experiment 3 loss, decoding and metrics

evaluation/     shared metrics, comparison plots, and README chart generation
                (make_readme_charts.py for Experiment 1, _exp2, _scaling and
                _exp3), plus identifiability_exp3.py (which pedals are audible)
optimization/   CMA-ES refinement of the CNN's predicted parameters
                (evaluate_optimization.py / _exp2.py / _exp3.py); for
                Experiment 2 the discrete choices can be held fixed, searched
                (every amp is tried, the best one refined), or set to the true
                ones as a ceiling (--mode fixed / search / oracle)
configs/        experiment configs
data/           generated datasets (not versioned)
```

### Running Experiment 1

```bash
python -m training.generate_dataset --n-examples 10000 --seed 42 --out-name exp1_smoke
python -m training.train --config configs/experiment1.yaml
python -m training.evaluate --config configs/experiment1.yaml --checkpoint best.pt
python -m optimization.evaluate_optimization --config configs/experiment1.yaml --n-examples 1000
python -m evaluation.make_readme_charts --run exp1_smoke
```

### Running Experiment 2

```bash
python -m training.generate_dataset_exp2 --n-examples 10000 --seed 42 --out-name exp2_smoke
python -m training.train_exp2 --config configs/experiment2.yaml
python -m training.evaluate_exp2 --config configs/experiment2.yaml --checkpoint best.pt
python -m optimization.evaluate_optimization_exp2 --config configs/experiment2.yaml --n-examples 1000 --tag pseed
# amp search vs. fixed choices vs. the true-choice ceiling, all at 1800 renders per example
python -m optimization.evaluate_optimization_exp2 --config configs/experiment2.yaml --mode search --max-evals 1800 --n-examples 200 --tag b1800_pseed
python -m optimization.evaluate_optimization_exp2 --config configs/experiment2.yaml --mode fixed --max-evals 1800 --n-examples 200 --tag b1800_pseed
python -m optimization.evaluate_optimization_exp2 --config configs/experiment2.yaml --mode oracle --max-evals 1800 --n-examples 200 --tag b1800_pseed
python -m evaluation.make_readme_charts_exp2 --run exp2_smoke
```

### Running the data-scaling runs

```bash
# 1M-example train-only pool, about 12 GB of features, resumable (finished shards are skipped)
python -m training.generate_shards_exp2 --n-examples 1000000 --out-dir data/generated/exp2_pool --no-audio --workers 2
# one run per size: 8k, 32k, 100k, 300k, 1m (and 100k_cosine for the schedule test); --resume continues a killed run
python -m training.train_exp2 --config configs/experiment2_pool300k.yaml --resume
python -m training.evaluate_exp2 --config configs/experiment2_pool300k.yaml --checkpoint best.pt --n-plots 0
python -m optimization.evaluate_optimization_exp2 --config configs/experiment2_pool300k.yaml --n-examples 1000 --tag pseed
python -m evaluation.make_readme_charts_scaling
```

### Running Experiment 3

```bash
# fixed 1,000-example test and validation sets (with audio), then the 1M training pool (about 12 GB)
python -m training.generate_exp3 --split test --n-examples 1000 --keep-audio
python -m training.generate_exp3 --split val --n-examples 1000 --keep-audio
python -m training.generate_exp3 --split train --n-examples 1000000 --workers 3
python -m training.train_exp3 --config configs/experiment3_1m.yaml --resume
python -m training.evaluate_exp3 --config configs/experiment3_1m.yaml
python -m evaluation.identifiability_exp3 --run exp3_1m
python -m optimization.evaluate_optimization_exp3 --config configs/experiment3_1m.yaml --mode fixed --n-examples 1000
python -m optimization.evaluate_optimization_exp3 --config configs/experiment3_1m.yaml --mode oracle --n-examples 1000
python -m optimization.evaluate_optimization_exp3 --config configs/experiment3_1m.yaml --mode fixed --n-examples 200 --max-evals 1800 --tag b1800
python -m optimization.evaluate_optimization_exp3 --config configs/experiment3_1m.yaml --mode oracle --n-examples 200 --max-evals 1800 --tag b1800
# search schedule picked on validation (see the table above), then run once on the test set
python -m optimization.evaluate_optimization_exp3 --config configs/experiment3_1m.yaml --mode search --vary amp,comp_on --split val --n-examples 100 --max-evals 1800 --n-candidates 2 --screen-evals 300
python -m optimization.evaluate_optimization_exp3 --config configs/experiment3_1m.yaml --mode search --vary amp,comp_on --n-examples 200 --max-evals 1800 --n-candidates 2 --screen-evals 300 --tag b1800
python -m evaluation.make_readme_charts_exp3 --run exp3_1m
```

`configs/experiment3_300k.yaml` is the run without BatchNorm recalibration, kept as the evidence for that fix; `experiment3_300k_bnrecal.yaml` is the same run with it.

The optimization evaluators and trainers resume from their last checkpoint if interrupted (`*_partial.json`, `--resume`), so a long run can simply be restarted with the same command.

### Running Experiment 4

```bash
python -m training.generate_dataset_exp4 --n-examples 3000 --out-name exp4_smoke
python -m training.train_exp4 --config configs/experiment4.yaml
```

---

## License

Tone Engine 7 is licensed under [Apache 2.0](LICENSE).
