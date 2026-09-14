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

**First milestone: proven on a single virtual amp, including the optimization loop.** Stage 2 (signal chains) is in progress: an overdrive, amp, and cabinet chain is working, the optimizer can change which amp is in the chain as well as its knobs, and training on up to a million synthetic examples lifted the CNN's amp identification from 58% to 82% (see Experiment 2). Compression, a noise gate, mic modeling, real recordings, and song input are still ahead.

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

## Repository Layout

```text
audio/
    synth/         synthetic DI guitar source generation
    rendering/     Experiment 1's single virtual amp, plus Experiment 2's
                    overdrive/amp/cabinet signal chain (amp_models.py,
                    cabinets.py, chain_params.py, signal_chain.py)
    preprocessing/ wav I/O and normalization
    features/      log-mel spectrogram extraction
    similarity/    multi-resolution STFT audio similarity

models/
    tone_predictor/ audio -> parameters CNN (model.py) and the multi-head
                    audio -> signal chain predictor (chain_model.py)
    checkpoints/    trained weights (not versioned)

training/
    generate_dataset.py, generate_dataset_exp2.py   build a dataset from the renderer
    generate_shards_exp2.py                          build a large train-only pool as feature shards
    dataset.py, dataset_exp2.py, pool_exp2.py        torch Datasets / shard-pool batches
    train.py, train_exp2.py                          training loops
    evaluate.py, evaluate_exp2.py                    held-out test set milestone reports

evaluation/     shared metrics, comparison plots, and README chart generation
                (make_readme_charts.py for Experiment 1, _exp2 for Experiment 2,
                _scaling for the training-data chart)
optimization/   CMA-ES refinement of the CNN's predicted parameters
                (evaluate_optimization.py / _exp2.py); for Experiment 2 the
                discrete choices can be held fixed, searched (every amp is
                tried, the best one refined), or set to the true ones as a
                ceiling (--mode fixed / search / oracle)
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

The optimization evaluator resumes from its last checkpoint (`*_partial.json`) if it is interrupted, so a long run can simply be restarted with the same command.

---

## License

Tone Engine 7 is licensed under [Apache 2.0](LICENSE).
