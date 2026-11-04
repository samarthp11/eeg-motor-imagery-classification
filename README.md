# Machine Learning for Left and Right Motor Imagery EEG Classification

> An auditable EEG classification study using the public BCI Competition IV Dataset 2b and locally recorded Qneuro H100 data, with classical machine learning, EEGNet, transfer learning, statistical testing, and signal-quality diagnostics.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EEGNet-EE4C2C?logo=pytorch&logoColor=white)

## Overview

A brain-computer interface (BCI) converts patterns in brain activity into commands for a computer or another device. In a motor-imagery BCI, a participant imagines moving a body part without physically performing the movement. This project studies a two-class task: identifying whether a participant imagined moving the **left hand** or the **right hand** from a short electroencephalography (EEG) recording.

The project has two main goals:

1. Build a transparent and reproducible machine-learning pipeline and validate it on an established public EEG benchmark.
2. Test whether the same methods and representations transfer to recordings collected using a lower-cost Qneuro H100 headset.

The notebook compares spectral, spatial, covariance-based, and deep-learning methods. It also includes negative controls, participant-level and group-level statistical tests, calibration learning curves, cross-person evaluation, physiological checks, and an audit of the exported H100 signal.

The key lesson is that **classification accuracy alone is insufficient evidence of genuine motor-imagery decoding**. The H100 exports contain patterns that can separate left and right trials, but several signal-quality checks raise questions about the physiological origin of those patterns.

## Project Questions

The analysis addresses the following questions:

- Can a common analysis pipeline recover left/right motor-imagery information from a recognised public benchmark?
- Which classical model or compact neural network performs best under held-out-session and held-out-person evaluation?
- Can left/right information be decoded from the Qneuro H100 recordings when chronological blocks are held out?
- At what point in the four-second imagery window does class information appear?
- Can a model trained on public EEG reduce the number of labelled H100 calibration trials?
- Does a model trained on eight H100 participants generalise to the ninth participant?
- Are the H100 signals physiologically consistent with expected motor-imagery patterns?
- Could trial ordering, pre-cue drift, signal export, filtering, gain, or channel mapping explain part of the observed performance?

## Datasets

### 1. BCI Competition IV Dataset 2b

The public benchmark is loaded through [MOABB](https://moabb.neurotechx.com/) using `BNCI2014_004`.

| Property | Value |
|---|---:|
| Participants | 9 |
| Trials | 6,520 |
| Left trials | 3,260 |
| Right trials | 3,260 |
| EEG channels used | C3, Cz, C4 |
| Sessions per participant | 5 |
| Sampling rate used | 250 Hz |
| Epoch duration | 4 seconds |
| Epoch shape | 3 channels × 1,000 samples |

For cross-session evaluation, sessions `0train`, `1train`, and `2train` are used for training, while `3test` and `4test` are held out for testing.

### 2. Qneuro H100 Recordings

The local dataset contains continuous filtered EEG exports from nine pseudonymous participants, labelled `S01` to `S09`.

| Property | Value |
|---|---:|
| Participants | 9 |
| Accepted trials | 337 |
| Left trials | 169 |
| Right trials | 168 |
| Primary export columns used | F-C3, F-CZ, F-C4 |
| Trials per participant | 20–43 |
| Sampling rate | 250 Hz |
| Analysis window | 0.5–4.5 s after the Start marker |
| Epoch shape | 3 channels × 1,000 samples |

The notebook expects files with names matching:

```text
filtered_eeg_01.csv
filtered_eeg_02.csv
...
filtered_eeg_09.csv
```

Each CSV must contain at least:

```text
PseudoTimer, Event, F-C3, F-CZ, F-C4
```

The actual exports inspected by the notebook also contained:

```text
F-FP1, F-FP2, F-O1, F-O2, F-C3, F-CZ, F-FZ, F-C4
```

Only `F-C3`, `F-CZ`, and `F-C4` were used in the reported three-channel experiments. The full channel-ablation experiment remains optional and runs only after `CH_ALL` is configured with verified EEG column names.

> **Data availability:** The private H100 recordings are not included in this repository. They contain research-participant data and should be published only when the necessary consent and institutional permissions allow it. The public BCI dataset is downloaded by MOABB when the notebook runs.

## Signal Processing

The H100 pipeline performs the following steps:

1. Read continuous EEG and event markers from each CSV export.
2. Parse left/right `Start` and `Stop` event pairs.
3. Reject incomplete, mismatched, too-short, or invalid trials.
4. Reconstruct sample timing using the 250 Hz row spacing after checking the exported timer.
5. Apply a fourth-order, zero-phase Butterworth band-pass filter from 8 to 30 Hz.
6. Extract the interval from 0.5 to 4.5 seconds after each Start marker.
7. Produce a trial tensor with shape `(n_trials, 3, 1000)`.

The loader also audits:

- accepted and rejected markers;
- left/right class counts;
- recording duration;
- timer discrepancy;
- number of class switches; and
- longest run of identical cues.

The cue-sequence audit helps detect a strongly blocked experimental design in which slow recording drift could mimic the class labels.

## Models

Seven main pipelines are evaluated.

| Model | Input representation | Classifier or learned model | Purpose |
|---|---|---|---|
| Bandpower + LDA | 15 standardised log-bandpower features | Shrinkage LDA | Simple and interpretable linear baseline |
| Bandpower + RBF SVM | Same 15 features | RBF support-vector machine | Tests a nonlinear decision boundary |
| Bandpower + Extra Trees | Same 15 features | 200-tree ensemble | Flexible nonlinear comparison |
| CSP + LDA | Spatially filtered EEG variance | Shrinkage LDA | Established motor-imagery spatial baseline |
| Filter-bank CSP + LDA | CSP features from five frequency bands | Shrinkage LDA | Tests frequency-specific spatial patterns |
| Riemannian + LR | OAS covariance matrices mapped to tangent space | Logistic regression | Models relationships between EEG channels |
| EEGNet | Scaled 3 × 1,000 EEG epoch | Compact convolutional neural network | Learns temporal and spatial filters from data |

### Bandpower Features

Welch's method estimates signal power in five bands:

```text
8–12 Hz
12–16 Hz
16–20 Hz
20–24 Hz
24–30 Hz
```

Five bands across three channels produce **15 log-bandpower features per trial**.

### EEGNet Configuration

The notebook implements a compact EEGNet-style architecture with:

- temporal convolution;
- depthwise spatial convolution;
- separable convolution;
- dropout;
- a final linear classifier;
- Adam optimisation;
- learning rate `0.001` for training from scratch;
- learning rate `0.0003` for fine-tuning;
- batch size `32`;
- 40 epochs in the full run; and
- time-shift and Gaussian-noise augmentation.

## Evaluation Design

### Public Cross-Session Evaluation

A separate model is evaluated for each public participant:

```text
Train: sessions 0, 1, 2
Test:  sessions 3, 4
```

This measures generalisation to later sessions from a participant already represented in training.

### Public Leave-One-Subject-Out Evaluation

Models train on eight public participants and test on the ninth. The process repeats until every participant has served as the test participant.

This protocol measures generalisation to a person with no labelled training trials.

### H100 Blocked Cross-Validation

Trials remain in chronological order and are divided into five contiguous blocks. Each block serves as the test block once. The four remaining blocks form the training set, excluding one trial next to each side of the test block.

```text
Fold 1: TEST | train | train | train | train
Fold 2: train | TEST | train | train | train
Fold 3: train | train | TEST | train | train
Fold 4: train | train | train | TEST | train
Fold 5: train | train | train | train | TEST
```

Pooling the five test blocks gives one out-of-block prediction for each accepted trial. This is more informative than randomly mixing nearby EEG trials between training and testing, although it still evaluates different time periods within the same recording.

### H100 Leave-One-Participant-Out Evaluation

The classical pipelines train on eight H100 participants and test on the ninth. The primary methods do not use labelled data from the held-out participant.

The separately reported aligned Riemannian method uses the held-out person's **unlabelled** covariance distribution. It is therefore an unsupervised transductive result rather than a strict zero-adaptation result.

### Primary Metric

The primary metric is balanced accuracy:

```text
Balanced accuracy = (left recall + right recall) / 2
```

Balanced accuracy gives equal weight to both classes. A chance-level classifier is expected to score approximately 50% in this two-class setting.

## Experiments

The notebook is organised into the following experiments:

| Experiment | Description |
|---|---|
| A | H100 blocked cross-validation, participant significance, permutation testing, model comparison, and negative controls |
| B | ERD/ERS and lateralisation physiology check |
| C | Time-resolved H100 decoding and information per trial |
| D | Public benchmark comparison of classical models and EEGNet |
| E | Public-to-H100 transfer learning and calibration curves |
| F | Optional three-channel versus all-channel ablation |
| G | H100 leave-one-participant-out evaluation |
| H | Signal comparison inside and outside marked trial windows |
| I | Baseline-free C3/C4 power lateralisation |
| J | Pooled confusion matrix and per-class recall |

## Main Results

### Public Benchmark

| Evaluation | Best model | Balanced accuracy, mean ± SD |
|---|---|---:|
| Held-out sessions | EEGNet | **79.2% ± 16.9%** |
| Leave one subject out | EEGNet | **71.3% ± 8.6%** |

EEGNet did not significantly outperform the best classical pipeline:

- Cross-session: EEGNet exceeded FBCSP + LDA by 2.2 points, `p = 0.238`.
- Leave-one-subject-out: EEGNet exceeded Bandpower + LDA by 0.9 points, `p = 0.496`.

These results validate that the implementation can recover motor-imagery information from an established benchmark under held-out evaluation.

### H100 Blocked Cross-Validation

| Rank | Model | Balanced accuracy, mean ± SD |
|---:|---|---:|
| 1 | Bandpower + LDA | **68.2% ± 6.5%** |
| 2 | Filter-bank CSP + LDA | 61.2% ± 6.6% |
| 3 | Bandpower + Extra Trees | 59.4% ± 9.1% |
| 4 | Bandpower + RBF SVM | 58.1% ± 12.5% |
| 5 | Riemannian + LR | 54.1% ± 10.2% |
| 6 | CSP + LDA | 49.8% ± 10.5% |

For Bandpower + LDA:

- participant scores ranged from 59.5% to 80.0%;
- the group permutation test used 200 circular label shifts;
- the permutation null mean was 48.3%;
- the 95th percentile of the null distribution was 54.8%; and
- the observed group result produced `p = 0.005`.

After correcting nine participant-level tests, only `S03` and `S06` were individually significant. The group result therefore does not imply reliable decoding for every participant.

### Negative Controls

| Control | Mean balanced accuracy |
|---|---:|
| Trial order and timing only | 43.1% |
| Pre-cue EEG only | 52.1% |
| Task-window Bandpower + LDA | 68.2% |

The controls do not explain the main score through cue order or pre-cue drift. They do not rule out artifacts or processing changes that occur inside the marked task window.

### Time-Resolved Decoding

Bandpower + LDA was evaluated in one-second windows moving through the four-second epoch in steps of 0.25 seconds.

- The group curve peaked at **69.7%** near 3.23 seconds after the Start marker.
- The peak corresponded to approximately **0.12 bits per trial** using the Wolpaw metric.

This information rate remains too low for a practical communication BCI.

### Public-to-H100 Transfer Learning

| Method | k = 0 | k = 5 | k = 10 | k = 20 | k = all |
|---|---:|---:|---:|---:|---:|
| H100-only Bandpower + LDA | — | 57.7 | 63.9 | 65.4 | **68.2** |
| H100-only EEGNet | — | 56.9 | 59.1 | 59.4 | 59.4 |
| Public EEGNet, zero-shot | 49.9 | — | — | — | — |
| Public EEGNet + fine-tuning | — | 55.3 | 53.7 | 54.9 | 55.1 |
| Public Riemannian, no alignment | 43.5 | — | — | — | — |
| Public Riemannian with alignment | 40.4 | — | — | — | — |
| Riemannian alignment with H100 labels | — | 39.0 | 38.1 | 41.7 | 40.9 |

`k` is the number of labelled H100 calibration trials selected from the training blocks.

The tested public pretraining and alignment methods did not improve H100 decoding. Direct H100 training with Bandpower + LDA performed best at every labelled calibration size.

### H100 Cross-Person Generalisation

| Rank | Model | Balanced accuracy, mean ± SD |
|---:|---|---:|
| 1 | Bandpower + RBF SVM | **73.2% ± 4.9%** |
| 2 | Bandpower + LDA | **72.6% ± 5.3%** |
| 3 | Filter-bank CSP + LDA | 70.0% ± 5.9% |
| 4 | Bandpower + Extra Trees | 69.6% ± 4.8% |
| 5 | Riemannian + LR with unlabelled target alignment | 60.4% ± 4.9% |
| 6 | Riemannian + LR | 59.0% ± 6.9% |
| 7 | CSP + LDA | 56.6% ± 8.5% |

The H100 recordings contain a left/right pattern shared across participants. This result does not establish that the shared pattern originates from motor-cortex activity; a device or export-related pattern could also generalise across users.

### Pooled Confusion Matrix

| Actual / Predicted | Left | Right |
|---|---:|---:|
| Left | 114 | 55 |
| Right | 53 | 115 |

- Left recall: 67.5%
- Right recall: 68.5%
- Pooled balanced accuracy: 68.0%

The similar recalls show that the model did not strongly favour one class.

## Signal-Quality Findings

The notebook includes several checks intended to prevent overinterpretation of accuracy.

### ERD/ERS Check

Expected motor-imagery physiology involves class-dependent changes in mu and beta power over motor-related electrodes. The H100 analysis instead found:

- unusually large baseline-relative changes;
- beta-power drops approaching approximately −97%;
- rectangular changes closely aligned with marked task windows; and
- an unexpected direction of mu-band lateralisation.

### Baseline-Free Lateralisation

The notebook compares within-epoch C3/C4 power ratios, avoiding dependence on the pre-cue baseline.

- All 9 participants showed a positive right-minus-left mu lateralisation difference.
- The expected direction under the assumed label and channel mapping was negative.
- The median mu difference was `+1.28` in log-ratio units.

### Inside-versus-Outside Audit

The exported signal inside the marked analysis windows was compared with the rest of each recording.

- Beta power was lower inside the marked window in 25 of 27 participant-channel combinations.
- Several changes spanned two to three orders of magnitude.
- Channels did not show identical inside/outside behaviour.

These checks identify a serious interpretation issue but do not identify its cause. Possible factors include export processing, filtering, gain changes, an unsuitable baseline period, or incorrect physical channel mapping. The available filtered exports do not allow these possibilities to be separated conclusively.

## Interpretation

The project supports the following conclusions:

1. The analysis pipeline works on the established public benchmark.
2. Left/right labels are decodable from the supplied H100 exports above chance at the group level.
3. The simplest H100 model, Bandpower + LDA, outperformed more flexible classical approaches under blocked cross-validation.
4. Public-to-H100 transfer did not help under the methods tested.
5. A left/right pattern generalised across H100 participants.
6. The H100 signal audit found non-physiological and task-window-linked characteristics.
7. The current evidence does not establish a reliable real-time, clinical, or motor-cortex BCI.

## Repository Structure

Recommended repository layout:

```text
eeg-motor-imagery-classification/
├── README.md
├── EEG_Motor_Imagery_Classification.ipynb
├── requirements.txt                 # optional but recommended
├── results/                         # optional selected outputs
│   ├── figures/
│   └── tables/
└── LICENSE                          # add only after choosing a licence
```

The private H100 CSV files should remain outside the public repository unless their release is explicitly permitted.

## Running the Notebook

### Recommended: Kaggle

The notebook was designed to run on Kaggle and can use a GPU for EEGNet.

1. Create a new Kaggle notebook.
2. Upload `EEG_Motor_Imagery_Classification.ipynb`.
3. Open **Notebook options**.
4. Enable **Internet** so MOABB can download the public benchmark.
5. Select a GPU accelerator if available.
6. Add the H100 CSV dataset privately as notebook input.
7. Confirm the configuration cell points to the input directory:

```python
H100_DIR = '/kaggle/input'
H100_GLOB = '**/filtered_eeg_*.csv'
```

8. Keep `SMOKE_TEST = False` for reportable results.
9. Run all cells from top to bottom.
10. Download the generated result ZIP from the Kaggle Output panel.

### Local Jupyter Environment

Clone the repository:

```bash
git clone https://github.com/samarthp11/eeg-motor-imagery-classification.git
cd eeg-motor-imagery-classification
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the main dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install jupyter numpy pandas scipy matplotlib scikit-learn mne moabb pyriemann statsmodels torch
```

Place authorised H100 data outside the Git repository, then update the configuration cell:

```python
H100_DIR = '/path/to/private/h100/data'
```

Start Jupyter:

```bash
jupyter notebook
```

Open `EEG_Motor_Imagery_Classification.ipynb` and run the cells in order.

## Important Configuration Options

| Option | Default | Meaning |
|---|---|---|
| `SMOKE_TEST` | `False` | Uses simulated data only for checking that code runs |
| `FAST_MODE` | Same as `SMOKE_TEST` | Reduces permutations, seeds, and epochs for a quick test |
| `H100_DIR` | Kaggle input or local `upload` | Directory containing H100 CSV files |
| `CH3` | `F-C3, F-CZ, F-C4` | Three analysis channels |
| `CH_ALL` | Empty | Optional verified list for channel ablation |
| `FS` | `250` | Sampling rate in Hz |
| `TMIN`, `TMAX` | `0.5, 4.5` | Analysis interval after Start marker |
| `BAND` | `8, 30` | Main band-pass range in Hz |
| `K_FOLDS` | `5` | Number of H100 chronological blocks |
| `GAP` | `1` | Adjacent trials excluded around a test block |
| `N_PERM` | `200` | Group permutation repetitions |
| `SEEDS` | `0, 1, 2` | EEGNet random seeds in the full run |
| `EEG_EPOCHS_PRE` | `40` | Public pretraining epochs |
| `EEG_EPOCHS_FT` | `40` | H100 scratch/fine-tuning epochs |
| `CALIB_SIZES` | `0, 5, 10, 20, all` | H100 calibration sizes |

Never report results generated with `SMOKE_TEST = True`. Smoke-test outputs are based on simulated data and are written to a directory marked `SMOKE_TEST_DO_NOT_REPORT`.

## Generated Outputs

The notebook writes figures and CSV summaries to `eeg_results/` locally or `/kaggle/working/eeg_results/` on Kaggle. Major outputs include:

```text
h100_audit.csv
A_h100_blockedcv_per_participant.csv
A_h100_blockedcv_summary.csv
A_h100_blockedcv.png
A_per_participant_significance.csv
A_permutation_test.png
A_model_comparison.csv
A_negative_controls.csv
B_erd_lateralisation.csv
B_lateralisation_stats.csv
B_erd_timecourse.png
C_time_resolved.csv
C_time_resolved.png
D_public_per_subject.csv
D_public_summary.csv
D_public_models.png
E_transfer_per_participant.csv
E_transfer_summary_mean_BA.csv
E_transfer_summary_long.csv
E_transfer_learning_curve.png
E_transfer_paired_tests.csv
F_channel_ablation.csv                 # only when CH_ALL is configured
G_h100_leave_one_participant_out.csv
G_h100_lopo_summary.csv
G_h100_leave_one_participant_out.png
H_inside_vs_outside_window.csv
H_inside_vs_outside_window.png
I_lateralisation_baseline_free.csv
I_lateralisation_baseline_free.png
J_confusion_matrix.png
```

At the end of a run, the output directory is packaged into a timestamped ZIP archive.

## Reproducibility and Statistical Notes

- All preprocessing and fitted transformations are learned from training data only.
- H100 trials remain in chronological order during blocked evaluation.
- The H100 test block is excluded from calibration, fine-tuning, alignment fitting, and classifier fitting unless a method is explicitly identified as transductive.
- The permutation test preserves label order structure through participant-wise circular shifts while breaking alignment between EEG and class.
- Participant-level binomial tests assume independent trials, which is only approximate for EEG data.
- Multiple model inspection makes uncorrected statistical tests exploratory.
- Holm correction is used for specified families of multiple comparisons.
- Between-participant standard deviation describes variability across people; it is not the uncertainty of a single participant's accuracy.
- The public and H100 protocols use different data sources and splits, so their percentages should not be treated as a direct device comparison.

## Limitations

- Only nine H100 participants were included.
- Each participant contributed one recording and 20–43 accepted trials.
- Only three of eight exported EEG channels were used in the reported results.
- The physical mapping of the H100 export fields to electrode positions was not independently verified.
- The internal filtering and processing applied before export were unknown.
- No dedicated ocular or muscle-artifact rejection pipeline was applied.
- H100 blocked cross-validation tests different blocks from the same recording, not a different day.
- Only two participants were individually significant after correction.
- The physiological and signal-window audits produced unresolved anomalies.
- The study evaluates offline classification and does not demonstrate real-time BCI control or clinical use.

## Future Work

- Obtain raw, minimally processed H100 signals.
- Verify the physical electrode layout and export-channel mapping.
- Run the planned full-channel ablation using all verified EEG channels.
- Collect more trials from more participants.
- Repeat recordings across multiple days and sessions.
- Add explicit eye, muscle, and movement-artifact monitoring.
- Investigate why signal statistics change around marked task windows.
- Evaluate real-time latency and information-transfer rate only after resolving signal provenance.
- Test transfer methods after confirming compatible channel locations and preprocessing across devices.

## Ethical and Data-Handling Statement

The H100 participants are represented only by pseudonymous IDs (`S01`–`S09`). Do not commit participant names, contact information, consent documents, private CSV recordings, or identifying photographs to a public repository. Keep restricted data in an access-controlled location and follow the applicable consent form and institutional requirements.

## Authors

- **Vishnuprasad Pai K** — 230907268
- **Ashutosh S Maakam** — 230907458
- **Samarth Pal** — 230907022

Project guide: **Dr. Ananthakrishna T**  
School of Electrical Engineering  
Manipal Institute of Technology, Manipal

## References

The notebook and final report build on the following key work:

1. Tangermann, M. et al. (2012). Review of the BCI Competition IV. *Frontiers in Neuroscience*, 6, 55.
2. Jayaram, V. and Barachant, A. (2018). MOABB: Trustworthy algorithm benchmarking for BCIs. *Journal of Neural Engineering*, 15, 066011.
3. Pfurtscheller, G. and Lopes da Silva, F. H. (1999). Event-related EEG/MEG synchronization and desynchronization. *Clinical Neurophysiology*, 110, 1842–1857.
4. Welch, P. D. (1967). The use of fast Fourier transform for the estimation of power spectra. *IEEE Transactions on Audio and Electroacoustics*, 15, 70–73.
5. Ramoser, H., Müller-Gerking, J., and Pfurtscheller, G. (2000). Optimal spatial filtering of single-trial EEG during imagined hand movement. *IEEE Transactions on Rehabilitation Engineering*, 8, 441–446.
6. Ang, K. K. et al. (2012). Filter bank common spatial pattern algorithm on BCI Competition IV datasets 2a and 2b. *Frontiers in Neuroscience*, 6, 39.
7. Barachant, A. et al. (2012). Multiclass brain-computer interface classification by Riemannian geometry. *IEEE Transactions on Biomedical Engineering*, 59, 920–928.
8. Lawhern, V. J. et al. (2018). EEGNet: A compact convolutional neural network for EEG-based brain-computer interfaces. *Journal of Neural Engineering*, 15, 056013.
9. Zanini, P. et al. (2018). Transfer learning: A Riemannian geometry framework with applications to BCIs. *IEEE Transactions on Biomedical Engineering*, 65, 1107–1116.
10. Combrisson, E. and Jerbi, K. (2015). Exceeding chance level by chance. *Journal of Neuroscience Methods*, 250, 126–136.

## Citation

If you use or adapt this repository, cite the project as:

```text
Pai, V. P. K., Maakam, A. S., and Pal, S. (2026).
Machine Learning for Left and Right Motor Imagery EEG Classification:
Public Benchmark, H100 Recordings and Transfer Analysis.
Minor Specialization Project in Computational Intelligence,
Manipal Institute of Technology.
```

## Licence

No licence has been assigned by default. The source code remains protected by copyright until the project authors choose and add a licence. Before adding an open-source licence, confirm that all contributors agree and that third-party data and institutional requirements permit the intended use.

---

This repository presents an exploratory feasibility study. It does not provide a medical device, diagnosis, clinical recommendation, or evidence of reliable real-time BCI control.
