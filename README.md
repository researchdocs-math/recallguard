# RecallGuard

Reproduce **Certifying Recall When Reviewers Err**, by Cemil Emre Yavas and Lei Chen.

RecallGuard certifies reference-relative recall while a review system screens documents with fallible labels, repairs earlier decisions, and draws uniform reference audits from every still-unverified document. The project includes the experiment data, fixed configurations, statistical certificate, review simulator, complete reference trajectories, and numerical analyses.

## Results reproduced

- 6,600 primary workflows: 22 evaluation topics × 10 seeds × 6 first-pass error conditions × 5 policies.
- 1,320 secondary exact hypergeometric audit comparisons.
- 270,000 informative-removal stress paths, with matched CPU double-precision checks.
- Topic-level bootstrap intervals and paired randomization tests.

At the 0.95 recall target and 0.05 risk level, adaptive repair with the shrinking certificate produced six false certifications in 1,320 runs and 99.95% mean reference-relative recall. Its mean effort was 1.132 times exhaustive reference adjudication under the primary cost ratio. The invalid error-blind diagnostic produced 927 false certifications; the distinction between provisional and reference labels is essential.

## Environment

Use Python 3.12 and the pinned dependencies in `requirements.txt`. The reference runs used NumPy 1.26.3, SciPy 1.11.4, scikit-learn 1.4.0, pandas 2.2.1, and PyTorch 2.4.1. Eight CPU workers run independent review simulations; six workers build features. Start with fewer workers if memory is limited. A GPU is optional: the stress validator automatically selects an available Metal backend, otherwise CPU. Its `--device cpu` option explicitly selects CPU execution.

Allow approximately 1 GB for the extracted project and generated feature caches, plus space for fresh outputs. Several gigabytes of available memory are useful for parallel sparse fitting. Runtime depends on worker count, corpus size, and processor throughput. The primary reference runs accumulated 3.10 worker-hours; elapsed parallel time also includes scheduling and feature preparation.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
```

Run all commands from the `recallguard` directory. Thread limits prevent each worker from independently occupying the entire CPU.

## Full reproduction

The normalized public data are included. Feature construction and review simulation run locally.

```sh
python src/features.py --workers 6
python -m unittest discover -s tests -v
python src/experiment.py --config configs/main.json --workers 8 --output results/reproduced --force
python src/experiment.py --config configs/frozen_exact.json --workers 8 --output results/reproduced --force
python src/check_reproduction.py --runs results/reproduced
python src/analyze.py --runs results/reproduced --out results/reproduced_analysis
python src/analyze_secondary.py --runs results/reproduced --out results/reproduced_analysis
python src/validate_martingale.py --paths 30000 --device auto
```

`--force` performs fresh simulations even when an output exists. Omitting it resumes a partially completed experiment by using existing outputs. `check_reproduction.py` verifies all 26 processed-data checksums and compares every field in each complete experiment trace, excluding the machine-dependent elapsed-seconds field. Matching software versions and numeric behavior are needed for exact trace equality. The stress validator uses the same random tape for its CPU/GPU cross-check and asserts identical threshold decisions on the matched paths.

## Quick reproducibility check

This smaller configuration covers three contrasting evaluation topics, all six error mechanisms, and all six workflows using seed 100.

```sh
python src/features.py --workers 3 --topics Bos_2018 Muthu_2021 Walker_2018
python src/experiment.py --config configs/smoke.json --workers 8 --output results/reproduced --force
python src/check_reproduction.py --runs results/reproduced --config configs/smoke.json
```

## Project contents

| Location | Purpose |
|---|---|
| `src/certificates.py` | Shrinking-population betting certificate, PRN prior/posterior ratio, and reference-error sensitivity bound. |
| `src/experiment.py` | Deterministic review simulator, error mechanisms, ranking, repair, random audits, and stopping rules. |
| `src/features.py` | Per-topic sparse TF–IDF construction from title and abstract text. |
| `src/prepare_data.py` | Download source files with checksum verification and reconstruct normalized records. |
| `src/analyze.py` | Primary effort/recall summaries, paired topic bootstrap, randomization tests, and Holm correction. |
| `src/analyze_secondary.py` | Exact-audit summaries and equality checks against primary frozen trajectories. |
| `src/validate_martingale.py` | Vectorized informative-removal stress experiment with CPU/GPU consistency checks. |
| `src/check_reproduction.py` | Processed-data integrity and full-trace comparison against reference outputs. |
| `tests/test_certificates.py` | Algebraic checks, exact small-population path enumeration, and error-budget checks. |
| `configs/main.json` | The primary 6,600-workflow design. |
| `configs/frozen_exact.json` | The secondary 1,320-workflow exact-audit design. |
| `configs/protocol.json` | Primary design, development split, analysis choices, and implementation hashes recorded before the primary runs. |
| `configs/smoke.json` | The 108-workflow quick check. |
| `data/processed/*.jsonl.gz` | Normalized SYNERGY version-1 records. |
| `data/datasets.json` | Per-topic sizes, inclusion counts, missing abstracts, and SHA-256 checksums. |
| `data/source_manifest.json` | Immutable source version, file identifiers, original checksums, and license. |
| `data/citations/` | Original topic and collection attribution. |
| `results/runs/*.json.gz` | Complete reference trajectories and certificate stopping states. |
| `results/analysis/` | Numerical CSV summaries, paired comparisons, and validation reports. |
| `results/martingale_validation.json` | Reference stress counts and numerical agreement measurements. |
| `CITATION.cff` | Machine-readable software citation. |

Each processed JSON line contains `id`, `title`, `abstract`, `year`, and binary `label`. `label=1` means included according to the original review. Missing abstracts are empty strings. Split JSON lines at newline characters; Unicode line-separator characters can occur within quoted text.

Each experiment output identifies its topic, seed, error condition, policy, target, and risk level. `stops` records the first successful certificate checkpoint. `trajectory` records all evaluated cycle boundaries. Call counters distinguish initial screening, uniform reference audits, and targeted reference repairs. `reference_calls` also includes verification of provisional positives and the two supplied seed examples. The `blind` policy is a deliberately invalid diagnostic under screening error and should be interpreted accordingly.

Normalized effort is `(screen_calls + rho * reference_calls) / (rho * N)`, with `rho=4` in the primary analysis. The denominator is exhaustive reference adjudication; values above one are meaningful. Topic averages, rather than individual seeds, are the primary paired statistical units.

## Data provenance and attribution

The underlying source is **SYNERGY, version 1.0**, released by Jonathan de Bruin, Yongchao Ma, Gerbrich Ferdinands, Jelle Teijema, and Rens van de Schoot (2023), DataverseNL, DOI [10.34894/HE6NAQ](https://doi.org/10.34894/HE6NAQ). The source release is licensed CC0 1.0. The full source metadata and topic-specific citations are preserved under `data/`.

All 26 topics are included: four development topics and 22 evaluation topics. Together they contain 169,288 topic–document records and 2,834 reference positives. The 22 evaluation topics contain 114,907 records and 2,070 positives. Review inclusion labels are retrospective reference decisions. Simulated first-pass errors are controlled experimental mechanisms, not measured human error rates.

To reconstruct the normalized files from the source release, run:

```sh
python src/prepare_data.py --workers 6
```

This optional acquisition step downloads the pinned source files, verifies their SHA-1 checksums, joins inclusion labels to OpenAlex records, reconstructs abstracts, checks conflicting duplicate labels, sorts identifiers, and writes deterministic compressed JSON lines.

## License and citation

Original software is released under the MIT license in `LICENSE`. Source data retain their CC0 release terms, described in `data/LICENSE.txt`. Cite the software using `CITATION.cff` and credit SYNERGY when using the data. The two authors share the project credit; correspondence is directed to Cemil Emre Yavas at cemil@fairesresearch.org.
