# PET-VNS-prediction

Research code accompanying **Regional FDG uptake and response to vagus nerve stimulation in drug-resistant epilepsy: Associations and incremental prediction**.

This repository is a portable, source-adapted release of numerical methods and feature definitions from the study. It is not a distribution of the private study workspace, a pretrained model, or a clinical prediction service. Patient data are not included.

## Scope

This initial package focuses on the paper's regional feature definitions, training-fold preprocessing, deterministic elastic-net logistic regression, repeated nested cross-validation, and regional association analysis. It is not yet the complete set of scripts for every manuscript sensitivity analysis and figure. See [Release scope](docs/release_scope.md) and [Reproducibility](docs/reproducibility.md) before using it to support a manuscript code-availability statement.

The clinical-only model is denoted C; CP adds six PET regional composites; CPT additionally includes six morphometric features. The regional features represent anatomical composites, not estimates of individual functional connectivity or mechanistic pathways.

## Installation

Use a separate Python environment, then from this directory run:

```sh
python -m pip install -e ".[imaging]"
python -m unittest discover -s tests -v
python examples/synthetic_demo.py
```

Tests use synthetic inputs. Passing them checks the specified software properties; it does not reproduce the manuscript results or establish external validity. The allowed dependency ranges in `pyproject.toml` are installation constraints, not a claim that all versions in those ranges have been tested.

The demonstration uses one repetition and one penalty setting for a short run. The library's full study defaults remain available from `paper_settings()`. See the [input schema](docs/data_schema.md) before using authorized research data.

`requirements-tested.txt` records the numerical dependency versions used for software checks. These are distinct from the full imaging-processing environment.

## Methods and interpretation

- Preprocessing is learned from the training data within each fold. Held-out observations are not used to fit imputation or scaling.
- The prediction implementation retains the study's deterministic penalized solver; it is not replaced by an unrelated off-the-shelf estimator.
- Model tuning uses the mean inner-validation Brier score. Compared models must use the same folds.
- The study's summary AUC is calculated from each participant's mean held-out probability across repetitions. It is not the measured performance of a single final model in a new external cohort.
- Resampling already computed held-out predictions gives conditional performance intervals. This is distinct from refitting the complete modeling procedure in a development bootstrap.
- Regional associations and predictions answer different questions. Neither establishes a mechanism of VNS action.

## Data access and privacy

The clinical records, images, participant-level feature matrices, outcomes, predictions, and original participant-specific split plans are not publicly distributed. Research data access requires a request to and approval by the corresponding author, subject to applicable institutional and ethical requirements. Do not post patient information in GitHub issues or pull requests.

Synthetic examples are artificial and cannot be used to recover the manuscript's reported estimates. Reproducing those estimates requires authorized access to the frozen study inputs, their ordering and split plan, and the relevant execution environment.

## Citation and license

Use `CITATION.cff` and identify the exact commit or release used. No DOI is assigned by this repository itself. The authors' code is released under the MIT license; see [third-party notices](THIRD_PARTY_NOTICES.md) for external dependencies.
