# PET-scanner-adjusted four-model comparison

This separately invoked extension reproduces the completed scanner-adjusted
comparison. It does not replace the original C/CP/CPT analysis in the parent
STEP07 folder. It uses ordinary Python; there is no assistant service, scheduler,
automatic next step or automatic restart.

## Inputs and environment

Use the package's tested numerical environment in `environment/`. The numerical
fit runs single-threaded. No raw imaging is read or reprocessed by this extension.
Run ordinary Python, without `-O` or `-OO`; optimized execution is rejected so
the source implementation's numerical assertions cannot be silently disabled.

1. The private STEP05 `model_input.json`: ordered `study_ids`, numeric binary `y`,
   predictor `x`, original C/CP/CPT `columns`, and complete saved `plans`. The
   original model specification and outer split plan are validated before use.
2. A private CSV with exactly one row for each selected `study_id` and a `scanner`
   value of **Biograph HiRes** or **Biograph Vision 600**. Optional
   `scanner_indicator` must be 0 for HiRes and 1 for Vision 600. Other source
   columns are not copied into the prepared input. Join order is determined by
   study IDs, not the CSV row order. Never derive scanner membership from outcome.

The study used 57 people, 32 responders and 25 nonresponders, and 100 outer
partitions (20 repeats of 5 folds). There is no hardcoded participant count in
the portable runner. Replaying the study still requires its exact authorized
inputs and saved outer partitions. Applying this code to another cohort is not
an external validation of VNS response prediction by itself.

## Manual commands

Run from the code-package root. Replace `/private/...` with your own paths.
The work and summary directories must be new and outside the public code folder.

```sh
python STEP07_prediction_models/scanner_adjusted_four_models/run_four_models.py prepare --input /private/step05/model_input.json --scanner-metadata /private/scanner_metadata.csv --output /private/scanner_work

python STEP07_prediction_models/scanner_adjusted_four_models/run_four_models.py fit --workdir /private/scanner_work --model C
```

The `fit` command without `--execute` only prints the single-model plan. Invoke
each model explicitly when ready; none starts another model or the summary.

```sh
python STEP07_prediction_models/scanner_adjusted_four_models/run_four_models.py fit --workdir /private/scanner_work --model C --execute
python STEP07_prediction_models/scanner_adjusted_four_models/run_four_models.py fit --workdir /private/scanner_work --model CT --execute
python STEP07_prediction_models/scanner_adjusted_four_models/run_four_models.py fit --workdir /private/scanner_work --model CP --execute
python STEP07_prediction_models/scanner_adjusted_four_models/run_four_models.py fit --workdir /private/scanner_work --model CPT --execute

python STEP07_prediction_models/scanner_adjusted_four_models/run_four_models.py summarize --workdir /private/scanner_work --output /private/scanner_summary
```

The default runtime limit is 10,800 seconds for each named model. A failure stops
that model and retains its status, diagnostics and completed fold files. Existing
model folders are never silently resumed or overwritten. A summary is allowed
only when all four models have completed with matching frozen inputs and folds.
Changing inputs, code or a prepared fold record is detected; it is not repaired
by choosing another partition or solver.

## Fixed analysis

| Model | Predictors in this extension |
| --- | --- |
| C | Clinical five plus PET scanner |
| CT | C plus six corresponding T1 measures |
| CP | C plus six regional PET measures |
| CPT | C plus six PET and six T1 measures |

The five clinical predictors are age at implantation, sex, reported epilepsy
duration, log1p of the baseline monthly average seizure frequency, and MRI
lesional status. HiRes is the scanner reference category. The intercept and
scanner contrast have penalty factor 0; all remaining slopes have penalty factor
1. Penalty factors are **not renormalized**.

For n training observations, the objective is mean binary logistic loss plus
`l1_ratio/(C*n) * sum(abs(penalized slopes))` plus
`(1-l1_ratio)/(2*C*n) * sum(penalized slopes squared)`.

The original saved outer training/test lists and inner seeds are unchanged.
Within each outer training set, inner three-fold assignment is constructed once
using `StratifiedKFold(n_splits=3, shuffle=True, random_state=original inner_seed)`
with joint strata `2*scanner_indicator + outcome`. Thus the inner assignment is
not the original response-only inner assignment. It is shared by all four
models, with no performance-based fold selection. At least three observations
per joint cell are required in every outer training set, and both response
classes must occur within each scanner group in every fitting partition.

Imputation, continuous scaling and removal of constant predictors are fitted
inside the respective training fold. All 30 grid combinations are evaluated:
`C = [0.001, 0.01, 0.1, 1, 10, 100]` and
`l1_ratio = [0, 0.25, 0.5, 0.75, 1]`. Selection minimizes mean inner Brier score;
ties within 1e-8 favor smaller C, then smaller l1_ratio.

`scanner_solver.py` is the unchanged numerical implementation used in the
completed run. Its active target KKT residual is 1e-11; acceptance requires KKT
residual <=1e-8 and mean duality gap <=1e-10. The selected fit must repeat exactly
and agree with the independent SLSQP objective within 1e-10 and probabilities
within 1e-5. Active solver limits are defined in that file (`MAX_LBFGS=5000`,
`MAX_POLISH=80`), not the legacy tolerance/iteration fields retained in the
shared settings record. Those legacy fields are not passed into this solver.

## Outputs and interpretation

Preparation saves reduced private inputs, original-to-joint split fingerprints,
protocol, training cell checks and code/input hashes. Fits save one private
record per outer split, including preprocessing rows, penalty masks, all
candidate scores and numerical verification. These are participant-level
research records and must not be uploaded to the public repository.

The summary produces `performance.json`, `conditional_intervals.json`,
`paired_comparisons.json`, private individual predictions, and private numerical
QA. AUC, Brier, calibration and decision curves use each person's **average of
20 held-out predictions**; repeat-specific AUC and Brier are also retained.
They are not the known performance of one deployed final model on new patients.

The primary paired comparison is CT to CPT. C to CP, CP to CPT and C to CT are
descriptive comparisons. Bootstrap uses 2,000 shared participant draws,
seed 20260910, and fixed predictions without refitting; reported percentile
intervals are conditional descriptive intervals, not full model-development
uncertainty. The source runner's bootstrap arithmetic is retained exactly.

## Synthetic tests

```sh
python -m unittest discover -s tests -p test_scanner_four_models.py -v
```

Tests use generated data only: no study participant is embedded. They check the
solver across the penalty grid, unchanged all-penalized behavior, nuisance
separation rejection, ID joins, frozen outer assignments, scanner/outcome cells,
zero scanner penalty, opt-in fitting, preserved failures and output boundaries.
Numerical tests and saved-result replay do not constitute clinical image QC or
a new full analysis of raw images.

中文简述：这是独立调用的PET设备调整四模型扩展，保留既有外层划分，仅按已完成分析
固定的设备×结局规则构造内层三折。先prepare，再分别手动fit四个模型，最后summarize。
所有真实输入、逐折记录和个体预测都应保存在仓库之外的受控目录；不自动续跑或修改参数。
