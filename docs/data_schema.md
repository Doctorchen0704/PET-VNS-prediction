# In-memory analysis inputs

Real data must be supplied by an authorized user. The repository does not download or discover study records.

## Prediction

`run_nested_models(x, y, settings)` expects a pandas DataFrame and Series. Their indices must be unique **nonempty strings** and identical in the same order; use a local, de-identified research identifier. Numeric row indices are rejected before fitting because the training-isolation audit records string identifiers. `y` is explicitly coded 0 (nonresponse) or 1 (response). Missing outcomes are not coded as nonresponse and must not be imputed.

The five clinical columns are:

| Column | Meaning and encoding |
| --- | --- |
| `age_at_implant_years` | Age at implantation, in years |
| `sex` | `female` or `male`; female reference |
| `epilepsy_duration_reported_years` | History-reported duration, in years |
| `log1p_baseline_monthly_frequency` | Natural logarithm of one plus history-reported mean monthly total seizure frequency |
| `mri_lesional_status` | `nonlesional` or `lesional`; nonlesional reference |

The frequency is a monthly average recorded in the clinical history, not necessarily a count from one standardized 30-day observation window. Apply its declared log1p transformation once, before supplying this specifically named column. Do not round an average to an integer to fit a table format.

Get the exact imaging column names with:

```python
from pet_vns_prediction import paper_settings
settings = paper_settings()
print(settings.pet)
print(settings.t1)
print(settings.model_columns())
```

PET columns contain the six supratentorial-gray-matter-normalized regional composites. T1 columns contain the six declared morphometric features, not all MRI contrasts and not identically defined PET regions. See `configs/features.json` and `docs/preprocessing.md` for definitions.

Fold preprocessing handles declared missing predictors within the training folds, rejects unsupported categories, and does not silently exclude outliers. Review input quality before calling the numerical interface. Returned predictions, fold membership and audit objects are participant-level information and must be stored privately when real data are used.

## Associations

`association_family` accepts an n-by-6 regional matrix, an n-vector of 0/1 response labels, and a mapping of named numeric covariate designs. Pass covariates only: the function adds the intercept and response column. Encode categorical references explicitly and do not supply redundant dummy columns.

The study's primary association design adjusted for age, sex, epilepsy duration, log1p monthly seizure frequency, MRI lesion status **and PET scanner**. Scanner adjustment here must not be confused with the five clinical predictors used by the primary prediction model. The generic association function does not infer these variables from column position: the caller must supply the complete declared design and record its column names. The demo uses an artificial scanner indicator; it is not observed scanner data.

Declare the entire number of tests with `expected_tests`; do not filter to favorable regions before adjustment. One adjustment design produces the six-test primary family, while three designs jointly produce an 18-test family. Region-specific morphometric adjustments and the complete processing-sensitivity family require their separately specified designs; they are not automatically inferred by this interface.

## Reusing saved splits

The API can accept an explicit `plans` list. This preserves shared folds across models or feature-removal comparisons. The original participant-specific study plans are not public. A newly generated split plan on different data is not an exact reproduction of the manuscript, even if its seed is the same.
