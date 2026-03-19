# Credit Risk PD / LGD / EAD Pipeline Playbook

> **Purpose**: This document is the single source of truth for every AI agent in the credit-risk
> modelling pipeline. It is injected verbatim into each agent's system prompt. Every business rule,
> formula, threshold, and specification that an agent may need is recorded here. Agents MUST
> follow these instructions exactly.

---

## 1. Dataset Description

| Attribute | Value |
|-----------|-------|
| Source | LendingClub consumer loan data, 2007-2018 |
| Row count | 2,260,701 loans |
| Column count | 151 columns |
| Storage | SQLite database at `Data/Raw/RCM_Controls.db`, table `my_table` |
| Primary key | `id` (unique LC-assigned loan listing ID) |
| Temporal key | `issue_d` — the month the loan was funded, format `"Mon-YYYY"` (e.g., `"Apr-2008"`) |
| Label column | `loan_status` — current status of the loan |

---

## 2. Data Dictionary

All 117 fields from the LendingClub data dictionary are listed below.

| Field | Description |
|-------|-------------|
| acc_now_delinq | The number of accounts on which the borrower is now delinquent |
| acc_open_past_24mths | Number of trades opened in past 24 months |
| addr_state | The state provided by the borrower in the loan application |
| all_util | Balance to credit limit on all trades |
| annual_inc | The self-reported annual income provided by the borrower during registration |
| annual_inc_joint | The combined self-reported annual income provided by the co-borrowers during registration |
| application_type | Indicates whether the loan is an individual application or a joint application with two co-borrowers |
| avg_cur_bal | Average current balance of all accounts |
| bc_open_to_buy | Total open to buy on revolving bankcards |
| bc_util | Ratio of total current balance to high credit/credit limit for all bankcard accounts |
| chargeoff_within_12_mths | Number of charge-offs within 12 months |
| collection_recovery_fee | Post charge off collection fee |
| collections_12_mths_ex_med | Number of collections in 12 months excluding medical collections |
| delinq_2yrs | The number of 30+ days past-due incidences of delinquency in the borrower's credit file for the past 2 years |
| delinq_amnt | The past-due amount owed for the accounts on which the borrower is now delinquent |
| desc | Loan description provided by the borrower |
| dti | A ratio calculated using the borrower's total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower's self-reported monthly income |
| dti_joint | A ratio calculated using the co-borrowers' total monthly payments on the total debt obligations, excluding mortgages and the requested LC loan, divided by the co-borrowers' combined self-reported monthly income |
| earliest_cr_line | The month the borrower's earliest reported credit line was opened |
| emp_length | Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years |
| emp_title | The job title supplied by the Borrower when applying for the loan |
| fico_range_high | The upper boundary range the borrower's FICO at loan origination belongs to |
| fico_range_low | The lower boundary range the borrower's FICO at loan origination belongs to |
| funded_amnt | The total amount committed to that loan at that point in time |
| funded_amnt_inv | The total amount committed by investors for that loan at that point in time |
| grade | LC assigned loan grade |
| home_ownership | The home ownership status provided by the borrower during registration. Values: RENT, OWN, MORTGAGE, OTHER |
| id | A unique LC assigned ID for the loan listing |
| il_util | Ratio of total current balance to high credit/credit limit on all install acct |
| initial_list_status | The initial listing status of the loan. Possible values: W, F |
| inq_fi | Number of personal finance inquiries |
| inq_last_12m | Number of credit inquiries in past 12 months |
| inq_last_6mths | The number of inquiries in past 6 months (excluding auto and mortgage inquiries) |
| installment | The monthly payment owed by the borrower if the loan originates |
| int_rate | Interest Rate on the loan |
| issue_d | The month which the loan was funded |
| last_credit_pull_d | The most recent month LC pulled credit for this loan |
| last_fico_range_high | The upper boundary range the borrower's last FICO pulled belongs to |
| last_fico_range_low | The lower boundary range the borrower's last FICO pulled belongs to |
| last_pymnt_amnt | Last total payment amount received |
| last_pymnt_d | Last month payment was received |
| loan_amnt | The listed amount of the loan applied for by the borrower |
| loan_status | Current status of the loan |
| max_bal_bc | Maximum current balance owed on all revolving accounts |
| member_id | A unique LC assigned Id for the borrower member |
| mo_sin_old_il_acct | Months since oldest bank installment account opened |
| mo_sin_old_rev_tl_op | Months since oldest revolving account opened |
| mo_sin_rcnt_rev_tl_op | Months since most recent revolving account opened |
| mo_sin_rcnt_tl | Months since most recent account opened |
| mort_acc | Number of mortgage accounts |
| mths_since_last_delinq | The number of months since the borrower's last delinquency |
| mths_since_last_major_derog | Months since most recent 90-day or worse rating |
| mths_since_last_record | The number of months since the last public record |
| mths_since_rcnt_il | Months since most recent installment accounts opened |
| mths_since_recent_bc | Months since most recent bankcard account opened |
| mths_since_recent_bc_dlq | Months since most recent bankcard delinquency |
| mths_since_recent_inq | Months since most recent inquiry |
| mths_since_recent_revol_delinq | Months since most recent revolving delinquency |
| next_pymnt_d | Next scheduled payment date |
| num_accts_ever_120_pd | Number of accounts ever 120 or more days past due |
| num_actv_bc_tl | Number of currently active bankcard accounts |
| num_actv_rev_tl | Number of currently active revolving trades |
| num_bc_sats | Number of satisfactory bankcard accounts |
| num_bc_tl | Number of bankcard accounts |
| num_il_tl | Number of installment accounts |
| num_op_rev_tl | Number of open revolving accounts |
| num_rev_accts | Number of revolving accounts |
| num_rev_tl_bal_gt_0 | Number of revolving trades with balance >0 |
| num_sats | Number of satisfactory accounts |
| num_tl_120dpd_2m | Number of accounts currently 120 days past due (updated in past 2 months) |
| num_tl_30dpd | Number of accounts currently 30 days past due (updated in past 2 months) |
| num_tl_90g_dpd_24m | Number of accounts 90 or more days past due in last 24 months |
| num_tl_op_past_12m | Number of accounts opened in past 12 months |
| open_acc | The number of open credit lines in the borrower's credit file |
| open_acc_6m | Number of open trades in last 6 months |
| open_il_12m | Number of installment accounts opened in past 12 months |
| open_il_24m | Number of installment accounts opened in past 24 months |
| open_il_6m | Number of currently active installment trades |
| open_rv_12m | Number of revolving trades opened in past 12 months |
| open_rv_24m | Number of revolving trades opened in past 24 months |
| out_prncp | Remaining outstanding principal for total amount funded |
| out_prncp_inv | Remaining outstanding principal for portion of total amount funded by investors |
| pct_tl_nvr_dlq | Percent of trades never delinquent |
| percent_bc_gt_75 | Percentage of all bankcard accounts > 75% of limit |
| policy_code | Publicly available policy_code=1, new products not publicly available policy_code=2 |
| pub_rec | Number of derogatory public records |
| pub_rec_bankruptcies | Number of public record bankruptcies |
| purpose | A category provided by the borrower for the loan request |
| pymnt_plan | Indicates if a payment plan has been put in place for the loan |
| recoveries | Post charge off gross recovery |
| revol_bal | Total credit revolving balance |
| revol_util | Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit |
| sub_grade | LC assigned loan subgrade |
| tax_liens | Number of tax liens |
| term | The number of payments on the loan. Values are in months and can be either 36 or 60 |
| title | The loan title provided by the borrower |
| tot_coll_amt | Total collection amounts ever owed |
| tot_cur_bal | Total current balance of all accounts |
| tot_hi_cred_lim | Total high credit/credit limit |
| total_acc | The total number of credit lines currently in the borrower's credit file |
| total_bal_ex_mort | Total credit balance excluding mortgage |
| total_bal_il | Total current balance of all installment accounts |
| total_bc_limit | Total bankcard high credit/credit limit |
| total_cu_tl | Number of finance trades |
| total_il_high_credit_limit | Total installment high credit/credit limit |
| total_pymnt | Payments received to date for total amount funded |
| total_pymnt_inv | Payments received to date for portion of total amount funded by investors |
| total_rec_int | Interest received to date |
| total_rec_late_fee | Late fees received to date |
| total_rec_prncp | Principal received to date |
| total_rev_hi_lim | Total revolving high credit/credit limit |
| url | URL for the LC page with listing data |
| verification_status | Indicates if income was verified by LC, not verified, or if the income source was verified |
| verified_status_joint | Indicates if the co-borrowers' joint income was verified by LC, not verified, or if the income source was verified |
| zip_code | The first 3 numbers of the zip code provided by the borrower in the loan application |

---

## 3. Target Construction

### 3.1 PD Target (Probability of Default)

```
default_flag = 1  if loan_status in ('Charged Off', 'Default')
default_flag = 0  if loan_status == 'Fully Paid'
```

**Critical filtering rule**: Only resolved loans are used for training and evaluation. The dataset
MUST be filtered to rows where `loan_status in ('Fully Paid', 'Charged Off')`. Exclude all
current, in-grace-period, and late loans from the training population.

### 3.2 LGD Target (Loss Given Default)

```
lgd = 1 - (recoveries - collection_recovery_fee) / funded_amnt
lgd = clip(lgd, 0, 1)
```

- Computed **only** for defaulted loans (`default_flag = 1`).
- Binary target for Stage 1 classification: `any_loss = 1 if lgd > 0 else 0`.
- Continuous target for Stage 2 regression: the `lgd` value itself, filtered to rows where `any_loss = 1`.

### 3.3 EAD Target (Exposure at Default)

```
ead = out_prncp
```

`ead` is the remaining outstanding principal at the time of default.

**Credit Conversion Factor (CCF)**:

```
CCF = ead / funded_amnt
```

**Alternative approach**: Compute EAD from the amortization schedule using `funded_amnt`,
`installment`, `term`, and `total_pymnt`.

### 3.4 Expected Loss (EL)

```
EL = PD x LGD x EAD
```

This is computed at the loan level and combined at inference time after all three component
models have produced their predictions.

---

## 4. Vintage-Based Train / Validation / Test Split

Parse `issue_d` (format `"Mon-YYYY"`, e.g., `"Apr-2008"`) to extract `issue_year`.

| Partition | Rule | Purpose |
|-----------|------|---------|
| **Training** | `issue_year <= 2015` | Fit model parameters |
| **Validation** | `issue_year == 2016` | Tune hyperparameters, select champion |
| **Test** | `issue_year >= 2017` | Final hold-out evaluation |

This vintage-based split prevents temporal leakage by ensuring the model never trains on data
from periods that overlap with the validation or test sets.

---

## 5. Leakage Columns

The following columns contain post-origination information that leaks future default status.
They MUST be **dropped from PD feature engineering** but MUST be **preserved separately** for
LGD/EAD target construction.

### 5.1 Drop for PD Modeling (post-origination leakage)

- `recoveries`
- `collection_recovery_fee`
- `total_pymnt`
- `total_pymnt_inv`
- `total_rec_prncp`
- `total_rec_int`
- `total_rec_late_fee`
- `last_pymnt_d`
- `last_pymnt_amnt`
- `last_credit_pull_d`
- `last_fico_range_high`
- `last_fico_range_low`
- `out_prncp`
- `out_prncp_inv`
- `debt_settlement_flag`
- `settlement_*` (all settlement-related columns)
- `hardship_*` (all hardship-related columns)
- `next_pymnt_d`
- `pymnt_plan`

### 5.2 Also Drop (not useful for modeling)

- `id`
- `member_id`
- `url`
- `desc`
- `emp_title`
- `title`
- `zip_code`
- `policy_code`

---

## 6. Data Quality Tests

| Test ID | Test Name | Threshold | Action on Fail |
|---------|-----------|-----------|----------------|
| DQ-01 | Missing Value Audit | > 5% missing per column | Impute (median by grade) or flag for removal |
| DQ-02 | Class Balance (PD target) | Minority class < 5% | Apply `class_weight` or SMOTE |
| DQ-03 | High Cardinality Categoricals | > 50 unique values | Target encoding or frequency encoding |
| DQ-04 | Near-Zero Variance | > 95% same value | Drop feature |
| DQ-05 | Multicollinearity (Pearson + VIF) | r > 0.85 or VIF > 10 | Remove one feature or apply PCA |
| DQ-06 | Target Leakage Detection | Single-feature AUC > 0.95 | Investigate and remove leaking column |
| DQ-07 | Distribution Stability (PSI) | PSI > 0.25 | Flag for retraining; monitor if 0.10-0.25 |
| DQ-08 | Outlier Detection (Z + IQR) | Z > 4.0 or IQR 3x | Winsorize per `WINSORIZE_COLS` config |
| DQ-09 | Feature-Target Correlation (Gini/IV) | Gini < 0.02 or > 0.50 | Remove weak; investigate suspiciously strong |
| DQ-10 | Data Type Consistency | Object column > 90% numeric | Coerce to correct type |

---

## 7. Data Cleaning Steps (in order)

The following steps MUST be executed in the exact order listed.

### Step 1: Drop Post-Default Leakage Columns

Remove all columns listed in Section 5.1 from the feature set. **Preserve them in a separate
DataFrame** for LGD/EAD target construction.

### Step 2: Filter to Resolved Loans Only

Keep only rows where `loan_status in ('Fully Paid', 'Charged Off')`. All other statuses
(Current, In Grace Period, Late 16-30, Late 31-120, etc.) are excluded.

### Step 3: Type Coercion

| Column | Transformation |
|--------|---------------|
| `int_rate` | Strip `"%"` suffix, convert to `float` |
| `term` | Strip `" months"` suffix, convert to `int` |
| `emp_length` | Parse text: `"< 1 year"` -> `0`, `"10+ years"` -> `10`, `"n years"` -> `n` |
| `revol_util` | Strip `"%"` suffix, convert to `float` |
| `issue_d` | Parse `"Mon-YYYY"` to `datetime` |
| `earliest_cr_line` | Parse `"Mon-YYYY"` to `datetime` |

### Step 4: Missing Value Imputation

| Column Type | Strategy |
|-------------|----------|
| Numeric columns | Median grouped by `grade` |
| Categorical columns | Mode |
| `emp_length` | Global median |

### Step 5: Outlier Capping (Winsorization)

Apply 1st / 99th percentile winsorization to:

- `annual_inc`
- `revol_bal`
- `loan_amnt`
- `funded_amnt`
- `dti`
- `open_acc`

### Step 6: Categorical Encoding

| Column(s) | Method | Details |
|-----------|--------|---------|
| `grade`, `sub_grade` | Ordinal mapping | A=1, B=2, C=3, D=4, E=5, F=6, G=7 |
| `purpose` | Risk-based encoding | Map to historical default rate per category |
| `home_ownership` | One-hot encoding | |
| `verification_status` | One-hot encoding | |
| `initial_list_status` | One-hot encoding | |
| `application_type` | One-hot encoding | |

---

## 8. Feature Engineering

### 8.1 Ratio Features

| Feature Name | Formula |
|-------------|---------|
| `loan_to_income` | `loan_amnt / annual_inc` |
| `installment_to_income` | `installment / (annual_inc / 12)` |
| `revol_to_total` | `revol_bal / (tot_cur_bal + 1)` |
| `credit_utilization` | `revol_bal / (total_rev_hi_lim + 1)` |
| `open_acc_ratio` | `open_acc / (total_acc + 1)` |

### 8.2 WoE / IV Transformation

Use the **optbinning** library for optimal binning.

1. Compute **Weight of Evidence (WoE)** for each binned feature.
2. Compute **Information Value (IV)** for feature selection.

**IV Interpretation Scale:**

| IV Range | Interpretation |
|----------|---------------|
| IV < 0.02 | Not useful -- drop |
| IV 0.02 - 0.10 | Weak predictor |
| IV 0.10 - 0.30 | Medium predictor |
| IV 0.30 - 0.50 | Strong predictor |
| IV > 0.50 | Suspiciously strong -- investigate for leakage |

### 8.3 Feature Selection Criteria

1. **Drop** features with IV < 0.02.
2. **Drop** one of any pair with Pearson |r| > 0.85 (keep the one with higher IV).
3. **Drop** features with VIF > 10 (iteratively remove the feature with the highest VIF, recalculate, repeat until all VIF <= 10).
4. **Investigate** any feature with single-feature AUC > 0.95 for potential leakage.

---

## 9. Model Candidate Pools

### 9.1 PD Candidates (Classification)

| # | Model | Library | Baseline Config |
|---|-------|---------|-----------------|
| 1 | Logistic Regression (L2) | scikit-learn | `C=0.1, penalty='l2', class_weight='balanced', solver='lbfgs'` |
| 2 | Logistic Regression (L1) | scikit-learn | `C=0.05, penalty='l1', class_weight='balanced', solver='liblinear'` |
| 3 | Elastic Net Logistic | scikit-learn | `l1_ratio=0.5, C=0.1` |
| 4 | Decision Tree | scikit-learn | `max_depth=5, min_samples_leaf=200` |
| 5 | Random Forest | scikit-learn | `n_estimators=200, max_depth=6` |
| 6 | Gradient Boosting (GBM) | scikit-learn | `n_estimators=200, max_depth=3, learning_rate=0.05` |
| 7 | AdaBoost | scikit-learn | `n_estimators=200, learning_rate=0.05` |
| 8 | Extra Trees | scikit-learn | `n_estimators=200, max_depth=6` |
| 9 | XGBoost | xgboost | `n_estimators=500, max_depth=4, learning_rate=0.03, reg_alpha=0.1, scale_pos_weight=auto` |
| 10 | LightGBM | lightgbm | `n_estimators=500, num_leaves=31, learning_rate=0.03` |
| 11 | Logit (full stats) | statsmodels | `method='bfgs', maxiter=1000` |
| 12 | Probit | statsmodels | `method='bfgs'` |

### 9.2 LGD Candidates

#### Stage 1 (Binary: any loss?)

| # | Model | Library | Config |
|---|-------|---------|--------|
| 1 | Logistic Regression | scikit-learn | `C=0.1, class_weight='balanced'` |
| 2 | Random Forest Classifier | scikit-learn | `n_estimators=200, max_depth=5` |
| 3 | GBM Classifier | scikit-learn | `n_estimators=200, max_depth=3` |
| 4 | XGBoost Classifier | xgboost | `n_estimators=300, max_depth=4` |
| 5 | LightGBM Classifier | lightgbm | `n_estimators=300, num_leaves=31` |

#### Stage 2 (Severity: how much loss?)

| # | Model | Library | Config |
|---|-------|---------|--------|
| 6 | OLS Linear Regression | statsmodels | Full stats output |
| 7 | Ridge Regression | scikit-learn | `alpha=1.0` |
| 8 | Lasso Regression | scikit-learn | `alpha=0.1` |
| 9 | Elastic Net | scikit-learn | `alpha=0.5, l1_ratio=0.5` |
| 10 | GBM Regressor | scikit-learn | `n_estimators=200, loss='huber'` |
| 11 | XGBoost Regressor | xgboost | `n_estimators=300, max_depth=3` |
| 12 | LightGBM Regressor | lightgbm | `n_estimators=300, num_leaves=31` |
| 13 | Huber Regressor | scikit-learn | `epsilon=1.35, max_iter=500` |

### 9.3 EAD Candidates (Regression)

| # | Model | Library | Config |
|---|-------|---------|--------|
| 1 | OLS Linear Regression | statsmodels | Full stats output |
| 2 | Ridge Regression | scikit-learn | `alpha=1.0` |
| 3 | Lasso Regression | scikit-learn | `alpha=0.1` |
| 4 | Elastic Net | scikit-learn | `alpha=0.5, l1_ratio=0.5` |
| 5 | Huber Regressor | scikit-learn | `epsilon=1.35` |
| 6 | Random Forest Regressor | scikit-learn | `n_estimators=200, max_depth=6` |
| 7 | GBM Regressor | scikit-learn | `n_estimators=200, loss='huber'` |
| 8 | XGBoost Regressor | xgboost | `n_estimators=300, max_depth=3` |
| 9 | LightGBM Regressor | lightgbm | `n_estimators=300, num_leaves=31` |

---

## 10. Model Tournament Framework

### Phase 1: Broad Sweep

1. Train **ALL** candidates with their baseline configs from Section 9.
2. Score every model on the **validation set** (AUC for classification, RMSE for regression).
3. Record all metrics, feature importances, and training time.
4. **No model is excluded or pre-filtered** at this stage.

### Phase 2: Feature Importance Consensus

1. Extract importances from every trained model using its native method:
   - **Logistic Regression**: `abs(coefficients)` after `StandardScaler`
   - **Tree models**: `feature_importances_` (Gini impurity / gain)
   - **statsmodels**: `abs(z-scores / t-statistics)`
   - **Model-agnostic**: permutation importance
   - **Optional**: SHAP values
2. Normalize each model's importances to sum to `1.0`.
3. Compute weighted average rank where `weight = model's validation performance`.
4. Consensus Score for feature `f`:

```
Consensus(f) = sum(w_m * normalized_importance_m(f)) / sum(w_m)
```

5. Assign tiers:

| Tier | Percentile Range | Action |
|------|-----------------|--------|
| Tier 1 (Critical) | Top 20% | Always retain |
| Tier 2 (Important) | Next 30% | Retain by default |
| Tier 3 (Marginal) | Next 30% | Test inclusion/exclusion |
| Tier 4 (Noise) | Bottom 20% | Remove by default |

### Phase 3: Refinement Loop

- Select top-K models (default `K = 5`).
- `max_iterations = 5`, `convergence_threshold = 0.002`.
- Each iteration:
  1. `RandomizedSearchCV` with `TimeSeriesSplit(n_splits=5)` per model.
  2. Test 3 feature sets: **Tier 1+2 only**, **Tier 1+2+3**, **model-specific top features**.
  3. Score all combinations on the validation set.
  4. Check convergence: if improvement < `0.002`, exit the loop.
  5. Prune models that are more than `0.03` below the leader.

### Phase 4: Champion Selection

Apply the weighted scoring rubric below.

#### PD Classification Rubric

| Dimension | Metric | Weight (Regulatory) | Weight (Performance) |
|-----------|--------|---------------------|----------------------|
| Discriminatory Power | AUC-ROC | 0.20 | 0.35 |
| Discriminatory Power | Gini | 0.15 | 0.20 |
| Rank Ordering | KS Statistic | 0.15 | 0.15 |
| Calibration | Brier Score (inverted) | 0.10 | 0.05 |
| Calibration | Hosmer-Lemeshow p-value | 0.10 | 0.00 |
| Stability | PSI (inverted) | 0.10 | 0.05 |
| Interpretability | Coefficient availability | 0.10 | 0.00 |
| Generalization | Train-Val AUC gap (inverted) | 0.05 | 0.10 |
| Efficiency | Inference time (inverted) | 0.05 | 0.10 |

#### LGD / EAD Regression Rubric

| Dimension | Metric | Weight |
|-----------|--------|--------|
| Accuracy | RMSE (inverted) | 0.30 |
| Accuracy | MAE (inverted) | 0.15 |
| Explanatory Power | R-squared | 0.20 |
| Stability | PSI (inverted) | 0.10 |
| Decile Alignment | Mean abs decile deviation | 0.10 |
| Generalization | Train-Val RMSE gap (inverted) | 0.10 |
| Efficiency | Inference time (inverted) | 0.05 |

#### Mandatory Regulatory Output

**Always produce** a statsmodels `Logit` / `OLS` output with a full coefficient table
(coefficients, standard errors, z-scores, p-values, confidence intervals, odds ratios) for
regulatory documentation, **regardless of which model wins the tournament**.

---

## 11. Hyperparameter Search Spaces (Phase 3)

| Model | Hyperparameter | Distribution |
|-------|---------------|-------------|
| Logistic Regression | `C` | `loguniform(0.001, 10.0)` |
| Logistic Regression | `penalty` | `['l1', 'l2', 'elasticnet']` |
| Random Forest | `n_estimators` | `randint(100, 800)` |
| Random Forest | `max_depth` | `randint(3, 10)` |
| Random Forest | `min_samples_leaf` | `randint(50, 500)` |
| GBM / AdaBoost | `n_estimators` | `randint(100, 600)` |
| GBM / AdaBoost | `learning_rate` | `uniform(0.01, 0.15)` |
| GBM | `max_depth` | `randint(2, 6)` |
| GBM | `subsample` | `uniform(0.6, 0.4)` |
| XGBoost | `n_estimators` | `randint(200, 800)` |
| XGBoost | `max_depth` | `randint(3, 7)` |
| XGBoost | `learning_rate` | `uniform(0.01, 0.09)` |
| XGBoost | `reg_alpha` | `uniform(0, 1.0)` |
| XGBoost | `reg_lambda` | `uniform(0.5, 2.0)` |
| XGBoost | `colsample_bytree` | `uniform(0.5, 0.5)` |
| XGBoost | `min_child_weight` | `randint(50, 300)` |
| LightGBM | `n_estimators` | `randint(200, 800)` |
| LightGBM | `num_leaves` | `randint(15, 63)` |
| LightGBM | `learning_rate` | `uniform(0.01, 0.09)` |
| LightGBM | `min_child_samples` | `randint(50, 300)` |
| LightGBM | `colsample_bytree` | `uniform(0.5, 0.5)` |
| Ridge / Lasso / ElasticNet | `alpha` | `loguniform(0.01, 100.0)` |
| Huber | `epsilon` | `uniform(1.1, 0.9)` |

---

## 12. Evaluation Metric Thresholds

### PD Classification

| Metric | Green (Pass) | Yellow (Monitor) | Red (Fail) |
|--------|-------------|------------------|------------|
| AUC-ROC | > 0.75 | 0.65 - 0.75 | < 0.65 |
| Gini Coefficient | > 0.50 | 0.30 - 0.50 | < 0.30 |
| KS Statistic | > 0.35 | 0.20 - 0.35 | < 0.20 |
| Brier Score | < 0.15 | 0.15 - 0.25 | > 0.25 |
| Hosmer-Lemeshow p-value | > 0.10 | 0.05 - 0.10 | < 0.05 |
| PSI | < 0.10 | 0.10 - 0.25 | > 0.25 |

---

## 13. Stress Testing Scenarios

| Scenario | PD Adjustment | LGD Adjustment | Description |
|----------|---------------|----------------|-------------|
| Base | No adjustment | No adjustment | Current economic conditions |
| Adverse | PD x 1.5 | LGD floor = 0.45 | Moderate recession |
| Severe | PD x 2.0 | LGD floor = 0.60 | Severe economic downturn |

---

## 14. Report Structures

### 14.1 Data Quality Report (custom)

1. **Data Overview** -- source, shape, vintages, column inventory
2. **Data Quality Scorecard** -- all DQ tests (DQ-01 through DQ-10) with pass/fail/warn results
3. **Data Assumptions and Treatments** -- cleaning steps applied, before/after statistics
4. **Feature Profiling** -- distribution summaries, null rates, cardinality
5. **Fit for Model Development Decision** -- overall assessment with ranked issues
6. **Query Trace** -- all SQL queries executed with results summary

### 14.2 Model Reports -- C1 Standalone Use Case Template (for PD, LGD, EAD)

Each of the three component models (PD, LGD, EAD) produces a report following this structure:

1. **Introduction** -- use case metadata, model summary
2. **Purpose and Uses** -- model objective, business context
3. **Use Case Description** -- pipeline architecture, I/O summary
4. **Data**
   - 4.1 Description / Sources
   - 4.2 Treatments
   - 4.3 Assumptions / Limitations
5. **Methodology**
   - 5.1 Feature Engineering / Selection
   - 5.2 Methodology Selection
   - 5.3 Calibration Dataset
   - 5.4 Assumptions / Limitations
6. **Outputs**
   - 6.1 Output Description
   - 6.2 Performance Testing
     - Accuracy
     - Stability
     - Robustness
     - Explainability
     - Fairness
7. **Implementation** -- environment, packages, user manual
8. **Appendix** -- version control, test checklist, section mapping

### 14.3 EL Summary Report (custom)

1. **Executive Summary** with portfolio-level EL
2. **Model Performance Summary** -- PD, LGD, EAD champion metrics side by side
3. **Loan-Level EL Distribution**
4. **Portfolio Roll-Up** by grade, vintage, and purpose
5. **Stress Testing Results** (Base / Adverse / Severe)
6. **Regulatory Capital Implications**

---

## 15. Agent Handoff Protocol

### Root Directory

```
Data/Output/pipeline_run_{timestamp}/
```

### Subdirectories

| Directory | Contents |
|-----------|----------|
| `01_data_quality/` | Data quality report, cleaned data, DQ test results |
| `02_features/` | Engineered features, IV tables, WoE mappings, feature selection results |
| `03_pd_model/` | PD model artifacts, tournament results, champion model, PD report |
| `04_lgd_model/` | LGD model artifacts, Stage 1 + Stage 2 results, LGD report |
| `05_ead_model/` | EAD model artifacts, tournament results, EAD report |
| `06_expected_loss/` | Loan-level EL predictions, portfolio roll-ups, stress test results |
| `07_reports/` | Final consolidated reports (Data Quality, PD, LGD, EAD, EL Summary) |

### Handoff JSON

Each agent writes a `handoff.json` file in its output directory upon completion:

```json
{
  "agent": "<agent_name>",
  "status": "success|failed",
  "started_at": "<ISO 8601>",
  "completed_at": "<ISO 8601>",
  "duration_s": 0,
  "output_files": {},
  "metrics": {},
  "errors": []
}
```

---

## 16. Test Framework Coverage

| Dimension | Category | Test IDs | Agent |
|-----------|----------|----------|-------|
| Inputs | Data Appropriateness | I.DA.1 - I.DA.5 | Data_Agent |
| Inputs | Data Quality | I.DQ.1 - I.DQ.8 | Data_Agent |
| Inputs | Fairness | I.FB.1 - I.FB.3 | Data_Agent |
| Inputs | Data Treatments | I.DT.1 - I.DT.5 | Data_Agent / Feature_Agent |
| Modelling | Feature Engineering | M.FE.1 - M.FE.7 | Feature_Agent |
| Modelling | Methodology Selection | M.MS.1 - M.MS.8 | PD / LGD / EAD Agent |
| Modelling | Sampling Approach | M.SA.1 - M.SA.3 | PD / LGD / EAD Agent |
| Modelling | Parameter Estimation | M.PE.1 - M.PE.7 | PD / LGD / EAD Agent |
| Outputs | Accuracy | O.A.1 | PD / LGD / EAD Agent |
| Outputs | Stability | O.S.1 - O.S.2 | PD / LGD / EAD Agent |
| Outputs | Robustness | O.R.1 | EL_Agent |
| Outputs | Fairness | O.FB.1 - O.FB.2 | PD / LGD / EAD Agent |
| Outputs | Explainability | O.E.1 | PD / LGD / EAD Agent |
| Business | End-to-End | B.EE.1 - B.EE.2 | EL_Agent |
| Technical | Documentation | TI.D.1 | Report_Agent |

### Test Output Schema

Each test produces a structured result with the following fields:

| Field | Description |
|-------|-------------|
| `test_id` | Unique identifier (e.g., `I.DA.1`, `M.FE.3`) |
| `test_name` | Human-readable name of the test |
| `dimension` | One of: Inputs, Modelling, Outputs, Business, Technical |
| `status` | `PASS`, `WARN`, or `FAIL` |
| `detail` | Description of the finding |
| `evidence` | Supporting data, statistics, or file references |
| `remediation` | Recommended action if status is WARN or FAIL |
