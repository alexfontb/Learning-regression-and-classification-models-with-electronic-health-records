# LAB 2: Learning Regression and Classification Models with Electronic Health Records

**Authors:**
* Àlex Font Bartumeus 
* Iris Azlor Solé 

---

## Project Overview

This project explores machine learning approaches applied to Electronic Health Records (EHR) to solve two clinical prediction tasks:
1. **Classification:** Predict whether a given patient will be readmitted (`readmitted`).
2. **Regression:** Estimate the expected length of hospital stay in days (`time_in_hospital`).

---

## Dataset

The analysis uses the `real_patients.csv` dataset (semicolon-separated), containing **25,000 patient records** and **17 features** with no `NaN` values:

* **Target Variables:**
  * `readmitted` (Binary categorical: `yes` / `no`) — Classification target
  * `time_in_hospital` (Integer: `1` to `14` days) — Regression target
* **Numerical (Count) Features:** `n_lab_procedures`, `n_procedures`, `n_medications`, `n_outpatient`, `n_inpatient`, `n_emergency`
* **Categorical Features:** `age`, `medical_specialty`, `diag_1`, `diag_2`, `diag_3`, `glucose_test`, `A1Ctest`, `change`, `diabetes_med`

---

## Notebook Structure & Methodology

### 1. Exploratory Data Analysis (EDA)
* **Data Inspection:** Loaded the dataset and verified data types (`int64` and `str`/`object`) and missing values (note: `medical_specialty` includes `"Missing"` as an explicit category).
* **Target Variable Analysis:**
  * `readmitted`: Relatively balanced classes (`52.98%` no vs. `47.02%` yes), allowing model training without aggressive resampling techniques.
  * `time_in_hospital`: Right-skewed distribution with a mean stay of **4.45 days** ($\text{SD} \approx 3.00$), ranging from 1 to 14 days.
* **Feature Distributions & Outliers:** Histograms of count features revealed strong right-skewness with a high concentration of zeros (particularly in `n_outpatient`, `n_inpatient`, `n_emergency`, and `n_procedures`), highlighting the need for feature scaling and outlier consideration prior to modeling.
* **Feature Selection & Data Leakage Prevention:** Evaluated clinical relationships and identified potential sources of data leakage to exclude prior to model training.

---

## Requirements & Setup

To run the notebook locally, ensure Python 3 is installed along with the following dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn ucimlrepo
