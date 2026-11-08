# Smart Hostel Energy Advisor

A machine learning pipeline that analyzes daily and hourly electrical energy consumption data from a university residential campus across 4 hostel blocks (A, B, C, D) over two consecutive academic operational years. The system integrates supervised regression models for total daily energy forecasting with unsupervised manifold learning and clustering algorithms for automated diurnal load pattern discovery, accompanied by rule-based load curtailment advisories.

## Table of Contents

- [System Architecture](#system-architecture)
- [Repository Structure](#repository-structure)
- [Data Specifications](#data-specifications)
- [Environment Setup & Dependencies](#environment-setup--dependencies)
- [Data Preprocessing Pipeline](#data-preprocessing-pipeline)
- [Feature Engineering Framework](#feature-engineering-framework)
- [Machine Learning Models & Mathematical Formulations](#machine-learning-models--mathematical-formulations)
  - [1. Multivariate Linear Regression](#1-multivariate-linear-regression)
  - [2. Decision Tree Regressor](#2-decision-tree-regressor)
  - [3. Principal Component Analysis (PCA)](#3-principal-component-analysis-pca)
  - [4. K-Means Clustering & Diurnal Row Normalization](#4-k-means-clustering--diurnal-row-normalization)
- [Out-of-Sample Year 2 Evaluation](#out-of-sample-year-2-evaluation)
- [Quantitative Performance Benchmark](#quantitative-performance-benchmark)
- [Smart Energy Advisory Engine](#smart-energy-advisory-engine)
- [Execution Instructions](#execution-instructions)
- [Technical Implementation Details](#technical-implementation-details)

---

## System Architecture

```
                    ┌──────────────────────────────────────────────────┐
                    │               RAW DATA INGESTION                 │
                    │  daily_energy.csv + hourly_profiles.csv + ground │
                    └────────────────────────┬─────────────────────────┘
                                             │
                                  ┌──────────▼──────────┐
                                  │  DATA PREPROCESSING │
                                  │  • Median / mean    │
                                  │    imputation       │
                                  │  • Date parsing     │
                                  │  • Block alignment  │
                                  └──────────┬──────────┘
                                             │
                                  ┌──────────▼──────────┐
                                  │ FEATURE ENGINEERING │
                                  │  • temp_range       │
                                  │  • rolling_3d/7d    │
                                  │  • night_day_ratio  │
                                  │  • Categorical enc. │
                                  └────┬────────────┬───┘
                                       │            │
                          ┌────────────▼──┐    ┌────▼────────────┐
                          │  SUPERVISED   │    │  UNSUPERVISED   │
                          │  REGRESSION   │    │  CLUSTERING     │
                          │               │    │                 │
                          │ Linear Reg.   │    │ Row Normalizer  │
                          │ Decision Tree │    │ StandardScaler  │
                          │      ↓        │    │ PCA (24D -> 2D) │
                          │ Predict kWh   │    │ K-Means (K=3)   │
                          │               │    │ Pattern Labels  │
                          └──────┬────────┘    └────────┬────────┘
                                 │                      │
                          ┌──────▼──────────────────────▼──────┐
                          │     SMART ENERGY ADVISORY ENGINE   │
                          │  • Continuous daily forecasts      │
                          │  • Behavioral pattern profiling    │
                          │  • Heatwave surge alarm (>P95)     │
                          │  • Targeted block load shedding    │
                          └────────────────────────────────────┘
```

---

## Repository Structure

```
.
├── smart_hostel_advisor.ipynb     # End-to-end ML pipeline (60 cells, 7 phases)
├── README.md                     # Technical documentation
├── requirements.txt              # Python dependencies
├── .gitignore
│
└── data/
    ├── train/                    # Year 1 training data (2025–2026)
    │   ├── daily_energy.csv          # 1,460 rows × 17 columns
    │   ├── hourly_profiles.csv       # 1,460 rows × 26 columns (h00–h23)
    │   └── ground_truth_patterns.csv # 1,460 rows (pattern labels)
    │
    └── test/                     # Year 2 evaluation data (2026–2027)
        ├── test_daily_energy.csv
        ├── test_hourly_profiles.csv
        └── test_ground_truth_patterns.csv
```

---

## Data Specifications

### Operational Datasets

1. **Year 1 Training Data (2025-11-01 to 2026-10-31)** — `data/train/`:
   - `daily_energy.csv`: 1,460 rows, 17 columns (4 blocks × 365 calendar days).
   - `hourly_profiles.csv`: 1,460 rows, 26 columns (`date`, `block`, and continuous hourly values `h00` through `h23`).
   - `ground_truth_patterns.csv`: 1,460 rows (`date`, `block`, `pattern`: `daytime_peak`, `night_heavy`, `low_flat`).

2. **Year 2 Out-of-Sample Evaluation Data (2026-11-01 to 2027-10-31)** — `data/test/`:
   - `test_daily_energy.csv`: 1,460 rows, 17 columns.
   - `test_hourly_profiles.csv`: 1,460 rows, 26 columns.
   - `test_ground_truth_patterns.csv`: 1,460 rows.

### Variable Dictionary

| Field | Representation | Physical / Contextual Definition |
|:---|:---:|:---|
| `date` | `YYYY-MM-DD` | Calendar timestamp |
| `block` | `string` | Residential hostel block (`A`, `B`, `C`, `D`) |
| `day_of_week` | `string` | Day identifier (`Monday` through `Sunday`) |
| `month` | `integer` | Calendar month index (`1` to `12`) |
| `season` | `string` | Climatological phase (`Post-monsoon`, `Winter`, `Summer`, `Monsoon`) |
| `temp_C` | `float` | Daily peak dry-bulb ambient temperature (°C) |
| `temp_min_C` | `float` | Daily minimum dry-bulb ambient temperature (°C) |
| `humidity_pct` | `float` | Mean relative ambient humidity percentage (%) |
| `rainfall_mm` | `float` | Cumulative daily precipitation depth (mm) |
| `occupancy_pct` | `float` | Percentage of active registered student rooms occupied |
| `ac_units_on` | `integer` | Count of active air conditioning compressor units |
| `is_weekend` | `binary` | Indicator flag for Saturday and Sunday (`0` or `1`) |
| `is_holiday` | `binary` | University gazetted administrative holiday flag (`0` or `1`) |
| `is_exam_week` | `binary` | Mid-semester / end-semester examination period (`0` or `1`) |
| `is_vacation` | `binary` | Winter / summer institutional recess (`0` or `1`) |
| `prev_day_kwh` | `float` | Lag-1 daily total electricity demand (kWh) |
| `kwh` | `float` | **Target Variable**: Aggregate 24-hour electricity demand (kWh) |

---

## Environment Setup & Dependencies

```bash
# Recommended environment: Python 3.10+
pip install numpy pandas scikit-learn matplotlib seaborn scipy
```

### Module Requirements
- `numpy >= 1.24.0`: Tensor algebra and vector calculations
- `pandas >= 2.0.0`: DataFrame transformations and window operations
- `scikit-learn >= 1.3.0`: Statistical modeling, decomposition, and cluster estimation
- `matplotlib >= 3.7.0`: Core figure rendering
- `seaborn >= 0.12.0`: Statistical distribution visualization
- `scipy >= 1.10.0`: Discrete modal cluster-to-label alignment

---

## Data Preprocessing Pipeline

### Missing Value Imputation Strategy
- **`prev_day_kwh`** (4 missing values at dataset initiation dates): Imputed via block-level historical mean. In Year 2 test sequences, continuous autoregressive chaining assigns the final day of Year 1 (2026-10-31) to initialize the first day of Year 2 (2026-11-01).
- **Environmental & Occupancy Nulls** (`temp_C`: 14, `humidity_pct`: 14, `occupancy_pct`: 10): Imputed using historical training median values to prevent future data leakage.

### Validation Splitting
- **Chronological Holdout Partition**: To respect temporal causality in time-series forecasting, an 80/20 chronological split is applied to Year 1:
  - Training Partition: First 1,168 observations (2025-11-01 to 2026-08-15).
  - Validation Partition: Subsequent 292 observations (2026-08-16 to 2026-10-31).
- **Out-of-Sample Holdout**: The complete Year 2 operational sequence (1,460 observations) is strictly sequestered until inference testing.

---

## Feature Engineering Framework

A 17-dimensional feature matrix $\mathbf{X} \in \mathbb{R}^{N \times 17}$ is synthesized:

1. **Diurnal Temperature Range (`temp_range`)**:
   $$\text{temp\_range}_i = \text{temp\_C}_i - \text{temp\_min\_C}_i$$
2. **Short-Term Autoregressive Memory (`rolling_3d_kwh`)**:
   $$\text{rolling\_3d\_kwh}_t = \frac{1}{\min(t, 3)} \sum_{k=0}^{\min(t, 3)-1} \text{kwh}_{t-k}$$
3. **Medium-Term Autoregressive Memory (`rolling_7d_kwh`)**:
   $$\text{rolling\_7d\_kwh}_t = \frac{1}{\min(t, 7)} \sum_{k=0}^{\min(t, 7)-1} \text{kwh}_{t-k}$$
4. **Diurnal Load Ratio (`night_day_ratio`)**:
   Extracted from hourly profile vectors:
   $$\text{night\_kwh} = \sum_{h \in \{00..05, 22, 23\}} \text{load}_h, \quad \text{day\_kwh} = \sum_{h \in \{08..17\}} \text{load}_h$$
   $$\text{night\_day\_ratio} = \frac{\text{night\_kwh}}{\text{day\_kwh} + 10^{-6}}$$
5. **Categorical Encodings**:
   - `block_encoded` $\in \{0, 1, 2, 3\}$ via label encoding on hostel identifier.
   - `season_encoded` $\in \{0, 1, 2, 3\}$ via label encoding on climatological season.

---

## Machine Learning Models & Mathematical Formulations

### 1. Multivariate Linear Regression

#### Hypothesis Function
The expected daily energy consumption $\hat{y}_i$ for an observation vector $\mathbf{x}_i \in \mathbb{R}^{p}$ ($p=17$) is modeled as:
$$\hat{y}_i = w_0 + \sum_{j=1}^{p} w_j x_{ij} = \mathbf{w}^T \mathbf{x}_i + b$$

In matrix form across $N$ instances:
$$\hat{\mathbf{y}} = \mathbf{X} \mathbf{w}$$
where $\mathbf{X} \in \mathbb{R}^{N \times (p+1)}$ contains an appended column of ones for the intercept term $b$.

#### Loss Function & Parameter Estimation
Parameters $\mathbf{w}^*$ are estimated by Ordinary Least Squares (OLS), minimizing the residual Mean Squared Error (MSE):
$$J(\mathbf{w}) = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2 = \frac{1}{N} \|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2$$

Setting the gradient $\nabla_{\mathbf{w}} J(\mathbf{w}) = \mathbf{0}$ yields the closed-form Normal Equation:
$$\mathbf{w}^* = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

#### Evaluation Criteria
- **Coefficient of Determination ($R^2$)**:
  $$R^2 = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2} = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}$$
- **Mean Absolute Error (MAE)**:
  $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$
- **Root Mean Squared Error (RMSE)**:
  $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$$

---

### 2. Decision Tree Regressor

#### Partitioning Formulation
Given training subset $S$, the regressor constructs axis-aligned splits partitioning the input space into disjoint regions $R_1, R_2, \dots, R_M$. The prediction for region $R_m$ is the constant response:
$$\hat{y}_{R_m} = \frac{1}{|S_m|} \sum_{i \in S_m} y_i \quad \text{where } S_m = \{(\mathbf{x}_i, y_i) \mid \mathbf{x}_i \in R_m\}$$

#### Splitting Objective & Variance Reduction
At node $S$, the optimal splitting feature $j \in \{1, \dots, p\}$ and split threshold $t \in \mathbb{R}$ are determined by maximizing the reduction in Mean Squared Error (variance reduction):
$$\text{MSE}(S) = \frac{1}{|S|} \sum_{i \in S} (y_i - \bar{y}_S)^2$$
$$\Delta \text{MSE}(S, j, t) = \text{MSE}(S) - \left( \frac{|S_L|}{|S|} \text{MSE}(S_L) + \frac{|S_R|}{|S|} \text{MSE}(S_R) \right)$$
where:
$$S_L = \{(\mathbf{x}_i, y_i) \in S \mid x_{ij} \le t\}, \quad S_R = \{(\mathbf{x}_i, y_i) \in S \mid x_{ij} > t\}$$

The optimal split $(j^*, t^*)$ satisfies:
$$(j^*, t^*) = \arg\max_{j, t} \Delta \text{MSE}(S, j, t)$$

#### Feature Importance (Mean Decrease in Impurity)
The relative importance of feature $f$ is calculated by accumulating impurity reductions across all internal nodes $n$ where feature $f$ is the splitting variable:
$$\text{FI}(f) = \sum_{n \in \text{splits on } f} \frac{N_n}{N} \left[ \text{MSE}(n) - \frac{N_{n,L}}{N_n}\text{MSE}(n_L) - \frac{N_{n,R}}{N_n}\text{MSE}(n_R) \right]$$

---

### 3. Principal Component Analysis (PCA)

#### Mathematical Objective
Given hourly profile matrix $\mathbf{X}_{\text{norm}} \in \mathbb{R}^{N \times 24}$, features are centered and standardized:
$$z_{ij} = \frac{x_{ij}^{\text{norm}} - \mu_j}{\sigma_j}, \quad \mathbf{Z} \in \mathbb{R}^{N \times 24}$$

The sample covariance matrix $\mathbf{\Sigma} \in \mathbb{R}^{24 \times 24}$ is formulated as:
$$\mathbf{\Sigma} = \frac{1}{N - 1} \mathbf{Z}^T \mathbf{Z}$$

#### Eigendecomposition
Principal direction vectors $\mathbf{v}_k$ and corresponding variances $\lambda_k$ are obtained by solving the characteristic equation:
$$\mathbf{\Sigma} \mathbf{v}_k = \lambda_k \mathbf{v}_k, \quad k \in \{1, \dots, 24\}$$
sorted in descending order such that $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_{24} \ge 0$.

#### Subspace Projection
Projecting into the top 2 eigenvectors $\mathbf{W}_2 = [\mathbf{v}_1, \mathbf{v}_2] \in \mathbb{R}^{24 \times 2}$:
$$\mathbf{Z}_{\text{pca}} = \mathbf{Z} \mathbf{W}_2 \in \mathbb{R}^{N \times 2}$$

#### Explained Variance Ratio (EVR)
$$\text{EVR}_k = \frac{\lambda_k}{\sum_{j=1}^{24} \lambda_j}$$
The cumulative variance retained by $K=2$ components is:
$$\text{Cumulative EVR} = \text{EVR}_1 + \text{EVR}_2 = 65.3\% + 18.4\% = 83.7\%$$

---

### 4. K-Means Clustering & Diurnal Row Normalization

#### Diurnal Row Normalization (Shape Isolation)
To decouple raw consumption magnitude (kW scale) from the temporal shape of daily routines, each 24-dimensional profile vector $\mathbf{x}_i = [x_{i,00}, \dots, x_{i,23}]$ is row-normalized:
$$\tilde{x}_{it} = \frac{x_{it}}{\sum_{h=0}^{23} x_{ih} + 10^{-8}}, \quad \forall t \in \{0, \dots, 23\}$$
This ensures $\sum_{t=0}^{23} \tilde{x}_{it} = 1.0$, converting absolute energy consumption into a discrete diurnal density profile.

#### K-Means Objective Function (Inertia)
K-Means partitions the normalized 2D PCA representations $\mathbf{z}_i \in \mathbb{R}^2$ into $K=3$ Voronoi clusters $C = \{C_1, C_2, C_3\}$ by minimizing within-cluster sum of squares:
$$J_{\text{K-Means}} = \sum_{k=1}^{K} \sum_{\mathbf{z}_i \in C_k} \|\mathbf{z}_i - \boldsymbol{\mu}_k\|_2^2$$

#### Alternating Optimization (Lloyd's Algorithm)
1. **Cluster Membership Step (Voronoi Partitioning)**:
   $$c^{(i)} = \arg\min_{k \in \{1, \dots, K\}} \|\mathbf{z}_i - \boldsymbol{\mu}_k\|_2^2$$
2. **Centroid Update Step**:
   $$\boldsymbol{\mu}_k = \frac{1}{|C_k|} \sum_{\mathbf{z}_i \in C_k} \mathbf{z}_i$$

#### Silhouette Metric
For observation $i \in C_A$:
$$a(i) = \frac{1}{|C_A| - 1} \sum_{j \in C_A, j \ne i} \|\mathbf{z}_i - \mathbf{z}_j\|_2, \quad b(i) = \min_{B \ne A} \frac{1}{|C_B|} \sum_{j \in C_B} \|\mathbf{z}_i - \mathbf{z}_j\|_2$$
$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}, \quad s(i) \in [-1, 1]$$

---

## Out-of-Sample Year 2 Evaluation

Models trained on Year 1 were frozen and tested on Year 2 data (1,460 samples, 2026-11-01 to 2027-10-31) without parameter updating:

### Regression Generalization

| Model | Year 1 Val $R^2$ | Year 2 Test $R^2$ | Year 2 Test MAE | Year 2 Test RMSE | Year 2 MAPE |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Linear Regression** | 0.9329 | **0.9404** | **40.50 kWh** | **55.81 kWh** | **12.82%** |
| **Decision Tree ($d=5$)** | 0.9429 | **0.9040** | **51.45 kWh** | **70.78 kWh** | **15.12%** |

### Clustering Generalization

| Evaluation Dimension | Baseline (Year 1) | Out-of-Sample (Year 2) | Relative Change |
|:---|:---:|:---:|:---:|
| **Cluster Mapping Accuracy** | 96.71% | **98.77%** | +2.06% |
| **Mean Silhouette Index** | 0.6674 | **0.7525** | +0.0851 |
| **Misclassified Profiles** | 48 of 1,460 | **18 of 1,460** | -30 profiles |

### Year 2 Classification Matrix

| Profile Class | Precision | Recall | F1-Score | Support (Days) |
|:---|:---:|:---:|:---:|:---:|
| `daytime_peak` | 1.00 | 0.98 | 0.99 | 729 |
| `night_heavy` | 0.96 | 1.00 | 0.98 | 406 |
| `low_flat` | 1.00 | 0.99 | 1.00 | 325 |
| **Weighted Average** | **0.99** | **0.99** | **0.99** | **1,460** |

$$\text{Confusion Matrix} = \begin{pmatrix} 716 & 0 & 13 \\ 1 & 322 & 2 \\ 2 & 0 & 404 \end{pmatrix}$$

### Block-by-Block Metrics (Year 2)

| Residential Block | Mean Demand | LR Test $R^2$ | LR Test MAE | DT Test $R^2$ | DT Test MAE | Cluster Accuracy |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Block A** | 356.7 kWh | 0.9265 | 37.36 kWh | 0.8730 | 47.95 kWh | 98.63% |
| **Block B** | 434.2 kWh | 0.9346 | 43.20 kWh | 0.8852 | 56.81 kWh | 98.90% |
| **Block C** | 266.7 kWh | 0.8942 | 33.11 kWh | 0.8658 | 41.08 kWh | 99.18% |
| **Block D** | 512.9 kWh | 0.9355 | 48.35 kWh | 0.8854 | 64.71 kWh | 98.36% |

---

## Quantitative Performance Benchmark

```
   100% ──────────────────────────────────────────────────────────── 98.8%
    90% ─── 93.3% ─── 94.0% ─────── 94.3% ─── 90.4% ─────── 96.7% ────────
    80% ──────────────────────────────────────────────────────────────────
         Year 1     Year 2          Year 1     Year 2       Year 1   Year 2
         [Linear Regression R²]     [Decision Tree R²]      [K-Means Acc]
```

- **Linear Regression**: Explains 94.04% of variance on future data. The continuous parametric slope extrapolates well during summer heatwaves (ambient temperatures up to 42.8°C).
- **Decision Tree ($d=5$)**: Achieves 90.40% variance explanation on future data. Predictions plateau at extreme demand values (>1,000 kWh) due to leaf-level bounds.
- **PCA + K-Means**: Row normalization improves accuracy from 63.42% to 96.71% on Year 1, and reaches 98.77% on Year 2.

---

## Smart Energy Advisory Engine

### Automated Policy Actions

1. **Night-Heavy Load Shifting (`night_heavy`)**:
   - Detected across 27.8% of Year 2 days (concentrated in exam seasons and weekends).
   - Automated Policy: Trigger automated shutoff of common-room circuits at 23:00; activate passive infrared (PIR) lighting in corridors.
2. **Diurnal Peak Flattening (`daytime_peak`)**:
   - Detected across 49.9% of Year 2 days.
   - Automated Policy: Regulate air conditioning setpoints to 24.5°C from 12:00 to 16:00. Pre-cool study halls during lower morning tariff windows (06:00–08:00).
3. **Block D Structural Audit**:
   - Block D averages 512.9 kWh/day (92% above Block C).
   - Automated Policy: Flag architectural solar exposure; schedule HVAC compressor overhaul and thermal window retrofits.
4. **Extreme Surge Alarm (>95th Percentile = 852.7 kWh)**:
   - System flagged 73 surge occurrences in Year 2 (peak: 1,190.2 kWh on March 13, 2027 in Block D).
   - Automated Policy: Issue dynamic load-shedding notifications to facilities managers when temperature forecasts exceed 38.0°C and active AC counts exceed 25.

---

## Execution Instructions

### Setup
```bash
pip install -r requirements.txt
```

### Run the Complete Pipeline
Launch the notebook containing data ingestion, exploratory analysis, regression tuning, PCA decomposition, normalized clustering, and out-of-sample Year 2 generalization verification:
```bash
jupyter notebook smart_hostel_advisor.ipynb
```
Run all cells sequentially (Phases 1–7). Output plots are saved to `outputs/`.

---

## Technical Implementation Details

### Computational Profiling & Memory Optimization
- **Parser Optimization**: `pandas.read_csv(..., engine='python')` is utilized during memory-constrained executions to avoid large C-buffer allocations.
- **Garbage Collection**: Intermediate DataFrame joins and historical rolling matrices are cleared via explicit `gc.collect()` calls.
- **Multithreading Configuration**: Scikit-Learn KMeans clusters leverage `n_init=20` with fixed seeds (`random_state=42`) to guarantee strict determinism across diverse execution runtimes.
