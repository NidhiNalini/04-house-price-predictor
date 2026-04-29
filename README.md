# Project 04 — House Price Predictor

**Tools:** Python · Pandas · scikit-learn · Matplotlib · Seaborn  
**Model:** Linear Regression  
**Dataset:** California Housing Dataset (scikit-learn built-in, 1990 US Census)

---

## Research Question
Which neighbourhood-level factors most strongly predict median house values in California, and how accurately can a Linear Regression model estimate them?

---

## What I Built
An end-to-end ML pipeline that takes raw census data through EDA, cleaning, feature scaling, model training, and honest evaluation — with findings communicated in plain English.

---

## Dataset
- **Source:** California Housing Dataset (built into scikit-learn)
- **Rows:** ~20,000 block groups (after cleaning)
- **Features:** 8 numerical inputs — income, house age, rooms, bedrooms, population, occupancy, latitude, longitude
- **Target:** Median house value per block group (in $100,000s)
- **Known issue:** Values above $500,000 are capped at $500,000 — documented as a limitation

---

## Methodology
1. **Data audit** — inspected shape, data types, missing values, and target distribution
2. **EDA** — 4 charts: target distribution, correlation heatmap, income vs value scatter, geographic map
3. **Cleaning** — removed capped rows (MedHouseVal == 5.0), capped extreme outliers at 99th percentile
4. **Scaling** — StandardScaler applied to training data only (no data leakage)
5. **Modelling** — Linear Regression trained on 80% of data, evaluated on held-out 20%
6. **Evaluation** — RMSE, MAE, R², residual plot, actual vs predicted chart

---

## Key Findings

**1. Income is the strongest predictor**  
Median neighbourhood income has a correlation of 0.69 with house value — the highest of any single feature. Each standard deviation increase in income corresponds to approximately $84,000 increase in predicted house value.

**2. Geography carries real signal**  
Latitude and Longitude together reveal that coastal California blocks (San Francisco, Los Angeles) command significantly higher prices. This aligns with well-known real estate patterns in the region.

**3. Model explains 63% of variance (R² = 0.63)**  
The remaining 37% is driven by factors not in this dataset: school district quality, walkability scores, property condition, and proximity to amenities. This is expected for a simple linear model on macro census data.

**4. Average prediction error is ~$71,000 (RMSE)**  
For mid-range properties at $250,000, this represents a ~28% error margin. Suitable for regional trend analysis — not for individual property valuation.

---

## Limitations
- **Data cap at $500K** — the model cannot reliably predict high-end properties
- **1990 data** — applying this model to current markets would produce unreliable results (model drift)
- **Linear assumptions** — house price relationships are non-linear; Ridge Regression or tree-based models would improve R²
- **Missing features** — school quality, crime rates, and amenity proximity are significant drivers not captured here

---

## Charts
| Chart | What it shows |
|-------|---------------|
| `chart_01_target_distribution.png` | Histogram + boxplot of house values — reveals the $500K data cap |
| `chart_02_correlation_heatmap.png` | Feature correlations — income is the dominant predictor |
| `chart_03_income_vs_value.png` | Income vs house value scatter — clear positive relationship |
| `chart_04_geographic_distribution.png` | California map coloured by house value — coastal premium visible |
| `chart_05_residuals_and_predictions.png` | Residual plot + actual vs predicted — model diagnostic |


