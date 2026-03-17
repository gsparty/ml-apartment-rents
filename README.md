# Predicting Apartment Rents — Group 2

**Course:** Fundamentals of Python & Applications in Data Science
**Option:** B — Machine Learning

## Setup
pip install -r requirements.txt
# or
conda env create -f environment.yml
conda activate ml-apartment-rents

## Data
Download the **detailed** listings file (79 columns, ~3,400 rows) from:
https://insideairbnb.com/get-the-data/ — select Zurich, download listings.csv.gz
Extract and place at: data/raw/listings.csv

Important: do NOT use the simple listings file (18 columns). The pipeline requires
the detailed version with columns such as accommodates, bedrooms, bathrooms, etc.

## Notebook order
1. 01_data_ingestion.ipynb  — load raw data, first sanity check, save to processed/
2. 02_eda.ipynb             — price distributions, boxplots by room type, correlation matrix
3. 03_preprocessing.ipynb  — outlier cap at 99th percentile, log-transform price,
                              encode categoricals, StandardScaler, 80/20 train/test split
4. 04_modeling.ipynb        — Linear Regression baseline vs Random Forest (100 trees),
                              5-fold cross-validation, save best model
5. 05_evaluation_xai.ipynb — final metrics, residual plot, SHAP summary + bar plots

## Results
| Model             | MAE (CHF) | RMSE (CHF) | R²    |
|-------------------|-----------|------------|-------|
| Linear Regression | 77.40     | 117.30     | 0.254 |
| Random Forest     | 51.85     | 100.89     | 0.448 |

Random Forest outperforms Linear Regression across all metrics.
Prices were log-transformed before training and reversed for evaluation.

## Key findings (SHAP)
- Bedrooms is the strongest predictor (mean SHAP ~30)
- Accommodates and latitude are second and third
- Private room type reduces predicted price significantly
- Neighbourhood effects are present but individually small
- Low R2 reflects host pricing discretion not captured by structured features

## Output
- Trained model:  models/rent_pipeline.pkl
- Figures:        reports/figures/
