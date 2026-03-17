# Predicting Apartment Rents — Group 2

**Course:** Fundamentals of Python & Applications in Data Science
**Option:** B — Machine Learning

## Setup
pip install -r requirements.txt

## Data
Download the **detailed** listings file (79 columns) from:
https://insideairbnb.com/get-the-data/ — select Zurich, download listings.csv.gz
Extract and place at: data/raw/listings.csv

IMPORTANT: Use the detailed version (listings.csv.gz, ~79 columns).
Do NOT use the simple listings file (18 columns) — the pipeline will break.

## Reproduction — run notebooks in this exact order
1. 01_data_ingestion.ipynb  — loads raw data, saves to data/processed/
2. 02_eda.ipynb             — EDA plots, saves updated data/processed/listings_eda.csv
3. 03_preprocessing.ipynb  — cleaning, encoding, splitting, saves X_train/X_test/y_train/y_test
4. 04_modeling.ipynb        — trains models, saves models/rent_pipeline.pkl
5. 05_evaluation_xai.ipynb — metrics, residuals, SHAP plots

Each notebook depends on the previous one. Always restart kernel and run all cells.

## Results
| Model             | MAE (CHF) | RMSE (CHF) | R²    |
|-------------------|-----------|------------|-------|
| Linear Regression | 77.40     | 117.30     | 0.254 |
| Random Forest     | 51.85     | 100.89     | 0.448 |

## Key findings (SHAP)
- Bedrooms is the strongest price predictor (mean SHAP ~30)
- Accommodates and latitude follow as second and third
- Private room type reduces predicted price significantly
- Neighbourhood effects are present but individually small
- Remaining variance reflects host discretion not captured by structured features

## Output
- Trained model:  models/rent_pipeline.pkl
- Figures:        reports/figures/
