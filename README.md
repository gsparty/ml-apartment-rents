# Predicting Apartment Rents — Group 2

**Course:** Fundamentals of Python & Applications in Data Science
**Option:** B — Machine Learning

## Setup
pip install -r requirements.txt

## Data
Download listings.csv from https://insideairbnb.com/get-the-data/ (Zurich)
Place it in data/raw/listings.csv

## Notebook order
1. 01_data_ingestion.ipynb  — load & store data
2. 02_eda.ipynb             — exploratory analysis
3. 03_preprocessing.ipynb  — clean, encode, split
4. 04_modeling.ipynb        — Linear Regression + Random Forest
5. 05_evaluation_xai.ipynb — metrics + SHAP plots

## Output
- Trained model: models/rent_pipeline.pkl
- Figures:       reports/figures/
