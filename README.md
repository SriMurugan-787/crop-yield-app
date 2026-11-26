# 🌾 Impact of Fertilizer and Rainfall on Crop Yield: A Machine Learning Approach

A Streamlit web app that predicts crop yield and shows how rainfall and fertilizer usage are associated with agricultural production in India.

🔗 **Live app:** https://crop-yield-app-prime.streamlit.app/

<img width="1365" height="687" alt="Screenshot 2026-11-26 190721" src="https://github.com/user-attachments/assets/46d4558f-7bcd-497b-8327-302eb80347b4" />
<img width="1356" height="684" alt="Screenshot 2026-11-26 190755" src="https://github.com/user-attachments/assets/1bd66b76-bc0f-4edd-a9ec-8232057df600" />
<img width="1363" height="685" alt="Screenshot 2026-11-26 190834" src="https://github.com/user-attachments/assets/34ef6c48-42c6-45e9-89d1-405255683574" />



## Features

- **Predict yield and total production** for a chosen state, crop, season, year, land area, rainfall and fertilizer usage
- **Compare with usual yield**: shows whether the prediction is better than, lower than, or about the same as past records for that state, crop and season
- **Rainfall × fertilizer scenarios**: a heatmap of predicted yield for dry / normal / wet years and low / typical / high fertilizer
- **Effect curves**: how predicted yield changes when only rainfall or only fertilizer is varied, plotted against real past records
- **Factor influence**: permutation importance chart showing which inputs the model relies on most

## Tech Stack

Python · Pandas · NumPy · Scikit-learn (Random Forest) · Altair · Streamlit

## Dataset

`crop_yield.csv`: 19,689 historical records of Indian crop production (1997–2020) covering 30 states and 55 crops.

| Column | Description |
|---|---|
| Crop, State, Season | Categorical inputs |
| Crop_Year | Year of the record |
| Area | Cultivated area (hectares) |
| Annual_Rainfall | Rainfall in the year (mm) |
| Fertilizer | Total fertilizer used (kg), converted to kg per hectare in the app |
| Pesticide | Pesticide used (not used by the model) |
| Production, Yield | Output and the target variable (tonnes per hectare) |

Source: [Crop Yield in Indian States Dataset](https://www.kaggle.com/datasets/akshatgupta7/crop-yield-in-indian-states-dataset) on Kaggle (by Akshat Gupta).

## Model

- **Algorithm:** Random Forest Regressor (80 trees, `min_samples_leaf=5`)
- **Features (7):** crop, season, state, year, land area, rainfall, fertilizer per hectare
- **Preprocessing:** one-hot encoding for categorical features
- **Target:** yield (tonnes per hectare)
- **Evaluation:** 80/20 random train-test split, **R² ≈ 0.90** on the test set
- **Interpretation:** permutation importance

## Run Locally

```bash
git clone https://github.com/prime787-Sri/crop-yield-app.git
cd crop-yield-app
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

```
├── app.py              # Streamlit app (data loading, model, UI, charts)
├── crop_yield.csv      # Dataset
├── requirements.txt    # Dependencies
└── README.md
```

## Key Findings

- **Crop type dominates the model (about 82% of influence)**, followed by season (8%) and state (4%). Rainfall (3%) and fertilizer (about 0%) have a much smaller role in this dataset.
- In the scenario heatmap, changes from dry to wet rainfall or low to high fertilizer move the predicted yield only by a few percent for the example tested.
- A main reason is that yield units differ across crops (for example, coconut is recorded in nuts, so its "yield" is in the thousands while most crops are below 10), so the crop label explains most of the variation.

These are patterns in historical data, not proof of cause and effect.

## Limitations

- The model learns **associations** in historical data. It does not prove that changing rainfall or fertilizer will cause a change in yield.
- The train-test split is random, so the R² may be optimistic for completely new years or regions.
- Soil quality, pests, irrigation and farming practices are not in the data, so predictions are estimates only.
- Yield units are not consistent across crops (e.g., coconut), which inflates the importance of crop type.
- Reliability is lower for state-crop-season combinations with few past records (the app warns about this).

## Acknowledgement

This was a learning project. As a beginner in Machine Learning and Python, I used AI tools to help understand concepts, write and debug code, and build the app.

## Author

**T Sri Murugan**  
Diploma student, Electrical and Electronics Engineering, SASTRA University  
[LinkedIn](https://www.linkedin.com/in/srimurugan07/)
