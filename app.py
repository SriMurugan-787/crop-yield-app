import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingRegressor

st.set_page_config(page_title="Crop Yield Predictor", page_icon="🌾")

CAT = ["Crop", "Season", "State"]
NUM = ["Crop_Year", "log_Area", "Annual_Rainfall", "Fert_per_ha", "Pest_per_ha"]
DATA_PATH = "crop_yield.csv"


@st.cache_data
def load_data(path_or_file):
    raw = pd.read_csv(path_or_file)
    df = raw.copy()
    for c in CAT:
        df[c] = df[c].str.strip()
    df = df[(df.Production > 0) & (df.Yield > 0)].reset_index(drop=True)
    df["Fert_per_ha"] = df.Fertilizer / df.Area
    df["Pest_per_ha"] = df.Pesticide / df.Area
    df["log_Area"] = np.log(df.Area)
    df["log_Yield"] = np.log(df.Yield)
    return df


@st.cache_resource
def train_model(df):
    pre = ColumnTransformer([
        ("c", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT),
        ("n", "passthrough", NUM),
    ])
    model = HistGradientBoostingRegressor(
        max_iter=250, learning_rate=0.1, random_state=0,
        categorical_features=[0, 1, 2],
    )
    pipe = Pipeline([("p", pre), ("m", model)])
    pipe.fit(df[CAT + NUM], df.log_Yield)
    return pipe


st.title("🌾 Crop Yield Predictor")
st.caption("Machine learning model trained on Indian crop data (1997-2020).")

# ---- load data ----
if os.path.exists(DATA_PATH):
    df = load_data(DATA_PATH)
else:
    up = st.file_uploader("Upload crop_yield.csv", type="csv")
    if up is None:
        st.info("Please upload crop_yield.csv to continue.")
        st.stop()
    df = load_data(up)

with st.spinner("Training the model (first time only)..."):
    model = train_model(df)

# ---- inputs ----
state = st.selectbox("State", sorted(df.State.unique()))
d_state = df[df.State == state]

crop = st.selectbox("Crop", sorted(d_state.Crop.unique()))
d_crop = d_state[d_state.Crop == crop]

season = st.selectbox("Season", sorted(d_crop.Season.unique()))

col1, col2, col3 = st.columns(3)
year = col1.number_input("Year", min_value=1997, max_value=2030, value=2020, step=1)
area = col2.number_input("Area (hectares)", min_value=1.0, value=1000.0, step=100.0)
rain_default = float(round(d_state.Annual_Rainfall.median(), 0))
rainfall = col3.number_input("Annual rainfall (mm)", min_value=0.0, value=rain_default, step=50.0)

if st.button("Predict yield", type="primary"):
    # Fertilizer and pesticide per hectare are state-year averages in this
    # dataset, so use the nearest years available for the chosen state.
    ref = d_state.iloc[(d_state.Crop_Year - year).abs().argsort()[:50]]
    row = pd.DataFrame([{
        "Crop": crop, "Season": season, "State": state,
        "Crop_Year": year, "log_Area": np.log(area),
        "Annual_Rainfall": rainfall,
        "Fert_per_ha": ref.Fert_per_ha.median(),
        "Pest_per_ha": ref.Pest_per_ha.median(),
    }])
    pred = float(np.exp(model.predict(row)[0]))

    st.success(f"Predicted yield: **{pred:.2f}** per hectare")
    typical = d_crop[d_crop.Season == season].Yield.median()
    st.write(f"For comparison, the typical historical yield for {crop} in {state} ({season}) is **{typical:.2f}**.")
    if crop.lower() == "coconut":
        st.caption("Coconut yield is measured in nuts per hectare; other crops in tonnes per hectare.")

st.divider()
st.caption(
    "Note: fertilizer and rainfall in this dataset are state-level yearly figures, "
    "so the prediction mainly reflects the crop, state and season."
)
