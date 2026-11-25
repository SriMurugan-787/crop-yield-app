"""Crop Yield Predictor - Streamlit app with a clear, readable output."""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ---------------------------------------------------------------
# SETTINGS: change these if your CSV uses different column names
# ---------------------------------------------------------------
DATA_FILE = "crop_yield.csv"
COL_CROP, COL_SEASON, COL_STATE = "Crop", "Season", "State"
COL_YEAR, COL_AREA, COL_RAIN = "Crop_Year", "Area", "Annual_Rainfall"
COL_YIELD = "Yield"
UNIT = "t/ha"  # unit of the Yield column (tonnes per hectare)

CAT = [COL_CROP, COL_SEASON, COL_STATE]
FEATURES = CAT + [COL_YEAR, COL_AREA, COL_RAIN]

st.set_page_config(page_title="Crop Yield Predictor", page_icon="🌾", layout="wide")


# ---------------------------------------------------------------
# DATA + MODEL
# ---------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)
    for c in CAT:
        df[c] = df[c].astype(str).str.strip()
    if COL_YIELD not in df.columns and "Production" in df.columns:
        df[COL_YIELD] = df["Production"] / df[COL_AREA]
    return df.dropna(subset=FEATURES + [COL_YIELD]).reset_index(drop=True)


@st.cache_resource
def train_model(_df):
    X, y = _df[FEATURES], _df[COL_YIELD]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CAT)],
        remainder="passthrough",
    )
    model = Pipeline(
        [
            ("pre", pre),
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=150, min_samples_leaf=2, random_state=42, n_jobs=-1
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    r2 = r2_score(y_test, model.predict(X_test))
    return model, r2


def predict_yield(model, crop, season, state, year, area, rainfall):
    """Predict yield (UNIT) for one combination of inputs."""
    row = pd.DataFrame(
        [
            {
                COL_CROP: crop,
                COL_SEASON: season,
                COL_STATE: state,
                COL_YEAR: year,
                COL_AREA: area,
                COL_RAIN: rainfall,
            }
        ]
    )
    return float(model.predict(row)[0])


df = load_data()
model, r2 = train_model(df)

# ---------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------
st.title("🌾 Crop Yield Predictor")
st.write(
    "Pick a state, crop and season, set the year, area and rainfall, "
    "then click **Predict yield**. The result is compared with the "
    "historical average so you can see whether it is good or bad."
)

# ---------------------------------------------------------------
# INPUTS (sidebar). Crop and season lists only show combinations
# that actually exist in the data for the chosen state.
# ---------------------------------------------------------------
with st.sidebar:
    st.header("Inputs")
    state = st.selectbox("State", sorted(df[COL_STATE].unique()))
    state_df = df[df[COL_STATE] == state]

    crop = st.selectbox("Crop", sorted(state_df[COL_CROP].unique()))
    crop_df = state_df[state_df[COL_CROP] == crop]

    season = st.selectbox("Season", sorted(crop_df[COL_SEASON].unique()))

    year = st.number_input(
        "Year",
        min_value=int(df[COL_YEAR].min()),
        max_value=2035,
        value=int(df[COL_YEAR].max()),
        step=1,
    )
    area = st.number_input("Area (hectares)", min_value=1.0, value=100.0, step=10.0)

    r_min = float(np.floor(df[COL_RAIN].min()))
    r_max = float(np.ceil(df[COL_RAIN].max()))
    default_rain = float(np.clip(round(crop_df[COL_RAIN].mean()), r_min, r_max))
    rainfall = st.slider("Annual rainfall (mm)", r_min, r_max, default_rain, step=1.0)

    go = st.button("Predict yield", type="primary")

if not go:
    st.info("Choose your inputs on the left and click **Predict yield**.")
    st.stop()

# ---------------------------------------------------------------
# PREDICTION + COMPARISON
# ---------------------------------------------------------------
pred = predict_yield(model, crop, season, state, year, area, rainfall)
hist = crop_df[COL_YIELD].mean()
n_records = len(crop_df)
diff = (pred - hist) / hist * 100 if hist > 0 else 0.0
total = pred * area

# 1. Headline numbers
st.subheader("Result")
c1, c2, c3 = st.columns(3)
c1.metric("Predicted yield", f"{pred:.2f} {UNIT}", f"{diff:+.1f}% vs average")
c2.metric(f"Average for {crop} in {state}", f"{hist:.2f} {UNIT}")
c3.metric("Expected total production", f"{total:,.0f} tonnes")

# 2. Plain-language verdict
if diff > 10:
    st.success(
        f"Good outlook: expected yield is {diff:.0f}% above the usual "
        f"{crop} yield in {state}."
    )
elif diff < -10:
    st.warning(
        f"Below-average outlook: expected yield is {abs(diff):.0f}% below "
        f"the usual {crop} yield in {state}."
    )
else:
    st.info("Expected yield is close to the usual average for this crop and state.")

st.write(
    f"For **{area:,.0f} hectares** of **{crop}** in **{state}** "
    f"({season} season, {year}) with **{rainfall:,.0f} mm** of rainfall, "
    f"the model expects about **{pred:.2f} {UNIT}**, which is roughly "
    f"**{total:,.0f} tonnes** in total."
)

if n_records < 5:
    st.caption(
        f"Note: only {n_records} historical record(s) exist for this crop in "
        f"{state}, so the comparison is less reliable."
    )

# 3. Two simple charts
left, right = st.columns(2)

with left:
    st.markdown("**Predicted vs historical average**")
    bars = pd.DataFrame(
        {f"Yield ({UNIT})": [pred, hist]},
        index=["Predicted", "Historical average"],
    )
    st.bar_chart(bars)

with right:
    st.markdown("**How rainfall changes the prediction**")
    lo = float(crop_df[COL_RAIN].min())
    hi = float(crop_df[COL_RAIN].max())
    if hi - lo < 50:  # too little spread in the data, use the full range
        lo, hi = r_min, r_max
    grid = np.linspace(lo, hi, 40)
    sweep = pd.DataFrame(
        {
            COL_CROP: crop,
            COL_SEASON: season,
            COL_STATE: state,
            COL_YEAR: year,
            COL_AREA: area,
            COL_RAIN: grid,
        }
    )
    curve = pd.DataFrame(
        {f"Predicted yield ({UNIT})": model.predict(sweep)},
        index=pd.Index(grid.round(0), name="Rainfall (mm)"),
    )
    st.line_chart(curve)
    st.caption(
        "Everything except rainfall is held fixed, so you can see how "
        "sensitive the prediction is to rain."
    )

# 4. Transparency: inputs, model quality, caveat
with st.expander("Inputs used and model details"):
    st.table(
        pd.DataFrame(
            {
                "Input": ["State", "Crop", "Season", "Year", "Area", "Rainfall"],
                "Value": [
                    state,
                    crop,
                    season,
                    str(year),
                    f"{area:,.0f} ha",
                    f"{rainfall:,.0f} mm",
                ],
            }
        )
    )
    st.write(
        f"Model: Random Forest regression trained on {len(df):,} historical "
        f"records. R² on held-out test data: **{r2:.2f}**."
    )

st.caption(
    "This is an estimate based on historical data. Soil quality, pests, "
    "fertilizer and farming practice also affect real yield."
)
