"""Crop Yield Predictor - simple, easy-to-understand Streamlit app."""
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
UNIT = "tonnes per hectare"  # unit of the Yield column

CAT = [COL_CROP, COL_SEASON, COL_STATE]
FEATURES = CAT + [COL_YEAR, COL_AREA, COL_RAIN]

st.set_page_config(page_title="Crop Yield Predictor", page_icon="🌾", layout="centered")


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
    return model, r2_score(y_test, model.predict(X_test))


def predict_yield(model, crop, season, state, year, area, rainfall):
    """Predict yield for one or many rainfall values."""
    rain = np.atleast_1d(rainfall)
    rows = pd.DataFrame(
        {
            COL_CROP: crop,
            COL_SEASON: season,
            COL_STATE: state,
            COL_YEAR: year,
            COL_AREA: area,
            COL_RAIN: rain,
        }
    )
    return model.predict(rows)


df = load_data()
model, r2 = train_model(df)

# ---------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------
st.title("🌾 Crop Yield Predictor")
st.write(
    "Find out how much of a crop you can expect to harvest. "
    "Answer the three steps below, then click **Predict yield**."
)

# ---------------------------------------------------------------
# STEP 1: WHAT AND WHERE
# ---------------------------------------------------------------
st.subheader("Step 1: What and where?")
c1, c2, c3 = st.columns(3)
with c1:
    state = st.selectbox("State", sorted(df[COL_STATE].unique()))
state_df = df[df[COL_STATE] == state]
with c2:
    crop = st.selectbox("Crop", sorted(state_df[COL_CROP].unique()))
crop_df = state_df[state_df[COL_CROP] == crop]
with c3:
    season = st.selectbox(
        "Season",
        sorted(crop_df[COL_SEASON].unique()),
        help="Kharif = monsoon season, Rabi = winter season, "
        "Summer = hot dry season. Only seasons that exist in the data are shown.",
    )

# ---------------------------------------------------------------
# STEP 2: FARM DETAILS
# ---------------------------------------------------------------
st.subheader("Step 2: Your farm")
c4, c5 = st.columns(2)
with c4:
    year = st.number_input(
        "Year",
        min_value=int(df[COL_YEAR].min()),
        max_value=2035,
        value=int(df[COL_YEAR].max()),
        step=1,
    )
with c5:
    area = st.number_input(
        "Land area (hectares)",
        min_value=1.0,
        value=100.0,
        step=10.0,
        help="1 hectare is about 2.5 acres.",
    )

# ---------------------------------------------------------------
# STEP 3: RAINFALL (simple choices instead of a raw slider)
# ---------------------------------------------------------------
st.subheader("Step 3: Rainfall")
rain_hist = crop_df[COL_RAIN]
scenarios = {
    "Dry year": float(round(rain_hist.quantile(0.2))),
    "Normal year": float(round(rain_hist.median())),
    "Wet year": float(round(rain_hist.quantile(0.8))),
}
choice = st.radio(
    "How much rain do you expect this year?",
    list(scenarios) + ["Enter my own"],
    index=1,
    horizontal=True,
    help="Dry, normal and wet are based on past years for this crop in this "
    "state. Nobody can know the exact rainfall in advance, so try all three.",
)
if choice == "Enter my own":
    rainfall = st.number_input(
        "Rainfall for the whole year (mm)",
        min_value=0.0,
        max_value=float(np.ceil(df[COL_RAIN].max())),
        value=scenarios["Normal year"],
        step=10.0,
    )
else:
    rainfall = scenarios[choice]
    st.caption(
        f"That means about **{rainfall:,.0f} mm** of rain in the year. "
        f"Past range for {crop} in {state}: "
        f"{rain_hist.min():,.0f} to {rain_hist.max():,.0f} mm."
    )

if st.button("Predict yield", type="primary"):
    st.session_state["show_result"] = True

if not st.session_state.get("show_result"):
    st.stop()

# ---------------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------------
pred = float(predict_yield(model, crop, season, state, year, area, rainfall)[0])
usual = crop_df[COL_YIELD].mean()
low_typ = crop_df[COL_YIELD].quantile(0.25)
high_typ = crop_df[COL_YIELD].quantile(0.75)
diff = (pred - usual) / usual * 100 if usual > 0 else 0.0
total = pred * area

st.divider()
st.header("Your result")

# 1. The answer in one sentence
st.success(
    f"You can expect about **{pred:,.2f} {UNIT}** of {crop}. "
    f"On {area:,.0f} hectares, that is about **{total:,.0f} tonnes** in total."
)

# 2. Is that good or bad?
if diff > 10:
    st.info(f"👍 **Better than usual.** About {diff:.0f}% above the usual yield of {usual:,.2f}.")
elif diff < -10:
    st.warning(f"⚠️ **Lower than usual.** About {abs(diff):.0f}% below the usual yield of {usual:,.2f}.")
else:
    st.info(f"👌 **About usual.** The usual yield for {crop} in {state} is {usual:,.2f}.")

st.write(
    f"Most past years for {crop} in {state} ranged from "
    f"**{low_typ:,.2f} to {high_typ:,.2f} {UNIT}**."
)

# 3. Simple chart: what if rainfall changes?
st.subheader("What if the rainfall is different?")
sc_names = list(scenarios)
sc_preds = predict_yield(
    model, crop, season, state, year, area, np.array(list(scenarios.values()))
)
sc_df = pd.DataFrame(
    {f"Yield ({UNIT})": sc_preds},
    index=[f"{n} ({scenarios[n]:,.0f} mm)" for n in sc_names],
)
st.bar_chart(sc_df)
best = sc_names[int(np.argmax(sc_preds))]
st.caption(
    f"With all other inputs unchanged, the model predicts the highest yield in a "
    f"**{best.lower()}**."
)

# 4. Compare with usual
st.subheader("Your prediction vs the usual yield")
st.bar_chart(
    pd.DataFrame(
        {f"Yield ({UNIT})": [pred, usual]},
        index=["Predicted", "Usual average"],
    )
)

# 5. Explanations
with st.expander("How to read these numbers"):
    st.markdown(
        f"""
- **Yield** is how much crop you get from one hectare of land. 1 tonne = 1,000 kg,
  so a yield of 2.5 means about 2,500 kg from every hectare.
- **Total production** = yield × your land area.
- **Usual yield** is the average of past records for {crop} in {state}.
- **Dry / Normal / Wet** rainfall come from past years in the data for this crop and state.
"""
    )

with st.expander("Technical details (for reviewers)"):
    st.write(
        f"Model: Random Forest regression trained on {len(df):,} historical records "
        f"using six inputs: crop, season, state, year, area and rainfall. "
        f"Accuracy on unseen test data: R² = **{r2:.2f}**."
    )
    st.write(
        f"This crop-state combination has {len(crop_df)} past record(s) in the data."
    )

st.caption(
    "This is an estimate based on historical data. Soil quality, pests, fertilizer "
    "and farming practice also affect real yield."
)
