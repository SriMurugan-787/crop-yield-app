"""Crop Yield Predictor - simple app with detailed, labelled charts."""
import altair as alt
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

BLUE, GREEN, ORANGE, RED, GREY = "#4c8bf5", "#2e9e5b", "#e8833a", "#e45756", "#9aa0a6"
YLAB = f"Yield ({UNIT})"

st.set_page_config(page_title="Crop Yield Predictor", page_icon="🌾", layout="centered")


def _is_dark():
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return False


TEXT = "#fafafa" if _is_dark() else "#31333f"


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
    rows = pd.DataFrame(
        {
            COL_CROP: crop,
            COL_SEASON: season,
            COL_STATE: state,
            COL_YEAR: year,
            COL_AREA: area,
            COL_RAIN: np.atleast_1d(rainfall),
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
# past records for exactly this state + crop + season (used for all comparisons)
ref_df = crop_df[crop_df[COL_SEASON] == season]

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
rain_hist = ref_df[COL_RAIN]
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
    "state and season. Nobody can know the exact rainfall in advance, so "
    "try all three.",
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
        f"Past range for {crop} in {state} ({season}): "
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
usual = ref_df[COL_YIELD].mean()
low_typ = ref_df[COL_YIELD].quantile(0.25)
high_typ = ref_df[COL_YIELD].quantile(0.75)
diff = (pred - usual) / usual * 100 if usual > 0 else 0.0
total = pred * area
where = f"{crop} in {state} ({season})"

st.divider()
st.header("Your result")

st.success(
    f"You can expect about **{pred:,.2f} {UNIT}** of {crop}. "
    f"On {area:,.0f} hectares, that is about **{total:,.0f} tonnes** in total."
)

if diff > 10:
    st.info(f"👍 **Better than usual.** About {diff:.0f}% above the usual yield of {usual:,.2f}.")
elif diff < -10:
    st.warning(f"⚠️ **Lower than usual.** About {abs(diff):.0f}% below the usual yield of {usual:,.2f}.")
else:
    st.info(f"👌 **About usual.** The usual yield for {where} is {usual:,.2f}.")

st.write(
    f"Most past years for {where} ranged from "
    f"**{low_typ:,.2f} to {high_typ:,.2f} {UNIT}**."
)
if len(ref_df) < 5:
    st.caption(
        f"Note: only {len(ref_df)} past record(s) exist for this exact "
        f"combination, so the comparisons are less reliable."
    )

# ---------------------------------------------------------------
# CHART 1: dry / normal / wet bars with value labels
# ---------------------------------------------------------------
st.subheader("1. What if the rainfall is different?")
sc_names = list(scenarios)
sc_rain = np.array(list(scenarios.values()))
sc_preds = predict_yield(model, crop, season, state, year, area, sc_rain)
sc_df = pd.DataFrame(
    {"Scenario": sc_names, "Rainfall (mm)": sc_rain, "Yield": sc_preds}
)
avg_df = pd.DataFrame({"Yield": [usual], "label": [f"Usual average: {usual:.2f}"]})

bars = (
    alt.Chart(sc_df)
    .mark_bar(size=70)
    .encode(
        x=alt.X("Scenario:N", sort=sc_names, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y(
            "Yield:Q",
            title=YLAB,
            scale=alt.Scale(domain=[0, float(max(sc_preds.max(), usual)) * 1.15]),
        ),
        color=alt.Color(
            "Scenario:N",
            scale=alt.Scale(domain=sc_names, range=[ORANGE, BLUE, GREEN]),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("Scenario:N"),
            alt.Tooltip("Rainfall (mm):Q", format=",.0f"),
            alt.Tooltip("Yield:Q", format=".2f", title=YLAB),
        ],
    )
)
bar_labels = (
    alt.Chart(sc_df)
    .mark_text(dy=-10, fontSize=15, fontWeight="bold", color=TEXT)
    .encode(
        x=alt.X("Scenario:N", sort=sc_names),
        y="Yield:Q",
        text=alt.Text("Yield:Q", format=".2f"),
    )
)
avg_rule = (
    alt.Chart(avg_df)
    .mark_rule(strokeDash=[6, 4], color=TEXT, opacity=0.7)
    .encode(y="Yield:Q")
)
avg_text = (
    alt.Chart(avg_df)
    .mark_text(align="left", dy=-8, color=TEXT, fontSize=12)
    .encode(x=alt.value(4), y="Yield:Q", text="label:N")
)
st.altair_chart((bars + bar_labels + avg_rule + avg_text).properties(height=320))
st.caption(
    "Rainfall used: "
    + ", ".join(f"{n} = {r:,.0f} mm" for n, r in zip(sc_names, sc_rain))
    + ". The dashed line is the usual average yield."
)

# ---------------------------------------------------------------
# CHART 2: past years with your prediction highlighted
# ---------------------------------------------------------------
st.subheader("2. Your prediction compared with past years")
hy = (
    ref_df.groupby(COL_YEAR, as_index=False)[COL_YIELD]
    .mean()
    .rename(columns={COL_YEAR: "Year", COL_YIELD: "Yield"})
)
pred_df = pd.DataFrame(
    {"Year": [year], "Yield": [pred], "label": [f"Predicted: {pred:.2f}"]}
)
x_year = alt.X(
    "Year:Q",
    axis=alt.Axis(format="d", tickMinStep=1),
    scale=alt.Scale(zero=False, nice=False, padding=15),
)
past_line = (
    alt.Chart(hy)
    .mark_line(point=True, color=BLUE)
    .encode(
        x=x_year,
        y=alt.Y("Yield:Q", title=YLAB),
        tooltip=[
            alt.Tooltip("Year:Q", format="d"),
            alt.Tooltip("Yield:Q", format=".2f", title=YLAB),
        ],
    )
)
pred_pt = (
    alt.Chart(pred_df)
    .mark_point(shape="diamond", size=320, filled=True, color=RED, opacity=1)
    .encode(x="Year:Q", y="Yield:Q")
)
pred_txt = (
    alt.Chart(pred_df)
    .mark_text(dy=-20, dx=8, align="right", fontSize=14, fontWeight="bold", color=TEXT)
    .encode(x="Year:Q", y="Yield:Q", text="label:N")
)
st.altair_chart(
    (past_line + pred_pt + pred_txt + avg_rule + avg_text).properties(height=320)
)
st.caption(
    "Blue line = average yield in each past year. Red diamond = your prediction. "
    "Dashed line = usual average."
)

# ---------------------------------------------------------------
# CHART 3: rainfall vs yield (real records + model curve)
# ---------------------------------------------------------------
st.subheader("3. How rainfall changes the yield")
lo, hi = float(rain_hist.min()), float(rain_hist.max())
if hi - lo < 50:  # too little spread in the data, use the full range
    lo, hi = float(df[COL_RAIN].min()), float(df[COL_RAIN].max())
grid = np.linspace(lo, hi, 60)
curve_df = pd.DataFrame(
    {
        "Rainfall": grid,
        "Yield": predict_yield(model, crop, season, state, year, area, grid),
    }
)
past_df = ref_df[[COL_RAIN, COL_YIELD]].rename(
    columns={COL_RAIN: "Rainfall", COL_YIELD: "Yield"}
)
scen_pts = pd.DataFrame(
    {"Rainfall": sc_rain, "Yield": sc_preds, "label": sc_names}
)
you_df = pd.DataFrame({"Rainfall": [rainfall], "Yield": [pred]})

x_rain = alt.X(
    "Rainfall:Q",
    title="Rainfall in the year (mm)",
    scale=alt.Scale(zero=False, nice=False, padding=15),
)
past_dots = (
    alt.Chart(past_df)
    .mark_circle(size=45, color=GREY, opacity=0.45)
    .encode(
        x=x_rain,
        y=alt.Y("Yield:Q", title=YLAB, scale=alt.Scale(zero=False)),
        tooltip=[
            alt.Tooltip("Rainfall:Q", format=",.0f", title="Rainfall (mm)"),
            alt.Tooltip("Yield:Q", format=".2f", title="Past yield"),
        ],
    )
)
curve = (
    alt.Chart(curve_df)
    .mark_line(color=GREEN, strokeWidth=3)
    .encode(x="Rainfall:Q", y="Yield:Q")
)
scen_marks = (
    alt.Chart(scen_pts)
    .mark_point(size=110, filled=True, color=ORANGE, opacity=1)
    .encode(x="Rainfall:Q", y="Yield:Q")
)
scen_txt = (
    alt.Chart(scen_pts)
    .mark_text(dy=-16, fontSize=12, color=TEXT)
    .encode(x="Rainfall:Q", y="Yield:Q", text="label:N")
)
you_mark = (
    alt.Chart(you_df)
    .mark_point(shape="diamond", size=320, filled=False, strokeWidth=3, color=RED)
    .encode(x="Rainfall:Q", y="Yield:Q")
)
st.altair_chart(
    (past_dots + curve + scen_marks + scen_txt + you_mark).properties(height=340)
)
st.caption(
    "Grey dots = real past records. Green line = what the model predicts for your "
    "inputs at different rainfall levels. Orange dots = dry / normal / wet year. "
    "Red diamond = your choice."
)

# ---------------------------------------------------------------
# DETAILS
# ---------------------------------------------------------------
with st.expander("See the past records used for comparison"):
    table = ref_df[[COL_YEAR, COL_RAIN, COL_AREA, COL_YIELD]].sort_values(COL_YEAR)
    table.columns = ["Year", "Rainfall (mm)", "Area (ha)", YLAB]
    st.dataframe(table, hide_index=True)

with st.expander("How to read these numbers"):
    st.markdown(
        f"""
- **Yield** is how much crop you get from one hectare of land. 1 tonne = 1,000 kg,
  so a yield of 2.5 means about 2,500 kg from every hectare.
- **Total production** = yield × your land area.
- **Usual yield** is the average of past records for {where}.
- **Dry / Normal / Wet** rainfall come from past years in the data for {where}.
"""
    )

with st.expander("Technical details (for reviewers)"):
    st.write(
        f"Model: Random Forest regression trained on {len(df):,} historical records "
        f"using six inputs: crop, season, state, year, area and rainfall. "
        f"Accuracy on unseen test data: R² = **{r2:.2f}**."
    )
    st.write(f"This combination has {len(ref_df)} past record(s) in the data.")

st.caption(
    "This is an estimate based on historical data. Soil quality, pests, fertilizer "
    "and farming practice also affect real yield."
)
