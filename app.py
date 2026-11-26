"""Impact of Fertilizer and Rainfall on Crop Yield - Streamlit app."""
import base64
import io
import os
import re
import traceback

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ---------------------------------------------------------------
# SETTINGS: change these if your CSV uses different column names
# ---------------------------------------------------------------
DATA_FILE = "crop_yield.csv"
BACKGROUND_FILE = "background.jpg"  # used only when there are no slide pictures
FONT_NAME = "Work Sans" # any font name from fonts.google.com, or None for the default font
# Background slideshow pictures: put them in the folder below, OR name them slide1.jpg, slide2.jpg ... next to app.py
SLIDES_DIR = "slides"
SLIDE_SECONDS = 6  # seconds each background picture stays on screen
FADE_SECONDS = 1.5  # seconds for each fade in / fade out
SLIDE_MAX_WIDTH = 1600  # bigger pictures are shrunk to this width so the page loads fast
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."
COL_CROP, COL_SEASON, COL_STATE = "Crop", "Season", "State"
COL_YEAR, COL_AREA, COL_RAIN = "Crop_Year", "Area", "Annual_Rainfall"
COL_FERT = "Fertilizer"
COL_YIELD = "Yield"
# True  -> the Fertilizer column is the TOTAL kg used on the whole area
# False -> the Fertilizer column is already kg per hectare
FERT_IS_TOTAL = True

COL_FPH = "Fertilizer_per_ha"  # created below: fertilizer in kg per hectare
UNIT = "tonnes per hectare"  # unit of the Yield column
FUNIT = "kg per hectare"

CAT = [COL_CROP, COL_SEASON, COL_STATE]
FEATURES = CAT + [COL_YEAR, COL_AREA, COL_RAIN, COL_FPH]
FRIENDLY = {
    COL_CROP: "Crop type",
    COL_SEASON: "Season",
    COL_STATE: "State",
    COL_YEAR: "Year",
    COL_AREA: "Land area",
    COL_RAIN: "Rainfall",
    COL_FPH: "Fertilizer",
}

BLUE, GREEN, ORANGE, RED, GREY = "#4c8bf5", "#2e9e5b", "#e8833a", "#e45756", "#9aa0a6"
YLAB = f"Yield ({UNIT})"

st.set_page_config(page_title="Fertilizer, Rainfall & Crop Yield", page_icon="🌾", layout="wide")


def _is_dark():
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return False


TEXT = "#ffffff"  # the content box is always black, so chart text is white


IMAGE_TYPES = (".jpg", ".jpeg", ".png", ".webp")


def _natural(name):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name.lower())]


def _prepare(path):
    """Read a picture, shrink it if it is huge, and return it as base64 JPEG text."""
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if img.width > SLIDE_MAX_WIDTH:
        new_h = round(img.height * SLIDE_MAX_WIDTH / img.width)
        img = img.resize((SLIDE_MAX_WIDTH, new_h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=80, optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


@st.cache_data
def load_background_images():
    """Slide pictures for the background (or background.jpg if there are none)."""
    paths = []
    folder = os.path.join(HERE, SLIDES_DIR)
    if os.path.isdir(folder):
        paths += [os.path.join(folder, n) for n in os.listdir(folder)]
    paths += [
        os.path.join(HERE, n)
        for n in os.listdir(HERE)
        if n.lower().startswith("slide") and os.path.isfile(os.path.join(HERE, n))
    ]
    paths = [p for p in paths if p.lower().endswith(IMAGE_TYPES)]
    paths.sort(key=lambda p: _natural(os.path.basename(p)))
    images = []
    for p in paths[:12]:
        try:
            images.append(_prepare(p))
        except Exception:
            continue  # skip any file that is not a readable picture
    if not images:
        fallback = os.path.join(HERE, BACKGROUND_FILE)
        if os.path.exists(fallback):
            try:
                images.append(_prepare(fallback))
            except Exception:
                pass
    return images


def _background_css(images):
    """CSS for a full-page background: one picture, or pictures that fade in and out."""
    if not images:
        return ""
    n = len(images)
    variables = "".join(
        f'--s{i}: url("data:image/jpeg;base64,{b64}");' for i, b64 in enumerate(images)
    )
    if n == 1:
        layer = "opacity: 1; background-image: var(--s0);"
        keyframes = ""
        reduced = ""
    else:
        t = float(SLIDE_SECONDS)
        f = max(0.2, min(float(FADE_SECONDS), t / 2 - 0.1))
        total = n * t
        eps = 0.02  # picture is swapped while it is fully transparent

        def pct(sec):
            return f"{sec / total * 100:.4f}%"

        frames = ""
        for k in range(n):
            start, end = k * t, (k + 1) * t
            frames += f"{pct(start)} {{ opacity: 0; background-image: var(--s{k}); }}\n"
            frames += f"{pct(start + f)} {{ opacity: 1; }}\n"
            frames += f"{pct(end - f)} {{ opacity: 1; }}\n"
            frames += f"{pct(end - eps)} {{ opacity: 0; background-image: var(--s{k}); }}\n"
        frames += f"100% {{ opacity: 0; background-image: var(--s{n - 1}); }}\n"
        keyframes = "@keyframes bgslides {\n" + frames + "}"
        layer = f"opacity: 0; animation: bgslides {total:g}s linear infinite;"
        reduced = (
            "@media (prefers-reduced-motion: reduce) { .stApp::before "
            "{ animation: none; opacity: 1; background-image: var(--s0); } }"
        )
    return f"""
.stApp {{ {variables} isolation: isolate; background: #000 !important; }}
.stApp::before {{
    content: "";
    position: fixed;
    top: 0; left: 0; width: 100%; height: 100%;
    z-index: -1;
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    {layer}
}}
[data-testid="stAppViewContainer"] {{ background: transparent !important; }}
{keyframes}
{reduced}
"""


def apply_background():
    """Fading picture slideshow behind the page + a large black box so every text is readable."""
    photo_css = _background_css(load_background_images())
    text_css = ""
    if not _is_dark():  # light-theme viewers: force white text on the black box
        text_css = """
[data-testid="stMainBlockContainer"] { color: #ffffff; }
[data-testid="stMainBlockContainer"] :is(h1, h2, h3, h4, h5, h6, p, li, label, summary, a,
    [data-testid="stCaptionContainer"]) { color: #ffffff !important; }
[data-testid="stMainBlockContainer"] [data-testid="stAlert"] :is(p, li, a, strong, code) {
    color: #1f2937 !important;
}"""
    st.html(
        f"""
<style>
{photo_css}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stMain"] {{ background: transparent; }}
[data-testid="stMainBlockContainer"], .block-container {{
    background: rgba(0, 0, 0, 0.90);
    border: 1px solid rgba(255, 255, 255, 0.10);
    border-radius: 18px;
    max-width: 1100px !important;
    margin: 1.5rem auto !important;
    padding: 2.5rem 2.5rem 3rem 2.5rem !important;
    box-shadow: 0 6px 30px rgba(0, 0, 0, 0.45);
}}
[data-testid="stExpander"] details {{ border-color: rgba(255, 255, 255, 0.25) !important; }}
@media (max-width: 640px) {{
    [data-testid="stMainBlockContainer"], .block-container {{
        padding: 1.5rem 1rem 2rem 1rem !important;
        border-radius: 12px;
    }}
}}
{text_css}
</style>
"""
    )


def show_chart(chart):
    """Draw an Altair chart with light text/lines so it is readable on the black box."""
    chart = (
        chart.configure(background="transparent")
        .configure_view(stroke=None)
        .configure_axis(
            labelColor="#e8e8e8",
            titleColor="#ffffff",
            gridColor="rgba(255,255,255,0.15)",
            domainColor="rgba(255,255,255,0.45)",
            tickColor="rgba(255,255,255,0.45)",
        )
        .configure_legend(labelColor="#e8e8e8", titleColor="#ffffff")
    )
    st.altair_chart(chart, theme=None)


def apply_font():
    """Load a Google Font and use it for the page text (code blocks and icons keep their own fonts)."""
    if not FONT_NAME:
        return
    family = FONT_NAME.replace(" ", "+")
    st.html(
        f"""
<style>
@import url('https://fonts.googleapis.com/css?family={family}:400,500,600,700&display=swap');
html, body, .stApp,
.stApp p, .stApp li, .stApp label, .stApp a, .stApp summary,
.stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp button, .stApp input, .stApp textarea,
.stApp [data-baseweb="select"] div,
.stApp [data-baseweb="radio"] div,
.stApp [data-testid="stCaptionContainer"] {{
    font-family: '{FONT_NAME}', sans-serif;
}}
</style>
"""
    )


apply_background()
apply_font()


# ---------------------------------------------------------------
# DATA + MODEL
# ---------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(HERE, DATA_FILE))
    for c in CAT:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    if COL_YIELD not in df.columns and {"Production", COL_AREA} <= set(df.columns):
        df[COL_YIELD] = df["Production"] / df[COL_AREA]
    if COL_FERT in df.columns and COL_AREA in df.columns:
        df = df[df[COL_AREA] > 0]
        df[COL_FPH] = df[COL_FERT] / df[COL_AREA] if FERT_IS_TOTAL else df[COL_FERT]
    needed = [c for c in FEATURES + [COL_YIELD] if c in df.columns]
    return df.dropna(subset=needed).reset_index(drop=True)


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
                    n_estimators=80,
                    min_samples_leaf=5,
                    max_features=0.5,
                    max_samples=0.7,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    r2 = r2_score(y_test, model.predict(X_test))
    # How much does accuracy drop when each factor is shuffled? (bigger = more influence)
    sample = X_test.sample(min(1500, len(X_test)), random_state=0)
    perm = permutation_importance(
        model, sample, y_test.loc[sample.index], n_repeats=3, random_state=42, n_jobs=1
    )
    importance = dict(zip(FEATURES, np.clip(perm.importances_mean, 0, None)))
    return model, r2, importance


def predict_yield(model, crop, season, state, year, area, rainfall, fert_ha):
    """Predict yield. rainfall and fert_ha can be single values or arrays."""
    rain, fert = np.broadcast_arrays(
        np.atleast_1d(rainfall).astype(float), np.atleast_1d(fert_ha).astype(float)
    )
    rows = pd.DataFrame(
        {
            COL_CROP: crop,
            COL_SEASON: season,
            COL_STATE: state,
            COL_YEAR: year,
            COL_AREA: area,
            COL_RAIN: rain,
            COL_FPH: fert,
        }
    )
    return model.predict(rows)


try:
    df = load_data()
    missing = [c for c in FEATURES + [COL_YIELD] if c not in df.columns]
    missing = [COL_FERT if c == COL_FPH else c for c in missing]  # show the CSV column name
    if missing:
        raise ValueError(
            f"These columns are missing from {DATA_FILE}: {', '.join(missing)}. "
            "Check the column names in the SETTINGS section at the top of app.py."
        )
    with st.spinner("Getting ready: training the model (only on first load)..."):
        model, r2, importance = train_model(df)
except Exception:
    st.error("The app could not start. The details below show why.")
    st.code(traceback.format_exc())
    try:
        st.write("Files found next to app.py:", sorted(os.listdir(HERE)))
    except Exception:
        pass
    st.stop()

# ---------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------
st.title("Impact of Fertilizer and Rainfall on Crop Yield")
st.write(
    "Find out how much of a crop you can expect to harvest, and how much rainfall "
    "and fertilizer change it. Answer the four steps below, then click "
    "**Predict yield**."
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
# STEP 3: RAINFALL
# ---------------------------------------------------------------
st.subheader("Step 3: Rainfall")
rain_hist = ref_df[COL_RAIN]
rain_sc = {
    "Dry year": float(round(rain_hist.quantile(0.2))),
    "Normal year": float(round(rain_hist.median())),
    "Wet year": float(round(rain_hist.quantile(0.8))),
}
rain_choice = st.radio(
    "How much rain do you expect this year?",
    list(rain_sc) + ["Enter my own"],
    index=1,
    horizontal=True,
    help="Dry, normal and wet are based on past years for this crop in this "
    "state and season. Nobody can know the exact rainfall in advance, so try all three.",
)
if rain_choice == "Enter my own":
    rainfall = st.number_input(
        "Rainfall for the whole year (mm)",
        min_value=0.0,
        max_value=float(np.ceil(df[COL_RAIN].max())),
        value=rain_sc["Normal year"],
        step=10.0,
    )
else:
    rainfall = rain_sc[rain_choice]
    st.caption(
        f"That means about **{rainfall:,.0f} mm** of rain in the year. "
        f"Past range for {crop} in {state} ({season}): "
        f"{rain_hist.min():,.0f} to {rain_hist.max():,.0f} mm."
    )

# ---------------------------------------------------------------
# STEP 4: FERTILIZER
# ---------------------------------------------------------------
st.subheader("Step 4: Fertilizer")
fert_hist = ref_df[COL_FPH]
fert_sc = {
    "Low fertilizer": float(round(fert_hist.quantile(0.2), 1)),
    "Typical fertilizer": float(round(fert_hist.median(), 1)),
    "High fertilizer": float(round(fert_hist.quantile(0.8), 1)),
}
fert_choice = st.radio(
    "How much fertilizer will you use?",
    list(fert_sc) + ["Enter my own"],
    index=1,
    horizontal=True,
    help="Low, typical and high are based on the fertilizer used per hectare in "
    "past years for this crop in this state and season.",
)
if fert_choice == "Enter my own":
    fert = st.number_input(
        f"Fertilizer ({FUNIT})",
        min_value=0.0,
        max_value=float(np.ceil(df[COL_FPH].max())),
        value=fert_sc["Typical fertilizer"],
        step=5.0,
    )
else:
    fert = fert_sc[fert_choice]
    st.caption(
        f"That means about **{fert:,.1f} {FUNIT}**. "
        f"Past range for {crop} in {state} ({season}): "
        f"{fert_hist.min():,.1f} to {fert_hist.max():,.1f} {FUNIT}."
    )
st.caption(
    f"On {area:,.0f} hectares, that is about **{fert * area:,.0f} kg "
    f"({fert * area / 1000:,.1f} tonnes)** of fertilizer in total."
)

if st.button("Predict yield", type="primary"):
    st.session_state["show_result"] = True

if not st.session_state.get("show_result"):
    st.stop()

# ---------------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------------
pred = float(predict_yield(model, crop, season, state, year, area, rainfall, fert)[0])
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
st.write(
    f"This assumes **{rainfall:,.0f} mm** of rainfall and **{fert:,.1f} {FUNIT}** "
    f"of fertilizer."
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
# CHART 1: rainfall x fertilizer heatmap
# ---------------------------------------------------------------
st.subheader("1. Rainfall and fertilizer together")
cells = []
for fn, fv in fert_sc.items():
    for rn, rv in rain_sc.items():
        cells.append(
            {
                "Rain": f"{rn.split()[0]} ({rv:,.0f} mm)",
                "Fert": f"{fn.split()[0]} ({fv:,.0f} kg/ha)",
                "rain_val": rv,
                "fert_val": fv,
            }
        )
cells_df = pd.DataFrame(cells)
cells_df["Yield"] = predict_yield(
    model, crop, season, state, year, area,
    cells_df["rain_val"].values, cells_df["fert_val"].values,
)
rain_order = [f"{n.split()[0]} ({v:,.0f} mm)" for n, v in rain_sc.items()]
fert_order = [f"{n.split()[0]} ({v:,.0f} kg/ha)" for n, v in fert_sc.items()]

x_enc = alt.X("Rain:N", sort=rain_order, title="Rainfall", axis=alt.Axis(labelAngle=0))
y_enc = alt.Y("Fert:N", sort=fert_order[::-1], title="Fertilizer")
mid = float((cells_df["Yield"].min() + cells_df["Yield"].max()) / 2)

rect = (
    alt.Chart(cells_df)
    .mark_rect()
    .encode(
        x=x_enc,
        y=y_enc,
        color=alt.Color(
            "Yield:Q",
            scale=alt.Scale(scheme="greens"),
            legend=alt.Legend(title="Yield"),
        ),
        tooltip=[
            alt.Tooltip("Rain:N", title="Rainfall"),
            alt.Tooltip("Fert:N", title="Fertilizer"),
            alt.Tooltip("Yield:Q", format=".2f", title=YLAB),
        ],
    )
)
cell_text = (
    alt.Chart(cells_df)
    .mark_text(fontSize=17, fontWeight="bold")
    .encode(
        x=alt.X("Rain:N", sort=rain_order),
        y=alt.Y("Fert:N", sort=fert_order[::-1]),
        text=alt.Text("Yield:Q", format=".2f"),
        color=alt.condition(
            alt.datum.Yield > mid, alt.value("white"), alt.value("black")
        ),
    )
)
layers = rect + cell_text
sel = cells_df[(cells_df["rain_val"] == rainfall) & (cells_df["fert_val"] == fert)]
if len(sel):
    outline = (
        alt.Chart(sel)
        .mark_rect(filled=False, stroke=RED, strokeWidth=4)
        .encode(x=alt.X("Rain:N", sort=rain_order), y=alt.Y("Fert:N", sort=fert_order[::-1]))
    )
    layers = layers + outline
show_chart(layers.properties(height=300))
st.caption(
    f"Each box is the predicted yield ({UNIT}) for {where} on {area:,.0f} hectares "
    "for that mix of rainfall and fertilizer. Darker green = higher yield."
    + (" Red outline = your choice." if len(sel) else "")
)

best = cells_df.loc[cells_df["Yield"].idxmax()]
worst = cells_df.loc[cells_df["Yield"].idxmin()]
normal_rain = rain_sc["Normal year"]
f_low = cells_df[(cells_df["rain_val"] == normal_rain) & (cells_df["fert_val"] == fert_sc["Low fertilizer"])]["Yield"].iloc[0]
f_high = cells_df[(cells_df["rain_val"] == normal_rain) & (cells_df["fert_val"] == fert_sc["High fertilizer"])]["Yield"].iloc[0]
f_pct = (f_high - f_low) / f_low * 100 if f_low > 0 else 0.0
r_low = cells_df[(cells_df["rain_val"] == rain_sc["Dry year"]) & (cells_df["fert_val"] == fert_sc["Typical fertilizer"])]["Yield"].iloc[0]
r_high = cells_df[(cells_df["rain_val"] == rain_sc["Wet year"]) & (cells_df["fert_val"] == fert_sc["Typical fertilizer"])]["Yield"].iloc[0]
r_pct = (r_high - r_low) / r_low * 100 if r_low > 0 else 0.0
st.write(
    f"- Highest yield in this table: **{best['Yield']:.2f}** "
    f"({best['Rain']} rainfall, {best['Fert']} fertilizer).\n"
    f"- Lowest yield: **{worst['Yield']:.2f}** "
    f"({worst['Rain']} rainfall, {worst['Fert']} fertilizer).\n"
    f"- With typical fertilizer, going from a dry to a wet year changes the "
    f"predicted yield by **{r_pct:+.0f}%**.\n"
    f"- In a normal year, going from low to high fertilizer changes it by "
    f"**{f_pct:+.0f}%**."
)

# ---------------------------------------------------------------
# CHART 2: past years with your prediction highlighted
# ---------------------------------------------------------------
st.subheader("2. Your prediction compared with past years")
avg_df = pd.DataFrame({"Yield": [usual], "label": [f"Usual average: {usual:.2f}"]})
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
hy = (
    ref_df.groupby(COL_YEAR, as_index=False)[COL_YIELD]
    .mean()
    .rename(columns={COL_YEAR: "Year", COL_YIELD: "Yield"})
)
pred_df = pd.DataFrame(
    {"Year": [year], "Yield": [pred], "label": [f"Predicted: {pred:.2f}"]}
)
past_line = (
    alt.Chart(hy)
    .mark_line(point=True, color=BLUE)
    .encode(
        x=alt.X(
            "Year:Q",
            axis=alt.Axis(format="d", tickMinStep=1),
            scale=alt.Scale(zero=False, nice=False, padding=15),
        ),
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
show_chart(
    (past_line + pred_pt + pred_txt + avg_rule + avg_text).properties(height=320)
)
st.caption(
    "Blue line = average yield in each past year. Red diamond = your prediction. "
    "Dashed line = usual average."
)


# ---------------------------------------------------------------
# CHARTS 3 + 4: effect of rainfall and effect of fertilizer
# ---------------------------------------------------------------
def effect_chart(vary):
    """Real past records + model curve when only rainfall OR fertilizer changes."""
    if vary == "rain":
        col, xt, scen = COL_RAIN, "Rainfall in the year (mm)", rain_sc
    else:
        col, xt, scen = COL_FPH, f"Fertilizer ({FUNIT})", fert_sc
    you_x = rainfall if vary == "rain" else fert

    s = ref_df[col]
    lo, hi = float(s.min()), float(s.max())
    if hi - lo < 0.15 * max(hi, 1.0):  # too little spread in the data
        pad = 0.2 * max(hi, 1.0)
        lo, hi = max(lo - pad, 0.0), hi + pad
    grid = np.linspace(lo, hi, 60)
    sv = np.array(list(scen.values()))
    if vary == "rain":
        curve_y = predict_yield(model, crop, season, state, year, area, grid, fert)
        scen_y = predict_yield(model, crop, season, state, year, area, sv, fert)
    else:
        curve_y = predict_yield(model, crop, season, state, year, area, rainfall, grid)
        scen_y = predict_yield(model, crop, season, state, year, area, rainfall, sv)

    curve_df = pd.DataFrame({"X": grid, "Yield": curve_y})
    past_df = ref_df[[col, COL_YIELD]].rename(columns={col: "X", COL_YIELD: "Yield"})
    scen_df = pd.DataFrame(
        {"X": sv, "Yield": scen_y, "label": [n.split()[0] for n in scen]}
    )
    you_df = pd.DataFrame({"X": [you_x], "Yield": [pred]})

    dots = (
        alt.Chart(past_df)
        .mark_circle(size=45, color=GREY, opacity=0.45)
        .encode(
            x=alt.X("X:Q", title=xt, scale=alt.Scale(zero=False, nice=False, padding=15)),
            y=alt.Y("Yield:Q", title=YLAB, scale=alt.Scale(zero=False)),
            tooltip=[
                alt.Tooltip("X:Q", format=",.1f", title=xt),
                alt.Tooltip("Yield:Q", format=".2f", title="Past yield"),
            ],
        )
    )
    line = (
        alt.Chart(curve_df)
        .mark_line(color=GREEN, strokeWidth=3)
        .encode(x="X:Q", y="Yield:Q")
    )
    marks = (
        alt.Chart(scen_df)
        .mark_point(size=110, filled=True, color=ORANGE, opacity=1)
        .encode(x="X:Q", y="Yield:Q")
    )
    names = (
        alt.Chart(scen_df)
        .mark_text(dy=-16, fontSize=12, color=TEXT)
        .encode(x="X:Q", y="Yield:Q", text="label:N")
    )
    you = (
        alt.Chart(you_df)
        .mark_point(shape="diamond", size=320, filled=False, strokeWidth=3, color=RED)
        .encode(x="X:Q", y="Yield:Q")
    )
    return (dots + line + marks + names + you).properties(height=340)


st.subheader("3. How rainfall changes the yield")
show_chart(effect_chart("rain"))
st.caption(
    "Grey dots = real past records. Green line = what the model predicts for your "
    "inputs at different rainfall levels (fertilizer kept at your choice). "
    "Orange dots = dry / normal / wet. Red diamond = your choice."
)

st.subheader("4. How fertilizer changes the yield")
show_chart(effect_chart("fert"))
st.caption(
    "Grey dots = real past records. Green line = what the model predicts for your "
    "inputs at different fertilizer levels (rainfall kept at your choice). "
    "Orange dots = low / typical / high. Red diamond = your choice. This shows what "
    "the model learned from past data, not a guaranteed result."
)

# ---------------------------------------------------------------
# CHART 5: which factors matter most (permutation importance)
# ---------------------------------------------------------------
st.subheader("5. Which factors matter most?")
tot = sum(importance.values())
imp_df = pd.DataFrame(
    {
        "Factor": [FRIENDLY[c] for c in importance],
        "Share": [v / tot * 100 if tot > 0 else 0.0 for v in importance.values()],
    }
)
imp_df["Group"] = imp_df["Factor"].map(
    lambda f: f if f in ("Rainfall", "Fertilizer") else "Other factors"
)
imp_df["label"] = imp_df["Share"].map(lambda v: f"{v:.0f}%")
order = imp_df.sort_values("Share", ascending=False)["Factor"].tolist()
imp_bars = (
    alt.Chart(imp_df)
    .mark_bar()
    .encode(
        y=alt.Y("Factor:N", sort=order, title=None),
        x=alt.X("Share:Q", title="Share of influence on yield (%)"),
        color=alt.Color(
            "Group:N",
            scale=alt.Scale(
                domain=["Rainfall", "Fertilizer", "Other factors"],
                range=[BLUE, GREEN, GREY],
            ),
            legend=None,
        ),
        tooltip=[alt.Tooltip("Factor:N"), alt.Tooltip("Share:Q", format=".1f")],
    )
)
imp_text = (
    alt.Chart(imp_df)
    .mark_text(align="left", dx=5, fontSize=13, color=TEXT)
    .encode(y=alt.Y("Factor:N", sort=order), x="Share:Q", text="label:N")
)
show_chart((imp_bars + imp_text).properties(height=280))
st.caption(
    "Measured by shuffling each factor and checking how much the model's accuracy "
    "drops: the bigger the drop, the more the model relies on that factor. "
    "Blue = rainfall, green = fertilizer."
)

# ---------------------------------------------------------------
# DETAILS
# ---------------------------------------------------------------
with st.expander("See the past records used for comparison"):
    table = ref_df[[COL_YEAR, COL_RAIN, COL_FPH, COL_AREA, COL_YIELD]].sort_values(
        COL_YEAR
    )
    table.columns = ["Year", "Rainfall (mm)", f"Fertilizer ({FUNIT})", "Area (ha)", YLAB]
    st.dataframe(table, hide_index=True)

with st.expander("How to read these numbers"):
    st.markdown(
        f"""
- **Yield** is how much crop you get from one hectare of land. 1 tonne = 1,000 kg,
  so a yield of 2.5 means about 2,500 kg from every hectare.
- **Total production** = yield × your land area.
- **Fertilizer** is shown per hectare so it can be compared between farms of any size.
- **Usual yield** is the average of past records for {where}.
- **Dry / Normal / Wet** and **Low / Typical / High** come from past years in the data for {where}.
"""
    )

with st.expander("Technical details (for reviewers)"):
    st.write(
        f"Model: Random Forest regression trained on {len(df):,} historical records "
        f"using seven inputs: crop, season, state, year, land area, rainfall and "
        f"fertilizer per hectare. Accuracy on unseen test data: R² = **{r2:.2f}**."
    )
    st.write(
        "Factor influence uses permutation importance on the test data. The charts "
        "show relationships the model found in past data (correlation), not proof "
        "that changing one factor will cause the yield to change."
    )
    st.write(f"This combination has {len(ref_df)} past record(s) in the data.")

st.caption(
    "This is an estimate based on historical data. Soil quality, pests and farming "
    "practice also affect real yield."
)
