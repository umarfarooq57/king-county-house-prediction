import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------
# King County Home Value Studio
# Expects house_price_model.pkl in the same folder as app.py.
# The saved model should be the fitted TransformedTargetRegressor
# pipeline (including preprocessing) trained in Colab.
# ------------------------------------------------------------

st.set_page_config(
    page_title="NestWise | King County Home Value",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "house_price_model.pkl")

# Remote photos are used as decorative visuals. App functionality does not depend on them.
HERO_IMAGE = "https://images.unsplash.com/photo-1600597928310-0e3f4a3d2f4b?auto=format&fit=crop&w=2200&q=85"
CARD_IMAGE_1 = "https://images.unsplash.com/photo-1600585154340-be6161a56a0c?auto=format&fit=crop&w=900&q=80"
CARD_IMAGE_2 = "https://images.unsplash.com/photo-1600607687939-ce8a6c25118c?auto=format&fit=crop&w=900&q=80"
CARD_IMAGE_3 = "https://images.unsplash.com/photo-1600566753086-00f18fb6b3ea?auto=format&fit=crop&w=900&q=80"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --ink:#15233d; --muted:#66758d; --violet:#7557f6; --teal:#17b9a4; --coral:#ff8b70; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:linear-gradient(135deg,#f6f8ff 0%,#f7fbfa 48%,#fff7f3 100%); color:var(--ink); }
.block-container { max-width:1280px; padding-top:1.2rem; padding-bottom:3rem; }
#MainMenu, footer, header {visibility:hidden;}
div[data-testid="stToolbar"] {visibility:hidden;height:0;}
.brand {font-family:'Manrope',sans-serif;font-size:1.2rem;font-weight:800;color:#172747;letter-spacing:-.5px;}
.brand span {color:#7557f6;}
.top-pill {display:inline-block;padding:8px 13px;border:1px solid #e1e5f4;border-radius:99px;background:#ffffffc9;color:#5c6580;font-size:.78rem;font-weight:700;}
.hero {margin-top:1rem;border-radius:28px;overflow:hidden;min-height:350px;position:relative;background:
linear-gradient(90deg,rgba(15,25,52,.93) 0%,rgba(27,34,69,.78) 48%,rgba(27,34,69,.08) 100%),
url('""" + HERO_IMAGE + r"""') center 52%/cover; padding:42px 44px; box-shadow:0 24px 60px rgba(33,42,82,.16);}
.hero-kicker {color:#8ff1dc;text-transform:uppercase;letter-spacing:2.4px;font-size:.72rem;font-weight:800;}
.hero h1 {font-family:'Manrope',sans-serif;color:white;font-size:clamp(2.1rem,4vw,3.55rem);line-height:1.08;letter-spacing:-1.7px;margin:.7rem 0 1rem;max-width:620px;}
.hero p {color:#e1e6f8;max-width:540px;font-size:1.02rem;line-height:1.7;}
.hero-chip {display:inline-block;margin-top:15px;margin-right:8px;padding:9px 13px;border-radius:12px;background:#ffffff19;border:1px solid #ffffff30;color:white;font-size:.78rem;}
.section-title {font-family:'Manrope',sans-serif;font-weight:800;font-size:1.45rem;letter-spacing:-.5px;color:#182746;margin:0;}
.section-sub {color:#738098;font-size:.9rem;margin-top:4px;margin-bottom:16px;}
.panel {background:rgba(255,255,255,.88);border:1px solid #e8eaf4;border-radius:22px;padding:22px 24px;box-shadow:0 12px 35px rgba(33,42,82,.055);}
.mini-label {color:#7a8499;font-size:.73rem;text-transform:uppercase;letter-spacing:1.2px;font-weight:800;}
.metric-card {background:white;border:1px solid #e8eaf4;border-radius:18px;padding:18px 20px;box-shadow:0 10px 24px rgba(33,42,82,.05);min-height:105px;}
.metric-num {font-family:'Manrope',sans-serif;font-size:1.45rem;font-weight:800;color:#182746;margin-top:8px;}
.metric-note {font-size:.78rem;color:#7c879b;margin-top:4px;}
div[data-testid="stNumberInput"] label, div[data-testid="stSelectbox"] label, div[data-testid="stSlider"] label {font-weight:700;color:#263653;}
.stButton > button {background:linear-gradient(100deg,#7557f6,#4f79f7);color:white;border:0;border-radius:13px;padding:.78rem 1.2rem;font-weight:800;box-shadow:0 10px 22px #7557f633;transition:all .2s ease;}
.stButton > button:hover {color:white;transform:translateY(-2px);box-shadow:0 14px 26px #7557f644;}
div[data-testid="stAlert"] {border-radius:15px;}
.footer {text-align:center;color:#8791a5;font-size:.78rem;padding:24px 0 0;}
div[data-testid="stExpander"] {border:1px solid #e5e8f3;border-radius:14px;background:#ffffffb8;}
@media(max-width:700px){.hero{padding:28px 23px;min-height:390px}.block-container{padding-left:1rem;padding-right:1rem}.panel{padding:17px}}
</style>
""", unsafe_allow_html=True)

# Header
h1, h2 = st.columns([3, 1])
with h1:
    st.markdown('<div class="brand">nest<span>wise</span> <span style="color:#9aa5bc;font-weight:500">/ HOME VALUE STUDIO</span></div>', unsafe_allow_html=True)
with h2:
    st.markdown('<div style="text-align:right"><span class="top-pill">✦ AI-POWERED ESTIMATE</span></div>', unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="hero-kicker">King County · Washington</div>
  <h1>Find the value<br>behind every home.</h1>
  <p>Explore an estimated property value with a machine-learning model trained on King County home sales. Adjust the home details and get an instant estimate.</p>
  <span class="hero-chip">⌂ Property insights</span>
  <span class="hero-chip">✦ Regression-powered</span>
  <span class="hero-chip">↗ Instant estimate</span>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)
left, right = st.columns([1.16, .84], gap="large")

with left:
    st.markdown('<div class="section-title">Tell us about the property</div><div class="section-sub">Add the home features below to calculate an estimated value.</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("#### 🏠 Home basics")
        c1, c2, c3 = st.columns(3)
        with c1:
            bedrooms = st.number_input("Bedrooms", min_value=1, max_value=9, value=3, step=1)
        with c2:
            bathrooms = st.number_input("Bathrooms", min_value=0.5, max_value=8.0, value=2.0, step=0.25)
        with c3:
            floors = st.selectbox("Floors", [1.0, 1.5, 2.0, 2.5, 3.0], index=1)

        c1, c2 = st.columns(2)
        with c1:
            sqft_living = st.number_input("Interior living area (sq ft)", min_value=300, max_value=12000, value=1800, step=100)
        with c2:
            sqft_lot = st.number_input("Lot size (sq ft)", min_value=500, max_value=1000000, value=5000, step=500)

        c1, c2, c3 = st.columns(3)
        with c1:
            grade = st.slider("Construction / design grade", 1, 13, 7)
        with c2:
            condition = st.slider("Property condition", 1, 5, 3)
        with c3:
            view = st.selectbox("View rating", [0, 1, 2, 3, 4], index=0)

        st.markdown("---")
        st.markdown("#### 📍 Location & extra details")
        c1, c2 = st.columns(2)
        with c1:
            zipcode = st.selectbox("King County ZIP code", [
                98001,98002,98003,98004,98005,98006,98007,98008,98010,98011,98014,98019,
                98022,98023,98024,98027,98028,98029,98030,98031,98032,98033,98034,98038,
                98039,98040,98042,98045,98052,98053,98055,98056,98058,98059,98065,98070,
                98072,98074,98075,98077,98092,98102,98103,98105,98106,98107,98108,98109,
                98112,98115,98116,98117,98118,98119,98122,98125,98126,98133,98136,98144,
                98146,98148,98155,98166,98168,98177,98178,98188,98198,98199
            ], index=28)
        with c2:
            waterfront = st.selectbox("Waterfront", ["No", "Yes"], index=0)

        c1, c2 = st.columns(2)
        with c1:
            yr_built = st.number_input("Year built", min_value=1900, max_value=2026, value=1990, step=1)
        with c2:
            yr_renovated = st.number_input("Year renovated (0 = never)", min_value=0, max_value=2026, value=0, step=1)

        st.markdown("##### Nearby-home averages")
        c1, c2 = st.columns(2)
        with c1:
            sqft_living15 = st.number_input("Nearby living area (sq ft)", min_value=300, max_value=10000, value=1800, step=100)
        with c2:
            sqft_lot15 = st.number_input("Nearby lot size (sq ft)", min_value=500, max_value=500000, value=5000, step=500)

        st.markdown("##### More property details")
        c1, c2, c3 = st.columns(3)
        with c1:
            sqft_basement = st.number_input("Basement area (sq ft)", min_value=0, max_value=6000, value=0, step=100)
        with c2:
            latitude = st.number_input("Latitude", min_value=47.15, max_value=47.80, value=47.50, step=0.001, format="%.3f")
        with c3:
            longitude = st.number_input("Longitude", min_value=-122.55, max_value=-121.30, value=-122.20, step=0.001, format="%.3f")

        predict_clicked = st.button("✨  Estimate home value", use_container_width=True)

with right:
    st.markdown('<div class="section-title">Your home, visualized</div><div class="section-sub">A little inspiration for your next property decision.</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="border-radius:20px;overflow:hidden;background:white;border:1px solid #e8eaf4;box-shadow:0 12px 30px #20294b0c;">
      <img src="{CARD_IMAGE_1}" style="width:100%;height:190px;object-fit:cover;display:block;">
      <div style="padding:17px 19px 20px">
        <div style="color:#7557f6;font-size:.72rem;font-weight:800;letter-spacing:1.2px;text-transform:uppercase">01 / HOME PROFILE</div>
        <div style="font-family:Manrope,sans-serif;font-size:1.12rem;font-weight:800;margin-top:6px;color:#182746">Space that fits your life</div>
        <div style="color:#748098;font-size:.86rem;margin-top:5px">Bedrooms, living space, grade and condition all help describe the property.</div>
      </div>
    </div>
    <div style="height:14px"></div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px">
      <div style="border-radius:18px;overflow:hidden;background:white;border:1px solid #e8eaf4">
        <img src="{CARD_IMAGE_2}" style="width:100%;height:115px;object-fit:cover;display:block">
        <div style="padding:12px"><b style="color:#182746">Design & comfort</b><div style="font-size:.78rem;color:#748098;margin-top:4px">Features shape the estimate</div></div>
      </div>
      <div style="border-radius:18px;overflow:hidden;background:white;border:1px solid #e8eaf4">
        <img src="{CARD_IMAGE_3}" style="width:100%;height:115px;object-fit:cover;display:block">
        <div style="padding:12px"><b style="color:#182746">Location matters</b><div style="font-size:.78rem;color:#748098;margin-top:4px">ZIP code & coordinates</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='height:15px'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="padding:17px 19px;border-radius:18px;background:linear-gradient(120deg,#e9e5ff,#e4fbf5);border:1px solid #deddf8">
      <div style="font-weight:800;color:#3d3976">✦ Prediction tip</div>
      <div style="color:#5e6681;font-size:.84rem;line-height:1.6;margin-top:6px">Use realistic values. Estimates are based on historical sales data and are not a formal appraisal or a guarantee of market value.</div>
    </div>
    """, unsafe_allow_html=True)

if predict_clicked:
    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found. Please place house_price_model.pkl in the same folder as app.py.")
        st.info("Expected files: app.py, requirements.txt, and house_price_model.pkl")
    else:
        try:
            model = joblib.load(MODEL_PATH)
            input_row = pd.DataFrame([{
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "sqft_living": sqft_living,
                "sqft_lot": sqft_lot,
                "floors": floors,
                "waterfront": 1 if waterfront == "Yes" else 0,
                "view": view,
                "condition": condition,
                "grade": grade,
                "sqft_basement": sqft_basement,
                "yr_built": yr_built,
                "yr_renovated": yr_renovated,
                "lat": latitude,
                "long": longitude,
                "sqft_living15": sqft_living15,
                "sqft_lot15": sqft_lot15,
                "zipcode": str(zipcode),
            }])
            # Keep ZIP-code dtype aligned with training data where possible.
            predicted = float(np.asarray(model.predict(input_row)).reshape(-1)[0])
            if not np.isfinite(predicted) or predicted <= 0:
                st.error("The model returned an invalid estimate. Check the model and input features.")
            else:
                st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
                st.markdown(f"""
                <div style="border-radius:24px;padding:27px;background:linear-gradient(120deg,#171f43,#343a77 65%,#7557f6);color:white;box-shadow:0 18px 45px #30286e30">
                  <div style="font-size:.75rem;text-transform:uppercase;letter-spacing:2px;color:#a8f1e2;font-weight:800">Your estimated home value</div>
                  <div style="font-family:Manrope,sans-serif;font-size:clamp(2rem,4vw,3rem);font-weight:800;letter-spacing:-1px;margin:10px 0">${predicted:,.0f}</div>
                  <div style="color:#e1e6fa;font-size:.87rem">Estimated by the saved King County regression model</div>
                </div>
                """, unsafe_allow_html=True)
                st.caption("This is a model-based estimate using historical data, not a certified appraisal. Real market prices can differ.")
        except Exception as exc:
            st.error("Prediction could not be completed.")
            st.exception(exc)

st.markdown("<div class='footer'>NESTWISE · KING COUNTY HOME VALUE STUDIO &nbsp; • &nbsp; Machine Learning Project</div>", unsafe_allow_html=True)
