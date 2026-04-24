import os
import json
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Food Vision AI",
    page_icon="🍱",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp { background: #0b0e1a; color: #dde1f0; }

[data-testid="stSidebar"] {
    background: #101422;
    border-right: 1px solid #1e2340;
}

.hero {
    background: linear-gradient(135deg, #1a2040 0%, #0f1428 50%, #1a1030 100%);
    border: 1px solid #2a305a;
    border-radius: 20px;
    padding: 36px 40px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, #f0900033 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 40px;
    font-weight: 800;
    background: linear-gradient(90deg, #f0a500, #ff6b35);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 6px 0;
}
.hero-sub { color: #8892b0; font-size: 15px; margin: 0; }

.sec-header {
    font-family: 'Syne', sans-serif;
    font-size: 20px;
    font-weight: 700;
    color: #f0a500;
    margin: 8px 0 18px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid #1e2340;
}

.card {
    background: #131829;
    border: 1px solid #1e2340;
    border-radius: 14px;
    padding: 22px 24px;
    margin-bottom: 14px;
}

.pill {
    display: inline-block;
    background: linear-gradient(135deg, #f0a500, #ff6b35);
    color: #fff;
    font-family: 'Syne', sans-serif;
    font-size: 17px;
    font-weight: 700;
    padding: 7px 22px;
    border-radius: 50px;
    margin: 8px 0 4px 0;
}

.bar-wrap {
    background: #1e2340;
    border-radius: 6px;
    height: 8px;
    margin: 4px 0 10px 0;
    overflow: hidden;
}

.bmi-num {
    font-family: 'Syne', sans-serif;
    font-size: 52px;
    font-weight: 800;
    line-height: 1;
}
.tag-under  { color: #4fc3f7; }
.tag-normal { color: #69f0ae; }
.tag-over   { color: #ffb74d; }
.tag-obese  { color: #ff5252; }

[data-testid="stFileUploader"] {
    background: #131829;
    border: 2px dashed #2a305a;
    border-radius: 12px;
    padding: 10px;
}

.stButton > button {
    background: linear-gradient(135deg, #f0a500, #ff6b35);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 10px 28px;
}

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0b0e1a; }
::-webkit-scrollbar-thumb { background: #2a305a; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ── Demo fallback data  (matches the actual 40 trained classes from UECFOOD100) ──

DEMO_CLASSES = [
    "Japanese-style_pancake",
    "beef_curry",
    "beef_noodle",
    "bibimbap",
    "chicken-'n'-egg_on_rice",
    "chicken_rice",
    "chip_butty",
    "croissant",
    "croquette",
    "eels_on_rice",
    "fried_noodle",
    "fried_rice",
    "gratin",
    "grilled_eggplant",
    "hamburger",
    "miso_soup",
    "oden",
    "omelet",
    "pilaf",
    "pizza",
    "pork_cutlet_on_rice",
    "potage",
    "raisin_bread",
    "ramen_noodle",
    "rice",
    "roll_bread",
    "sandwiches",
    "sausage",
    "sauteed_spinach",
    "sauteed_vegetables",
    "soba_noodle",
    "spaghetti",
    "sushi",
    "takoyaki",
    "tempura_bowl",
    "tempura_udon",
    "tensin_noodle",
    "toast",
    "udon_noodle",
    "vegetable_tempura",
]

DEMO_CALORIES = {
    "Japanese-style_pancake": 200,
    "beef_curry":             150,
    "beef_noodle":            200,
    "bibimbap":               120,
    "chicken-'n'-egg_on_rice": 130,   # mapped from chicken_and_egg_on_rice ≈ 175 → using rice base
    "chicken_rice":           160,
    "chip_butty":             200,
    "croissant":              406,
    "croquette":              243,
    "eels_on_rice":           165,
    "fried_noodle":           200,
    "fried_rice":             185,
    "gratin":                 136,
    "grilled_eggplant":       200,
    "hamburger":              295,
    "miso_soup":              200,
    "oden":                   200,
    "omelet":                 200,
    "pilaf":                  180,
    "pizza":                  266,
    "pork_cutlet_on_rice":    248,
    "potage":                 200,
    "raisin_bread":           200,
    "ramen_noodle":           136,
    "rice":                   130,
    "roll_bread":             280,
    "sandwiches":             233,
    "sausage":                200,
    "sauteed_spinach":        200,
    "sauteed_vegetables":      70,
    "soba_noodle":            200,
    "spaghetti":              200,
    "sushi":                  143,
    "takoyaki":               165,
    "tempura_bowl":           290,
    "tempura_udon":           290,
    "tensin_noodle":          200,
    "toast":                  265,
    "udon_noodle":             96,
    "vegetable_tempura":      290,
}


# ── Load resources ─────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model...")
def load_resources():
    model        = None
    class_names  = DEMO_CLASSES
    calorie_dict = DEMO_CALORIES
    demo_mode    = False

    if os.path.exists("class_names.json"):
        with open("class_names.json") as f:
            class_names = json.load(f)
    else:
        demo_mode = True

    if os.path.exists("calorie_dict.json"):
        with open("calorie_dict.json") as f:
            calorie_dict = json.load(f)
    else:
        demo_mode = True

    if os.path.exists("food_model.h5"):
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model("food_model.h5")
        except Exception as e:
            st.error(f"Model load error: {e}")
            demo_mode = True
    else:
        demo_mode = True

    return model, class_names, calorie_dict, demo_mode


model, class_names, calorie_dict, DEMO_MODE = load_resources()


# ── Helpers ────────────────────────────────────────────────────────

def preprocess(pil_img, size=299):          # InceptionV3 native size = 299
    img = pil_img.convert("RGB").resize((size, size))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, 0)

def real_predict(pil_img, top_k=3):
    probs = model.predict(preprocess(pil_img), verbose=0)[0]
    idxs  = np.argsort(probs)[::-1][:top_k]
    return [(class_names[i], float(probs[i]) * 100) for i in idxs]

def demo_predict(top_k=3):
    import random
    picks = random.sample(class_names, top_k)
    raw   = sorted([random.uniform(40,80), random.uniform(10,25), random.uniform(3,10)], reverse=True)
    total = sum(raw)
    return [(p, r/total*100) for p, r in zip(picks, raw)]

def fmt(name):
    return name.replace("_", " ").replace("-", " ").replace("'", "").title()

def bmi_info(bmi):
    if bmi < 18.5: return "Underweight", "tag-under",  "💙", "Consider increasing calorie intake with nutrient-dense foods. Consult a nutritionist."
    if bmi < 25.0: return "Normal",      "tag-normal", "💚", "Great! Maintain your healthy routine with a balanced diet and regular exercise."
    if bmi < 30.0: return "Overweight",  "tag-over",   "🟠", "Aim for 150 min of moderate exercise per week and a mindful, balanced diet."
    return               "Obese",       "tag-obese",  "❤️", "Please consult a healthcare provider for a personalised weight-management plan."


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🍱 Food Vision AI")
    st.divider()

    if DEMO_MODE:
        st.warning("**Demo Mode**\n\nPlace these files next to app.py to enable real predictions:\n- food_model.h5\n- class_names.json\n- calorie_dict.json")
    else:
        st.success(f"✅ Model ready  \n**{len(class_names)} food classes**")

    st.divider()

    st.markdown("**How to use**")
    st.markdown("1. Upload a food photo\n2. See the predicted food & calories\n3. Enter your portion size\n4. Check your BMI below")


# ══════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero">
  <p class="hero-title">🍱 Food Vision AI</p>
  <p class="hero-sub">Upload a food photo to instantly identify it, estimate calories, and check your BMI.</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# SECTION 1 — FOOD RECOGNITION + CALORIES
# ══════════════════════════════════════════════════════════════════

st.markdown('<div class="sec-header">📸 Food Recognition & Calories</div>', unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    uploaded = st.file_uploader("Upload a food image", type=["jpg","jpeg","png","webp"])
    if uploaded:
        pil_img = Image.open(uploaded)
        st.image(pil_img, use_container_width=True)

with col_right:
    if uploaded:
        with st.spinner("Analysing..."):
            top3 = demo_predict() if (DEMO_MODE or model is None) else real_predict(pil_img)

        best_cls, best_conf = top3[0]
        cal_100g = calorie_dict.get(best_cls, 200)

        st.markdown(f"""
        <div class="card">
          <div style="color:#8892b0;font-size:12px;letter-spacing:1px;">PREDICTED FOOD</div>
          <div class="pill">🍽️ {fmt(best_cls)}</div>
          <div style="color:#8892b0;font-size:13px;margin-top:6px;">
            Confidence: <strong style="color:#dde1f0">{best_conf:.1f}%</strong>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Top 3 predictions**")
        medals = ["🥇","🥈","🥉"]
        fills  = ["linear-gradient(90deg,#f0a500,#ff6b35)", "#3a4070", "#2a2f50"]
        colors = ["#f0a500", "#8892b0", "#4a5070"]
        for i, (cls, conf) in enumerate(top3):
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;color:{colors[i]};font-size:14px;margin-top:6px;">
              <span>{medals[i]} {fmt(cls)}</span><span>{conf:.1f}%</span>
            </div>
            <div class="bar-wrap">
              <div style="background:{fills[i]};width:{min(conf,100):.1f}%;height:8px;border-radius:6px;"></div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        st.markdown("**🔥 Calorie Calculator**")
        st.markdown(f"**{fmt(best_cls)}** — {cal_100g} kcal per 100 g")

        portion   = st.number_input("Your portion size (grams)", min_value=10, max_value=2000, value=150, step=10)
        total_cal = (portion / 100) * cal_100g

        c1, c2, c3 = st.columns(3)
        c1.metric("Per 100 g",      f"{cal_100g} kcal")
        c2.metric("Portion",        f"{portion} g")
        c3.metric("Total Calories", f"{total_cal:.0f} kcal")

        pct_daily = min(total_cal / 2000 * 100, 100)
        st.markdown(f"""
        <div style="margin-top:10px;">
          <div style="display:flex;justify-content:space-between;font-size:12px;color:#8892b0;">
            <span>% of daily 2000 kcal goal</span><span>{pct_daily:.1f}%</span>
          </div>
          <div class="bar-wrap" style="height:12px;">
            <div style="background:linear-gradient(90deg,#f0a500,#ff6b35);width:{pct_daily:.1f}%;height:12px;border-radius:6px;"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="card" style="text-align:center;padding:60px 20px;">
          <div style="font-size:64px;">🍱</div>
          <div style="color:#8892b0;margin-top:14px;font-size:16px;">Upload a food image to get started</div>
          <div style="color:#4a5070;margin-top:6px;font-size:13px;">JPG · PNG · WEBP supported</div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# SECTION 2 — BMI CALCULATOR
# ══════════════════════════════════════════════════════════════════

st.divider()
st.markdown('<div class="sec-header">⚖️ BMI Calculator</div>', unsafe_allow_html=True)

bl, br = st.columns([1, 1], gap="large")

with bl:
    height_cm = st.number_input("Height (cm)", min_value=50, max_value=250, value=170, step=1)
    weight_kg = st.number_input("Weight (kg)", min_value=10, max_value=300, value=65,  step=1)

with br:
    bmi = weight_kg / (height_cm / 100) ** 2
    cat, css, emoji, tip = bmi_info(bmi)

    st.markdown(f"""
    <div class="card">
      <div style="color:#8892b0;font-size:12px;letter-spacing:1px;">YOUR BMI</div>
      <div style="display:flex;align-items:center;gap:20px;margin-top:8px;">
        <div style="font-size:48px;">{emoji}</div>
        <div>
          <div class="bmi-num {css}">{bmi:.1f}</div>
          <div style="font-size:18px;font-weight:600;color:#dde1f0;">{cat}</div>
          <div style="font-size:12px;color:#8892b0;">{height_cm} cm &nbsp;|&nbsp; {weight_kg} kg</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    for label, rng, tag, active in [
        ("Underweight", "< 18.5",    "tag-under",  bmi < 18.5),
        ("Normal",      "18.5–24.9", "tag-normal", 18.5 <= bmi < 25),
        ("Overweight",  "25–29.9",   "tag-over",   25 <= bmi < 30),
        ("Obese",       "≥ 30",      "tag-obese",  bmi >= 30),
    ]:
        you = " ← You" if active else ""
        fw  = "700"     if active else "400"
        bg  = "#181e30" if active else "#10131f"
        st.markdown(f"""
        <div style="background:{bg};border:1px solid #1e2340;border-radius:8px;
                    padding:9px 14px;margin:4px 0;display:flex;
                    justify-content:space-between;font-weight:{fw};font-size:14px;">
          <span>{label}<span class="{tag}">{you}</span></span>
          <span class="{tag}">{rng}</span>
        </div>
        """, unsafe_allow_html=True)

    st.info(f"💡 {tip}")


# ══════════════════════════════════════════════════════════════════
# SECTION 3 — CALORIE REFERENCE TABLE
# ══════════════════════════════════════════════════════════════════

st.divider()
st.markdown('<div class="sec-header">📊 Calorie Reference Table</div>', unsafe_allow_html=True)

import pandas as pd

def cal_level(c):
    if c < 100: return "🟢 Low"
    if c < 250: return "🟡 Medium"
    if c < 400: return "🟠 High"
    return "🔴 Very High"

rows = [{"Food": fmt(k), "Kcal / 100g": v, "Level": cal_level(v)}
        for k, v in sorted(calorie_dict.items(), key=lambda x: x[1])]
df = pd.DataFrame(rows)

search = st.text_input("🔍 Search food", placeholder="e.g. rice, pizza, ramen")
if search:
    df = df[df["Food"].str.lower().str.contains(search.lower(), na=False)]

st.dataframe(df, use_container_width=True, hide_index=True,
    column_config={
        "Food":        st.column_config.TextColumn("🍽️ Food", width="large"),
        "Kcal / 100g": st.column_config.NumberColumn("🔥 Kcal/100g", format="%d kcal"),
        "Level":       st.column_config.TextColumn("Level"),
    })
st.caption(f"Showing {len(df)} of {len(calorie_dict)} items. Values are approximate.")
