import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Real Estate Advisor", layout="wide")
st.title("🏠 Real Estate Investment Advisor")

# =====================================================
# CACHED LOADERS
# =====================================================
@st.cache_resource
def load_models():
    pre = joblib.load("preprocessor.pkl")
    reg = joblib.load("RandomForest_reg_model.pkl")
    clf = joblib.load("RandomForest_clf_model.pkl")
    return pre, reg, clf

@st.cache_data
def load_data():
    return pd.read_csv("../dataset/india_housing_prices.csv")

@st.cache_resource
def load_feature_names():
    return pd.read_csv("feature_names.csv", header=None).squeeze().tolist()

@st.cache_resource
def build_map(df):
    m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
    subset = df.sample(300)

    for _, row in subset.iterrows():
        folium.CircleMarker(
            location=[19 + np.random.rand(), 72 + np.random.rand()],
            radius=3,
            color="blue",
            fill=True,
        ).add_to(m)

    return m

# Load artifacts
preprocessor, reg_model, clf_model = load_models()
sample_df = load_data()
feature_names = load_feature_names()
raw_feature_list = preprocessor.feature_names_in_.tolist()


# =====================================================
# BUILD COMPLETE RAW FEATURE ROW
# =====================================================
def build_full_feature_row(user_input, raw_feature_list):
    df = pd.DataFrame([user_input])

    for col in raw_feature_list:
        if col not in df.columns:
            if col in ["State"]: df[col] = "DefaultState"
            elif col in ["City"]: df[col] = "DefaultCity"
            elif col in ["Locality"]: df[col] = "DefaultLocality"
            elif col in ["Furnished_Status"]: df[col] = "Unfurnished"
            elif col in ["Security", "Parking_Space"]: df[col] = "No"
            elif col in ["Owner_Type"]: df[col] = "Owner"
            elif col in ["Facing"]: df[col] = "East"
            elif col in ["Age_Bucket"]: df[col] = "Medium"
            elif col in ["Size_Bucket"]: df[col] = "Medium"
            elif col in ["Public_Transport_Accessibility"]: df[col] = "Medium"
            elif col in ["Availability_Status"]: df[col] = "Ready_to_Move"
            elif col in ["Property_Type"]: df[col] = "Apartment"
            elif col in ["Amenities"]: df[col] = "None"
            else: df[col] = 0

    return df[raw_feature_list]


# =====================================================
# INPUT FORM
# =====================================================
with st.form("property_form"):
    st.header("Enter Property Details")
    col1, col2 = st.columns(2)

    with col1:
        city = st.selectbox("City", ["Mumbai","Bengaluru","Delhi","Other"])
        property_type = st.selectbox("Property Type", ["Apartment","Independent House","Villa"])
        bhk = st.number_input("BHK", 1, 10, 3)
        size = st.number_input("Size (SqFt)", 200, 10000, 1200)

    with col2:
        price = st.number_input("Current Price (Lakhs)", 1.0, 100000.0, 100.0)
        transport = st.selectbox("Public Transport Access", ["Low","Medium","High"])
        parking = st.selectbox("Parking Space", ["No","Yes"])
        avail = st.selectbox("Availability Status", ["Ready_to_Move","Under_Construction"])
        amenities = st.multiselect("Amenities", ["Pool","Gym","Garden","Clubhouse","Playground","Security"])

    submitted = st.form_submit_button("Predict")


# =====================================================
# SAVE PREDICTION STATE
# =====================================================
if submitted:
    st.session_state["run_prediction"] = True

    user_input = {
        "City": city,
        "State": "DefaultState",
        "Locality": "DefaultLocality",
        "Property_Type": property_type,
        "BHK": bhk,
        "Size_in_SqFt": size,
        "Price_in_Lakhs": price,
        "Price_per_SqFt": (price*100000)/size,
        "Public_Transport_Accessibility": transport,
        "Parking_Space": parking,
        "Availability_Status": avail,
        "Amenities": ",".join(amenities),
        "Amenity_Count": len(amenities)
    }

    X_full = build_full_feature_row(user_input, raw_feature_list)

    st.session_state["pred_result"] = {
        "pred_price_5y": reg_model.predict(X_full)[0],
        "pred_good": clf_model.predict(X_full)[0],
        "pred_proba": clf_model.predict_proba(X_full)[0][1],
        "X_full": X_full,
    }


# =====================================================
# SHOW PREDICTION (PERSISTENT)
# =====================================================
if st.session_state.get("run_prediction", False):

    res = st.session_state.get("pred_result", None)

    # If prediction not computed yet → avoid KeyError
    if res is None or "X_full" not in res:
        st.warning("Prediction data not found. Please click Predict again.")
        st.stop()

    X_full = res["X_full"]

    st.subheader("📊 Predictions")
    st.metric("Estimated Price in 5 Years", round(res["pred_price_5y"], 2))
    st.metric(
        "Good Investment?",
        "Yes" if res["pred_good"] == 1 else "No",
        delta=f"{round(res['pred_proba'] * 100, 2)}% confidence"
    )


    # =====================================================
    # SHAP — TOP 10 FEATURES
    # =====================================================
    st.subheader("🌟 Local Feature Importance (SHAP)")

    preprocess_only = clf_model.named_steps["pre"]
    model_only = clf_model.named_steps["model"]

    X_trans = preprocess_only.transform(X_full)

    explainer = shap.TreeExplainer(model_only)
    shap_values_all = explainer.shap_values(X_trans)

    if isinstance(shap_values_all, list):
        if len(shap_values_all) == 2:
            shap_values = shap_values_all[1][0]
        else:
            shap_values = shap_values_all[0][0]
    else:
        shap_values = shap_values_all[0]

    shap_values = np.array(shap_values).flatten()
    feature_list = np.array(feature_names).flatten()

    min_len = min(len(feature_list), len(shap_values))
    feature_list = feature_list[:min_len]
    shap_values = shap_values[:min_len]

    shap_df = pd.DataFrame({
        "feature": feature_list,
        "shap_value": shap_values
    }).sort_values("shap_value", key=abs, ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.barplot(data=shap_df, y="feature", x="shap_value", ax=ax)
    ax.set_title("Top 10 Important Features (Local SHAP)")
    plt.tight_layout()
    st.pyplot(fig)


# =====================================================
# FAST VISUAL INSIGHTS
# =====================================================
st.subheader("📈 Visual Insights")

df_vis = sample_df.sample(8000, random_state=42)

st.markdown("### BHK vs Price per SqFt")
fig1, ax1 = plt.subplots(figsize=(10,5))
sns.boxplot(x="BHK", y="Price_per_SqFt", data=df_vis, ax=ax1)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
st.pyplot(fig1)

st.markdown("### City-wise Median Price")
city_median = df_vis.groupby("City")["Price_in_Lakhs"].median().reset_index()
fig2, ax2 = plt.subplots(figsize=(10,5))
sns.barplot(x="City", y="Price_in_Lakhs", data=city_median, ax=ax2)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
st.pyplot(fig2)

# MAP (cached)
st.markdown("### Location Price Heatmap")
map_obj = build_map(df_vis)

with st.container():
    st_folium(map_obj, width=700, height=400)
