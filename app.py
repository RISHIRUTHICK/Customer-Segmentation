import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Customer Segmentation", page_icon="🛍️", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "kmeans_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"

FEATURES = [
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds",
    "NumDealsPurchases",
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases",
    "NumWebVisitsMonth",
]

SEGMENT_NAMES = {
    0: "High-Value Multi-Channel Customers",
    1: "Low-Spend, Low-Purchase Customers",
    2: "High-Value Store & Catalogue Customers",
    3: "Active Online & Deal-Oriented Customers",
}

SEGMENT_DESCRIPTIONS = {
    0: "This cluster shows relatively high spending across several product categories and activity across multiple purchase channels.",
    1: "This cluster generally has lower product spending and fewer purchases, while website visits may still be frequent.",
    2: "This cluster tends to show stronger spending and purchase activity, especially through store and catalogue channels.",
    3: "This cluster tends to show higher online engagement and deal-oriented purchasing behaviour.",
}

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

st.title("🛍️ Customer Segmentation")
st.write(
    "Enter a customer's spending and purchasing behaviour to predict their segment "
    "using the trained K-Means model."
)
st.caption("The prediction uses the same 11 features and StandardScaler as the training notebook.")

if not MODEL_PATH.exists() or not SCALER_PATH.exists():
    st.error("Model files are missing. Keep app.py, kmeans_model.pkl, and scaler.pkl in the same folder.")
    st.stop()

try:
    model, scaler = load_artifacts()
except Exception as exc:
    st.error(f"The model files could not be loaded: {exc}")
    st.info("Check that the scikit-learn version matches the version used to save the model.")
    st.stop()

with st.form("customer_form"):
    st.subheader("Customer details")
    st.markdown("### Product spending")
    a, b, c = st.columns(3)
    with a:
        wines = st.number_input("Wine spending", min_value=0, max_value=100000, value=300, step=10)
        fruits = st.number_input("Fruit spending", min_value=0, max_value=100000, value=20, step=5)
    with b:
        meat = st.number_input("Meat product spending", min_value=0, max_value=100000, value=150, step=10)
        fish = st.number_input("Fish product spending", min_value=0, max_value=100000, value=30, step=5)
    with c:
        sweets = st.number_input("Sweet product spending", min_value=0, max_value=100000, value=25, step=5)
        gold = st.number_input("Gold product spending", min_value=0, max_value=100000, value=40, step=5)

    st.markdown("### Purchase behaviour")
    d, e, f = st.columns(3)
    with d:
        deals = st.number_input("Purchases made with deals", min_value=0, max_value=1000, value=2, step=1)
        web_purchases = st.number_input("Web purchases", min_value=0, max_value=1000, value=4, step=1)
    with e:
        catalogue_purchases = st.number_input("Catalogue purchases", min_value=0, max_value=1000, value=2, step=1)
        store_purchases = st.number_input("Store purchases", min_value=0, max_value=1000, value=5, step=1)
    with f:
        web_visits = st.number_input("Website visits per month", min_value=0, max_value=1000, value=5, step=1)

    submitted = st.form_submit_button("Predict customer segment", type="primary", use_container_width=True)

if submitted:
    customer = pd.DataFrame([{
        "MntWines": wines,
        "MntFruits": fruits,
        "MntMeatProducts": meat,
        "MntFishProducts": fish,
        "MntSweetProducts": sweets,
        "MntGoldProds": gold,
        "NumDealsPurchases": deals,
        "NumWebPurchases": web_purchases,
        "NumCatalogPurchases": catalogue_purchases,
        "NumStorePurchases": store_purchases,
        "NumWebVisitsMonth": web_visits,
    }], columns=FEATURES)

    try:
        scaled_customer = scaler.transform(customer)
        cluster = int(model.predict(scaled_customer)[0])
        st.success("Prediction completed")
        st.subheader(f"Predicted Cluster: {cluster}")
        st.header(SEGMENT_NAMES.get(cluster, f"Customer Cluster {cluster}"))
        st.write(SEGMENT_DESCRIPTIONS.get(cluster, "This customer belongs to the predicted cluster."))
        with st.expander("Review the values used for prediction"):
            st.dataframe(customer, use_container_width=True, hide_index=True)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

st.divider()
st.caption("Model: K-Means clustering · Preprocessing: StandardScaler · Features: 11")
