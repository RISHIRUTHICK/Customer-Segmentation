import joblib
import pandas as pd
import streamlit as st
from pathlib import Path
import sklearn

st.set_page_config(page_title="Customer Segmentation", page_icon="🛍️", layout="wide")
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "kmeans_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"

# Keep the feature order identical to the model-training notebook.
FEATURES = [
    "MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts",
    "MntSweetProducts", "MntGoldProds", "NumDealsPurchases",
    "NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases",
    "NumWebVisitsMonth",
]

# Segment labels and values below follow the notebook's final cluster summary.
SEGMENT_NAMES = {
    0: "High-Value Multi-Channel Customers",
    1: "Low-Spend, Low-Purchase Customers",
    2: "High-Value Store & Catalogue Customers",
    3: "Active Online & Deal Customers",
}
SEGMENT_DESCRIPTIONS = {
    0: ("Notebook profile: 275 customers (12.28%). Highest average total spending "
        "(1427.17), with strong spending across product categories and activity across "
        "web, catalogue, and store channels."),
    1: ("Notebook profile: 1,014 customers (45.27%), the largest segment. Lowest "
        "average total spending (92.66) and total purchases (5.77), despite frequent "
        "website visits (6.29 per month)."),
    2: ("Notebook profile: 478 customers (21.34%). Average total spending is 1200.34 "
        "and average total purchases are the highest (19.66), with particularly strong "
        "store and catalogue activity."),
    3: ("Notebook profile: 473 customers (21.12%). Highest average web purchases (6.44), "
        "deal purchases (4.49), and website visits (6.67 per month), indicating active "
        "online and deal-oriented behaviour."),
}

@st.cache_resource
def load_artifacts():
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)

st.title("🛍️ Customer Segmentation")
st.write("Enter customer spending and purchasing behaviour to predict a segment using the trained K-Means model.")
st.caption(f"4-cluster K-Means · 11 input features · Runtime scikit-learn: {sklearn.__version__}")

if not MODEL_PATH.exists() or not SCALER_PATH.exists():
    st.error("Model files are missing. Keep app.py, kmeans_model.pkl, and scaler.pkl in the same repository folder.")
    st.stop()
try:
    model, scaler = load_artifacts()
except Exception as exc:
    st.error(f"The saved model files could not be loaded: {exc}")
    st.info("The model and scaler were saved with scikit-learn 1.7.2. Keep scikit-learn==1.7.2 in requirements.txt.")
    st.stop()

with st.expander("View the cluster profiles from the notebook"):
    profiles = pd.DataFrame([
        {"Cluster": 0, "Segment": SEGMENT_NAMES[0], "Customer Count": 275, "Customer %": "12.28%", "Avg Total Spent": 1427.17, "Avg Total Purchases": 19.50, "Web Visits / Month": 2.91, "Deal Purchases": 1.45, "Web Purchases": 5.63, "Catalogue Purchases": 5.67, "Store Purchases": 8.21},
        {"Cluster": 1, "Segment": SEGMENT_NAMES[1], "Customer Count": 1014, "Customer %": "45.27%", "Avg Total Spent": 92.66, "Avg Total Purchases": 5.77, "Web Visits / Month": 6.29, "Deal Purchases": 1.86, "Web Purchases": 2.00, "Catalogue Purchases": 0.54, "Store Purchases": 3.22},
        {"Cluster": 2, "Segment": SEGMENT_NAMES[2], "Customer Count": 478, "Customer %": "21.34%", "Avg Total Spent": 1200.34, "Avg Total Purchases": 19.66, "Web Visits / Month": 3.31, "Deal Purchases": 1.67, "Web Purchases": 5.29, "Catalogue Purchases": 5.64, "Store Purchases": 8.73},
        {"Cluster": 3, "Segment": SEGMENT_NAMES[3], "Customer Count": 473, "Customer %": "21.12%", "Avg Total Spent": 627.48, "Avg Total Purchases": 15.79, "Web Visits / Month": 6.67, "Deal Purchases": 4.49, "Web Purchases": 6.44, "Catalogue Purchases": 2.44, "Store Purchases": 6.91},
    ])
    st.dataframe(profiles, use_container_width=True, hide_index=True)
    st.markdown("### Customer Distribution by Cluster")
    st.bar_chart(profiles.set_index("Segment")["Customer Count"])

    st.markdown("### Average Spending by Cluster")
    st.bar_chart(profiles.set_index("Segment")["Avg Total Spent"])

    st.markdown("### Average Purchases by Cluster")
    st.bar_chart(profiles.set_index("Segment")["Avg Total Purchases"])
    st.caption("Cluster IDs and summary values are taken from the notebook's final K-Means cluster summary.")

with st.form("customer_form"):
    st.subheader("Customer details")
    st.markdown("#### Product spending")
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
    st.markdown("#### Purchase behaviour")
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
    customer = pd.DataFrame([[wines, fruits, meat, fish, sweets, gold, deals, web_purchases,
                               catalogue_purchases, store_purchases, web_visits]], columns=FEATURES)
    try:
        scaled_values = scaler.transform(customer)
        scaled_customer = pd.DataFrame(scaled_values, columns=FEATURES)
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
st.caption("Deployment files: app.py, requirements.txt, kmeans_model.pkl, scaler.pkl. The notebook is included as a project reference and is not required to run the app.")
