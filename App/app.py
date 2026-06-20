import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# Set page config
st.set_page_config(page_title="Shopper Spectrum", layout="centered")

st.title("🛒 Shopper Spectrum")
st.subheader("Customer Segmentation & Product Recommendations")

# ------------------- Load Models -------------------
@st.cache_resource
def load_models():
    # Determine the base directory (where this script is)
    base_dir = Path(__file__).parent.parent  # Goes up one level from App/
    models_dir = base_dir / "Models"

    # If Models folder doesn't exist, fallback to current directory (for local testing)
    if not models_dir.exists():
        models_dir = Path(".")

    def load_pickle(filename):
        with open(models_dir / filename, 'rb') as f:
            return pickle.load(f)

    scaler = load_pickle('scaler.pkl')
    kmeans = load_pickle('kmeans.pkl')
    sim_df = load_pickle('product_similarity.pkl')
    product_list = load_pickle('product_list.pkl')
    stock_to_desc = load_pickle('stock_to_desc.pkl')

    # Reverse mapping for display
    desc_to_stock = {v: k for k, v in stock_to_desc.items()}
    return scaler, kmeans, sim_df, product_list, stock_to_desc, desc_to_stock

scaler, kmeans, sim_df, product_list, stock_to_desc, desc_to_stock = load_models()

# ------------------- Helper Functions -------------------
def recommend_products(product_code, sim_df, top_n=5):
    if product_code not in sim_df.index:
        return None
    similarities = sim_df[product_code].sort_values(ascending=False)
    # Exclude the product itself
    recommendations = similarities.iloc[1:top_n+1]
    return recommendations.index.tolist()

def predict_cluster(recency, frequency, monetary):
    input_data = np.array([[recency, frequency, monetary]])
    scaled = scaler.transform(input_data)
    cluster = kmeans.predict(scaled)[0]
    return cluster

def get_cluster_label(cluster):
    labels = {
        0: "High-Value",
        1: "Regular",
        2: "At-Risk",
        3: "Occasional"
    }
    return labels.get(cluster, "Unknown")

# ------------------- UI Tabs -------------------
tab1, tab2 = st.tabs(["🔍 Product Recommendation", "👤 Customer Segmentation"])

with tab1:
    st.header("Find Similar Products")
    product_input = st.text_input("Enter product name or StockCode")
    
    if st.button("Get Recommendations"):
        if not product_input:
            st.warning("Please enter a product name or code.")
        else:
            # Check if input is a StockCode
            if product_input.upper() in product_list:
                stock_code = product_input.upper()
            else:
                # Try to map from description
                stock_code = desc_to_stock.get(product_input.strip())
                if stock_code is None:
                    # Try partial match (case-insensitive)
                    matches = [desc for desc in desc_to_stock.keys() if product_input.lower() in desc.lower()]
                    if matches:
                        selected_desc = matches[0]
                        stock_code = desc_to_stock[selected_desc]
                    else:
                        st.error("Product not found. Please check the name or code.")
                        st.stop()
            
            rec_codes = recommend_products(stock_code, sim_df)
            if rec_codes is None:
                st.error("Product not found in similarity matrix.")
            else:
                st.subheader("Top 5 Similar Products")
                for i, code in enumerate(rec_codes, 1):
                    desc = stock_to_desc.get(code, "Unknown description")
                    st.write(f"{i}. **{code}** – {desc}")

with tab2:
    st.header("Predict Customer Segment")
    st.markdown("Enter the RFM values for a customer to predict their segment.")
    recency = st.number_input("Recency (days since last purchase)", min_value=0, step=1)
    frequency = st.number_input("Frequency (number of purchases)", min_value=0, step=1)
    monetary = st.number_input("Monetary (total spend)", min_value=0.0, step=1.0)
    
    if st.button("Predict Cluster"):
        if recency < 0 or frequency < 0 or monetary < 0:
            st.error("Please enter non-negative values.")
        else:
            cluster = predict_cluster(recency, frequency, monetary)
            label = get_cluster_label(cluster)
            st.success(f"Predicted Segment: **{label}** (Cluster {cluster})")
            if label == "High-Value":
                st.info("High-Value customers are recent, frequent, and high spenders. They should be retained with loyalty programs.")
            elif label == "Regular":
                st.info("Regular customers are steady but not premium. Upselling and cross-selling can increase their value.")
            elif label == "At-Risk":
                st.warning("At-Risk customers haven't purchased recently. Consider re-engagement campaigns.")
            elif label == "Occasional":
                st.info("Occasional customers buy rarely. They may need incentives to increase frequency.")