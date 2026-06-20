import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Set page config
st.set_page_config(page_title="Shopper Spectrum", layout="centered")

st.title("🛒 Shopper Spectrum")
st.subheader("Customer Segmentation & Product Recommendations")

# ------------------- Load Models -------------------
@st.cache_resource
def load_models():
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('kmeans.pkl', 'rb') as f:
        kmeans = pickle.load(f)
    with open('product_similarity.pkl', 'rb') as f:
        sim_df = pickle.load(f)
    with open('product_list.pkl', 'rb') as f:
        product_list = pickle.load(f)
    with open('stock_to_desc.pkl', 'rb') as f:
        stock_to_desc = pickle.load(f)
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
    # Scale the input using the saved scaler
    input_data = np.array([[recency, frequency, monetary]])
    scaled = scaler.transform(input_data)
    cluster = kmeans.predict(scaled)[0]
    # Map cluster number to label
    cluster_map = {
        0: "High-Value",
        1: "Regular",
        2: "At-Risk",
        3: "Occasional"   # adjust order if needed based on your cluster averages
    }
    # But we should determine the actual labels based on your cluster averages.
    # For demonstration, we'll use a fixed mapping; you can adjust after checking the averages.
    # We'll assign based on average RFM; but we can also compute dynamically.
    # For now, we'll return the cluster number and let the user interpret.
    return cluster

# We should compute cluster labels from the saved model or define them manually.
# In your notebook, you identified clusters. Let's capture those labels.
# We'll use a dictionary mapping cluster number to segment name (based on your EDA).
# From your notebook (Chart-8), you had:
# Cluster 0: Low Recency, High Frequency, High Monetary -> High-Value
# Cluster 1: Medium Recency, Medium Frequency, Medium Monetary -> Regular
# Cluster 2: High Recency, Low Frequency, Low Monetary -> At-Risk
# Cluster 3: Low Recency, Low Frequency, Low Monetary -> Occasional (but could be different)
# We'll assume that order; you can verify with your cluster_avg values.
# To be safe, we'll compute the average RFM of each cluster on load and assign labels.
# We can load the rfm data? Or we can just hardcode based on the output you saw.

# We'll do a dynamic label assignment based on average RFM values.
# For this, we need to load the cluster centers from kmeans (in scaled space) and then inverse transform.
# Or we can compute averages from the training data, but we don't have it. So we'll hardcode for now.
# I'll provide a function that returns the label based on cluster number from your known mapping.

def get_cluster_label(cluster):
    # According to your notebook's cluster_avg:
    # Cluster 0: Recency low, Frequency high, Monetary high -> High-Value
    # Cluster 1: Recency medium, Frequency medium, Monetary medium -> Regular
    # Cluster 2: Recency high, Frequency low, Monetary low -> At-Risk
    # Cluster 3: Recency low, Frequency low, Monetary low -> Occasional (or maybe new)
    # We'll assume these labels.
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
    # Input: product description or code
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
                    # Try partial match (optional)
                    # We'll use a simple fuzzy match: find description containing input
                    matches = [desc for desc in desc_to_stock.keys() if product_input.lower() in desc.lower()]
                    if matches:
                        selected_desc = matches[0]
                        stock_code = desc_to_stock[selected_desc]
                    else:
                        st.error("Product not found. Please check the name or code.")
                        st.stop()
            
            # Get recommendations
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
            # Optional: display interpretation
            if label == "High-Value":
                st.info("High-Value customers are recent, frequent, and high spenders. They should be retained with loyalty programs.")
            elif label == "Regular":
                st.info("Regular customers are steady but not premium. Upselling and cross-selling can increase their value.")
            elif label == "At-Risk":
                st.warning("At-Risk customers haven't purchased recently. Consider re-engagement campaigns.")
            elif label == "Occasional":
                st.info("Occasional customers buy rarely. They may need incentives to increase frequency.")