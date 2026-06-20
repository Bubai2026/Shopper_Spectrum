```markdown
# Shopper Spectrum
## Customer Segmentation and Product Recommendations in E-Commerce

---

## 📖 Project Overview

Shopper Spectrum is an end-to-end data science project that analyzes e-commerce transaction data to:

- **Segment customers** based on **Recency, Frequency, and Monetary (RFM)** analysis using KMeans clustering.
- **Recommend similar products** using **item-based collaborative filtering** (cosine similarity).
- Provide an interactive **Streamlit web application** for real-time predictions.

This project demonstrates practical applications in targeted marketing, personalized recommendations, and customer retention.

---

## 🎯 Problem Statement

The global e-commerce industry generates vast amounts of transaction data daily. Analyzing this data is essential for identifying meaningful customer segments and recommending relevant products to enhance customer experience and drive business growth. This project aims to:

- Uncover patterns in customer purchase behavior.
- Segment customers based on RFM analysis.
- Develop a product recommendation system using collaborative filtering.

---

## 📁 Dataset

- **Source:** [Online Retail Dataset](https://drive.google.com/file/d/1rzRwxm_CJxcRzfoo9Ix37A2JTlMummY-/view?usp=sharing)
- **Description:** Transactional data from a UK-based online retailer (2022–2023).
- **Features:** InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country.

---

## 🔧 Methodology

### 1. Data Preprocessing
- Removed rows with missing `CustomerID`.
- Excluded cancelled invoices (InvoiceNo starting with 'C').
- Filtered out negative quantities and prices.
- Created `TotalPrice` = Quantity × UnitPrice.

### 2. Exploratory Data Analysis (EDA)
- 15+ visualizations (country sales, top products, seasonality, RFM distributions, etc.).
- Insights: UK dominates transactions, sales are seasonal, customer spending is highly skewed.

### 3. Customer Segmentation (RFM + Clustering)
- **RFM Calculation:**
  - **Recency:** Days since last purchase.
  - **Frequency:** Number of unique invoices.
  - **Monetary:** Total spend.
- **Scaling:** StandardScaler.
- **Clustering:** KMeans (k=4) determined by Elbow Method and Silhouette Score.
- **Segment Labels:**
  | Cluster | Characteristics                | Segment Label |
  |---------|--------------------------------|---------------|
  | 0       | Low Recency, High Frequency, High Monetary | **High-Value** |
  | 1       | Medium Recency, Medium Frequency, Medium Monetary | **Regular**   |
  | 2       | High Recency, Low Frequency, Low Monetary | **At-Risk**   |
  | 3       | Low Recency, Low Frequency, Low Monetary | **Occasional**|

### 4. Product Recommendation System
- Built a customer–product matrix (CustomerID × StockCode).
- Computed **cosine similarity** between products.
- Returns top-5 similar products for any given product code.

### 5. Model Deployment (Streamlit App)
- Two modules:
  - **Product Recommendation:** Enter product name/code → get 5 similar products.
  - **Customer Segmentation:** Input Recency, Frequency, Monetary → predict segment label.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Bubai2026/shopper-spectrum.git
   cd shopper-spectrum
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset:**
   The notebook will automatically download it using `gdown`. Alternatively, download it manually and place `online_retail.csv` in the project folder.

### Running the Jupyter Notebook
Launch the notebook to see all analysis, visualizations, and model training:
```bash
jupyter notebook Shopper_Spectrum.ipynb
```

### Running the Streamlit App
```bash
streamlit run App/app.py
```

The app will open in your browser.

---

## 📊 Streamlit App Features

### 🔍 Product Recommendation
- Input a product name (e.g., "WHITE HANGING HEART") or StockCode.
- Click "Get Recommendations".
- Displays 5 similar products with their names and codes.

### 👤 Customer Segmentation
- Input:
  - **Recency** (days since last purchase)
  - **Frequency** (number of purchases)
  - **Monetary** (total spend)
- Click "Predict Cluster".
- Outputs the segment label (High-Value, Regular, At-Risk, Occasional) with a short business insight.

---

## 📁 Project Structure

```
shopper-spectrum/
├── App/
│   ├── app.py                  # Streamlit application
│   └── requirements.txt        # Python dependencies
├── Dataset/
│   └── online_retail.csv       # Dataset (not included in repo, download separately)
├── Models/
│   ├── desc_list.pkl           # List of product descriptions (for search)
│   ├── kmeans.pkl              # Trained KMeans model
│   ├── product_list.pkl        # List of all StockCodes
│   ├── product_similarity.pkl  # Cosine similarity matrix (products)
│   ├── scaler.pkl              # StandardScaler for RFM
│   └── stock_to_desc.pkl       # Mapping StockCode → Description
├── Notebook/
│   └── Shopper_Spectrum.ipynb  # Main Jupyter notebook with all analysis
├── .gitattributes              # Git LFS configuration (for large files)
├── .gitignore                  # Git ignore rules
└── README.md                   # Project documentation
```

---

## 🧰 Technical Stack

- **Data Handling:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn (KMeans, StandardScaler, cosine_similarity)
- **Statistical Testing:** SciPy
- **Model Deployment:** Streamlit
- **Others:** Gdown (data download), Pickle (model serialization)

---

## 📈 Key Results

- **Optimal number of clusters:** 4 (based on Elbow & Silhouette).
- **Silhouette Score:** ~0.45 (indicating reasonable separation).
- **Cluster sizes:** Balanced, with a majority of Regular customers.
- **Recommendation quality:** The system retrieves relevant similar products, which can be validated manually.

---

## 💡 Business Impact

- **Targeted Marketing:** High-Value customers can receive premium offers; At-Risk customers can be re-engaged.
- **Cross-Selling:** Product recommendations increase average order value.
- **Inventory Management:** Understand demand patterns for popular items.
- **Customer Retention:** Identify and intervene for At-Risk segments.

---

## 🧪 Hypothesis Testing

Three statistical tests were conducted:
1. **High-Value vs Regular Monetary:** Significant difference (p < 0.05).
2. **UK vs Non-UK Transaction Value:** Significant difference (Welch's t-test, p < 0.05).
3. **Recency vs Monetary Correlation:** Significant negative correlation (Pearson r = -0.3, p < 0.05).

---

## 🛠 Future Improvements

- Implement **customer-specific recommendations** (user-based CF or matrix factorization).
- Add **hyperparameter tuning** for clustering (e.g., DBSCAN, hierarchical).
- Incorporate **real-time data streaming** for dynamic segmentation.
- Enhance the app with **product images** and **purchase history visualization**.

---

## 📝 License

This project is for educational purposes. Feel free to use and modify it.

---

## 👤 Author

**Bubai Sou**  
[GitHub](https://github.com/Bubai2026)  
[LinkedIn](https://linkedin.com/in/bubaisou)

---

## 🙏 Acknowledgements

- UCI Machine Learning Repository for the original dataset.
- Streamlit for the amazing app framework.

---

### ⭐ If you found this project useful, please give it a star!
#