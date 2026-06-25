# Customer Segmentation & Recommendation System

## Project Overview

An end-to-end customer analytics project that segments customers based on purchasing behaviour and generates personalized product recommendations using unsupervised machine learning.

The project performs data cleaning, RFM analysis, behavioural feature engineering, dimensionality reduction using PCA, customer segmentation using K-Means and Hierarchical Clustering, and builds a recommendation engine along with an interactive dashboard and automated business report.

---

## Business Problem

Businesses often treat all customers similarly despite differences in purchasing behaviour.

This project helps answer questions like:

- Which customers generate the highest revenue?
- Which customers are at risk of churn?
- How can marketing campaigns be personalized?
- Which products should be recommended to different customer groups?

---

## Dataset

**Dataset:** UCI Online Retail Dataset

- 541,909 transactions
- 4,334 customers after preprocessing
- 3,661 products
- Online retail transactions from a UK-based retailer

Dataset is downloaded automatically during execution.

---

## Project Pipeline

```
Raw Dataset
      │
      ▼
Data Cleaning
      │
      ▼
RFM Analysis
      │
      ▼
Behavioural Feature Engineering
      │
      ▼
Log Transformation + Robust Scaling
      │
      ▼
Principal Component Analysis (PCA)
      │
      ▼
K-Means & Hierarchical Clustering
      │
      ▼
Customer Segments
      │
      ├──────────────┐
      ▼              ▼
Recommendation    Dashboard
Engine            Business Report
```

---

## Features

### Data Preprocessing

- Missing value handling
- Cancellation removal
- Invalid transaction filtering
- Feature creation
- Date parsing
- Revenue calculation

### Customer Feature Engineering

- Recency
- Frequency
- Monetary Value
- Average Order Value
- Average Items per Order
- Product Diversity
- Category Diversity
- Preferred Shopping Hour
- Purchase Span
- Country Count

### Machine Learning

- RobustScaler
- Log Transformation
- PCA
- K-Means Clustering
- Hierarchical Clustering

### Recommendation System

- Segment-based recommendations
- Similarity-based recommendations
- Popularity baseline comparison

---

## Results

### PCA

- Reduced customer features using Principal Component Analysis
- **4 Principal Components retained**
- **89.1% variance explained**

### Customer Segments

The project automatically identified **2 customer segments**:

- High-Value Champions
- At-Risk / Hibernating Customers

### Recommendation Engine

Provides:

- Personalized recommendations
- Segment-specific products
- Similarity-based recommendations
- Popularity baseline comparison

---

## Project Structure

```
customer_segmentation_project/

│── data/
│── notebooks/
│── outputs/
│── src/
│     ├── clustering.py
│     ├── data_loader.py
│     ├── dashboard.py
│     ├── eda.py
│     ├── feature_engineering.py
│     ├── pca_analysis.py
│     ├── preprocessing.py
│     ├── recommendation.py
│     └── report.py
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- SciPy
- Matplotlib
- Seaborn
- Plotly
- OpenPyXL

---

## Installation

Clone the repository

```bash
git clone https://github.com/Ishikaaa23/customer-segmentation-recommendation.git
```

Create a virtual environment

```bash
python -m venv venv
```

Activate it

Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the project

```bash
python main.py
```

---

## Outputs

The project automatically generates:

- Customer Segments
- Recommendation Engine
- Interactive Dashboard
- Business Report
- PCA Visualizations
- Cluster Analysis
- Marketing Insights

---

## Future Improvements

- DBSCAN and Gaussian Mixture clustering
- Customer Lifetime Value (CLV)
- Real-time recommendation API
- Streamlit web application
- Model deployment

---