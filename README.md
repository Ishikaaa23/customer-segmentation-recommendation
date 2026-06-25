# Customer Segmentation & Recommendation System

> An end-to-end customer analytics project that segments customers based on purchasing behaviour and generates personalized product recommendations using unsupervised machine learning.

## Overview

Understanding customer behaviour is one of the most important challenges in retail analytics. Instead of treating every customer the same, businesses can group customers with similar purchasing patterns and design targeted marketing strategies.

In this project, I built a complete customer segmentation pipeline using the UCI Online Retail dataset. The workflow starts with cleaning raw transaction data, engineering customer-level features using RFM analysis and behavioural metrics, reducing feature dimensionality with Principal Component Analysis (PCA), and identifying customer segments using K-Means and Hierarchical Clustering. Finally, a recommendation engine generates personalized product suggestions, supported by an interactive dashboard and an automated business report.

## Business Objectives

- Which customers generate the highest revenue?
- Which customers are at risk of becoming inactive?
- How can customers be segmented based on purchasing behaviour?
- Which products should be recommended to different customer groups?
- How can these insights support targeted marketing campaigns?

## Dataset

**Source:** UCI Machine Learning Repository – Online Retail Dataset

| Description | Value |
|---|---:|
| Raw Transactions | 541,909 |
| Customers (after preprocessing) | 4,334 |
| Products | 3,661 |

The dataset is downloaded automatically during execution.

## Project Workflow

```text
Online Retail Dataset
        |
Data Cleaning & Validation
        |
RFM + Behavioural Feature Engineering
        |
Log Transformation + Robust Scaling
        |
Principal Component Analysis (PCA)
        |
K-Means + Hierarchical Clustering
        |
Customer Segments + Recommendation Engine
        |
Dashboard & Business Report
```

## Key Features

- Automated data preprocessing
- RFM analysis
- Behavioural feature engineering
- PCA-based dimensionality reduction
- K-Means and Hierarchical Clustering
- Segment-based recommendation engine
- Similarity-based recommendation engine
- Interactive dashboard
- Automated business report

## Results

- 4 Principal Components retained (89.1% explained variance)
- 2 customer segments identified:
  - High-Value Champions
  - At-Risk / Hibernating
- Personalized recommendation engine with popularity baseline comparison

## Repository Structure

```text
customer_segmentation_project/
├── data/
├── notebooks/
├── outputs/
├── src/
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Data | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Visualization | Matplotlib, Seaborn, Plotly |

## Installation

```bash
git clone https://github.com/Ishikaaa23/customer-segmentation-recommendation.git
cd customer-segmentation-recommendation
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

## Outputs

- Customer Segments
- Product Recommendations
- Interactive Dashboard
- Business Report
- PCA Visualizations
- Marketing Insights
