# src/rfm_analysis.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

class RFMCalculator:
    """
    Calculates Recency, Frequency, and Monetary value for each customer.
    """
    def __init__(self, transaction_data):
        self.df = transaction_data.copy()

    def calculate_rfm(self, customer_col='CustomerId', date_col='TransactionStartTime', amount_col='Amount'):
        print("📊 Calculating RFM Metrics...")
        
        # Ensure date column is datetime
        self.df[date_col] = pd.to_datetime(self.df[date_col])
        
        # Define Snapshot date (the day after the last transaction in the dataset)
        snapshot_date = self.df[date_col].max() + pd.Timedelta(days=1)
        
        # Calculate RFM
        rfm = self.df.groupby(customer_col).agg({
            date_col: lambda x: (snapshot_date - x.max()).days, # Recency: Days since last txn
            customer_col: 'count',                               # Frequency: Total count
            amount_col: 'sum'                                    # Monetary: Total Value
        })
        
        # Rename columns
        rfm.rename(columns={
            date_col: 'Recency',
            customer_col: 'Frequency',
            amount_col: 'Monetary'
        }, inplace=True)
        
        # Handle outliers/negatives (Credit Risk usually looks at spending power)
        # We ensure Monetary is absolute (magnitude of flow) or handle negatives if they imply debt
        rfm['Monetary'] = rfm['Monetary'].abs() 
        
        print(f"✅ RFM Calculated. Shape: {rfm.shape}")
        return rfm

class CustomerSegmenter:
    """
    Segments customers using K-Means Clustering based on RFM scores.
    """
    def __init__(self, rfm_data, n_clusters=3, random_state=42):
        self.rfm = rfm_data.copy()
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.scaler = StandardScaler()

    def segment(self):
        print(f"🧩 Segmenting customers into {self.n_clusters} clusters...")
        
        # 1. Scaling (K-Means is sensitive to scale)
        rfm_scaled = self.scaler.fit_transform(self.rfm[['Recency', 'Frequency', 'Monetary']])
        rfm_scaled_df = pd.DataFrame(rfm_scaled, columns=['Recency', 'Frequency', 'Monetary'])
        
        # 2. Apply K-Means
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        clusters = self.kmeans.fit_predict(rfm_scaled_df)
        
        # 3. Assign Cluster Labels
        self.rfm['RFM_Cluster'] = clusters
        
        print("✅ Segmentation Complete.")
        return self.rfm

    def define_risk_label(self, segmented_data):
        """
        Logic to decide which cluster is 'High Risk' (Bad) and which is 'Low Risk' (Good).
        High Risk usually means: High Recency (Inactive), Low Frequency, Low Monetary.
        """
        # Calculate centroids (average values per cluster)
        centroids = segmented_data.groupby('RFM_Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
        print("\n📊 Cluster Centroids (Averages):")
        print(centroids)
        
        # Logic: Find the cluster with the Highest Recency and Lowest Frequency
        # We assume High Recency = Risk (Customer churned or is inactive)
        
        # Scoring the clusters: 
        # Rank Recency descending (High Recency = High Score)
        # Rank Freq ascending (Low Freq = High Score)
        # Rank Mon ascending (Low Mon = High Score)
        
        # Simple heuristic: The cluster with the MAX Recency is usually the 'Bad' / 'High Risk' one
        high_risk_cluster = centroids['Recency'].idxmax()
        
        print(f"\n⚠️ Identified Cluster {high_risk_cluster} as 'High Risk' (Highest Recency).")
        
        # Assign Label: 1 = High Risk, 0 = Low Risk
        segmented_data['is_high_risk'] = segmented_data['RFM_Cluster'].apply(
            lambda x: 1 if x == high_risk_cluster else 0
        )
        
        return segmented_data

def visualize_clusters(df, x, y, hue):
    """Helper to visualize 2D slices of the clusters."""
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x=x, y=y, hue=hue, palette='viridis', alpha=0.6)
    plt.title(f'Customer Segments: {x} vs {y}')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()