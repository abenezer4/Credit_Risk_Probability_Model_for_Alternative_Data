# src/data_processing.py

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


class TimeFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, time_col='TransactionStartTime'):
        self.time_col = time_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.time_col in X.columns:
            X[self.time_col] = pd.to_datetime(X[self.time_col])
            X['TransactionHour'] = X[self.time_col].dt.hour
            X['TransactionDay'] = X[self.time_col].dt.day
            X['TransactionMonth'] = X[self.time_col].dt.month
            X['TransactionYear'] = X[self.time_col].dt.year
        return X

class NumericAggregator(BaseEstimator, TransformerMixin):
    """Aggregates numeric transaction data to customer level."""
    def __init__(self, customer_col='CustomerId', amount_col='Amount'):
        self.customer_col = customer_col
        self.amount_col = amount_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print("🔄 Aggregating Numeric Features...")
        X = X.copy()
        
        # Define aggregations
        aggs = {
            self.amount_col: ['sum', 'mean', 'std', 'count', 'min', 'max'],
            'Value': ['sum', 'mean'],
            'TransactionHour': ['mean', 'std']
        }
        
        df_agg = X.groupby(self.customer_col).agg(aggs)
        df_agg.columns = ["_".join(x) for x in df_agg.columns.ravel()]
        df_agg.reset_index(inplace=True)
        
        # Rename
        df_agg.rename(columns={
            f'{self.amount_col}_sum': 'total_amount',
            f'{self.amount_col}_mean': 'avg_amount',
            f'{self.amount_col}_std': 'std_amount',
            f'{self.amount_col}_count': 'txn_count',
            'TransactionHour_mean': 'avg_txn_hour',
            'TransactionHour_std': 'std_txn_hour'
        }, inplace=True)
        
        return df_agg.fillna(0)

class CategoricalAggregator(BaseEstimator, TransformerMixin):
    """
    Creates ratios for categorical columns (e.g., % of transactions that were 'Airtime').
    """
    def __init__(self, customer_col='CustomerId', cat_cols=['ProductCategory', 'ChannelId', 'ProviderId']):
        self.customer_col = customer_col
        self.cat_cols = cat_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print("🔄 Aggregating Categorical Features...")
        X = X.copy()
        
        customer_ids = X[[self.customer_col]].drop_duplicates().sort_values(self.customer_col)
        
        # We merge everything onto this base
        df_final = customer_ids
        
        # Get total counts per customer for denominator
        total_counts = X.groupby(self.customer_col).size()
        
        for col in self.cat_cols:
            if col in X.columns:
                # Get dummies and sum by customer
                dummies = pd.get_dummies(X[[self.customer_col, col]], columns=[col])
                # Group by customer and sum
                agg_cats = dummies.groupby(self.customer_col).sum()
                
                # Calculate Ratios (Count of Specific Cat / Total Txns)
                # Align indices to ensure division works
                agg_cats = agg_cats.div(total_counts, axis=0)
                
                # Add suffix to avoid name collisions
                agg_cats = agg_cats.add_suffix('_ratio')
                
                # Merge
                df_final = pd.merge(df_final, agg_cats, on=self.customer_col, how='left')
                
        return df_final.fillna(0)
class WoE_IV_Calculator:
    """
    Helper class to calculate Weight of Evidence (WoE) and Information Value (IV).
    Note: Requires a binary target variable (Good=0, Bad=1).
    """
    def __init__(self, df, feature, target):
        self.df = df
        self.feature = feature
        self.target = target

    def calculate(self):
        # Calculate distribution of Good (0) and Bad (1)
        lst = []
        for i in range(self.df[self.feature].nunique()):
            val = list(self.df[self.feature].unique())[i]
            lst.append({
                'Value': val,
                'All': self.df[self.df[self.feature] == val].count()[self.feature],
                'Good': self.df[(self.df[self.feature] == val) & (self.df[self.target] == 0)].count()[self.feature],
                'Bad': self.df[(self.df[self.feature] == val) & (self.df[self.target] == 1)].count()[self.feature]
            })
            
        dset = pd.DataFrame(lst)
        dset['Distr_Good'] = dset['Good'] / dset['Good'].sum()
        dset['Distr_Bad'] = dset['Bad'] / dset['Bad'].sum()
        
        # Avoid division by zero with small epsilon
        dset['WoE'] = np.log((dset['Distr_Good'] + 0.0001) / (dset['Distr_Bad'] + 0.0001))
        dset['IV'] = (dset['Distr_Good'] - dset['Distr_Bad']) * dset['WoE']
        
        return dset.sort_values(by='IV', ascending=False)
def build_preprocessing_pipeline(numeric_features):
    """
    Builds the pipeline to Impute and Scale numeric features.
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), # Handle missing
        ('scaler', StandardScaler())                   # Normalize
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ],
        remainder='drop' # IMPORTANT: Drop columns we aren't scaling (like IDs) to avoid shape errors
    )
    
    return preprocessor