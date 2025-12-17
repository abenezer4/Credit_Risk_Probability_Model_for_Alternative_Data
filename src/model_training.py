# src/model_training.py

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, 
    confusion_matrix, roc_curve, precision_recall_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns

# Import the calculator we made in Task 3
try:
    from data_preprocessing import WoE_IV_Calculator
except ImportError:
    # Fallback if running from a different directory context
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from data_preprocessing import WoE_IV_Calculator

class ModelTrainer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = pd.read_csv(data_path)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.feature_names = [] 

    def prepare_data(self, target_col='is_high_risk', test_size=0.2, random_state=42):
        X = self.df.drop(columns=[target_col, 'CustomerId'])
        y = self.df[target_col]
        self.feature_names = X.columns.tolist()
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"✅ Data Split: Train shape {self.X_train.shape}, Test shape {self.X_test.shape}")
        return self.X_train, self.X_test, self.y_train, self.y_test

    # ... (Keep train_model, hyperparameter_tuning, find_best_model, register_model) ...
    # ... (Ensure you have the latest versions from previous steps) ...
    
    def train_model(self, model_type='logistic', params=None, experiment_name="Credit_Risk_Experiment"):
        """
        Trains a model and logs it to MLflow.
        """
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run():
            # 1. Initialize Model
            if model_type == 'logistic':
                self.model = LogisticRegression(**(params or {}))
            elif model_type == 'random_forest':
                self.model = RandomForestClassifier(**(params or {}))
            elif model_type == 'xgboost':
                # XGBoost Classifier is a Scikit-Learn wrapper
                self.model = XGBClassifier(eval_metric='logloss', **(params or {}))
            else:
                raise ValueError("Unsupported model type")

            # 2. Train
            print(f"🔄 Training {model_type}...")
            self.model.fit(self.X_train, self.y_train)
            
            # 3. Predict
            y_pred = self.model.predict(self.X_test)
            y_prob = self.model.predict_proba(self.X_test)[:, 1]
            
            # 4. Calculate Metrics
            metrics = {
                "accuracy": accuracy_score(self.y_test, y_pred),
                "precision": precision_score(self.y_test, y_pred, zero_division=0),
                "recall": recall_score(self.y_test, y_pred),
                "f1_score": f1_score(self.y_test, y_pred),
                "roc_auc": roc_auc_score(self.y_test, y_prob)
            }
            
            # 5. Log to MLflow
            mlflow.log_params(params or {})
            mlflow.log_param("model_type", model_type)
            mlflow.log_metrics(metrics)
            
            # FIX: Use sklearn logger for ALL models (including XGBClassifier)
            # This handles the wrapper class correctly
            mlflow.sklearn.log_model(self.model, "model")
            
            print(f"✅ Training Complete. Metrics: {metrics}")
            return metrics

    def hyperparameter_tuning(self, model_type, param_grid, cv=3):
        # ... (Same code as before) ...
        print(f"🔄 Starting Hyperparameter Tuning for {model_type}...")
        if model_type == 'random_forest':
            estimator = RandomForestClassifier(random_state=42)
        elif model_type == 'xgboost':
            estimator = XGBClassifier(eval_metric='logloss', random_state=42)
        else:
            raise ValueError("Tuning not implemented for this model type")

        grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=1)
        grid_search.fit(self.X_train, self.y_train)
        best_params = grid_search.best_params_
        print(f"✅ Best Parameters found: {best_params}")
        return best_params

    def find_best_model(self, experiment_name="Credit_Risk_Experiment", metric="roc_auc"):
        # ... (Same code as before) ...
        print(f"🔍 Searching for best model in experiment: '{experiment_name}' based on {metric}...")
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None: return None
            exp_id = experiment.experiment_id
        except Exception as e:
            return None
        runs_df = mlflow.search_runs(experiment_ids=[exp_id])
        metric_col = f"metrics.{metric}"
        if metric_col not in runs_df.columns: return None
        best_run = runs_df.sort_values(by=metric_col, ascending=False).iloc[0]
        return best_run["run_id"], best_run["params.model_type"], best_run[metric_col]

    def register_model(self, run_id, model_name="Credit_Risk_Best_Model"):
        # ... (Same code as before) ...
        print(f"📝 Registering Run ID {run_id} as '{model_name}'...")
        model_uri = f"runs:/{run_id}/model"
        mlflow.register_model(model_uri, model_name)
        print(f"✅ Model registered successfully.")

    # --- VISUALIZATION METHODS ---
    
    def analyze_feature_iv(self, target_col='is_high_risk'):
        """
        Calculates Information Value (IV) for all features against the target.
        Displays a bar chart ranking features by predictive power.
        """
        print("📊 Calculating IV for all features...")
        
        # Combine X and y temporarily for calculation
        df_temp = self.X_train.copy()
        df_temp[target_col] = self.y_train
        
        iv_results = []
        
        for feature in self.feature_names:
            # Bin numeric features (qcut) to calculate WoE
            try:
                df_temp[f'{feature}_bin'] = pd.qcut(df_temp[feature], q=10, duplicates='drop')
                calc = WoE_IV_Calculator(df_temp, f'{feature}_bin', target_col)
                woe_df = calc.calculate()
                total_iv = woe_df['IV'].sum()
                iv_results.append({'Feature': feature, 'IV': total_iv})
            except Exception as e:
                print(f"Skipping {feature} due to binning error: {e}")

        iv_df = pd.DataFrame(iv_results).sort_values(by='IV', ascending=False)
        
        # Interpret IV
        def get_iv_strength(iv):
            if iv < 0.02: return 'Useless'
            elif iv < 0.1: return 'Weak'
            elif iv < 0.3: return 'Medium'
            elif iv < 0.5: return 'Strong'
            else: return 'Suspicious (>0.5)'
            
        iv_df['Strength'] = iv_df['IV'].apply(get_iv_strength)
        
        print(iv_df)
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.barplot(x='IV', y='Feature', data=iv_df, palette='viridis')
        plt.title('Feature Importance via Information Value (IV)')
        plt.axvline(x=0.02, color='red', linestyle='--', label='Weak Predictor Threshold')
        plt.axvline(x=0.5, color='orange', linestyle='--', label='Suspicious Threshold')
        plt.legend()
        plt.show()
        
        return iv_df

    def plot_feature_importance(self):
        """Plots feature importance for Tree-based models."""
        if self.model is None: return
        importances = None
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_[0])
        
        if importances is not None:
            indices = np.argsort(importances)[::-1]
            top_n = min(15, len(self.feature_names))
            plt.figure(figsize=(10, 6))
            plt.title("Top 15 Feature Importances (Model Internal)")
            plt.bar(range(top_n), importances[indices[:top_n]], align="center")
            plt.xticks(range(top_n), [self.feature_names[i] for i in indices[:top_n]], rotation=45, ha='right')
            plt.xlim([-1, top_n])
            plt.tight_layout()
            plt.show()

    def plot_roc_curve(self):
        if self.model is None: return
        y_prob = self.model.predict_proba(self.X_test)[:, 1]
        fpr, tpr, _ = roc_curve(self.y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.show()

    def plot_precision_recall_curve(self):
        if self.model is None: return
        y_prob = self.model.predict_proba(self.X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(self.y_test, y_prob)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.show()
    
    def plot_confusion_matrix(self):
        y_pred = self.model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()