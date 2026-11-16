"""
Project: Supervised Learning - Credit Card Fraud Detection Pipeline
Description: A complete Machine Learning pipeline handling highly imbalanced data
             using SMOTE, evaluating Logistic Regression and Random Forest.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-Learn modules
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             precision_recall_curve, roc_curve, auc)
from sklearn.datasets import make_classification

# Imbalanced-Learn modules
# Note: install via `pip install imbalanced-learn` if not present
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
except ImportError:
    print("Please install imbalanced-learn using: pip install imbalanced-learn")
    exit()

import warnings
warnings.filterwarnings('ignore')

def load_or_generate_data():
    """
    Attempts to load 'creditcard.csv'. If not found, generates a synthetic dataset
    that mimics the highly imbalanced nature of the Kaggle fraud dataset.
    """
    file_path = 'creditcard.csv'
    if os.path.exists(file_path):
        print("Loading local 'creditcard.csv'...")
        df = pd.read_csv(file_path)
    else:
        print(f"'{file_path}' not found. Generating a synthetic imbalanced dataset for demonstration...")
        # Generating 50,000 samples, with 0.5% fraud (highly imbalanced)
        X_synth, y_synth = make_classification(n_samples=50000, n_features=30, 
                                               n_informative=20, n_redundant=2, 
                                               weights=[0.995, 0.005], random_state=42)
        
        # Convert to DataFrame to match Kaggle structure
        feature_names = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
        df = pd.DataFrame(X_synth, columns=feature_names)
        
        # Make Time and Amount somewhat realistic
        df['Time'] = np.random.randint(0, 100000, size=len(df))
        df['Amount'] = np.abs(np.random.normal(100, 200, size=len(df)))
        df['Class'] = y_synth
        
    return df

def plot_confusion_matrix(y_true, y_pred, title):
    """Utility function to plot a styled confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.show()

def main():
    print("="*50)
    print("💳 CREDIT CARD FRAUD DETECTION PIPELINE")
    print("="*50)

    # 1. Load Data
    df = load_or_generate_data()
    print(f"\nDataset Shape: {df.shape}")
    
    # 2. EDA & Class Imbalance
    fraud_count = df['Class'].sum()
    legit_count = len(df) - fraud_count
    print(f"\nClass Distribution:")
    print(f"Legitimate (0): {legit_count} ({legit_count/len(df)*100:.2f}%)")
    print(f"Fraudulent (1): {fraud_count} ({fraud_count/len(df)*100:.2f}%)")
    
    # Drop duplicates if any
    df = df.drop_duplicates()

    # 3. Data Preprocessing (Scaling)
    print("\nScaling Features (Time and Amount)...")
    scaler = StandardScaler()
    df['Scaled_Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1,1))
    df['Scaled_Time'] = scaler.fit_transform(df['Time'].values.reshape(-1,1))
    df.drop(['Time', 'Amount'], axis=1, inplace=True)

    # 4. Train-Test Split (Stratified to maintain class ratio)
    X = df.drop('Class', axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training Set Shape: {X_train.shape}")
    print(f"Testing Set Shape:  {X_test.shape}")

    # 5. Model 1: Logistic Regression with SMOTE Pipeline
    print("\n" + "-"*40)
    print("Training Logistic Regression with SMOTE...")
    print("-"*40)
    
    # Using ImbPipeline prevents data leakage during cross-validation
    lr_pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('lr', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    # Train directly (Simplified for demonstration, add RandomizedSearchCV here for tuning)
    lr_pipeline.fit(X_train, y_train)
    y_pred_lr = lr_pipeline.predict(X_test)
    y_prob_lr = lr_pipeline.predict_proba(X_test)[:, 1]

    print("Logistic Regression Results:")
    print(classification_report(y_test, y_pred_lr))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob_lr):.4f}")
    plot_confusion_matrix(y_test, y_pred_lr, "Logistic Regression Confusion Matrix")

    # 6. Model 2: Random Forest with SMOTE Pipeline
    print("\n" + "-"*40)
    print("Training Random Forest with SMOTE...")
    print("*(This may take a moment depending on dataset size)*")
    print("-"*40)

    # Using limited estimators/depth to ensure it runs quickly in demo mode
    rf_pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1))
    ])

    rf_pipeline.fit(X_train, y_train)
    y_pred_rf = rf_pipeline.predict(X_test)
    y_prob_rf = rf_pipeline.predict_proba(X_test)[:, 1]

    print("Random Forest Results:")
    print(classification_report(y_test, y_pred_rf))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob_rf):.4f}")
    plot_confusion_matrix(y_test, y_pred_rf, "Random Forest Confusion Matrix")

    # 7. Model Comparison Visualizations
    print("\nGenerating Model Comparison Curves...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ROC Curve Comparison
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
    
    axes[0].plot(fpr_lr, tpr_lr, label=f'LogReg (AUC = {auc(fpr_lr, tpr_lr):.3f})')
    axes[0].plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc(fpr_rf, tpr_rf):.3f})')
    axes[0].plot([0, 1], [0, 1], 'k--')
    axes[0].set_title('ROC Curve Comparison')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].legend()

    # Precision-Recall Curve Comparison
    prec_lr, rec_lr, _ = precision_recall_curve(y_test, y_prob_lr)
    prec_rf, rec_rf, _ = precision_recall_curve(y_test, y_prob_rf)

    axes[1].plot(rec_lr, prec_lr, label=f'LogReg (PR-AUC = {auc(rec_lr, prec_lr):.3f})')
    axes[1].plot(rec_rf, prec_rf, label=f'Random Forest (PR-AUC = {auc(rec_rf, prec_rf):.3f})')
    axes[1].set_title('Precision-Recall Curve Comparison')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    print("\n✅ Pipeline Execution Complete!")

if __name__ == "__main__":
    main()