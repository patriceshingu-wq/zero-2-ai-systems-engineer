import pandas as pd, mlflow, mlflow.sklearn
import os, subprocess, sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, recall_score, accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, '..', 'data', 'patient_encounters.csv')
if not os.path.exists(DATA_PATH):
    subprocess.run([sys.executable, os.path.join(HERE, 'generate_usecase_data.py')], check=True)

df = pd.read_csv(DATA_PATH)
X = df[['age', 'length_of_stay', 'prior_admissions', 'chronic_conditions']]
y = df['readmitted_30d']
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment('readmission-risk-v1')
with mlflow.start_run():
    mlflow.set_tag('data_snapshot', '2026-Q2')
    mlflow.set_tag('reviewed_by', 'pending')

    mlflow.log_param('n_estimators', 200)
    mlflow.log_param('max_depth', 8)

    clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    y_proba = clf.predict_proba(X_te)[:, 1]

    mlflow.log_metric('accuracy', accuracy_score(y_te, y_pred))
    mlflow.log_metric('auc', roc_auc_score(y_te, y_proba))
    mlflow.log_metric('recall_high_risk', recall_score(y_te, y_pred))
    mlflow.sklearn.log_model(clf, name='readmission_model')
