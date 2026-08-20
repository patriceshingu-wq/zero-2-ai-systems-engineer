import pandas as pd, mlflow, mlflow.sklearn
import os, subprocess, sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, recall_score, accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, '..', 'data', 'subscriber_activity.csv')
if not os.path.exists(DATA_PATH):
    subprocess.run([sys.executable, os.path.join(HERE, 'generate_usecase_data.py')], check=True)

df = pd.read_csv(DATA_PATH)
X = df[['avg_watch_hours_week', 'days_since_last_login',
        'titles_completed_30d', 'support_tickets_90d']]
y = df['churned_next_30d']
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment('subscriber-churn-v1')
with mlflow.start_run():
    mlflow.log_param('n_estimators', 150)
    mlflow.log_param('max_depth', 7)

    clf = RandomForestClassifier(n_estimators=150, max_depth=7, random_state=42)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    y_proba = clf.predict_proba(X_te)[:, 1]

    mlflow.log_metric('accuracy', accuracy_score(y_te, y_pred))
    mlflow.log_metric('auc', roc_auc_score(y_te, y_proba))
    mlflow.log_metric('recall_churn', recall_score(y_te, y_pred))
    mlflow.sklearn.log_model(clf, name='churn_model')
