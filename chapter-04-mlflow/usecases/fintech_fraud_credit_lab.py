import pandas as pd, mlflow, mlflow.sklearn
import os, subprocess, sys
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, '..', 'data', 'transactions.csv')
if not os.path.exists(DATA_PATH):
    subprocess.run([sys.executable, os.path.join(HERE, '..', 'src', 'generate_data.py')], check=True)

df = pd.read_csv(DATA_PATH)
X = df[['amount', 'time']]
y = df['is_fraud']
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment('fraud-detection-v1')
with mlflow.start_run():
    mlflow.log_param('max_depth', 3)
    mlflow.log_param('criterion', 'gini')

    clf = DecisionTreeClassifier(max_depth=3, criterion='gini')
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)

    mlflow.log_metric('accuracy', accuracy_score(y_te, y_pred))
    mlflow.log_metric('recall', recall_score(y_te, y_pred))
    mlflow.log_metric('f1', f1_score(y_te, y_pred))
    mlflow.sklearn.log_model(clf, name='fraud_model')
