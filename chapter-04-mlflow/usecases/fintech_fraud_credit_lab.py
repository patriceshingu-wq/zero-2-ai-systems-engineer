import pandas as pd, mlflow, mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score

df = pd.read_csv('data/transactions.csv')
X = df[['amount', 'time']]
y = df['is_fraud']
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment('fraud-detection-v1')
with mlflow.start_run():
    mlflow.log_param('max_depth', 6)
    mlflow.log_param('criterion', 'gini')

    clf = DecisionTreeClassifier(max_depth=6, criterion='gini')
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)

    mlflow.log_metric('accuracy', accuracy_score(y_te, y_pred))
    mlflow.log_metric('recall', recall_score(y_te, y_pred))
    mlflow.log_metric('f1', f1_score(y_te, y_pred))
    mlflow.sklearn.log_model(clf, name='fraud_model')
