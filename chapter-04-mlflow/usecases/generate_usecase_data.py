# generate_usecase_data.py
# Makes the practice data for the 4.10 (healthcare) and 4.11 (entertainment)
# guided labs. The 4.9 (fintech) lab reuses data/transactions.csv, which
# src/generate_data.py already builds.
import pandas as pd
import numpy as np, os

HERE = os.path.dirname(os.path.abspath(__file__))
# scripts live in usecases/, so the data folder is one level up: ../data
DATA_DIR = os.path.join(HERE, '..', 'data')

np.random.seed(42)
n = 1000

# --- Healthcare: patient_encounters.csv ---
age = np.random.randint(18, 90, size=n)
length_of_stay = np.random.exponential(scale=4, size=n).round(1)
prior_admissions = np.random.poisson(lam=1.2, size=n)
chronic_conditions = np.random.poisson(lam=1.5, size=n)
risk_score = (0.02 * age + 0.4 * length_of_stay
              + 0.8 * prior_admissions + 0.6 * chronic_conditions
              + np.random.normal(0, 2, size=n))
readmitted_30d = (risk_score > np.percentile(risk_score, 75)).astype(int)

patients = pd.DataFrame({
    'age': age,
    'length_of_stay': length_of_stay,
    'prior_admissions': prior_admissions,
    'chronic_conditions': chronic_conditions,
    'readmitted_30d': readmitted_30d,
})

# --- Entertainment: subscriber_activity.csv ---
avg_watch_hours_week = np.random.exponential(scale=5, size=n).round(2)
days_since_last_login = np.random.exponential(scale=7, size=n).round(1)
titles_completed_30d = np.random.poisson(lam=3, size=n)
support_tickets_90d = np.random.poisson(lam=0.5, size=n)
churn_score = (-0.3 * avg_watch_hours_week + 0.5 * days_since_last_login
               - 0.4 * titles_completed_30d + 0.7 * support_tickets_90d
               + np.random.normal(0, 2, size=n))
churned_next_30d = (churn_score > np.percentile(churn_score, 75)).astype(int)

subscribers = pd.DataFrame({
    'avg_watch_hours_week': avg_watch_hours_week,
    'days_since_last_login': days_since_last_login,
    'titles_completed_30d': titles_completed_30d,
    'support_tickets_90d': support_tickets_90d,
    'churned_next_30d': churned_next_30d,
})

os.makedirs(DATA_DIR, exist_ok=True)
patients.to_csv(os.path.join(DATA_DIR, 'patient_encounters.csv'), index=False)
subscribers.to_csv(os.path.join(DATA_DIR, 'subscriber_activity.csv'), index=False)
print(f'Generated {len(patients)} rows, {readmitted_30d.sum()} readmitted -> patient_encounters.csv')
print(f'Generated {len(subscribers)} rows, {churned_next_30d.sum()} churned -> subscriber_activity.csv')
