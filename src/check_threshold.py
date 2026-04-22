import joblib, pandas as pd, numpy as np

model  = joblib.load('models/ids_model.pkl')
scaler = joblib.load('models/scaler.pkl')
df     = pd.read_csv('data/processed/scaled_data.csv')
scores = model.decision_function(df.values)
print('2nd  pct:', np.percentile(scores, 2).round(4))
print('5th  pct:', np.percentile(scores, 5).round(4))
print('10th pct:', np.percentile(scores, 10).round(4))