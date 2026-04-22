import pandas as pd, numpy as np, joblib

model  = joblib.load('models/ids_model.pkl')
scaler = joblib.load('models/scaler.pkl')
df     = pd.read_csv('data/processed/scaled_data.csv')

scores = model.decision_function(df.values)
print('min   :', scores.min().round(4))
print('p5    :', np.percentile(scores, 5).round(4))
print('median:', np.median(scores).round(4))
print('max   :', scores.max().round(4))

thresh = float(open('models/threshold.txt').read())
print('current threshold:', thresh)
print('flagged as anomaly:', (scores < thresh).sum(), '/', len(scores))