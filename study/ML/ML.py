import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(100)

raw = pd.read_csv('data/aiif_eikon_eod_data.csv', index_col=0, parse_dates=True)['EUR=']

labels = raw.resample('1M').last()
labels = labels.values
labels -= labels.mean()
features = np.linspace(-2, 2, len(labels))
plt.figure(figsize=(10, 6))
plt.plot(features, labels, 'ro')
plt.title('Sample Data Set')
plt.xlabel('features')
plt.ylabel('labels')
plt.show()