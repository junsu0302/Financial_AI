import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(100)

# 원본 데이터 불러오기
raw = pd.read_csv('data/aiif_eikon_eod_data.csv', index_col=0, parse_dates=True)['EUR=']

# 데이터를 리샘플링하고 전처리
labels = raw.resample('1M').last()
labels = labels.values
labels -= labels.mean()
features = np.linspace(-2, 2, len(labels))

# 데이터를 훈련, 검증, 테스트 세트로 분할
split_s = int(0.25 * len(features))
split_e = int(0.25 * len(features))

index = np.arange(len(features))
np.random.shuffle(index)
index_test = np.sort(index[:split_s])
index_validation = np.sort(index[split_s:split_s+split_e])
index_train = np.sort(index[split_s+split_e:])

f_test = features[index_test]
f_validation = features[index_validation]
f_train = features[index_train]
l_test = labels[index_test]
l_validation = labels[index_validation]
l_train = labels[index_train]

# 데이터 시각화
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# 전체 데이터셋
axs[0, 0].plot(features, labels, 'ro')
axs[0, 0].set_title('Main Dataset')
axs[0, 0].set_xlabel('features')
axs[0, 0].set_ylabel('labels')

# 훈련 데이터셋
axs[0, 1].plot(f_train, l_train, 'go')
axs[0, 1].set_title('Training Set')
axs[0, 1].set_xlabel('features')
axs[0, 1].set_ylabel('labels')

# 검증 데이터셋
axs[1, 0].plot(f_validation, l_validation, 'bo')
axs[1, 0].set_title('Validation Set')
axs[1, 0].set_xlabel('features')
axs[1, 0].set_ylabel('labels')

# 테스트 데이터셋
axs[1, 1].plot(f_test, l_test, 'mo')
axs[1, 1].set_title('Test Set')
axs[1, 1].set_xlabel('features')
axs[1, 1].set_ylabel('labels')

plt.tight_layout()
plt.savefig('assets/ML/data.png')
plt.show()
