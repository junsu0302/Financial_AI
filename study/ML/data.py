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
split = int(0.25 * len(features))

index = np.arange(len(features))
np.random.shuffle(index)
index_test = np.sort(index[:split])
index_train = np.sort(index[split:])

f_test = features[index_test]
f_train = features[index_train]
l_test = labels[index_test]
l_train = labels[index_train]

def plot():
  # 데이터 시각화
  fig, axs = plt.subplots(1, 3, figsize=(20, 5))

  # 전체 데이터셋
  axs[0].plot(features, labels, 'ro')
  axs[0].set_title('Main Dataset')
  axs[0].set_xlabel('features')
  axs[0].set_ylabel('labels')

  # 훈련 데이터셋
  axs[1].plot(f_train, l_train, 'go')
  axs[1].set_title('Training Set')
  axs[1].set_xlabel('features')
  axs[1].set_ylabel('labels')

  # 테스트 데이터셋
  axs[2].plot(f_test, l_test, 'mo')
  axs[2].set_title('Test Set')
  axs[2].set_xlabel('features')
  axs[2].set_ylabel('labels')

  plt.tight_layout()
  plt.savefig('assets/ML/data.png')
  plt.show()

# plot()