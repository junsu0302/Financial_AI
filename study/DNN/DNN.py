import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

raw = pd.read_csv('data/aiif_eikon_id_eur_usd.csv', index_col=0, parse_dates=True)
symbol = 'EUR_USD'

data = pd.DataFrame(raw['CLOSE'].loc[:])
data.columns = [symbol]
data = data.resample('1h', label='right').last().ffill()
