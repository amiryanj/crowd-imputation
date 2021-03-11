# Author: Javad Amirian
# Email: amiryan.j@gmail.com

import numpy as np
import pandas as pd
from scipy.stats import linregress

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # sudo apt-get install python3.7-tk

results_file = '/home/cyrus/Music/num-results/num_results-Hermes.csv'

results_df = pd.read_csv(results_file, delimiter=',', names=['time', 'robot_id', 'frame_id', 'hypo', 'density',
                                                             'mse_ours', 'bce_ours',
                                                             'mse_pcf', 'bce_pcf',
                                                             'mse_bl', 'bce_bl'])
results_df = results_df[pd.to_numeric(results_df['hypo'], errors='coerce').notnull()]
results_df = results_df.dropna()
results_df['density'] = results_df['density'].astype(float)
results_df['frame_id'] = results_df['frame_id'].astype(int)
results_df['mse_ours'] = results_df['mse_ours'].astype(float)
results_df['bce_ours'] = results_df['bce_ours'].astype(float)
results_df['mse_pcf'] = results_df['mse_pcf'].astype(float)
results_df['bce_pcf'] = results_df['bce_pcf'].astype(float)
results_df['mse_bl'] = results_df['mse_bl'].astype(float)
results_df['bce_bl'] = results_df['bce_bl'].astype(float)

results_df_mean = results_df.groupby('robot_id').mean()


sns.set(style="whitegrid")
fig, ax = plt.subplots()
fig.set_figheight(8)
fig.set_figwidth(12)

# sns.scatterplot(x='density', y='mse_ours', data=results_df_mean, ax=ax, label='Ours')
# sns.scatterplot(x='density', y='mse_pcf', data=results_df_mean, ax=ax, label='PCF')
# sns.scatterplot(x='density', y='mse_bl', data=results_df_mean, ax=ax, label='Vanilla-MOT')

sns.scatterplot(x='density', y='bce_ours', data=results_df_mean, ax=ax, label='Ours')
sns.scatterplot(x='density', y='bce_pcf', data=results_df_mean, ax=ax, label='PCF')
sns.scatterplot(x='density', y='bce_bl', data=results_df_mean, ax=ax, label='Vanilla-MOT')

plt.gca().set_ylim(bottom=0)
plt.ylabel("BCE")

# regression
x = results_df_mean['density']

reg_coef_mse_ours = np.polyfit(x, results_df_mean['mse_ours'], 1)
poly1d_fn_ours = np.poly1d(reg_coef_mse_ours)

reg_coef_mse_pcf = np.polyfit(x, results_df_mean['mse_pcf'], 1)
poly1d_fn_pcf = np.poly1d(reg_coef_mse_pcf)

reg_coef_mse_vMOT = np.polyfit(x, results_df_mean['mse_bl'], 1)
poly1d_fn_vMOT = np.poly1d(reg_coef_mse_vMOT)

# plt.plot([min(x), max(x)], poly1d_fn_ours([min(x), max(x)]), '--b')
# plt.plot([min(x), max(x)], poly1d_fn_pcf([min(x), max(x)]), '--r')
# plt.plot([min(x), max(x)], poly1d_fn_vMOT([min(x), max(x)]), '--g')

plt.legend(loc='lower right')
plt.show()
