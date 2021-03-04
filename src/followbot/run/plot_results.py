# Author: Javad Amirian
# Email: amiryan.j@gmail.com

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # sudo apt-get install python3.7-tk


results_file = '/home/cyrus/Music/num-results/num_results.csv'

results_df = pd.read_csv(results_file, delimiter=',', names=['time', 'robot_id', 'frame_id', 'hypo', 'density', 'mse', 'bce'])
results_df = results_df[pd.to_numeric(results_df['hypo'], errors='coerce').notnull()]
results_df = results_df.dropna()
results_df['density'] = results_df['density'].astype(float)
results_df['frame_id'] = results_df['frame_id'].astype(float)
results_df['mse'] = results_df['mse'].astype(float)
results_df['bce'] = results_df['bce'].astype(float)

results_df_mean = results_df.groupby('robot_id').mean()


sns.set(style="whitegrid")
fig, ax = plt.subplots()
# plt.scatter(results_df['density'].astype(float), results_df['mse'].astype(float))
sns.scatterplot(x='density', y='mse', data=results_df_mean, ax=ax)
plt.ylim([0, 0.14])
plt.show()
