import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

df = pd.read_csv("/home/kandelaki/git/SAM-Adapter-PyTorch/figures/dataframes/df.csv")

df  = df[df["metric"] == "IoU"]

df_trained_on_1 = df[df["trained_on"] == 1.0]

df_trained_on_2 = df[df["trained_on"] == 2.0]

df_trained_on_4 = df[df["trained_on"] == 4.0]

df_trained_on_6 = df[df["trained_on"] == 6.0]


df_trained_on_2 = df_trained_on_2.sort_values(by=['tested_on'])
df_trained_on_1 = df_trained_on_1.sort_values(by=['tested_on'])
df_trained_on_4 = df_trained_on_4.sort_values(by=['tested_on'])
df_trained_on_6= df_trained_on_6.sort_values(by=['tested_on'])




fig, ax = plt.subplots()


ax.plot("tested_on","mean", data=df_trained_on_1, marker='*', markerfacecolor='blue', markersize=8, color='skyblue', linewidth=2)

ax.plot("tested_on","mean", data=df_trained_on_2, marker='o', markerfacecolor='brown', markersize=8, color='gray', linewidth=2)

ax.plot("tested_on","mean", data=df_trained_on_4, marker='+', markerfacecolor='black', markersize=8, color='blue', linewidth=2)

ax.plot("tested_on","mean", data=df_trained_on_6, marker='x', markerfacecolor='black', markersize=8, color='red', linewidth=2)


ax.legend()

ax.grid(alpha=.4,linestyle='--')


fig.tight_layout()

plt.savefig('plot.png')