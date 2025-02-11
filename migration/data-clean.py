import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from helpers import gen_histograms

PATH='D:'
PATH_FIG=r'C:\Users\renan\Documents\Migration_Model\migration\figures'

df=pd.read_csv(f'{PATH}\\od-est.csv')

cov_list=['inc_pc_d', 'population91_d', 'PT_votes_98_d', 'share_landless_d', 'illiter_d',
          'urbanization_d', 'agro_share_d', 'elevation_d', 'dist', 'travel_time']

# plot histograms of all covariates in the same plot
gen_histograms(df, cov_list)
plt.savefig(f'{PATH_FIG}\\histograms.png')
plt.show()

# log transformation
for cov in ['inc_pc_d', 'population91_d', 'inc_pc_o', 'population91_o']:
    df[cov]=np.log(df[cov])

# heatmap correlation
# sns.heatmap(df[cov_list].corr(), annot=True)
# plt.savefig(f'{PATH_FIG}\\heatmap.png')
# plt.show()

df['same_muni']=np.where(df['muni_id_o']==df['muni_id_d'], 1, 0)

df.to_csv(f'{PATH}\\od-est-clean.csv', index=False)