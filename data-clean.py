import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

PATH='D:'
df=pd.read_csv(f'{PATH}\\od-est.csv')

cov_list=['inc_pc_x', 'population91_x', 'PT_votes_98_x', 'share_landless_x', 'illiter_x', 
          'urbanization_x', 'agro_share_x', 'elevation_x', 'dist', 'travel_time']

# plot histograms of all covariates in the same plot
fig, axs = plt.subplots(5, 2, figsize=(20, 20))
for i, ax in enumerate(axs.flat):
    cov=cov_list[i]
    sns.histplot(df[cov], bins=20, ax=ax)
    ax.set_title(cov)
plt.show()

df[cov_list].describe()


# log transformation
for cov in ['inc_pc_x', 'population91_x', 'inc_pc_y', 'population91_y']:
    df[cov]=np.log(df[cov])

# boxplot outlier check
fig, axs = plt.subplots(5, 2, figsize=(30, 20))
for i, ax in enumerate(axs.flat):
    cov=cov_list[i]
    sns.boxplot(x=df[cov], ax=ax)
    ax.set_title(cov)
    
plt.show()

# remove outliers



# heatmap correlation
import seaborn as sns
sns.heatmap(df.corr(), annot=True)
plt.show()

df['same_muni']=np.where(df['muni_id_o']==df['muni_id_d'], 1, 0)

df.to_csv(f'{PATH}\\od-est-clean.csv', index=False)