import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

from paths import PATH_OUT, PATH_FIG, PATH_IN
PATH='D:'

# === PREP DATA ===
sig=pd.read_csv(f'{PATH_OUT}\\signal-clean.csv')
occs=pd.read_csv(f'{PATH_OUT}\\land-merged.csv')
origins=sig[['muni_id']]  # to get coord, muni id  

# get count of od pairs for munis with origin
pairs=occs.groupby(['muni_id_land', 'muni_id_origin'])['muni_id_land'].count()
pairs=pairs.reset_index(name='movers')
pairs['muni_id_origin']=pairs['muni_id_origin'].astype(int)
pairs.value_counts('movers')

# get only lands without origin, production set
occs['has_origin']=np.where(occs['muni_id_origin'].isna(), 0, 1)
occs['has_origin']=occs.groupby('muni_id_land')['has_origin'].transform('max')
occ_prod=occs.loc[occs['has_origin']==0, 'muni_id_land'].drop_duplicates().reset_index().drop('index', axis=1) # occupied munis without origin (red)
dests=occs.loc[occs['has_origin']==1, 'muni_id_land'].drop_duplicates().reset_index(name='muni_id').drop('index', axis=1) # occupied munis with origin (blue)

# 2nd step. create all OD pairs with occ, merge with 1st step
df_panel = pd.merge(origins.assign(key=1), dests.assign(key=1), on='key', suffixes=('_o', '_d')).drop('key', axis=1)
prod_df = pd.merge(origins.assign(key=1), occ_prod.assign(key=1), on='key', suffixes=('_o', '_d')).drop('key', axis=1)

# Merge with OD pairs we know
df = df_panel.merge(pairs, left_on=['muni_id_d', 'muni_id_o'],right_on=['muni_id_land', 'muni_id_origin'], how='left')
df['movers']=df['movers'].fillna(0) 
df=df.drop(columns=['muni_id_land', 'muni_id_origin']) 

est_df=df[['movers', 'muni_id_o', 'muni_id_d']]


# adding features 
covs=pd.read_csv(f'{PATH_OUT}\\tv-cross-section.csv')
cov_list=['muni_id','inc_pc','population91', 'PT_votes_98','share_landless', 'illiter', 'urbanization','agro_share', 'pasture_share', 'elevation']
covs=covs[cov_list]

est_df=est_df.merge(covs[cov_list], left_on='muni_id_d', right_on='muni_id', how='left')    
est_df=est_df.merge(covs[cov_list], left_on='muni_id_o', right_on='muni_id', how='left')
est_df=est_df.drop(columns=['muni_id_x', 'muni_id_y'])

# merge with distance and travel time
dist_df=pd.read_csv(f'{PATH}\\mun_dist\\mun_dist.csv', sep=',')
dist_df.rename(columns={'cod_municipio1':'muni_id_o', 'cod_municipio2':'muni_id_d', 'distanciaGeo':'dist'}, inplace=True)
est_df=est_df.merge(dist_df[['muni_id_o', 'muni_id_d', 'dist']], on=['muni_id_o', 'muni_id_d'], how='left')


travel_df=pd.read_csv(f'{PATH}\\dist_brasil\\dist_brasil.csv', sep=';')
travel_df['travel_time']=travel_df['dur'].str.replace(',', '.').astype(float)
travel_df.rename(columns={'orig':'muni_id_o', 'dest':'muni_id_d'}, inplace=True)

est_df=est_df.merge(travel_df[['muni_id_o', 'muni_id_d', 'travel_time']], on=['muni_id_o', 'muni_id_d'], how='left')
est_df.info()

est_df['has_road']=np.where(est_df['travel_time'].isna(), 0, 1)
est_df['travel_time']=est_df['travel_time'].fillna(est_df['travel_time'].mean())
est_df=est_df.dropna()

est_df['movers']=1*(est_df['movers']>0)

est_df.to_csv(f'{PATH}\\od-est.csv', index=False)
#prod_df.to_csv(f'{PATH}\\od-prod.csv', index=False)
 


