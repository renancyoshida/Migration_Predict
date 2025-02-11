
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
# generate yearly occupation plot
import matplotlib.pyplot as plt
def gen_occ_plot(df, xlabel, ylabel, title):
    
    xaxis= df['ANO'].unique()
    yaxis= df.groupby('ANO')['OCUPAÇÃO'].sum()
    
    # keep years<2010
    xaxis=xaxis[xaxis<=2010]
    yaxis=yaxis[yaxis.index<=2010]
    
    # plot
    plt.figure(figsize=(10,5))
    ax = sns.barplot(x=xaxis, y=yaxis, color="blue")
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.set_xticklabels(xaxis,rotation=45, fontsize=8)
    plt.tight_layout()
    #plt.show()
    
def signal_map(gdf, transm_gdf, station, state):   
    """
    creates map of signal strength and free space for a given station and state
    gdf: geodataframe with signal strength
    transm: geodataframe with transmitter locations
    station: 'record', 'globo' or 'any'
    state: 'SP', 'RJ', 'MG', or none for all states
    """
    if station in ['record', 'globo']:
        transm_gdf=transm_gdf[transm_gdf['station']==station]
    
    if state:
        gdf=gdf[gdf['SIGLA_UF']==state]
        transm_gdf=transm_gdf[transm_gdf['uf']==state]
    
    fig, axes = plt.subplots(1,2, figsize=(10, 10))
    ax=axes[0]
    cmap='viridis'
    norm = mcolors.BoundaryNorm(boundaries=[-180, -150, -120, -90, -60, -30, 0], ncolors=256)

    if station=='record':
        signal_vars=['signal_r', 'free_space_r']        
    elif station=='globo':
        signal_vars=['signal_g', 'free_space_g']
    elif station=='any':
        signal_vars=['signal', 'free_space']
    else:
        print('Station not found')
    
    gdf.plot(column=signal_vars[0], cmap=cmap, linewidth=0.4, ax=ax, norm=norm)
    ax.axis("off")
    ax.set_title("Signal Strength")
    ax=axes[1]
    gdf.plot(column=signal_vars[1], cmap=cmap, linewidth=0.4, ax=ax, norm=norm)
    ax.axis("off")
    ax.set_title("Free Space")

    # Plot centroids as dots on both axes
    for ax in axes:
        transm_gdf.plot(ax=ax, color='black', marker='o', markersize=.1, label='Transmitters')

    bar_info = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    bar_info._A = []  # Required to create a colorbar
    cbaxes = fig.add_axes([0.3, 0.3, 0.4, 0.01]) # left, bottom, width, height
    cbar = fig.colorbar(bar_info, cax=cbaxes, orientation='horizontal')

if __name__ == "__main__":
    # load data
    path_out= r'C:\Users\renan\OneDrive - Leland Stanford Junior University\Research\IL\conflict\data\clean'
    df=pd.read_csv(f"{path_out}\\occupations-clean.csv") # muni year level, only occs>0
    gen_occ_plot(df, 'Year', 'Occupations', 'Occupations by Year')
