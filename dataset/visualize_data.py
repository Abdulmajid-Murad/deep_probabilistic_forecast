import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
import seaborn as sns
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
sns.set_theme()
sns.set(font_scale=1.2)
sns.set_style("whitegrid", {'grid.linestyle': '--'})


def scatter_color(class_lst):
    class_color_dict = {0: 'green', 1: 'lime', 2: 'yellow', 3: 'orange', 4: 'crimson'}
    return [class_color_dict[item] for item in class_lst]

def visualize_aq_index():
    df_aq = pd.read_csv('./dataset/air_quality_measurements.csv',index_col='time',  parse_dates=True)
    cmap=ListedColormap(['green', 'lime', 'yellow', 'orange', 'crimson'])
    bin_edges = {'pm25':[0, 15, 30, 55, 110, 1000], 'pm10': [0, 25, 50, 90, 180, 1000] }
    all_labels = {'pm25':['Very Low(0--15)', 'Low(15-30)', 'Medium(30--55)', 'High(55--110)', 'Very High(>110)'],
            'pm10': ['Very Low(0 --25)', 'Low(25--50)', 'Medium(50--90)', 'High(90--180)', 'Very Hig(>180)'] }
    columns = df_aq.columns.values.copy()

    columns = [col for col in columns if ('_class' not in col) and ('_threshold' not in col)]

    fig3, axs3 = plt.subplots(int(len(columns)/2), 2,  figsize=(20, 2.0*len(columns)))

    fig2, axs2 = plt.subplots(2, 1, figsize=(20, 7))
    axs2_counter=0

    fig, axs = plt.subplots(len(columns), 1,  figsize=(20, 3.5*len(columns)))
    xlim = [pd.to_datetime('2019-01-01'), pd.to_datetime('2020-12-31')]
    for i, col in enumerate(columns):
  
        split = col.split('_')
        labels = all_labels[split[1]]
        axs[i].plot(df_aq.index, df_aq[col], color='k')
        axs[i].set_title('Monitoring Station: '+col.split('_')[0] + ('($PM_{10}$)' if split[1]=='pm10' else '($PM_{2.5}$)'))
        axs[i].set_ylabel('Air pollutant ' + ('$PM_{10}$' if split[1]=='pm10' else '$PM_{2.5}$') + ' (${\mu}g/m^3 $)')
        axs[i].set_xlabel('Time')
        bar_colors = df_aq[str(col+'_class')].values
        bar_colors = bar_colors[None, :-1]
        psm = axs[i].pcolormesh(df_aq.index, axs[i].get_ylim() ,bar_colors, alpha=1.0 ,cmap=cmap, vmin=-0.5, vmax=4.5)
        
        
        divider = make_axes_locatable(axs[i])
        cax = divider.append_axes("right", size="1%", pad=0.05)
        cbar=plt.colorbar(psm, cax=cax, ticks=[0, 1, 2, 3, 4])
        cbar.ax.set_yticklabels(labels)
        cbar.ax.set_title('CAQI')
        cbar.solids.set(alpha=1)
        axs[i].set_xlim(xlim)

        if col.split('_')[0] == 'Elgeseter':
            axs2[axs2_counter].plot(df_aq.index, df_aq[col], color='k')
            axs2[axs2_counter].set_title('Monitoring Station: '+col.split('_')[0] + ('($PM_{10}$)' if split[1]=='pm10' else '($PM_{2.5}$)'))
            axs2[axs2_counter].set_ylabel('Air pollutant ' + ('$PM_{10}$' if split[1]=='pm10' else '$PM_{2.5}$') + ' (${\mu}g/m^3 $)')
            axs2[axs2_counter].set_xlabel('Time')
            bar_colors = df_aq[str(col+'_class')].values
            bar_colors = bar_colors[None, :-1]
            psm = axs2[axs2_counter].pcolormesh(df_aq.index, axs2[axs2_counter].get_ylim() ,bar_colors, alpha=1.0 ,cmap=cmap, vmin=-0.5, vmax=4.5)
            divider = make_axes_locatable(axs2[axs2_counter])
            cax = divider.append_axes("right", size="1%", pad=0.05)
            cbar=plt.colorbar(psm, cax=cax, ticks=[0, 1, 2, 3, 4])
            cbar.ax.set_yticklabels(labels)
            cbar.ax.set_title('CAQI')
            cbar.solids.set(alpha=1)
            axs2[axs2_counter].set_xlim(xlim)
            axs2_counter += 1
            

        ax = axs3[int(i/2)][i%2]
        psm2 = axs[i].pcolormesh(df_aq.index, axs[i].get_ylim() ,bar_colors, alpha=1.0 ,cmap=cmap, vmin=-0.5, vmax=4.5)
        if split[1] == 'pm10':
            n, bins, patches = ax.hist(df_aq[col], bins=200, color='b')
            ax.set_title(col.split('_')[0] + '(PM10)')
            ax.set_xlabel('PM10 pollutant level (${\mu}g/m^3 $)')
            ax.set_ylabel('Frequency of pollutant level')
            ax.set_xlim(-3, 80)
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            bin_color = cmap(np.array(pd.cut(x=bin_centers, bins=bin_edges[split[1]], right=False, labels=[0, 1, 2, 3, 4] )))
            for c, p in zip(bin_color, patches):
                plt.setp(p, 'facecolor', c)
        else:
            n, bins, patches = ax.hist(df_aq[col], bins=135, color='b')
            ax.set_title(col.split('_')[0] + '(PM2.5)')
            ax.set_xlabel('PM2.5 pollutat level (${\mu}g/m^3 $)')
            ax.set_ylabel('Frequency of pollutant level')
            ax.set_xlim(-3, 40)
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            bin_color = cmap(np.array(pd.cut(x=bin_centers, bins=bin_edges[split[1]], right=False, labels=[0, 1, 2, 3, 4] )))
            for c, p in zip(bin_color, patches):
                plt.setp(p, 'facecolor', c) 
                
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.05)
        cbar=plt.colorbar(psm2, cax=cax, ticks=[0, 1, 2, 3, 4])
        cbar.ax.set_yticklabels(labels)
        cbar.ax.set_title('CAQI')
        cbar.solids.set(alpha=1)

    fig.tight_layout()
    fig.savefig('./dataset/aq_index_all_stations.jpg', bbox_inches='tight')
    fig2.tight_layout()
    fig2.savefig('./dataset/aq_index.jpg', bbox_inches='tight')
    fig3.tight_layout()
    fig3.savefig('./dataset/right_skewed.jpg', bbox_inches='tight')


def visualize_aq():

    df_aq = pd.read_csv('./dataset/air_quality_measurements.csv',index_col='time',  parse_dates=True)
    columns =list(df_aq.columns)
    columns = [col for col in columns if ('_class' not in col) and ('_threshold' not in col)]
    colors = ['b', 'g', 'r', 'c', 'm' , 'y', 'orange']
    colors = np.resize(colors, len(columns))
    fig, axs = plt.subplots(len(columns), 1,  figsize=(20, 2.5*len(columns)))
    xlim = [pd.to_datetime('2019-01-01'), pd.to_datetime('2020-12-31')]
    for i, col in enumerate(columns):
        axs[i].plot(df_aq.index, df_aq[col], color=colors[int(i/2)])
        axs[i].set_title('Industrial Sensor: ' + col.split('_',  1)[0])
        axs[i].set_xlim(xlim)
        if i%2 ==0:
            axs[i].set_ylabel('PM2.5 ($ {\mu}g/m^3 $)')
        else:
            axs[i].set_ylabel('PM10 ($ {\mu}g/m^3 $)')
    fig.tight_layout()
    fig.savefig('./dataset/aq_data.jpg', bbox_inches='tight')

def visualize_weather():

    df_weather= pd.read_csv('./dataset/weather.csv', index_col='time', parse_dates=True)
    columns =list(df_weather.columns)
    feature_names = ['Air temperature',
                 'Relative humidity',
                 'Precipitation',
                 'Air pressure',
                 'Wind speed',
                 'Wind direction',
                 'Air temperature',
                 'Relative humidity',
                 'Precipitation',
                 'Snow thickness',
                 'Duration of sunshine',
                 'Relative humidity',
                 'Precipitation',
                 'Air temperature']

    colors = ['b', 'g', 'r', 'c', 'm' , 'y', 'orange']
    colors = np.resize(colors, len(columns))
    xlim = [pd.to_datetime('2019-01-01'), pd.to_datetime('2020-12-31')]
    fig, axs = plt.subplots(len(columns), 1,  figsize=(20, 2*len(columns)))
    for i, col in enumerate(columns):
        axs[i].plot(df_weather.index, df_weather[col], color=colors[i])
        axs[i].set_title('Monitoring station: ' + (col.split('_',  1)[0]).title())
        axs[i].set_ylabel(feature_names[i])
        axs[i].set_xlim(xlim)
    fig.tight_layout()
    fig.savefig('./dataset/weather_data.jpg', bbox_inches='tight')

def visualize_traffic():

    df_traffic = pd.read_csv('./dataset/traffic.csv', index_col='Time', parse_dates=True)
    columns =list(df_traffic.columns)
    colors = ['b', 'g', 'r', 'c', 'm' , 'y', 'orange']
    colors = np.resize(colors, len(columns))
    fig, axs = plt.subplots(len(columns), 1,  figsize=(20, 2.5*len(columns)))
    xlim = [pd.to_datetime('2019-01-01'), pd.to_datetime('2020-12-31')]
    for i, col in enumerate(columns):
        axs[i].plot(df_traffic.index, df_traffic[col], color=colors[i])
        axs[i].set_title('Street: ' +col)
        axs[i].set_ylabel('Traffic volume')
        axs[i].set_xlim(xlim)
    fig.tight_layout()
    fig.savefig('./dataset/traffic_data.jpg', bbox_inches='tight')

def visualize_street_cleaning():

    df_street_cleaning = pd.read_csv('./dataset/street_cleaning.csv', index_col='time', parse_dates=True)
    columns =list(df_street_cleaning.columns)
    colors = ['b', 'g', 'r', 'c', 'm' , 'y', 'orange']
    colors = np.resize(colors, len(columns))
    fig, axs = plt.subplots(len(columns), 1,  figsize=(20, 2*len(columns)))
    xlim = [pd.to_datetime('2019-01-01'), pd.to_datetime('2020-12-31')]
    for i, col in enumerate(columns):
        axs[i].plot(df_street_cleaning.index, df_street_cleaning[col], color=colors[i])
        axs[i].set_title('Street: ' + col)
        axs[i].set_xlim(xlim)
        axs[i].set_yticks([1])
        labels = [item.get_text() for item in axs[i].get_yticklabels()]
        labels[0] = 'Cleaning'
        axs[i].set_yticklabels(labels)
    fig.tight_layout()
    fig.savefig('./dataset/street_cleaning_data.jpg', bbox_inches='tight')



if __name__ == "__main__":
    visualize_aq()
    visualize_weather()
    visualize_traffic()
    visualize_street_cleaning()
    visualize_aq_index()