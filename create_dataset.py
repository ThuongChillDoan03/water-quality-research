import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
############


def fill_missing_data(WQI_list):
    columns_to_fill = ['BOD(㎎/L)', 'COD(㎎/L)', 'Conductivity(µS/㎝)', 'DO (㎎/L)',
                       'Dissolved Total P(㎎/L)', 'SS(㎎/L)', 'T-N(㎎/L)', 'T-P(㎎/L)', 'Temp.(℃)']
    for i, df in enumerate(WQI_list):
        df_filled = df.copy()
        
        for col in columns_to_fill:
            if col in df_filled.columns:
                df_filled[col].fillna(df_filled[col].mean(), inplace=True)
        
        WQI_list[i] = df_filled

    return WQI_list


if __name__ == '__main__':

    WQI = pd.read_csv('D://2024.2//ĐANCCN//ĐATN code//water-quality-research//data convert//data_train_test.csv')
    WQI_list = [group for _, group in WQI.groupby("Name (E)")]
    WQI_list = fill_missing_data(WQI_list)
    WQI_interpolation = pd.concat(WQI_list, ignore_index=True)
    WQI_sub = WQI_interpolation[['YY/MM','BOD(㎎/L)','COD(㎎/L)', 'Conductivity(µS/㎝)', 'DO (㎎/L)',
                       'Dissolved Total P(㎎/L)', 'SS(㎎/L)', 'T-N(㎎/L)', 'T-P(㎎/L)', 'Temp.(℃)', 'TSI(Chl-a)']]
    WQI_sub['TSI(Chl-a)'].replace([np.inf, -np.inf], np.nan, inplace=True)
    WQI_sub = WQI_sub.dropna(subset=['TSI(Chl-a)'])
    WQI_sub = WQI_sub.sort_values(by='YY/MM', ascending=True)
    WQI_sub.reset_index(drop=True, inplace=True)
    WQI_sub = WQI_sub.drop(columns = ['YY/MM'])

    WQI_sub.to_csv('D://2024.2//ĐANCCN//ĐATN code//water-quality-research//data convert//WQI_dataset.csv', index=False)
