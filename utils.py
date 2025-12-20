import pandas as pd
import yaml

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

def engineer_features(df, sensors_list, window_size):

    df_with_features = df.copy()
    grouped = df_with_features.groupby('unit_nr')[sensors_list]

    roll_mean = grouped.rolling(window=window_size).mean().reset_index(0, drop=True)
    roll_std = grouped.rolling(window=window_size).std().reset_index(0, drop=True)

    roll_mean.columns = [f"{c}_mean" for c in sensors_list]
    roll_std.columns = [f"{c}_std" for c in sensors_list]

    df_final = pd.concat([df_with_features, roll_mean, roll_std], axis=1)

    features_final = sensors_list + list(roll_mean.columns) + list(roll_std.columns)

    return df_final, features_final
