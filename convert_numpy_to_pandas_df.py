import numpy as np
import pandas as pd
import os

param_sens_study_dir = "param_sens_study\\L03\\"
param_sens_study_file_numpy = "param_sens_study_file.npy"
param_sens_study_file_pandas = "param_sens_study_file.csv"

pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 20)
pd.set_option('max_colwidth', 2000)

numpy_data = np.load(param_sens_study_dir + param_sens_study_file_numpy)
print(numpy_data)

df = pd.DataFrame(numpy_data)
print(df)


df.columns = ['board_state_100chars_code', 'conf_level', 'margin_est', 'margin_highest', 'rel_error', 'time_mc']
print(df)
print(df.dtypes)

# exit()

df[['conf_level', 'margin_est', 'margin_highest', 'rel_error', 'time_mc']] =\
    df[['conf_level', 'margin_est', 'margin_highest', 'rel_error', 'time_mc']].apply(pd.to_numeric, errors='coerce')

df.to_csv(param_sens_study_dir + param_sens_study_file_pandas, index=False)

print(df)
print(df.dtypes)