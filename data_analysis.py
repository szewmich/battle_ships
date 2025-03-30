import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats

pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 400)
pd.set_option('display.max_columns', 20)
pd.set_option('max_colwidth', 2000)

param_sens_study_dir = "param_sens_study\\"
param_sens_study_file = "param_sens_study_file_2.npy"
param_sens_study_path = os.path.join(param_sens_study_dir, param_sens_study_file)

times_adv_dir = "times_adv\\"
times_adv_file = "times_adv_file.npy"
times_adv_path = os.path.join(times_adv_dir, times_adv_file)

data_numpy = np.load(param_sens_study_path)
adv_times_numpy = np.load(times_adv_path)

uniq = np.unique_counts(data_numpy[:, 0])
print(uniq.values)
print(uniq.counts)

# dup = np.argwhere(data_numpy == "0737700000073707000007370077770777007227700000777700007000000000070000000000700077777700007444470000")
# print(dup)
#
# modified = np.delete(data_numpy, slice(58800, 88200), 0)
# uniq = np.unique_counts(modified[:, 0])
# print(uniq.values)
# print(uniq.counts)

# np.save(param_sens_study_path, modified)

df = pd.DataFrame(data_numpy)
df.dtypes
# print(df.head())
# print(df.tail())
df.columns = ['game_code', 'conf_level', 'margin_est', 'margin_highest', 'rel_error', 'time_mc']

#game_df = df.set_index('game_code', inplace = False)
game_df = df

adv_times_df = pd.DataFrame(adv_times_numpy)
adv_times_df.columns = ['game_code', 'time_adv']
adv_times_df['time_adv'] = adv_times_df['time_adv'].apply(pd.to_numeric, errors='coerce')
print(adv_times_df)
# game_df = df.set_index('game_code', inplace = False)
#game_df = game_df.loc["0000777000000073700000007370000000737007000077707077000007077277777770727555557072777777707770070000"]
#game_df = game_df.loc["0072700000007277000000777000007770700000727000000172700707777777000074000007007477777000747333700074"]
#game_df = game_df.loc["0737700000073707000007370077770777007227700000777700007000000000070000000000700077777700007444470000"]
#game_df = game_df.loc["7227070000777777770007555557000777777777001000007200000000727000777077007073700000007370000007737000"]
#game_df = game_df.loc["7227077370777777737000000073700000007777707001007200000000727000777077007073700000007370000007737000"]
#game_df = game_df.loc["7270000000727007000077707000000007777777007755555707077777777077700000007370000700737707000073770000"]
#game_df = game_df.loc["7700707077270000007327070000737700000073000007007770007070000070000000000001070000070000000000070000"]

# uniq = np.unique_counts(game_df[:, 0])
# print(uniq.values)
# print(uniq.counts)

#print(game_df ['time_mc'])

# game_df ['time_mc'].replace(0.00, 0.50)
#game_df.loc[game_df.time_mc == '0.00', 'time_mc'] = '0.50'

#print(game_df['time_mc'].sample(n=50))

game_df[['rel_error', 'time_mc']] = game_df[['rel_error', 'time_mc']].apply(pd.to_numeric, errors='coerce')
# game_df[['rel_error', 'time_mc']] = pd.to_numeric(game_df[['rel_error', 'time_mc']], errors='coerce')
# game_df['time_mc'] = pd.to_numeric(game_df['time_mc'], errors='coerce')
# game_df['rel_error'] = pd.to_numeric(game_df['rel_error'], errors='coerce')


# game_df = game_df.sample(100, replace = True)
#avg_res = game_df.groupby(['conf_level', 'margin_est', 'margin_highest'], as_index=False)[['rel_error', 'time_mc']].mean()
# avg_res = game_df.groupby(['conf_level', 'margin_est', 'margin_highest'], as_index=False).agg(
#     rel_error = ('rel_error', 'mean'),
#     error_rms = ('rel_error', lambda x: stats.pmean(x, p=2)),
#     time_mc = ('time_mc', 'mean'),
#     max_error = ('rel_error', 'max'),
#
#     )


avg_res = game_df.groupby(['game_code', 'conf_level', 'margin_est', 'margin_highest'], as_index=False).agg(
    rel_error = ('rel_error', 'mean'),
    time_mc = ('time_mc', 'mean'),
    max_error = ('rel_error', 'max'),
    )

print(avg_res)

avg_res = avg_res.loc[avg_res['max_error'] < 0.15]
print(avg_res.describe())

avg_res['error_normalized'] = avg_res['rel_error'] / 0.005 * 100
print('avg_res - before changes')
print(avg_res)

# For each game find minimum time_mc for which ['error_normalized'] < 100
avg_res_per_game = avg_res.loc[(avg_res['error_normalized'] < 100)]
print('avg_res_per_game - all times where error_normalized < 100')
print(avg_res_per_game)

avg_res_per_game = avg_res_per_game.groupby(['game_code'], as_index=False).agg(
    time_at_100_error = ('time_mc', 'min'),
    )
print('avg_res_per_game - min times where error_normalized < 100')
print(avg_res_per_game)

avg_res['time_at_100_error'] = avg_res['game_code'].map(avg_res_per_game.set_index('game_code')['time_at_100_error'])
avg_res['time_adv'] = avg_res['game_code'].map(adv_times_df.set_index('game_code')['time_adv'])
print('avg_res - min times withing game where error_normalized < 100')
print(avg_res)

avg_res['time_normalized'] = avg_res['time_mc'] / avg_res['time_at_100_error'] * 100



t_norm = avg_res['time_normalized']
t_value = avg_res['time_mc']
e_norm = avg_res['error_normalized']
time_vs_error_factor = 2

# avg_res['score'] = 1000 / (((t_value/10)**2 * t_norm**2 + e_norm**2)**0.5)
avg_res['score'] = (100 - e_norm) * time_vs_error_factor - (t_norm - 100)

avg_res['score_1'] = (100 - e_norm) * 1 - (t_norm - 100)
avg_res['score_2'] = (100 - e_norm) * 2 - (t_norm - 100)
avg_res['score_4'] = (100 - e_norm) * 4 - (t_norm - 100)

print('avg_res - normalized results per game 0074707270007470727000747077700074707077007770007207777700727072270077007777000000000000000000707000')
print(avg_res.loc[avg_res['game_code'] == '0074707270007470727000747077700074707077007770007207777700727072270077007777000000000000000000707000'].sort_values(by = 'score'))

res_per_comb = avg_res.loc[(avg_res['conf_level'] == '2.170') & (avg_res['margin_est'] == '1.000') & (avg_res['margin_highest'] == '0.100')]


avg_res = avg_res.groupby(['conf_level', 'margin_est', 'margin_highest'], as_index=False).agg(
    avg_error_normalized = ('error_normalized', 'mean'),
    avg_time_normalized = ('time_normalized', 'mean'),
    avg_score = ('score', 'mean'),
    avg_score_1 = ('score_1', 'mean'),
    avg_score_2 = ('score_2', 'mean'),
    avg_score_4 = ('score_4', 'mean'),
    )
print('avg_res - averaged normalized results between games')
# print(avg_res)
print(avg_res.sort_values(by = 'avg_score'))
print(avg_res.describe())

groups_conf_level = avg_res.groupby('conf_level')
#groups_margin_est = avg_res.groupby('margin_est')
groups_margin_highest = avg_res.groupby('margin_highest')

# groups_game = avg_res.groupby('game_code')

#avg_res2 = game_df.groupby('conf_level', as_index=False)[['rel_error', 'time_mc']]

# print(avg_res)
#reduced = avg_res.loc[(avg_res['time_mc'] < 2) & (avg_res['rel_error'] < 0.010)]
# reduced = avg_res.loc[avg_res['avg_score'] >= -400000]
sc_limit = -1000
reduced = avg_res.loc[(avg_res['avg_score_1'] >= sc_limit) & (avg_res['avg_score_2'] >= sc_limit) & (avg_res['avg_score_4'] >= sc_limit) & (avg_res['avg_time_normalized'] < 5000)]
# reduced = avg_res.loc[(avg_res['margin_est'] != '0.020') | (avg_res['margin_highest'] != '0.020')][['rel_error', 'time_mc']].mean()
#reduced = avg_res.loc[(avg_res['conf_level'] == '2.170') & (avg_res['margin_highest'] == '0.060')]
#reduced = avg_res.loc[(avg_res['conf_level'] == '1.960')]
#reduced = avg_res.loc[(avg_res['margin_highest'] == '0.020')]
# reduced = avg_res.loc[(avg_res['rel_error'] < 0.005)]

groups_margin_est = reduced.groupby('margin_est')
groups_score = reduced.groupby('avg_score')
#print(avg_res.sort_values(by = 'score'))
print(reduced.sort_values(by = 'avg_score'))
print(reduced.describe())

print(reduced.sort_values(by = ['conf_level', 'margin_est', 'margin_highest']))
# print(res_per_comb)
#print(reduced.describe())
# print(game_df.describe())
# print(groups_game)
# print(reduced)
# print (avg_res.loc[(avg_res['conf_level'] == '1.150') & (avg_res['margin_est'] == '0.100') & (avg_res['margin_highest'] == '0.060')])
# print (avg_res.loc[(avg_res['conf_level'] == '1.280') & (avg_res['margin_est'] == '0.120') & (avg_res['margin_highest'] == '0.120')])
# print (avg_res.loc[(avg_res['conf_level'] == '1.440') & (avg_res['margin_est'] == '0.100') & (avg_res['margin_highest'] == '0.060')])

#print (avg_res.loc[(avg_res['conf_level'] == '1.440') & (avg_res['margin_est'] == '0.100') & (avg_res['margin_highest'] == '0.060')])

# avg_res.to_csv('avg_res_file.csv')

#Scatter plot
# plt.figure(figsize=(8, 6))
# plt.scatter(avg_res.time_normalized, avg_res.error_normalized, alpha=0.7)
#
# # Labels and title
# plt.xlabel('Average time_mc')
# plt.ylabel('Average rel_error')
# plt.title('Scatter Plot of Averaged Results')
# plt.grid(True)
# # plt.legend()
# plt.show()


#avg_res.plot(x = 'time_mc', y = 'rel_error', title = 'Scatter Plot of Averaged Results')
# for name, group in groups_conf_level:
#     plt.plot(group.time_mc, group.rel_error, marker='o', linestyle='', markersize=6, label=name)
# plt.legend()
# plt.show()
#
# for name, group in groups_margin_est:
#     plt.plot(group.time_mc, group.rel_error, marker='o', linestyle='', markersize=6, label=name)
# plt.xlim([0,8])
# plt.ylim([0,0.015])
# plt.legend()
# plt.show()
#
# for name, group in groups_margin_highest:
#     plt.plot(group.time_mc, group.rel_error, marker='o', linestyle='', markersize=6, label=name)
# plt.legend()
# plt.show()

# for name, group in groups_game:
#     plt.plot(group.time_mc, group.rel_error, marker='o', linestyle='', markersize=6, label=name)
# plt.legend()
# plt.show()

# for name, group in groups_score:
#     plt.plot(group.time_normalized, group.error_normalized, marker='o', linestyle='', markersize=6, label=name)
# plt.legend()
# plt.show()


# x = reduced['avg_time_normalized']
# y = reduced['avg_error_normalized']
# plt.xlim([0,300])
# plt.ylim([0,300])
# # x = reduced['rel_error']
# # y = reduced['error_rms']
# c  =  reduced['avg_score']
#
# # plt.xlim([0,0.002])
# # plt.ylim([0,0.01])
# plt.scatter(x, y, c=c, alpha=0.5, cmap='Blues')
# plt.show()

# print('reduced:')
# print(reduced)

hist_data = game_df.query('game_code == "0074707270007470727000747077700074707077007770007207777700727072270077007777000000000000000000707000"'
                          ' and conf_level == "2.170"'
                          ' and margin_est == "1.000"'
                          ' and margin_highest == "0.100"')
                          # and conf_level == "2.170" and
                          # margin_est == "1.000" and
                          # margin_highest == "0.06"', inplace = False)

print(hist_data)

plt.hist(hist_data.time_mc, bins = 20)
plt.title('results for one chosen game code')
plt.xlabel('time_mc')
plt.ylabel("Number of results out of 100")
# plt.xticks(np.arange(0,9)*0.1)
plt.show()