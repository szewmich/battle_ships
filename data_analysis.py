import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 1700)
pd.set_option('display.max_columns', 20)
pd.set_option('max_colwidth', 2000)

param_sens_study_dir = "param_sens_study\\L04"
param_sens_study_file = "param_sens_study_file_n100.csv"
param_sens_study_path = os.path.join(param_sens_study_dir, param_sens_study_file)

times_adv_dir = "times_adv\\"
times_adv_file = "times_adv_file.npy"
times_adv_path = os.path.join(times_adv_dir, times_adv_file)

adv_times_numpy = np.load(times_adv_path)

df = pd.read_csv(param_sens_study_path)
# print(df.dtypes)


# lookup = df.query("board_state_100chars_code == '0000000000000000000077700000007470000000747700000074700700007470701100777700070072277777777777555557'")
# print(lookup)
# lookup = df.query("board_state_100chars_code == '0000007770000000757700000075730000007573700000757307000075777777777770727333700772777777707777444470'")
# print(lookup)
# lookup = df.query("board_state_100chars_code == '7700707077270000007327070000737700000073000007007770007070000070000000000001070000070000000000070000'")
# print(lookup)
#
# r = r + 2



# df = df[df.board_state_100chars_code != '0000000000000000000077700000007470000000747700000074700700007470701100777700070072277777777777555557']
# df = df[df.board_state_100chars_code != '0000007770000000757700000075730000007573700000757307000075777777777770727333700772777777707777444470']
# df = df[df.board_state_100chars_code != '7700707077270000007327070000737700000073000007007770007070000070000000000001070000070000000000070000']
# df = df[df.board_state_100chars_code != '0000074700770007477027000747002700074777770007772777777777277555557777777777700000733377770077777333']
#df = df[df.board_state_100chars_code != '0777777000074444700077777770003337000700777777000770722700000077770000000710000000000000000000070000']

# print(df['board_state_100chars_code'].unique())
print(df.groupby('board_state_100chars_code').count())
# print(df.groupby('board_state_100chars_code').count())
# exit()
##########################################################################
# COUNT AND VISUALISE INCONSISTENT RESULTS
##########################################################################
tot_per_conf = df.groupby('conf_level', as_index=False).count()
# print(tot_per_conf)

df['is_inconsistent'] = (df['rel_error'] > df['margin_highest'])
df['exceedance'] = np.where(df['rel_error'] > df['margin_highest'],
                            (df['rel_error'] / df['margin_highest'] - 1) * 100,
                            np.nan)

inc_plot = df.groupby('conf_level', as_index=False).agg(
    inc_count = ('is_inconsistent', 'sum'),
    avg_perc_exceedance = ('exceedance', 'mean')
)

inc_plot['total'] = inc_plot['conf_level'].map(tot_per_conf.set_index('conf_level')['rel_error'])
inc_plot['perc_inc'] = inc_plot['inc_count'] / inc_plot['total'] * 100

inc_plot['avg_perc_exceedance'] = inc_plot['avg_perc_exceedance'].round(0)
inc_plot['perc_inc'] = inc_plot['perc_inc'].round(2)
# print(inc_plot)

###### PLOT INCONSISTENT RESULTS PER CONF_LEVEL #######
plt.bar(inc_plot['conf_level'], inc_plot['perc_inc'], width=0.1)
plt.title('Percentage of inconsistent results per confidence level')
plt.xlabel('conf_level')
plt.ylabel("% inconsistent results")
plt.xticks(inc_plot['conf_level'])
# plt.show()
# print(inc_plot)

##########################################################################

game_df = df

#####################################
# LOAD ADVANCED TIMES
# TODO: convert numpy into dataframe in csv
adv_times_df = pd.DataFrame(adv_times_numpy)
adv_times_df.columns = ['board_state_100chars_code', 'time_adv']
adv_times_df['time_adv'] = adv_times_df['time_adv'].apply(pd.to_numeric, errors='coerce')
# print(adv_times_df)
# print(adv_times_df.dtypes)
#####################################

# Aggregate series of repetitive runs for the same board_state and parameters
avg_res = game_df.groupby(['board_state_100chars_code', 'conf_level', 'margin_est', 'margin_highest'], as_index=False).agg(
    rel_error = ('rel_error', lambda x: np.sqrt(np.mean(np.square(x)))),
    #rel_error = ('rel_error', 'mean'),
    time_mc = ('time_mc', 'mean'),
    max_error = ('rel_error', 'max'),
)
# print(avg_res)
# print(avg_res.describe())

# # Normalize average error to 0.5%
# avg_res['error_normalized'] = avg_res['rel_error'] / 0.005 * 100

# print(avg_res.query('board_state_100chars_code == "0777770000007470000070747000000774700000007477007777777007750100007075000007007500007007750007007075"'))
# exit()

#################################################################################
# SCALE TIME AND ERROR
#################################################################################
avg_res_per_code = avg_res.groupby(['board_state_100chars_code'], as_index=False).agg(
    max_error_per_code = ('rel_error', 'max'),
    max_time_per_code = ('time_mc', 'max'),
    )
# Map 'max_error_per_code' to original avg_res dataframe
avg_res['max_error_per_code'] = avg_res['board_state_100chars_code'].map(avg_res_per_code.set_index('board_state_100chars_code')['max_error_per_code'])
avg_res['max_time_per_code'] = avg_res['board_state_100chars_code'].map(avg_res_per_code.set_index('board_state_100chars_code')['max_time_per_code'])

# Drop board state codes where max_error_per_code is 0
avg_res = avg_res.query('max_error_per_code > 0')
print(avg_res.describe())

# Normalize error and time as percentage of their max values per game code
avg_res['error_normalized'] = avg_res['rel_error'] / avg_res['max_error_per_code'] * 100
avg_res['time_normalized'] = avg_res['time_mc'] / avg_res['max_time_per_code'] * 100

print(avg_res.sample(50))
# exit()

# MinMax scale error
# mm_scaler = MinMaxScaler()
# avg_res["rel_error"] = mm_scaler.fit_transform(avg_res[["rel_error"]])
# print(avg_res.sample(50))
#################################################################################



#################################################################################
# SET NORMALIZED TIME
#################################################################################
# # For each game find minimum time_mc for which ['error_normalized'] < 100
# # This will be reference time
# avg_res_per_game = avg_res.loc[(avg_res['error_normalized'] < 100)]

# avg_res_per_game = avg_res_per_game.groupby(['board_state_100chars_code'], as_index=False).agg(
#     time_at_100_error = ('time_mc', 'min'),
#     )

# # Map 'time_at_100_error' and 'time_adv' to original avg_res dataframe
# avg_res['time_at_100_error'] = avg_res['board_state_100chars_code'].map(avg_res_per_game.set_index('board_state_100chars_code')['time_at_100_error'])
# avg_res['time_adv'] = avg_res['board_state_100chars_code'].map(adv_times_df.set_index('board_state_100chars_code')['time_adv'])

# # Normalize 'time_mc' as percentage of 'time_at_100_error'
# avg_res['time_normalized'] = avg_res['time_mc'] / avg_res['time_at_100_error'] * 100

# # Normalize advanced time as percentage of 'time_at_100_error'
# avg_res['time_adv'] = avg_res['board_state_100chars_code'].map(adv_times_df.set_index('board_state_100chars_code')['time_adv'])
# avg_res['time_adv_normalized'] = avg_res['time_adv'] / avg_res['time_at_100_error'] * 100

# # MinMax scale time
# mm_scaler = MinMaxScaler()
# avg_res["time_mc"] = mm_scaler.fit_transform(avg_res[["time_mc"]])

# #################################################################################
# # RELATION BETWEEN BOARD STATE and TIME

# avg_res['n_zeros'] = avg_res['board_state_100chars_code'].apply(lambda x: x.count('0'))
# avg_res['n_hits'] = avg_res['board_state_100chars_code'].apply(lambda x: (100 - x.count('0') - x.count('7')))
# avg_res['n_ones'] = avg_res['board_state_100chars_code'].apply(lambda x: x.count('1'))
# print(avg_res.sample(10))

# def plot_time_vs_board_state_stats(avg_res):

#     n_zeros_plot = avg_res.groupby('n_zeros', as_index = False).agg(
#         time_adv_avg = ('time_adv', 'mean'),
#         time_mc_avg = ('time_mc', 'mean')
#     )
#     n_zeros_plot = n_zeros_plot.query('n_zeros < 55')

#     n_hits_plot = avg_res.groupby('n_hits', as_index = False).agg(
#         time_adv_avg = ('time_adv', 'mean'),
#         time_mc_avg = ('time_mc', 'mean')
#     )
#     n_hits_plot = n_hits_plot.query('n_hits > 9')

#     n_ones_plot = avg_res.groupby('n_ones', as_index = False).agg(
#         time_adv_avg = ('time_adv', 'mean'),
#         time_mc_avg = ('time_mc', 'mean')
#     )

#     # Scatter plot
#     plt.figure(figsize=(18, 6))

#     plt.subplot(1, 3, 2)
#     # plt.figure(figsize=(8, 6))
#     plt.scatter(n_hits_plot.n_hits, n_hits_plot.time_adv_avg, label = 'adv')
#     plt.scatter(n_hits_plot.n_hits, n_hits_plot.time_mc_avg, label = 'mc')
#     plt.xlabel('n_hits')
#     plt.ylabel('time')
#     plt.title('Scatter Plot of time vs n_hits')
#     plt.grid(True)
#     plt.legend()

#     plt.subplot(1, 3, 3)
#     # plt.figure(figsize=(8, 6))
#     plt.scatter(n_ones_plot.n_ones, n_ones_plot.time_adv_avg, label = 'adv')
#     plt.scatter(n_ones_plot.n_ones, n_ones_plot.time_mc_avg, label = 'mc')
#     plt.xlabel('n_ones')
#     plt.ylabel('time')
#     plt.title('Scatter Plot of time vs n_ones')
#     plt.grid(True)
#     plt.legend()

#     plt.subplot(1, 3, 1)
#     # plt.figure(figsize=(8, 6))
#     plt.scatter(n_zeros_plot.n_zeros, n_zeros_plot.time_adv_avg, label = 'adv')
#     plt.scatter(n_zeros_plot.n_zeros, n_zeros_plot.time_mc_avg, label = 'mc')

#     # Fit mc exponential curve
#     x = n_zeros_plot.n_zeros
#     y = n_zeros_plot.time_mc_avg
#     a_guess = 0.15
#     b_guess = 0.1
#     c_guess = 4
#     popt, pcov = curve_fit(lambda t, a, b, c: a * np.exp(b * t) + c, x, y, p0=(a_guess, b_guess, c_guess))
#     a = popt[0]
#     b = popt[1]
#     c = popt[2]
#     x_fitted_mc = np.linspace(np.min(x), np.max(x), 100)
#     y_fitted_mc = a * np.exp(b * x_fitted_mc) + c
#     plt.plot(x_fitted_mc, y_fitted_mc, color='red', label='mc_fitted')

#     # Fit adv exponential curve
#     x = n_zeros_plot.n_zeros
#     y = n_zeros_plot.time_adv_avg
#     a_guess = 0.15
#     b_guess = 0.1
#     c_guess = 4
#     popt, pcov = curve_fit(lambda t, a, b, c: a * np.exp(b * t) + c, x, y, p0=(a_guess, b_guess, c_guess))
#     a = popt[0]
#     b = popt[1]
#     c = popt[2]
#     x_fitted_adv = np.linspace(np.min(x), np.max(x), 100)
#     y_fitted_adv = a * np.exp(b * x_fitted_adv) + c
#     plt.plot(x_fitted_adv, y_fitted_adv, color='darkblue', label='adv_fitted')

#     plt.xlabel('n_zeros')
#     plt.ylabel('time')
#     plt.title('Scatter Plot of time vs n_zeros')
#     plt.grid(True)
#     plt.legend()

#     # plt.show()

# plot_time_vs_board_state_stats(avg_res)


# ######################################

# n_zeros_40 = avg_res.query("n_zeros == 40")
# print(n_zeros_40)
# print(n_zeros_40.describe())    
# print(n_zeros_40.query("time_mc > 200"))

# avg_res_cleaned = avg_res.query('board_state_100chars_code != "5555577333777777777777777777777777777777777777777777777777770000000000000000000000000000000000000000"')
# # plot_time_vs_board_state_stats(avg_res_cleaned)

# ######################################

# by_board_state_stats = avg_res.groupby(['n_zeros', 'n_hits'], as_index=True).agg(
#     avg_time_adv = ('time_adv', 'mean'),
#     avg_time_mc = ('time_mc', 'mean'),
#     count = ('board_state_100chars_code', 'count'),
# )
# by_board_state_stats['adv_to_mc_ratio'] = by_board_state_stats['avg_time_adv'] / by_board_state_stats['avg_time_mc']
# adv_suited = by_board_state_stats.query("adv_to_mc_ratio < 2")
# adv_suited = adv_suited.query("n_zeros > 45")

# print("adv_suited: \n")
# print(adv_suited)

# mc_suited = by_board_state_stats.query("adv_to_mc_ratio >= 2")
# mc_suited = mc_suited.query("n_zeros < 56")

# print("mcsuited: \n")
# print(mc_suited)

########################################################################################
# For each board state code create scatter plot of rel_error vs time_mc per parameter set
red_highlight_condition = 'conf_level == 2.17 and margin_est == 1 and margin_highest == 0.10'
green_highlight_condition = 'conf_level == 2.58 and margin_est == 1 and margin_highest == 0.10'
orange_highlight_condition = 'conf_level == 2.58 and margin_est == 1 and margin_highest == 0.12'

plt.figure(figsize=(12, 12))
plt.suptitle('Scatter Plots of rel_error vs time_mc for Each Board State Code\n ' \
'Red-highlighted parameter set: ' + red_highlight_condition + '\n'
'green-highlighted parameter set: ' + green_highlight_condition + '\n'
'orange-highlighted parameter set: ' + orange_highlight_condition + '\n'
, fontsize=12)

i = 1
for name, group in avg_res.groupby('board_state_100chars_code'):
    plt.subplot(7, 7, i)

    temp_df = avg_res.query('board_state_100chars_code == @name')
    plt.scatter(temp_df.time_normalized, temp_df.error_normalized, marker='o', color='black', s=5)

    red_highlight = temp_df.query(red_highlight_condition)
    green_highlight = temp_df.query(green_highlight_condition)
    orange_highlight = temp_df.query(orange_highlight_condition)
    plt.scatter(red_highlight.time_normalized, red_highlight.error_normalized, marker='x', color='red', s=50)
    plt.scatter(green_highlight.time_normalized, green_highlight.error_normalized, marker='x', color='green', s=50)
    plt.scatter(orange_highlight.time_normalized, orange_highlight.error_normalized, marker='x', color='orange', s=50)

    i +=1

plt.show()
# exit()
#################################################################################
# SET SCORE METRICS
#################################################################################
t_norm = avg_res['time_normalized']
t_value = avg_res['time_mc']
e_norm = avg_res['error_normalized']

change_in_error = (e_norm - 100)  # Increases score if negative
change_in_time = (t_norm - 100)   # Increases score if negative
time_vs_error_factor = 1          # How much more we care about precision increase than time decrease

# Main score metrics by which results will be sorted
avg_res['score'] = -1 * (change_in_error * time_vs_error_factor + change_in_time)

# Different variants to compare
avg_res['score_1'] = -1 * (change_in_error * 1 + change_in_time)
avg_res['score_2'] = -1 * (change_in_error * 2 + change_in_time)
avg_res['score_4'] = -1 * (change_in_error * 4 + change_in_time)

#################################################################################
# Group by unique parameters set, average across board_states
avg_res = avg_res.groupby(['conf_level', 'margin_est', 'margin_highest'], as_index=False).agg(
    max_error = ('max_error', 'max'),
    avg_error_normalized = ('error_normalized', 'mean'),
    avg_time_normalized = ('time_normalized', 'mean'),
    avg_score = ('score', 'mean'),
    avg_score_1 = ('score_1', 'mean'),
    avg_score_2 = ('score_2', 'mean'),
    avg_score_4 = ('score_4', 'mean'),
    )
#################################################################################
# Discard parameters which resulted in max_error higher than 25%
avg_res = avg_res.query('max_error < 0.25')

# Discard parameters with given score metrics
sc_limit = -500
reduced = avg_res.query('avg_score_1 >= @sc_limit & avg_score_2 >= @sc_limit & avg_score_4 >= @sc_limit & avg_time_normalized < 5000')

# Discard confidence levels that produce inconsintent results
reduced = reduced.query('conf_level >= 1.96')

#################################################################################
# Scatter plot - all averaged results
plt.figure(figsize=(8, 6))
plt.scatter(avg_res.avg_time_normalized, avg_res.avg_error_normalized)
plt.xlabel('Average Normalized time_mc')
plt.ylabel('Average Normalized rel_error')
plt.title('Scatter Plot of Averaged and Normalized Results')
plt.grid(True)
# plt.legend()
plt.show()
#################################################################################
# Group results by different parameters to see their individual influence
groups_conf_level = avg_res.groupby('conf_level')
groups_margin_est = avg_res.groupby('margin_est')
groups_margin_highest = avg_res.groupby('margin_highest')

print('reduced results - sorted by score')
print(reduced.sort_values(by = 'avg_score'))

print('reduced results - sorted by parameters set')
print(reduced.sort_values(by = ['conf_level', 'margin_est', 'margin_highest']))

# Plots by group
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
for name, group in groups_conf_level:
    plt.plot(group.avg_time_normalized, group.avg_error_normalized, marker='o', linestyle='', markersize=6, label=name)
    plt.title('Influence of conf_level')
    plt.xlabel('avg_time_normalized')
    plt.ylabel("avg_error_normalized")
plt.legend()

plt.subplot(1, 3, 2)
for name, group in groups_margin_est:
    plt.plot(group.avg_time_normalized, group.avg_error_normalized, marker='o', linestyle='', markersize=6, label=name)
    plt.title('Influence of margin_est')
    plt.xlabel('avg_time_normalized')
    plt.ylabel("avg_error_normalized")
plt.legend()

plt.subplot(1, 3, 3)
for name, group in groups_margin_highest:
    plt.plot(group.avg_time_normalized, group.avg_error_normalized, marker='o', linestyle='', markersize=6, label=name)
    plt.title('Influence of margin_highest')
    plt.xlabel('avg_time_normalized')
    plt.ylabel("avg_error_normalized")
plt.legend()

plt.show()
#################################################################################
# Scatter plot - reduced results, focused on the best section

x = reduced['avg_time_normalized']
y = reduced['avg_error_normalized']
plt.xlim([0,100])
plt.ylim([0,100])
c  =  reduced['avg_score']

plt.scatter(x, y, c=c, alpha=0.5, cmap='Blues')
orange_highlight_condition = 'conf_level == 2.58 and margin_est == 1 and margin_highest == 0.12'
orange_highlight = reduced.query(orange_highlight_condition)
plt.scatter(orange_highlight.avg_time_normalized, orange_highlight.avg_error_normalized, marker='o', color='orange', s=80)

plt.show()
#################################################################################


# hist_data = game_df.query('board_state_100chars_code == "0074707270007470727000747077700074707077007770007207777700727072270077007777000000000000000000707000"'
#                           ' and conf_level == "2.170"'
#                           ' and margin_est == "1.000"'
#                           ' and margin_highest == "0.100"')
#                           # and conf_level == "2.170" and
#                           # margin_est == "1.000" and
#                           # margin_highest == "0.06"', inplace = False)
#
# print(hist_data)
#
# plt.hist(hist_data.time_mc, bins = 20)
# plt.title('results for one chosen game code')
# plt.xlabel('time_mc')
# plt.ylabel("Number of results out of 100")
# # plt.xticks(np.arange(0,9)*0.1)
# plt.show()


