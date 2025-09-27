import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

all_games_results_file_path = "all_games_results.csv"

if all_games_results_file_path in os.listdir(os.getcwd()):
    all_games_results = pd.read_csv(all_games_results_file_path)
    print('loaded existing all_games_results dataframe')

print(all_games_results.describe())

x_means = all_games_results['n_retrieved'].rolling(50).mean().dropna()

window_size = 500
n_windows = len(all_games_results) // window_size

retrieved_window_averages = [
    all_games_results['n_retrieved'][i*window_size:(i+1)*window_size].mean()
    for i in range(n_windows)
]
# retrieved_window_averages = pd.Series(retrieved_window_averages)

time_window_averages = [
    all_games_results['time_game'][i*window_size:(i+1)*window_size].mean()
    for i in range(n_windows)
]
time_window_averages = pd.Series(time_window_averages)

# print(retrieved_window_averages)

window_averages = pd.DataFrame({
    'retrieved': retrieved_window_averages,
    'time': time_window_averages
})

print(window_averages)
# plt.scatter(all_games_results.index, all_games_results['n_retrieved'])
# plt.xlabel('ID')
# plt.ylabel('n_retrieved')
# plt.title('n_retrieved vs ID')
# plt.show()

plt.scatter(window_averages.index * window_size, window_averages['time'])
# plt.show()


# Fit adv exponential curve
x = window_averages.index * window_size
y = window_averages['time']
a_guess = 150
b_guess = -0.0001
c_guess = 0
# , p0=(a_guess, b_guess, c_guess)
popt, pcov = curve_fit(lambda t, a, b, c: a * np.exp(b * t) + c, x, y , p0=(a_guess, b_guess, c_guess))
a = popt[0]
b = popt[1]
c = popt[2]
x_fitted_adv = np.linspace(np.min(x), np.max(x)*3, 100)
y_fitted_adv = a * np.exp(b * x_fitted_adv) + c
plt.plot(x_fitted_adv, y_fitted_adv, color='darkblue', label='adv_fitted')

plt.xlim(0, 25000)
plt.ylim(0, 170)
plt.show()