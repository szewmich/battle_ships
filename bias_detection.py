import numpy as np
import scipy.stats as stats
from scipy.stats import ks_2samp, wasserstein_distance
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression

import bias_metrics

pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 1700)
pd.set_option('display.max_columns', 20)
pd.set_option('max_colwidth', 2000)

def bootstrap_ci(x, metric_fn=np.mean, n_boot=2000):
    boots = []
    for i in range(n_boot):
        sample = np.random.choice(x, size=len(x), replace=True)
        boots.append(metric_fn(sample))
    lower = np.percentile(boots, 2.5)
    upper = np.percentile(boots, 97.5)
    return lower, upper



def wasserstein_similarity(x, y, scale=None):
    """
    Similarity = exp(-W / scale).
    If scale is None, use pooled MAD (robust).
    """
    w = wasserstein_distance(x, y)

    if scale is None:
        pooled = np.concatenate([x, y])
        mad = np.median(np.abs(pooled - np.median(pooled)))
        scale = mad if mad > 0 else np.std(pooled)
        if scale == 0:
            scale = 1.0

    return float(np.exp(-w / scale))


def get_metrics (boards_list):
    temp_boards_metrics = bias_metrics.make_metrics_dict()
    metrics_per_board = {}

    for board_code in boards_list:
        board_array = np.array([int(x) for x in board_code]).reshape((6,6))
        current_metrics = bias_metrics.calculate_bias_metrics(board_array)

        metrics_per_board[board_code] = current_metrics

    return metrics_per_board

# def prepare_new_data(hyp_board, scaler):
#     if type(hyp_board) == str:
#         board_array = np.array([int(x) for x in hyp_board]).reshape((6,6))
#     else:
#         board_array = hyp_board
#     current_metrics = bias_metrics.calculate_bias_metrics(board_array)
#     new_batch = pd.DataFrame([current_metrics])
#     new_batch_s = scaler.transform(new_batch)
#     return new_batch_s

# def convert_to_df(metrics):
#     return pd.DataFrame([metrics])

# def scale_new_data(new_batch, scaler):
#     return scaler.transform(new_batch)

def metrics_to_scaled_vector(metrics, scaler_params):
    # convert dict to vector
    center, scale, feature_order = scaler_params
    # print(feature_order)
    # print(center)
    # print(scale)
    # print('hello')

    x = np.fromiter((metrics[f] for f in feature_order), dtype=np.float64)
    # scale manually
    x_scaled = (x - center) / scale
    return x_scaled.reshape(1, -1)

def prepare_new_data(hyp_board, scaler_params):
    # if type(hyp_board) == str:
    #     board_array = np.array([int(x) for x in hyp_board]).reshape((6,6))
    # else:
    board_array = hyp_board
    current_metrics = bias_metrics.calculate_bias_metrics(board_array)
    new_batch_s = metrics_to_scaled_vector(current_metrics, scaler_params)
    return new_batch_s

def get_class_weights(df):
    n_R = len(df[df['label'] == 0])
    n_B = len(df[df['label'] == 1])

    pi_R = n_R / (n_R + n_B)
    pi_B = n_B / (n_R + n_B)

    return pi_R, pi_B

def predict_LR_new_board(new_board_s, class_weights, clf):

    pi_R, pi_B = class_weights
    c = clf.predict_proba(new_board_s)
    p = c[0][1]  # probability of being in biased class
    # print(c)

    odds = p / (1 - p)
    LR_hat = odds * (pi_R / pi_B)
    # LR_hat = odds

    # print(f'weight = {np.round(LR_hat, 3)}')
    return LR_hat

if False:
    random_boards_df = pd.read_csv("bias_checks\\random_boards_bruteforce_uniform.csv")
    random_boards_df['label'] = 0
    random_boards = random_boards_df['board'].tolist()

    biased_boards_df = pd.read_csv("bias_checks\\biased_relational_boards.csv")
    biased_boards_df['label'] = 1
    biased_boards = biased_boards_df['board'].tolist()

    # all_boards_df = pd.concat([random_boards_df, biased_boards_df], ignore_index=True)
    # all_boards_df = random_boards_df.copy()
    all_boards_df = pd.concat([random_boards_df, biased_boards_df], ignore_index=True)
    # make 'board' column the index of all_boards_df
    all_boards_df.set_index('board', inplace=True)

    # print(all_boards_df.sample(100))
    # exit()

    for board_code in all_boards_df.index:
        board_array = np.array([int(x) for x in board_code]).reshape((6,6))
        current_metrics = bias_metrics.calculate_bias_metrics(board_array)

        for metric_name, metric_value in current_metrics.items():
            all_boards_df.at[board_code, metric_name] = metric_value

    print(all_boards_df.head())

    # pd.to_csv("bias_checks\\all_boards_with_metrics.csv", all_boards_df)

    # save as csv
    all_boards_df.to_csv("bias_checks\\all_boards_with_metrics.csv")

if __name__ == "__main__":
    all_boards_df = pd.read_csv("bias_checks\\all_boards_with_metrics.csv", index_col=0)
    # get only first 10500 rows of df

    red_boards_df = all_boards_df[:10500]



    X = red_boards_df.drop(columns=['label'])

    # apply scaler
    scaler = RobustScaler().fit(X)
    Xs = scaler.transform(X)

    clf = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        max_iter=500
    )

    # clf = SGDClassifier(
    #     loss="log_loss",
    #     penalty="l2",      # <-- best choice
    #     alpha=1e-4,
    #     learning_rate="optimal",
    # )


    clf.fit(Xs, red_boards_df['label'])

    # clf.fit(Xs, red_boards_df['label'])

    new_board_s = prepare_new_data('000000000000020002020002000000000333', scaler)

    class_weights = get_class_weights(red_boards_df)
    LR_hat = predict_LR_new_board(new_board_s, class_weights, clf)
    print(f'weight = {np.round(LR_hat, 3)}')


    batch_size = 20
    n = int((11000 - 10500) / batch_size)

    for k in range(n):
        # new_batch = all_boards_df[10500 + (k) * batch_size : 10500 + (k+1) * batch_size]

        new_batch = all_boards_df[: 10500 + (k+1) * batch_size]
        class_weights = get_class_weights(new_batch)

        X_new = new_batch.drop(columns=['label'])
        X_new_s = scaler.transform(X_new)


        clf.fit(X_new_s, new_batch['label'])

        LR_hat = predict_LR_new_board(new_board_s, class_weights, clf)
        print(f'weight = {np.round(LR_hat, 3)}')

# save classifier and scaler using joblib
    import joblib
    joblib.dump(clf, "bias_checks\\logistic_regression_classifier.pkl")
    joblib.dump(scaler, "bias_checks\\robust_scaler.pkl")


    # metrics_per_board_r = get_metrics(random_boards)
    # metrics_per_board_b = get_metrics(biased_boards)

    # metrics_per_board = {**metrics_per_board_r, **metrics_per_board_b}

    # # for metric in bias_metrics.make_metrics_dict().keys():
    # #     all_boards_df[metric] = [metrics_per_board[board][metric] for board in all_boards_df['board']]

    # # print(random_boards_df)

    # # print(biased_boards_df)

    # print(all_boards_df.head())
    # # print(metrics_per_board)

    # # create a dataframe from metrics_per_board dictionary
    # # each key of this dict shall be an index . The values are dicts with metric names as keys. These metric names shall become columns in the new df and their values the corresponding column values.
    # metrics_per_board_df = pd.DataFrame.from_dict(metrics_per_board, orient='index')
    # print(metrics_per_board_df.head())
    # print(metrics_per_board_df.describe())
    # print(metrics_per_board_df.info())
    # print(metrics_per_board['000200000202000002003000003000003000'])
    # print(metrics_per_board_df.columns)
    # exit()

    # filter the metrics_per_board df to only include row with board '000200000202000002003000003000003000'
    # filtered_df = metrics_per_board_df.loc['000000000000000333000000022002000002']
    # print(filtered_df)

    # red = all_boards_df.loc['000000000000000333000000022002000002']
    # print(red)
    # # exit()
    # print(metrics_per_board_df)

    # # ??????????????????????????????????????????????????????????????????????????????????????
    # # add metrics columns to all_boards_df and map values from metrics_per_board_df
    # for metric in metrics_per_board_df.columns:
    #     all_boards_df[metric] = metrics_per_board_df[metric]
    # # ??????????????????????????????????????????????????????????????????????????????????????

    # for board in all_boards_df.index:
    #     for metric in bias_metrics.make_metrics_dict().keys():
    #         all_boards_df.loc[board][metric] = metrics_per_board_df.loc[board][metric]

    # print(all_boards_df.head())
    # exit()
    # # # save metrics_per_board to file
    # # pd.DataFrame.from_dict(metrics_per_board, orient='index').to_csv("bias_checks\\random_boards_metrics.csv")

    # # read metrics_per_board from file
    # metrics_per_board = pd.read_csv("bias_checks\\random_boards_metrics.csv", index_col=0).to_dict(orient='index')

    # # print(metrics_per_board)

    # metrics_list = list(bias_metrics.make_metrics_dict().keys())
    # #convert to list
    # # print(metrics_list)
    # # exit()

    # # n_boards = 1000
    # n_shuffles = 1_00

    # metrics_percentiles_per_n_boards = pd.DataFrame()


    # for exponent in range (5, 13):
    #     n_boards = 2 ** exponent

    #     metric_similarities_all = bias_metrics.make_metrics_dict()
    #     metric_similarities_all_min ={}
    #     metric_similarities_all_max ={}
    #     metric_similarities_all_avg ={}
    #     metric_similarities_all_perc ={}

    #     for i in range(n_shuffles):

    #         if (i+1) % 10 == 0:
    #             print(f'Shuffle {i+1} / {int(n_shuffles)}')

    #         chosen_boards_A = np.random.choice(random_boards, size=n_boards, replace=True).tolist()
    #         chosen_boards_B = np.random.choice(random_boards, size=n_boards, replace=True).tolist()

    #         # filter metrics_per_board to only include chosen_boards
    #         chosen_metrics_A = {board: metrics_per_board[board] for board in chosen_boards_A}
    #         chosen_metrics_B = {board: metrics_per_board[board] for board in chosen_boards_B}

    #         metric_similarities = {}

    #         for metric in metrics_list:
    #             metric_values_A = [chosen_metrics_A[board][metric] for board in chosen_boards_A]
    #             metric_values_B = [chosen_metrics_B[board][metric] for board in chosen_boards_B]

    #             # if metric == 'total_ships_whole_at_edge':
    #             #     print(f'Average total_ships_whole_at_edge in A: {np.mean(metric_values_A)}')
    #             #     print(f'Average total_ships_whole_at_edge in B: {np.mean(metric_values_B)}')

    #             # if metric == 'total_ships_touching_edge':
    #             #     print(f'Average total_ships_touching_edge in A: {np.mean(metric_values_A)}')
    #             #     print(f'Average total_ships_touching_edge in B: {np.mean(metric_values_B)}')

    #             sim = wasserstein_similarity(metric_values_A, metric_values_B)
    #             # sim = np.round(sim, 3)

    #             metric_similarities[metric] = sim
            
    #             metric_similarities_all[metric].append(sim)

    #     for metric in metrics_list:
    #         metric_similarities_all_min[metric] = np.round(np.min(metric_similarities_all[metric]), 3)
    #         metric_similarities_all_max[metric] = np.round(np.max(metric_similarities_all[metric]), 3)
    #         metric_similarities_all_avg[metric] = np.round(np.mean(metric_similarities_all[metric]), 3)

    #         metric_similarities_all_perc[metric] = np.round(np.percentile(metric_similarities_all[metric], 2.5), 3)

    #         # lower = np.percentile(boots, 2.5)


    #     metrics_sim = pd.DataFrame.from_dict(metric_similarities, orient='index', columns=['similarity'])
    #     metrics_sim_all = pd.DataFrame.from_dict(metric_similarities_all_avg, orient='index', columns=['similarity'])
    #     metrics_sim_all['min'] = pd.Series(metric_similarities_all_min)
    #     metrics_sim_all['max'] = pd.Series(metric_similarities_all_max)
    #     metrics_sim_all[f'{n_boards}'] = pd.Series(metric_similarities_all_perc)

    #     # print(metrics_sim)
    #     print("Metrics similarities for n_boards =", n_boards)
    #     print(metrics_sim_all)

    #     metrics_percentiles_per_n_boards = pd.concat([metrics_percentiles_per_n_boards, metrics_sim_all[f'{n_boards}']], axis=1)

        
    # print(metrics_percentiles_per_n_boards)
    # metrics_percentiles_per_n_boards.to_csv("bias_checks\\metrics_percentiles_per_n_boards.csv")

