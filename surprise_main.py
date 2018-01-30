
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import time
import datetime
import random

import numpy as np
from tabulate import tabulate
from collections import defaultdict

from surprise import Dataset
from surprise.model_selection import KFold
from surprise import NormalPredictor
from surprise import BaselineOnly
from surprise import KNNBasic
from surprise import KNNWithMeans
from surprise import KNNBaseline
from surprise import accuracy


def precision_recall_at_k(predictions, k=10, threshold=3.5):
    '''Return precision and recall at k metrics for each user.'''

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))
        # print(user_est_true[uid])

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():
        # print(user_ratings)
        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        precisions[uid] = (n_rel_and_rec_k + 0.0) / n_rec_k if n_rec_k != 0 else 1

        # Recall@K: Proportion of relevant items that are recommended
        recalls[uid] = (n_rel_and_rec_k + 0.0) / n_rel if n_rel != 0 else 1
    return precisions, recalls


if __name__ == '__main__':

    # The algorithms to cross-validate
    classes = ( KNNBasic, KNNWithMeans, KNNBaseline, BaselineOnly, NormalPredictor)


    # set RNG
    np.random.seed(0)
    random.seed(0)

    dataset = 'ml-100k'
    data = Dataset.load_builtin(dataset)
    kf = KFold(n_splits=5, random_state=0)  # folds will be the same for all algorithms.

    table = []
    for klass in classes:
        mean_dict = defaultdict(list)
        klass_run = klass()
        for trainset, testset in kf.split(data):
            start = time.time()
            klass_run.fit(trainset)
            predictions = klass_run.test(testset)
            cv_time = time.time() - start
            precisions, recalls = precision_recall_at_k(predictions, k=5, threshold=4)

            # Precision and recall can then be averaged over all users
            precision = (sum(prec for prec in precisions.values()) / len(precisions))
            recall = (sum(rec for rec in recalls.values()) / len(recalls))
            rmse = accuracy.rmse(predictions)
            mae = accuracy.mae(predictions)
            mean_dict['time'].append(cv_time)
            mean_dict['precision'].append(precision)
            mean_dict['recall'].append(recall)
            mean_dict['rmse'].append(rmse)
            mean_dict['mae'].append(mae)
        print(mean_dict)
        [mean_precision, mean_recall, mean_rmse, mean_mae, mean_cv_time] = [
            (sum(mean_dict['precision']) / len(mean_dict['precision'])),
            (sum(mean_dict['recall']) / len(mean_dict['recall'])), (sum(mean_dict['rmse']) / len(mean_dict['rmse'])),
            (sum(mean_dict['mae']) / len(mean_dict['mae'])),
            str(datetime.timedelta(seconds=int((sum(mean_dict['time']) / len(mean_dict['time'])))))]

        new_line = [klass.__name__, mean_precision, mean_recall, mean_rmse, mean_mae, mean_cv_time]
        print(tabulate([new_line], tablefmt="pipe"))  # print current algo perf
        table.append(new_line)

    header = ['Algs',
              'Precision',
              'Recall',
              'RMSE',
              'MAE',
              'Time'
              ]
    print(tabulate(table, header, tablefmt="pipe"))
