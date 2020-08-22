import logging

import os
import sys

import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from typing import Dict, Any
from explainers.utils import load_data

"""
This script pulls the Adult data from the ``data/`` directory and fits a logistic regression model to it. Model is 
saved under ``assets/predictor.pkl``. 
"""

sys.path.append('./utils')


def fit_adult_logistic_regression(data_dict: Dict[str, Any]):
    """
    Fit a logistic regression model to the processed Adult dataset.
    """

    logging.info("Fitting model ...")
    X_train_proc = data_dict['X']['processed']['train']
    X_test_proc = data_dict['X']['processed']['test']
    y_train = data_dict['y']['train']
    y_test = data_dict['y']['test']

    classifier = LogisticRegression(multi_class='multinomial',
                                    random_state=0,
                                    max_iter=500,
                                    verbose=0,
                                    )
    classifier.fit(X_train_proc, y_train)

    logging.info(f"Test accuracy: {accuracy_score(y_test, classifier.predict(X_test_proc))}")

    return classifier


def main():

    if not os.path.exists('assets'):
        os.mkdir('assets')

    data = load_data()
    lr_predictor = fit_adult_logistic_regression(data['all'])
    with open("assets/predictor.pkl", "wb") as f:
        pickle.dump(lr_predictor, f)


if __name__ == '__main__':
    main()
