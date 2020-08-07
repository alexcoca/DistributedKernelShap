import argparse
import logging
import os
import pickle
import ray

import numpy as np

from explainers.kernel_shap import KernelShap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from typing import Any, Dict
from timeit import default_timer as timer
from utils.utils import get_filename, load_data


logging.basicConfig(level=logging.INFO)


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


def group_adult_dataset(preprocessed_dataset: Dict):
    """
    This function:
        - Finds the numerical and categorical variables in the processed data, along with the encoding length for cat. \
        variables
        - Outputs a list of the same length as the number of variable, where each element is a list specifying the \ 
        indices occupied by each variable in the processed (aka encoded) dataset
    """  # noqa

    feature_names = preprocessed_dataset['orig_feature_names']
    preprocessor = preprocessed_dataset['preprocessor']
    numerical_feats_idx = preprocessor.transformers_[0][2]
    categorical_feats_idx = preprocessor.transformers_[1][2]
    ohe = preprocessor.transformers_[1][1]
    # compute encoded dimension; -1 as ohe is setup with drop='first'
    feat_enc_dim = [len(cat_enc) - 1 for cat_enc in ohe.categories_]
    num_feats_names = [feature_names[i] for i in numerical_feats_idx]
    cat_feats_names = [feature_names[i] for i in categorical_feats_idx]

    group_names = num_feats_names + cat_feats_names
    groups = []
    cat_var_idx = 0

    for name in group_names:
        if name in num_feats_names:
            groups.append(list(range(len(groups), len(groups) + 1)))
        else:
            start_idx = groups[-1][-1] + 1 if groups else 0
            groups.append(list(range(start_idx, start_idx + feat_enc_dim[cat_var_idx])))
            cat_var_idx += 1

    return group_names, groups


def fit_kernel_shap_explainer(clf, data: dict, distributed_opts: Dict[str, Any] = None):
    """Returns an a fitted explainer for the classifier `clf`"""

    pred_fcn = clf.predict_proba
    group_names, groups = group_adult_dataset(data['all'])
    explainer = KernelShap(pred_fcn, link='logit', feature_names=group_names, distributed_opts=distributed_opts)
    explainer.fit(data['background']['X']['preprocessed'], group_names=group_names, groups=groups)

    return explainer


def main():

    # load data
    data = load_data()
    # fit logistic reg predictor
    lr_predictor = fit_adult_logistic_regression(data['all'])
    # fit explainer
    if args.cores == -1:
        distributed_opts = {'batch_size': None, 'n_cpus': None, 'actor_cpu_fraction': 1.0}
    else:
        distributed_opts = {'batch_size': args.batch_size, 'n_cpus': args.cores, 'actor_cpu_fraction': 1.0}
    explainer = fit_kernel_shap_explainer(lr_predictor, data, distributed_opts=distributed_opts)
    # explain the test data
    X_explain = data['all']['X']['processed']['test'].toarray()
    nruns = args.nruns if args.benchmark == 1 else 1
    # run sequential benchmark
    if args.cores == -1:
        experiment(explainer, X_explain, distributed_opts, nruns)
    # run distributed benchmark or simply explain on a number of cores, depeding on args.benchmark value
    else:
        cores_range = range(2, args.cores + 1) if args.benchmark == 1 else range(args.cores, args.cores + 1)
        for ncores in cores_range:
            logging.info(f"Running experiment on {ncores}")
            experiment(explainer, X_explain, distributed_opts, nruns)
            ray.shutdown()
            distributed_opts['ncpus'] = ncores + 1
            explainer = fit_kernel_shap_explainer(lr_predictor, data, distributed_opts)


def experiment(explainer, X_explain: np.ndarray, distributed_opts: dict, nruns: int):
    """
    Explain `X_explain` with `explainer` configured with `distributed_opts` `nruns` times in order to obtain
    runtime statistics.
    """

    if not os.path.exists('results'):
        os.mkdir('results')

    result = {'t_elapsed': [], 'explanations': []}
    for run in range(nruns):
        logging.info(f"run: {run}")
        t_start = timer()
        explanation = explainer.explain(X_explain, silent=True)
        t_elapsed = timer() - t_start
        logging.info(f"Time elapsed: {t_elapsed}")
        result['t_elapsed'].append(t_elapsed)
        result['explanations'].append(explanation)

    with open(get_filename(distributed_opts), 'wb') as f:
        pickle.dump(result, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-batch_size",
        default=10,
        type=int,
        help="Minibatch size. How many explanations each worker has to execute."
    )
    parser.add_argument(
        "-cores",
        default=-1,
        type=int,
        help="The number of cores to distribute the explanations dataset on. Set to -1 to run sequenential version."
    )
    parser.add_argument(
        "-benchmark",
        default=0,
        type=int,
        help="Set to 1 to benchmark parallel computation. In this case, explanations are distributed over cores in "
             "range(2, args.cores).!"
    )
    parser.add_argument(
        "-nruns",
        default=5,
        type=int,
        help="Controls how many times an experiment is run (in benchmark mode) for a given number of cores to obtain "
             "run statistics."
    )
    args = parser.parse_args()
    main()
