import argparse
import logging
import os
import pickle
import ray

import numpy as np

from explainers.kernel_shap import KernelShap
from typing import Any, Dict
from timeit import default_timer as timer
from utils.utils import get_filename, load_data, load_model

logging.basicConfig(level=logging.INFO)


def fit_kernel_shap_explainer(clf, data: dict, distributed_opts: Dict[str, Any] = None):
    """
    Returns an a fitted KernelShap explainer for the classifier `clf`. The categorical variables are grouped according
    to the information specified in `data`.

    Parameters
    ----------
    clf
        Classifier whose predictions are to be explained.
    data
        Contains the background data as well as information about the features and the columns in the feature matrix
        they occupy.
    distributed_opts
        Options controlling the number of worker processes that will distribute the workload.
    """

    pred_fcn = clf.predict_proba
    group_names, groups = data['groups_names'], data['groups']
    explainer = KernelShap(pred_fcn, link='logit', feature_names=group_names, distributed_opts=distributed_opts)
    explainer.fit(data['background']['X']['preprocessed'], group_names=group_names, groups=groups)

    return explainer


def main():

    data = load_data()
    predictor = load_model('assets/predictor.pkl')
    if args.cores == -1:
        distributed_opts = {'batch_size': None, 'n_cpus': None, 'actor_cpu_fraction': 1.0}
    else:
        distributed_opts = {'batch_size': args.batch_size, 'n_cpus': args.cores, 'actor_cpu_fraction': 1.0}
    explainer = fit_kernel_shap_explainer(predictor, data, distributed_opts=distributed_opts)
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
            explainer = fit_kernel_shap_explainer(predictor, data, distributed_opts)


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
