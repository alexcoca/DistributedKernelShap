import argparse
import logging
import os
import pickle
import ray

import numpy as np

from explainers.kernel_shap import KernelShap
from explainers.utils import get_filename, load_data, load_model
from sklearn.metrics import accuracy_score
from typing import Any, Dict
from timeit import default_timer as timer

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
    group_names, groups = data['all']['group_names'], data['all']['groups']
    explainer = KernelShap(pred_fcn, link='logit', feature_names=group_names, distributed_opts=distributed_opts, seed=0)
    explainer.fit(data['background']['X']['preprocessed'], group_names=group_names, groups=groups)
    return explainer


def run_explainer(explainer, X_explain: np.ndarray, distributed_opts: dict, nruns: int):
    """
    Explain `X_explain` with `explainer` configured with `distributed_opts` `nruns` times in order to obtain
    runtime statistics.

    Parameters
    ---------
    explainer
        Fitted KernelShap explainer object
    X_explain
        Array containing instances to be explained, layed out row-wise. Split into minibatches that are distributed
        by the explainer.
    distributed_opts
        A dictionary of the form::

            {
            'n_cpus': int - controls the number of workers on which the instances are explained
            'batch_size': int - the size of a minibatch into which the dateset is split
            'actor_cpu_fraction': the fraction of CPU allocated to an actor
            }
    nruns
        Number of times `X_explain` is explained for a given workers and batch size setting.
    """

    if not os.path.exists('./results'):
        os.mkdir('./results')
    batch_size = distributed_opts['batch_size']
    result = {'t_elapsed': []}
    workers = distributed_opts['n_cpus']
    for run in range(nruns):
        logging.info(f"run: {run}")
        t_start = timer()
        explanation = explainer.explain(X_explain, silent=True)
        t_elapsed = timer() - t_start
        logging.info(f"Time elapsed: {t_elapsed}")
        result['t_elapsed'].append(t_elapsed)

        with open(get_filename(workers, batch_size, serve=False), 'wb') as f:
            pickle.dump(result, f)


def main():

    # experiment settings
    nruns = args.nruns
    batch_sizes = [int(elem) for elem in args.batch]

    # load data and instances to be explained
    data = load_data()
    predictor = load_model('assets/predictor.pkl')  # download if not available locally
    y_test, X_test_proc = data['all']['y']['test'], data['all']['X']['processed']['test']
    logging.info(f"Test accuracy: {accuracy_score(y_test, predictor.predict(X_test_proc))}")
    X_explain = data['all']['X']['processed']['test'].toarray()  # instances to be explained

    if args.workers == -1:  # sequential benchmark
        logging.info(f"Running sequential benchmark without ray ...")
        distributed_opts = {'batch_size': None, 'n_cpus': None, 'actor_cpu_fraction': 1.0}
        explainer = fit_kernel_shap_explainer(predictor, data, distributed_opts=distributed_opts)
        run_explainer(explainer, X_explain, distributed_opts, nruns)
    # run distributed benchmark or simply explain on a number of cores, depending on args.benchmark value
    else:
        workers_range = range(1, args.workers + 1) if args.benchmark == 1 else range(args.workers, args.workers + 1)
        for workers in workers_range:
            for batch_size in batch_sizes:
                logging.info(f"Running experiment using {workers} actors...")
                logging.info(f"Running experiment with batch size {batch_size}")
                distributed_opts = {'batch_size': int(batch_size), 'n_cpus': workers, 'actor_cpu_fraction': 1.0}
                explainer = fit_kernel_shap_explainer(predictor, data, distributed_opts)
                run_explainer(explainer, X_explain, distributed_opts, nruns)
                ray.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b",
        "--batch",
        nargs='+',
        help="A list of values representing the maximum batch size of instances sent to the same worker.",
        required=True,
    )
    parser.add_argument(
        "-w",
        "--workers",
        default=-1,
        type=int,
        help="The number of processes to distribute the explanations dataset on. Set to -1 to run sequenential (without"
             "ray) version."
    )
    parser.add_argument(
        "-benchmark",
        default=0,
        type=int,
        help="Set to 1 to benchmark parallel computation. In this case, explanations are distributed over cores in "
             "range(1, args.workers).!"
    )
    parser.add_argument(
        "-n",
        "--nruns",
        default=5,
        type=int,
        help="Controls how many times an experiment is run (in benchmark mode) for a given number of workers to obtain "
             "run statistics."
    )
    args = parser.parse_args()
    main()
