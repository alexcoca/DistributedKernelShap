import argparse
import logging
import os
import ray
import pickle
import requests

import numpy as np

from collections import namedtuple
from ray import serve
from timeit import default_timer as timer
from typing import Any, Dict, Tuple
from utils.wrappers import BatchKernelShapModel, KernelShapModel
from utils.utils import get_filename, load_data

logging.basicConfig(level=logging.INFO)

PREDICTOR_PATH = 'assets/predictor.pkl'
"""
str: The file containing the predictor. The predictor can be created by running `fit_adult_model.py` or output by 
calling `utils.utils.load_model()`, which will download a default predictor if `assets/` does not contain one. 
"""


def endpont_setup(tag: str, backend_tag: str, route: str = "/"):
    """
    Creates an endpoint for serving explanations.

    Parameters
    ----------
    tag
        Endpoint tag.
    backend_tag
        A tag for the backend this explainer will connect to.
    route
        The URL where the explainer can be queried.
    """
    serve.create_endpoint(tag, backend=backend_tag, route=route, methods=["GET"])


def backend_setup(tag: str, worker_args: Tuple, replicas: int, max_batch_size: int) -> None:
    """
    Setups the backend for the distributed explanation task.

    Parameters
    ----------
    tag
        A tag for the backend component. The same tag must be passed to `endpoint_setup`.
    worker_args
        A tuple containing the arguments for initialising the explainer and fitting it.
    replicas
        The number of backend replicas that serve explanations.
    max_batch_size
        Maximum number of requests to batch and send to a worker process.
    """

    serve.init()

    if max_batch_size == 1:
        config = {'num_replicas': max(replicas, 1)}
        serve.create_backend(tag, KernelShapModel, *worker_args)
    else:
        config = {'num_replicas': max(replicas, 1), 'max_batch_size': max_batch_size}
        serve.create_backend(tag, BatchKernelShapModel, *worker_args)
    serve.update_backend_config(tag, config)

    logging.info(f"Backends: {serve.list_backends()}")


def prepare_explainer_args(data: Dict[str, Any]) -> Tuple[str, np.ndarray, dict, dict]:
    """
    Extracts the name of the features (group_names) and the columns corresponding to each feature in the faeture matrix
    (group_names) from the `data` dict and defines the explainer arguments. The background data necessary to initialise
    the explainer is also extracted from the same dictionary.

    Parameters
    ----------
    data
        A dictionary that contains all information necessary to initialise the explainer.

    Returns
    -------
    A tuple containing the positional and keyword arguments necessary for initialising the explainers.
    """

    groups = data['all']['groups']
    group_names = data['all']['group_names']
    background_data = data['background']['X']['preprocessed']
    assert background_data.shape[0] == 100
    init_kwargs = {'link': 'logit', 'feature_names': group_names, 'seed': 0}
    fit_kwargs = {'groups': groups, 'group_names': group_names}
    worker_args = (PREDICTOR_PATH, background_data, init_kwargs, fit_kwargs)

    return worker_args


@ray.remote
def distribute_request(instance: np.ndarray, url: str = "http://localhost:8000") -> str:
    """
    Task for distributing the explanations across the backend.

    Parameters
    ----------
    instance
        Instance to be explained.
    url:
        The explainer URL.

    Returns
    -------
    A str representation of the explanation output json file.
    """

    resp = requests.get(url, json={"array": instance.tolist()})
    return resp.json()


def explain(data: np.ndarray, *, url: str) -> namedtuple:
    """
    Sends the requests to the explainer URL. The `data` array is split into sub-array containing only one instance.

    Parameters
    ----------
    data:
        Array of instances to be explained.
    url
        Explainer endpoint.

    Returns
    -------
    responses
        A named tuple with a `responses` field and a `t_elapsed` field.
    """

    run_output = namedtuple('run_output', 'responses t_elapsed')
    instances = np.split(data, data.shape[0])
    logging.info(f"Explaining {len(instances)} instances!")
    tstart = timer()
    responses_id = [distribute_request.remote(instance, url=url) for instance in instances]
    responses = [ray.get(resp_id) for resp_id in responses_id]
    t_elapsed = timer() - tstart
    logging.info(f"Time elapsed: {t_elapsed}")

    return run_output(responses=responses, t_elapsed=t_elapsed)


def distribute_explanations(n_runs: int, replicas: int, max_batch_size: int, address: str = "http://localhost:8000"):
    """
    Setup an endpoint and a backend and send requests to the endpoint.

    Parameters
    -----------
    n_runs
        Number of times to run an experiment where the entire set of explanations is sent to the explainer endpoint.
        Used to determine the average runtime given the number of cores.
    replicas
        How many backend replicas should be used for distributing the workload
    max_batch_size
        The maximum batch size the explainer accepts.
    address
        The url for the explainer endpoint.
    """

    result = {'t_elapsed': [], 'explanations': []}
    route = "explain"
    backend_tag = "kernel_shap:b100"  # b100 means 100 background samples
    endpoint_tag = f"{backend_tag}_endpoint"
    data = load_data()
    worker_args = prepare_explainer_args(data)
    backend_setup(backend_tag, worker_args, replicas, max_batch_size)
    endpont_setup(endpoint_tag, backend_tag, route=f"/{route}")
    # extract instances to be explained from the dataset
    X_explain = data['all']['X']['processed']['test'].toarray()
    assert X_explain.shape[0] == 2560
    for run in range(n_runs):
        logging.info(f"Experiment run: {run}")
        results = explain(X_explain, url=f"{address}/{route}")
        result['t_elapsed'].append(results.t_elapsed)
        result['explanations'].append(results.responses)

    with open(get_filename(replicas, max_batch_size), 'wb') as f:
        pickle.dump(result, f)

    ray.shutdown()


def main():

    if not os.path.exists('results'):
        os.mkdir('results')

    address = f"{args.host}:{args.port}"
    batch_size_limits = [int(elem) for elem in args.max_batch_size]
    if args.benchmark:
        for replicas in range(1, args.replicas + 1):
            logging.info(f"Running on {replicas} backend replicas!")
            for max_batch_size in batch_size_limits:
                logging.info(f"Batching with max_batch_size of {max_batch_size}")
                distribute_explanations(args.nruns, replicas, max_batch_size, address=address)
    else:
        nruns = 1
        for max_batch_size in batch_size_limits:
            distribute_explanations(nruns, args.replicas, max_batch_size, address=address)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r, --replicas",
        default=1,
        type=int,
        help="The number of backend replicas used to serve the explainer."
    )
    parser.add_argument(
        "-batch, --max_batch_size",
        nargs='+',
        help="A list of values representing the maximum batch size of pending queries sent to the same worker.",
        required=True,
    )
    parser.add_argument(
        "-benchmark",
        default=0,
        type=int,
        help="Set to 1 to benchmark parallel computation. In this case, explanations are distributed over replicas in "
             "range(1, args.replicas).!"
    )
    parser.add_argument(
        "-n, --nruns",
        default=5,
        type=int,
        help="Controls how many times an experiment is run (in benchmark mode) for a given number of cores to obtain "
             "run statistics."
    )
    parser.add_argument(
        "-h, --host",
        default="http://localhost",
        type=str,
        help="Hostname"
    )
    parser.add_argument(
        "-p, --port",
        default="8000",
        type=str,
        help="Port"
    )
    args = parser.parse_args()
    main()
