import argparse
import logging
import os
import ray
import pickle
import requests
import numpy as np

import explainers.wrappers as wrappers

from collections import namedtuple
from ray import serve
from timeit import default_timer as timer
from typing import Any, Dict, List, Tuple
from explainers.utils import get_filename, batch, load_data, load_model


logging.basicConfig(level=logging.INFO)

PREDICTOR_URL = 'https://storage.googleapis.com/seldon-models/alibi/distributed_kernel_shap/predictor.pkl'
PREDICTOR_PATH = 'assets/predictor.pkl'
"""
str: The file containing the predictor. The predictor can be created by running `fit_adult_model.py` or output by 
calling `explainers.utils.load_model()`, which will download a default predictor if `assets/` does not contain one. 
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

    if max_batch_size == 1:
        config = {'num_replicas': max(replicas, 1)}
        serve.create_backend(tag, wrappers.KernelShapModel, *worker_args)
    else:
        config = {'num_replicas': max(replicas, 1), 'max_batch_size': max_batch_size}
        serve.create_backend(tag, wrappers.BatchKernelShapModel, *worker_args)
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
    predictor = load_model(PREDICTOR_URL)
    worker_args = (predictor, background_data, init_kwargs, fit_kwargs)

    return worker_args


@ray.remote
def distribute_request(instance: np.ndarray, url: str = "http://localhost:8000/explain") -> str:
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


def request_explanations(instances: List[np.ndarray], *, url: str) -> namedtuple:
    """
    Sends the instances to the explainer URL.

    Parameters
    ----------
    instances:
        Array of instances to be explained.
    url
        Explainer endpoint.


    Returns
    -------
    responses
        A named tuple with a `responses` field and a `t_elapsed` field.
    """

    run_output = namedtuple('run_output', 'responses t_elapsed')
    tstart = timer()
    responses_id = [distribute_request.remote(instance, url=url) for instance in instances]
    responses = [ray.get(resp_id) for resp_id in responses_id]
    t_elapsed = timer() - tstart
    logging.info(f"Time elapsed: {t_elapsed}...")

    return run_output(responses=responses, t_elapsed=t_elapsed)


def run_explainer(X_explain: np.ndarray,
                  n_runs: int,
                  replicas: int,
                  max_batch_size: int,
                  batch_mode: str = 'ray',
                  url: str = "http://localhost:8000/explain"):
    """
    Setup an endpoint and a backend and send requests to the endpoint.

    Parameters
    -----------
    X_explain
        Instances to be explained. Each row is an instance that is explained independently of others.
    n_runs
        Number of times to run an experiment where the entire set of explanations is sent to the explainer endpoint.
        Used to determine the average runtime given the number of cores.
    replicas
        How many backend replicas should be used for distributing the workload
    max_batch_size
        The maximum batch size the explainer accepts.
    batch_mode : {'ray', 'default'}
        If 'ray', ray_serve components are leveraged for minibatches. Otherwise the input tensor is split into
        minibatches which are sent to the endpoint.
    url
        The url of the explainer endpoint.
    """

    result = {'t_elapsed': [], 'explanations': []}
    # extract instances to be explained from the dataset
    assert X_explain.shape[0] == 2560

    # split input into separate requests
    if batch_mode == 'ray':
        instances = np.split(X_explain, X_explain.shape[0])  # use ray serve to batch the requests
        logging.info(f"Explaining {len(instances)} instances...")
    else:
        instances = batch(X_explain, batch_size=max_batch_size)
        logging.info(f"Explaining {len(instances)} mini-batches of size {max_batch_size}...")

    # distribute it
    for run in range(n_runs):
        logging.info(f"Experiment run: {run}...")
        results = request_explanations(instances, url=url)
        result['t_elapsed'].append(results.t_elapsed)
        result['explanations'].append(results.responses)

    with open(get_filename(replicas, max_batch_size), 'wb') as f:
        pickle.dump(result, f)


def main():

    if not os.path.exists('results'):
        os.mkdir('results')

    data = load_data()
    X_explain = data['all']['X']['processed']['test'].toarray()

    max_batch_size = [int(elem) for elem in args.max_batch_size][0]
    batch_mode, replicas = args.batch_mode, args.replicas
    ray.init(address='auto')  # connect to the cluster
    serve.init(http_host='0.0.0.0')  # listen on 0.0.0.0 to make endpoint accessible from other machines
    host, route = os.environ.get("RAY_HEAD_SERVICE_HOST", args.host), "explain"
    url = f"http://{host}:{args.port}/{route}"
    backend_tag = "kernel_shap:b100"  # b100 means 100 background samples
    endpoint_tag = f"{backend_tag}_endpoint"
    worker_args = prepare_explainer_args(data)
    if batch_mode == 'ray':
        backend_setup(backend_tag, worker_args, replicas, max_batch_size)
        logging.info(f"Batching with max_batch_size of {max_batch_size} ...")
    else:  # minibatches are sent to the ray worker
        backend_setup(backend_tag, worker_args, replicas, 1)
        logging.info(f"Minibatches distributed of size {max_batch_size} ...")
    endpont_setup(endpoint_tag, backend_tag, route=f"/{route}")

    run_explainer(X_explain, args.n_runs, replicas, max_batch_size, batch_mode=batch_mode, url=url)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--replicas",
        default=1,
        type=int,
        help="The number of backend replicas used to serve the explainer."
    )
    parser.add_argument(
        "-batch",
        "--max_batch_size",
        nargs='+',
        help="A list of values representing the maximum batch size of pending queries sent to the same worker."
             "This should only contain one element as the backend is reset from `k8s_benchmark_serve.sh`.",
        required=True,
    )
    parser.add_argument(
        "-batch_mode",
        type=str,
        default='ray',
        help="If set to 'ray' the batching will be leveraging ray serve. Otherwise, the input array is split into "
             "minibatches that are sent to the endpoint.",
        required=True,
    )
    parser.add_argument(
        "-n",
        "--n_runs",
        default=5,
        type=int,
        help="Controls how many times an experiment is run (in benchmark mode) for a given number of cores to obtain "
             "run statistics."
    )
    parser.add_argument(
        "-ho",
        "--host",
        default="http://localhost",
        type=str,
        help="Hostname."
    )
    parser.add_argument(
        "-p",
        "--port",
        default="8000",
        type=str,
        help="Port."
    )
    args = parser.parse_args()
    main()
