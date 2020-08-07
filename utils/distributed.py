import ray

import numpy as np

from functools import partial
from scipy import sparse
from typing import Any, Callable, Dict, List, Optional, Union


def kernel_shap_target_fn(actor: Any, instances: tuple, kwargs: Optional[Dict] = None) -> Callable:
    """
    A target function that is executed in parallel given an actor pool. Its arguments must be an actor and a batch of
    values to be processed by the actor. Its role is to execute distributed computations when an actor is available.

    Parameters
    ----------
    actor
        A `ray` actor. This is typically a class decorated with the @ray.remote decorator, that has been subsequently
        instantiated using cls.remote(*args, **kwargs).
    instances
        A (batch_index, batch) tuple containing the batch of instances to be explained along with a batch index.
    kwargs
        A list of keyword arguments for the actor `shap_values` method.

    Returns
    -------
    A callable that can be used as a target process for a parallel pool of actor objects.
    """

    if kwargs is None:
        kwargs = {}

    return actor.get_explanation.remote(instances, **kwargs)


def kernel_shap_postprocess_fn(ordered_result: List[Union[np.ndarray, List[np.ndarray]]]) \
        -> List[Union[np.ndarray, List[np.ndarray]]]:
    """
    Merges the results of the batched computation for KernelShap.

    Parameters
    ----------
    ordered_result
        A list containing the results for each batch, in the order that the batch was submitted to the parallel pool.
        It may contain:
            - `np.ndarray` objects (single-output predictor)
            - lists of `np.ndarray` objects (multi-output predictors)

    Returns
    -------
    concatenated
        A list containing the concatenated results for all the batches.
    """
    if isinstance(ordered_result[0], np.ndarray):
        return np.concatenate(ordered_result, axis=0)

    # concatenate explanations for every class
    n_classes = len(ordered_result[0])
    to_concatenate = [list(zip(*ordered_result))[idx] for idx in range(n_classes)]
    concatenated = [np.concatenate(arrays, axis=0) for arrays in to_concatenate]
    return concatenated


def invert_permutation(p: list):
    """
    Inverts a permutation.

    Parameters:
    -----------
    p
        Some permutation of 0, 1, ..., len(p)-1. Returns an array s, where s[i] gives the index of i in p.

    Returns
    -------
    s
        `s[i]` gives the index of `i` in `p`.
    """

    s = np.empty_like(p)
    s[p] = np.arange(len(p))
    return s


class DistributedExplainer:
    """
    A class that orchestrates the execution of the execution of a batch of explanations in parallel.
    """

    def __init__(self, distributed_opts, cls, init_args, init_kwargs):

        self.n_jobs = distributed_opts['n_cpus']
        self.n_actors = int(distributed_opts['n_cpus'] // distributed_opts['actor_cpu_fraction'])
        self.actor_cpu_frac = distributed_opts['actor_cpu_fraction']
        self.batch_size = distributed_opts['batch_size']
        self.algorithm = distributed_opts['algorithm']
        self.target_fn = globals()[f"{distributed_opts['algorithm']}_target_fn"]
        try:
            self.post_process_fcn = globals()[f"{distributed_opts['algorithm']}_postprocess_fn"]
        except KeyError:
            self.post_process_fcn = None

        self.explainer = cls
        self.explainer_args = init_args
        self.explainer_kwargs = init_kwargs

        if not ray.is_initialized():
            print(f"Initialising ray on {distributed_opts['n_cpus']} cpus!")
            ray.init(num_cpus=distributed_opts['n_cpus'])

        self.pool = self.create_parallel_pool()

    def __getattr__(self, item):
        """
        Access to actor attributes. Should be used to retrieve only state that is shared by all actors in the pool.
        """
        actor = self.pool._idle_actors[0]
        return ray.get(actor.return_attribute.remote(item))

    def create_parallel_pool(self):
        """
        Creates a pool of actors (aka proceses containing explainers) that can execute explanations in parallel.
        """

        actor_handles = [ray.remote(self.explainer).options(num_cpus=self.actor_cpu_frac) for _ in range(self.n_actors)]

        actors = [handle.remote(*self.explainer_args, **self.explainer_kwargs) for handle in actor_handles]
        return ray.util.ActorPool(actors)

    def batch(self, X: np.ndarray) -> enumerate:
        """
        Splits the input into sub-arrays according to the following logic:

            - if `batch_size` is not `None`, batches of this size are created. The sizes of the batches created might \
            vary if the 0-th dimension of `X` is not divisible by `batch_size`. For an array of length l that should be
            split into n sections, it returns l % n sub-arrays of size l//n + 1 and the rest of size l//n.
            - if `batch_size` is `None`, then `X` is split into `n_jobs` sub-arrays

        Parameters
        ----------
        X
            Array to be split.
        Returns
        ------
            A list of sub-arrays of X.
        """

        n_records = X.shape[0]
        if isinstance(X, sparse.spmatrix):
            X = X.toarray()

        if self.batch_size:
            n_batches = n_records // self.batch_size
            if n_records % self.batch_size != 0:
                n_batches += 1
            slices = [self.batch_size * i for i in range(1, n_batches)]
            batches = np.array_split(X, slices)
        else:
            batches = np.array_split(X, self.n_jobs)
        return enumerate(batches)

    def get_explanation(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Performs distributed explanations of instances in `X`.

        Parameters
        ----------
        X
            A batch of instances to be explained. Split into batches according to the settings passed to the constructor.
        kwargs
            Any keyword-arguments for the explainer `explain` method. 

        Returns
        --------
            An array of explanations.
        """  # noqa E501

        if kwargs is not None:
            self.target_fn = partial(self.target_fn, kwargs=kwargs)
        batched_instances = self.batch(X)

        unordered_explanations = self.pool.map_unordered(self.target_fn, batched_instances)

        return self.order_result(unordered_explanations)

    def order_result(self, unordered_result: List[tuple]) -> np.ndarray:
        """
        Re-orders the result of a distributed explainer so that the explanations follow the same order as the input to
        the explainer.


        Parameters
        ----------
        unordered_result
            Each tuple contains the batch id as the first entry and the explanations for that batch as the second.

        Returns
        -------
        A numpy array where the the batches ordered according to their batch id are concatenated in a single array.
        """

        # TODO: THIS DOES NOT LEVERAGE THE FACT THAT THE RESULTS ARE RETURNED AS AVAILABLE. ISSUE TO BE RAISED.

        result_order, results = list(zip(*[(idx, res) for idx, res in unordered_result]))
        orig_order = invert_permutation(list(result_order))
        ordered_result = [results[idx] for idx in orig_order]
        if self.post_process_fcn is not None:
            return self.post_process_fcn(ordered_result)
        return ordered_result
