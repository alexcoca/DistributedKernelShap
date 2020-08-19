import io
import logging
import os
import pickle
import requests

from functools import singledispatch, update_wrapper
from typing import Callable


EXPLANATIONS_SET_URL = 'https://storage.googleapis.com/seldon-datasets/experiments/distributed_kernel_shap/adult_processed.pkl'
BACKGROUND_SET_URL = 'https://storage.googleapis.com/seldon-datasets/experiments/distributed_kernel_shap/adult_background.pkl'
MODEL_URL = 'https://storage.googleapis.com/seldon-models/alibi/distributed_kernel_shap/predictor.pkl'
EXPLANATIONS_SET_LOCAL = 'data/adult_processed.pkl'
BACKGROUND_SET_LOCAL = 'data/adult_background.pkl'
MODEL_LOCAL = 'assets/predictor.pkl'


class Bunch(dict):
    """
    Container object for internal datasets. Dictionary-like object that exposes its keys as attributes.
    """

    def __init__(self, **kwargs):
        super().__init__(kwargs)

    def __setattr__(self, key, value):
        self[key] = value

    def __dir__(self):
        return self.keys()

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)


def methdispatch(func: Callable):
    """
    A decorator that is used to support singledispatch style functionality
    for instance methods. By default, singledispatch selects a function to
    call from registered based on the type of args[0]:

        def wrapper(*args, **kw):
            return dispatch(args[0].__class__)(*args, **kw)

    This uses singledispatch to do achieve this but instead uses args[1]
    since args[0] will always be self.
    """

    dispatcher = singledispatch(func)

    def wrapper(*args, **kw):
        return dispatcher.dispatch(args[1].__class__)(*args, **kw)

    wrapper.register = dispatcher.register
    update_wrapper(wrapper, dispatcher)

    return wrapper


def download(path: str):
    """ Download from Seldon GC bucket indicated by `path`."""

    try:
        resp = requests.get(path)
        resp.raise_for_status()
    except requests.RequestException:
        logging.exception("Could not connect to bucket, URL may be out of service!")
        raise ConnectionError

    return resp


def load_data():
    """
    Load instances to be explained and background data from the data/ directory if they exist, otherwise download
    from Seldon Google Cloud bucket.
    """

    data = {'all': None, 'background': None}
    try:
        with open(BACKGROUND_SET_LOCAL, 'rb') as f:
            data['background'] = pickle.load(f)
        with open(EXPLANATIONS_SET_LOCAL, 'rb') as f:
            data['all'] = pickle.load(f)
    except FileNotFoundError:
        logging.info(f"Downloading data from {EXPLANATIONS_SET_URL}")
        all_data_raw = download(EXPLANATIONS_SET_URL)
        data['all'] = pickle.load(io.BytesIO(all_data_raw.content))
        logging.info(f"Downloading data from {BACKGROUND_SET_URL}")
        background_data_raw = download(BACKGROUND_SET_URL)
        data['background'] = pickle.load(io.BytesIO(background_data_raw.content))

        # save the data locally so we don't download it every time we run the main script
        if not os.path.exists('../data'):
            os.mkdir('../data')
        with open('../data/adult_background.pkl', 'wb') as f:
            pickle.dump(data['background'], f)
        with open('../data/adult_processed.pkl', 'wb') as f:
            pickle.dump(data['all'], f)

    return data


def load_model(path: str):
    """
    Load a model that has been saved locally or download a default model from a Seldon bucket.
    """

    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        logging.info(f"Could not find model {path}. Downloading from {MODEL_URL}...")
        model_raw = download(MODEL_URL)
        model = pickle.load(io.BytesIO(model_raw.content))

        if not os.path.exists('assets'):
            os.mkdir('assets')

        with open("assets/predictor.pkl", "wb") as f:
            pickle.dump(model, f)

        return model


def get_filename(workers: int, batch_size: int, cpu_fraction: int, serve: bool = True):
    """
    Creates a filename for an experiment given the inputs.

    Parameters
    ----------
    workers
        How many worker processes are used for the explanation task.
    batch_size
        Mini-batch size: how many explanations are sent to one worker process at a time.
    cpu_fraction
        CPU fraction utilized by a worker process.
    serve
        A different naming convention is used depending on whether ray serve is used to distribute the explanations or
        not.
    """

    if serve:
        return f"results/ray_replicas_{workers}_maxbatch_{batch_size}_actorfr_{cpu_fraction}.pkl"
    return f"results/ray_workers_{workers}_bsize_{batch_size}_actorfr_{cpu_fraction}.pkl"

