import logging

import numpy as np

from explainers.kernel_shap import KernelShap
from ray import serve
from typing import Any, Dict, List
from explainers.utils import load_model


class KernelShapModel:
    """Backend class for distributing explanations with Ray Serve."""
    def __init__(self,
                 predictor_path: str,
                 background_data: np.ndarray,
                 constructor_kwargs: Dict[str, Any],
                 fit_kwargs: Dict[str, Any]):
        """
        Initialises backend for distributed explanations.


        Parameters
        ----------
        predictor_path
            Path to the model to be explained.
        background_data
            Background data used for fitting the explainer.
        constructor_kwargs
            Any other arguments for explainer constructor. See `explainers.kernel_shap.KernelShap` for details.
        fit_kwargs
            Any other arguments for the explainer `fit` method. See `explainers.kernel_shap.KernelShap` for details.
        """

        predictor = load_model(predictor_path)
        predict_fcn = predictor.predict_proba
        if not hasattr(predictor, "predict_proba"):
            logging.warning("Predictor does not have predict_proba attribute, defaulting to predict")
            predict_fcn = predictor.predict
        self.explainer = KernelShap(predict_fcn, **constructor_kwargs)

        # TODO: REFACTOR THIS TO USE THE BACKEND METHOD CALLING FUNCTIONALITY
        self.explainer.fit(background_data, **fit_kwargs)

    def __call__(self, flask_request) -> str:
        """
        Serves explanations for a single instance.

        Parameters
        ---------
        flask_request
            A json flask request that contains a list with the instance to be explained in the ``array`` field.

        Returns
        -------
        A `str` object representing a json representation of the explainer output.
        """
        instance = np.array(flask_request.json["array"])
        explanations = self.explainer.explain(instance, silent=True)

        return explanations.to_json()


class BatchKernelShapModel(KernelShapModel):
    """Extends KernelShapModel to achieve batching of requests."""

    @serve.accept_batch
    def __call__(self, flask_requests: List) -> List[str]:
        """
        Serves explanations for a single instance.

        Parameters
        ----------
        flask_requests:
            A list of json flask requests. Each request should contain an instance to be explained in the ``array``
            field.

        Returns
        -------
        A `str` object representing a json representation of the explainer output.
        """

        instances = [request.json["array"] for request in flask_requests]
        explanations = []
        for instance in instances:
            explanations.append(
                self.explainer.explain(np.array(instance), silent=True).to_json()
            )

        return explanations
