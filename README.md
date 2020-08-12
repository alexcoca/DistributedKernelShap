# Running distributed KernelSHAP with Ray Serve

To create a virtual environment that allows you to run KernelSHAP in a distributed fashion with [`ray serve`](https://github.com/ray-project/ray) you need to configure your environment first, which requires [`conda`](https://problemsolvingwithpython.com/01-Orientation/01.05-Installing-Anaconda-on-Linux/) to be installed. You can then run the command::

`conda env create -f environment.yml -p /home/user/anaconda3/envs/env_name`

to create the environment and then activate it with `conda activate shap`. If you don not wish to change the installation path then you can skip the `-p` option. You are now ready to run the experiments. The steps involved are:

1. data processing 
2. predictor fitting
2. running benchmarking experiments

_**Step 1 (optional):**_ To process the data it is sufficient to run `python scripts/preprocess_data.py` with the default options. This will output a preprocessed version of the [`Adult`](http://archive.ics.uci.edu/ml/datasets/Adult) dataset and a partition of it that is used to initialise the KernelSHAP explainer. However, you can proceed to step 2 if you don't intend to change the default parameters as the same data will be automatically downloaded.

_**Step 2 (optional):**_ A logistic regression predictor can be fit on the preprocessed data by running `python scripts/fit_adult_model.py`. The predictor will be saved in the `assets/` directory under the `predictor.pkl` filename. If you did not alter the data processing script, it is not necessary to run this script as the predictor will be automatically downloaded and saved to `assets/`.


_**Step 3:**_ You can distribute the task of explaining `2560` examples for the Adult (our test split) with KernelSHAP configured with a background dataset of `100` samples by running the `serve_explanations` script. The configurable options are:

- `-replicas`: controls how many explainer replicas will serve the requests

- `-max_batch_size`: sending a batch of requests as opposed to a single request to one replica can improve performance. Use this argument to optimize the maximum size of a batch of requests sent to each replica. 
- `-benchmark`: if set to `1`, this algorithm will run the experiment over an increasingly large number of replicas. The replicas range is `range(1, -replicas + 1)`. For each number of replicas and each value in `-max_batch_size` the experiment is repeated to obtain runtime averages.
- `-nruns`: controls how many times an experiment with a given `-replicas` setting is run for each value in the `-max_batch_size` array. This allows obtaining the average runtimes for the task. This setting only takes effect only if the option `-benchmark 1` is specified.
