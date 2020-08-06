# Running distributed KernelSHAP

To create a virtual environment that allows you to run KernelSHAP in a distributed fashion with [`ray`](https://github.com/ray-project/ray) you need to configure your environment first, which requires [`conda`](https://problemsolvingwithpython.com/01-Orientation/01.05-Installing-Anaconda-on-Linux/) to be installed. You can then run the command::

`conda env create -f environment.yml -p /home/user/anaconda3/envs/env_name`

to create the environment and then activate it with `conda activate shap`. If you don not wish to change the installation path then you can skip the `-p` option. You are now ready to run the experiments. The steps involved are:

1. data processing 
2. running the experiments

To process the data it is sufficient to run `python preprocess_data.py` with the default options. This will output a preprocessed version of the [`Adult`](http://archive.ics.uci.edu/ml/datasets/Adult) dataset and a partition of it that is used to initialise the KernelSHAP explainer. However, you can proceed to step 2 if you don't intend to change the default parameters as the same data will be automatically downloaded.

You can run an experiment with the command `python experiment.py`. By default, this will run the explainer on the `2560` examples from the `Adult` dataset with a background dataset with `100` samples, sequentially (5 times if the `-benchmark 1` option is passed to it). The resuults are saved in the `results/` folder. If you wish to run the same explanations in parallel, then run the command

`python experiment.py -cores 3`

which will use `ray` to perform explanations across multiple cores.

Other options for the script are:

- `-benchmark`: if set to 1, `-cores` will be treated as the upper bound of number of cores to compute the explanations on. The lower bound is `2`, and the explanations are computed 5 times (by default) to provide runtime averages. The number of repetitions can be controlled using the `-nruns` argument.
- `-batch_size`: controls how many instances are explained by a core at once. This parameter has an important bearing to the code runtime performance
