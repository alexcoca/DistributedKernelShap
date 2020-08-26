# Distributing KernelSHAP using `ray`

This repository shows how to distribute explanations with KernelSHAP one a single node or a Kubernetes cluster using [`ray`](https://github.com/ray-project/ray). The predictions of a logistic regression model on `2560` instances from the [`Adult`](http://archive.ics.uci.edu/ml/datasets/Adult) dataset are explained using KernelSHAP configured with a background set of `100` samples from the same dataset. The data preprocessing and model fitting steps are available in the `scripts/` folder, but both the data and the model will be automatically downloaded by the benchmarking scripts.

## Distributed KernelSHAP on a single multicore node
### Setup

1. Install [`conda`](https://problemsolvingwithpython.com/01-Orientation/01.05-Installing-Anaconda-on-Linux/)
2. Create a virtual environment with `conda create --name shap python=3.7`
3. Activate the environment with `conda activate shap` 
4. Execute `pip install .` in order to install the dependencies needed to run the benchmarking scripts

### Running the benchmarks

Two code versions are available:

  - One using a parallel pool of `ray` actors, which consume small subsets of the `2560` dataset to be explained
  - One using `ray serve` instead of the parallel pool

The two methods can be run from the repository root, using the scripts `benchmarks/ray_pool.py` and `bechmarks/serve_explanations.py`, respectively. Options that can be configured are:
   - number of actors/replicas that the task is going to be distributed on (e.g., `--workers 5` (pool), `--replicas 5` (ray serve))
   - if a benchmark (i.e., redistributing the task over an increasingly large pool or number of replicas) is to be performed (`-benchmark 0` to disable or `benchmark 1` to enable)
   - the number of times the task is run for the same configuration in benchmarking mode (e.g, `--nruns 5`)
   - how many instances can be sent to an actor/replica at once (this is a required argument) (e.g., `-b 1 5 10` (pool) `-batch 1 5 10` (ray serve)). If more than one value is passed after the argument name, the task (or benchmarking) will be executed for different batch sizes
   
## Distributed KernelShap on a Kubernetes cluster
### Setup 

This requires you to have access to a Kubernetes cluster and have [`kubectl`](https://kubernetes.io/docs/tasks/tools/install-kubectl/) installed. Don't forget to export the path to the cluster configuration `.yaml` file in your `KUBECONFIG` environment variable, as described [here](https://auth0.com/blog/kubernetes-tutorial-step-by-step-introduction-to-basic-concepts/) before moving on to the next steps.

### Running the benchmarks

The `ray_pool.py` and `serve_explanations.py` have been modified to be deployable in the kubernetes and prefixed by `k8s_`. The benchmark experiments can be run via the `bash` scripts in the `benchmarks/` folder. These scripts:

  - Apply the appropriate k8s manifest in `cluster/` to the k8s cluster
  - Upload a `k8s*.py` file to it 
  - Run the script 
  - Pull the results and save them in the `results` directory
  
Specifically:

  - Calling `bash benchmarks/k8s_benchmark_pool.sh 10 20 ` will run the benchmark with increasing number of workers (the cluster is reset as the number of workers is increased). By default the experiment is run with batches of sizes `1 5` and `10`. This can be changed by updating the value of `BATCH` in `cluster/Makefile.pool`
  - Calling `bash benchmarks/k8s_benchmark_serve.sh 10 20 ray` will run the benchmark with increasing number of workers and batch size of `1 5` and `10` for each worker. The batch size setting can be modified from the `.sh` script itself. The `ray` argument means that `ray` is able to batch single requests together and dispatch them to the same worker. If replaced by `default`, minibatches will be distributed to each worker
  
## Sample results
### Single node
The experiments were run on a compute-optimized dedicated machine in Digital Ocean with 32vCPUs. This explains why the performance gains attenuation below.

The results obtained running the task using the `ray` parallel pool are below:

![alt text](https://github.com/alexcoca/DistributedKernelShap/blob/master/images/pool_1_node.PNG?raw=true)

Distributing using ray serve yields similar results:

![alt text](https://github.com/alexcoca/DistributedKernelShap/blob/master/images/serve_1_node.PNG?raw=true)
### Kubernetes cluster
The experiments were run on a cluster consisting of two compute-optimized dedicated machine in Digital Ocean with 32vCPUs each. This explains why the performance gains attenuation below.

The results obtained running the task using the `ray` parallel pool over a two-node cluster are shown below:

![alt text](https://github.com/alexcoca/DistributedKernelShap/blob/master/images/pool_k8s_32.PNG?raw=true)
![alt text](https://github.com/alexcoca/DistributedKernelShap/blob/master/images/pool_k8s_56.PNG?raw=true)

Distributing using ray serve yields similar results:

![alt text](https://github.com/alexcoca/DistributedKernelShap/blob/master/images/serve_k8s_32.PNG?raw=true)
![alt text](https://github.com/alexcoca/DistributedKernelShap/blob/master/images/serve_k8s_56.PNG?raw=true)
