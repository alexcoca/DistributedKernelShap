NAMESPACE ?= pool-ray-cluster
WORKERS ?= 2
BATCH ?= 1 2 5 10 15 20

SHELL := /bin/bash


.ONESHELL:

deploy:
	kubectl apply -f ray_pool_cluster.yaml
	kubectl rollout status deployment/ray-head -n pool-ray-cluster
	kubectl rollout status deployment/ray-worker -n pool-ray-cluster

destroy:
	kubectl delete -f ray_pool_cluster.yaml

reset:
	kubectl delete -n ${NAMESPACE} pod --all
	kubectl rollout status deployment/ray-head -n pool-ray-cluster
	kubectl rollout status deployment/ray-worker -n pool-ray-cluster

upload-script:
	POD=`kubectl -n ${NAMESPACE} get pod -l component=ray-head -o jsonpath="{.items[0].metadata.name}"`
	kubectl cp -n ${NAMESPACE} ../experiment.py $${POD}:experiment.py

run-experiment:
	POD=`kubectl -n ${NAMESPACE} get pod -l component=ray-head -o jsonpath="{.items[0].metadata.name}"`
	kubectl exec -it -n ${NAMESPACE} $${POD} -- python -W ignore experiment.py -batch ${BATCH} -workers ${WORKERS}

pull-results:
	POD=`kubectl -n ${NAMESPACE} get pod -l component=ray-head -o jsonpath="{.items[0].metadata.name}"`
	kubectl cp ${NAMESPACE}/$${POD}:/distributed_explainers/results ../results/
