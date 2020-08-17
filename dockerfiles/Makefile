DOCKER REPOSITORY ?= alexcoca/distributedkernelshap

POOL_IMAGE_NAME ?= pooldistributedkernelshap
POOL_IMAGE_VERSION ?= 0.1

pool-kernel-shap-image:
	docker build . -f Dockerfile.raypool -t ${POOL_IMAGE_NAME}:${POOL_IMAGE_VERSION}
	docker tag ${POOL_IMAGE_NAME}:${POOL_IMAGE_VERSION} ${DOCKER_REPOSITORY}/${POOL_IMAGE_NAME}:${POOL_IMAGE_VERSION}

push-pool-kernel-shap-image: pool-kernel-shap-image
	docker push ${DOCKER_REPOSITORY}/${POOL_IMAGE_NAME}:${POOL_IMAGE_VERSION}
