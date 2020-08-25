DOCKER_REPOSITORY ?= alexcoca

POOL_IMAGE_NAME ?= distributedkernelshap
POOL_IMAGE_VERSION ?= 0.1

kernel-shap-image:
	docker build ../ -f Dockerfile.raypool -t ${POOL_IMAGE_NAME}:${POOL_IMAGE_VERSION}
	docker tag ${POOL_IMAGE_NAME}:${POOL_IMAGE_VERSION} ${DOCKER_REPOSITORY}/${POOL_IMAGE_NAME}:${POOL_IMAGE_VERSION}

push-kernel-shap-image: kernel-shap-image
	docker push ${DOCKER_REPOSITORY}/${POOL_IMAGE_NAME}:${POOL_IMAGE_VERSION}
