DOCKER_REPOSITORY ?= alexcoca

IMAGE_NAME ?= distributedkernelshap
IMAGE_VERSION ?= 0.6

kernel-shap-image:
	docker build ../ -f Dockerfile -t ${IMAGE_NAME}:${IMAGE_VERSION}
	docker tag ${IMAGE_NAME}:${IMAGE_VERSION} ${DOCKER_REPOSITORY}/${IMAGE_NAME}:${IMAGE_VERSION}

push-kernel-shap-image: kernel-shap-image
	docker push ${DOCKER_REPOSITORY}/${IMAGE_NAME}:${IMAGE_VERSION}
