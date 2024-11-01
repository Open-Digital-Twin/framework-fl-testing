VERSION=latest
SCRIPTS_DIR=./scripts
DOCKER_DIR=./docker
CODE_DIR=source
CERTIFICATES_DIR=./.cache/certificates
LOCAL_CONFIG_DIR=./config/local
BUILD_DIR=./.cache/build
SERVER_DIR=./server
CLIENT_DIR=./client
DOCKER_USER=fschwanck
KUBERNETES_SERVER=
SSH_USER=

.PHONY: local-run-client local-run-server connect-kubernetes install install-pipenv



docker-build-client: Dockerfile.client certificates
	docker build -t '$(DOCKER_USER)/fl-framework-client:${VERSION}'  -f ./Dockerfile.client .
#	docker push $(DOCKER_USER)/fl-framework-client:${VERSION}

docker-build-server: Dockerfile.server certificates
	docker build -t '$(DOCKER_USER)/fl-framework-server:${VERSION}'  -f ./Dockerfile.server .
#	docker push $(DOCKER_USER)/fl-framework-server:${VERSION}


local-run-client: 
	pipenv run sh ${CLIENT_DIR}/run.sh ${CONFIG_FILE}


local-run-server: 
	pipenv run sh ${SERVER_DIR}/run.sh ${CONFIG_FILE}


connect-kubernetes:
	ssh ${SSH_USER}@${KUBERNETES_SERVER} -L 6443:localhost:6443

certificates: 
	sh ${SCRIPTS_DIR}/certificates/generate.sh

install-pipenv:
	pip install --user pipenv || sudo pacman -S python-pipenv

install: install-pipenv
	mkdir .venv || true
	pipenv install --dev

deploy:
	helm template 
	kubectl 
