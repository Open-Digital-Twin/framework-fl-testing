VERSION=latest
SCRIPTS_DIR=./scripts
DOCKER_DIR=./docker
CODE_DIR=./code
CERTIFICATES_DIR=./certificates

.PHONY: install-poetry
install-poetry:
	curl -sSL https://install.python-poetry.org | POETRY_HOME=/bin/poetry python3 -
	echo 'export PATH="/bin/poetry/bin:$PATH"' >> ~/.bashrc
	source ~/.bashrc


docker-build-client: ${DOCKER_DIR}/client/Dockerfile
	docker build -t 'fl-client:${VERSION}'  -f ${DOCKER_DIR}/client/Dockerfile .

docker-build-server: ./docker/server/Dockerfile
	docker build -t 'fl-server:${VERSION}'  -f ${DOCKER_DIR}/server/Dockerfile .

.PHONY: local-run-client
local-run-client: 
	poetry shell python3.10 ${CODE_DIR}/client/main.py

.PHONY: local-run-server
local-run-server: 
	poetry shell python3.10  ${CODE_DIR}/client/server.py


certificates: ${CERTIFICATES_DIR}/ca.crt ${CERTIFICATES_DIR}/ca.key ${CERTIFICATES_DIR}/server.csr ${CERTIFICATES_DIR}/server.key ${CERTIFICATES_DIR}/server.pem
	sh ${SCRIPTS_DIR}/certificates/generate.sh
