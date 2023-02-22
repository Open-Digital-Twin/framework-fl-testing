VERSION=latest
SCRIPTS_DIR=./scripts
DOCKER_DIR=./docker
CODE_DIR=source
CERTIFICATES_DIR=./.cache/certificates
LOCAL_CONFIG_DIR=./config/local
BUILD_DIR=./.cache/build
SERVER_DIR=./server
CLIENT_DIR=./client

.PHONY: install-poetry
install-poetry:
	curl -sSL https://install.python-poetry.org | POETRY_HOME=/bin/poetry python3 -
	echo 'export PATH="/bin/poetry/bin:$PATH"' >> ~/.bashrc
	source ~/.bashrc
	poetry config virtualenvs.in-project true

.PHONY: install-dependencies-server
install-dependencies-server:
	poetry install -C $(SERVER_DIR)

.PHONY: install-dependencies-client
install-dependencies-client:
	poetry install -C $(CLIENT_DIR)

.PHONY: install-dependencies
install-dependencies: install-dependencies-server install-dependencies-client
	


docker-build-client: ${DOCKER_DIR}/client/Dockerfile certificates
	poetry export -C $(CLIENT_DIR) --without-hashes --format=requirements.txt > ${DOCKER_DIR}/client/requirements.txt
	docker build -t 'fschwanck/fl-framework-client:${VERSION}'  -f ${DOCKER_DIR}/client/Dockerfile .

docker-build-server: ./docker/server/Dockerfile certificates
	poetry export -C $(SERVER_DIR) --without-hashes --format=requirements.txt > ${DOCKER_DIR}/server/requirements.txt
	docker build -t 'fschwanck/fl-framework-server:${VERSION}'  -f ${DOCKER_DIR}/server/Dockerfile .

.PHONY: local-run-client
local-run-client: 
	poetry run -C $(CLIENT_DIR) sh ${CLIENT_DIR}/run.sh ${CONFIG_FILE}

.PHONY: local-run-server
local-run-server: 
	poetry run -C $(SERVER_DIR) sh ${SERVER_DIR}/run.sh ${CONFIG_FILE}


certificates: ${CERTIFICATES_DIR}/ca.crt ${CERTIFICATES_DIR}/ca.key ${CERTIFICATES_DIR}/server.csr ${CERTIFICATES_DIR}/server.key ${CERTIFICATES_DIR}/server.pem
	sh ${SCRIPTS_DIR}/certificates/generate.sh
