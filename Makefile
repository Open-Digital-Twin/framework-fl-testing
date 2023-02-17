VERSION=latest

.PHONY: install-poetry
install-poetry:
	curl -sSL https://install.python-poetry.org | POETRY_HOME=/etc/poetry python3 -
	export PATH="/bin/poetry/bin:$PATH"

build-apache-superset: ./docker/apache-superset/Dockerfile
	docker build -t 'fl-apache-superset:${VERSION}'  -f ./docker/apache-superset/Dockerfile .