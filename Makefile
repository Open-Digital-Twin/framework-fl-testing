
.PHONY: install-poetry
install-poetry:
	curl -sSL https://install.python-poetry.org | POETRY_HOME=/etc/poetry python3 -
	export PATH="/bin/poetry/bin:$PATH"

