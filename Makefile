.DEFAULT_GOAL := help
.PHONY: help logs test stop build up install setup find admin

help:
	@perl -nle'print $& if m{^[a-zA-Z_-]+:.*?## .*$$}' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-25s\033[0m %s\n", $$1, $$2}'

install: ## install the package and dev dependencies (editable)
	pip install uv 2>/dev/null || true
	uv pip install -e ".[dev]"

setup: install ## install + migrate (the CRM bootstraps itself on the first run)
	python manage.py migrate --no-input

find: ## find leads: make find N=10 [UNIT=emails]
	python manage.py find $(or $(N),1) $(UNIT)

test: ## run the test suite
	pytest

admin: ## start the Django Admin web server
	@echo ""
	@echo "  Django Admin: http://localhost:8000/admin/"
	@echo "  No superuser yet? Run: python manage.py createsuperuser"
	@echo ""
	python manage.py runserver

# Docker targets — the server deploy only (docs/infrastructure.md §7).
# Development and tests run natively; there is no docker-test.
logs: ## follow the logs of the service
	docker compose -f local.yml logs -f

stop: ## stop all services defined in Docker Compose
	docker compose -f local.yml stop

build: ## build all services defined in Docker Compose
	docker compose -f local.yml build

up: ## run the defined service in Docker Compose
	docker compose -f local.yml up --build -d
	docker compose -f local.yml logs -f
