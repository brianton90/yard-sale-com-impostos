PYTHON ?= python3
VENV   ?= .venv
BIN    := $(VENV)/bin

.DEFAULT_GOAL := help
.PHONY: help install run test clean

help: ## Lista os alvos disponíveis
	@grep -E '^[a-z]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-10s %s\n", $$1, $$2}'

STAMP := $(VENV)/.installed

$(STAMP): requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install -r requirements.txt
	@touch $(STAMP)

install: $(STAMP) ## Cria o venv e instala as dependências

run: install ## Executa a simulação (gera gráficos em Results/)
	MPLBACKEND=Agg $(BIN)/python main.py

test: install ## Roda os testes com pytest
	$(BIN)/python -m pytest tests -q

clean: ## Remove venv e caches
	rm -rf $(VENV) .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
