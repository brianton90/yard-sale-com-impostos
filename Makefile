PYTHON ?= python3
VENV   ?= .venv
BIN    := $(VENV)/bin
EXPERIMENT_ID ?= main
RESULTS_ROOT ?= Results/experiments

.DEFAULT_GOAL := help
.PHONY: help install run test experiment analyze-experiment experiment-smoke clean

help: ## Lista os alvos disponíveis
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-20s %s\n", $$1, $$2}'

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

experiment: install ## Executa ou retoma o experimento científico completo
	MPLBACKEND=Agg $(BIN)/python -m experiments.run_experiment all --experiment-id "$(EXPERIMENT_ID)" --results-root "$(RESULTS_ROOT)"

analyze-experiment: install ## Recria tabelas e figuras dos dados brutos
	MPLBACKEND=Agg $(BIN)/python -m experiments.run_experiment analyze --experiment-id "$(EXPERIMENT_ID)" --results-root "$(RESULTS_ROOT)"

experiment-smoke: install ## Executa uma integração experimental pequena
	MPLBACKEND=Agg $(BIN)/python -m experiments.run_experiment all --experiment-id smoke --results-root "$(RESULTS_ROOT)" --n-realizations 2 --n-agents 20 --n-steps 8 --transient-start 3 --analysis-end 8 --mobility-delta 2 --snapshot-interval 4

clean: ## Remove venv e caches
	rm -rf $(VENV) .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
