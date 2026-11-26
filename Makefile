PYTHON ?= python3
VENV   ?= .venv
BIN    := $(VENV)/bin
EXPERIMENT_ID ?= main
RESULTS_ROOT ?= Results/experiments
PILOT_ID ?= main
PILOT_RESULTS_ROOT ?= Results/pilots
STABILITY_CONFIG ?= config/stability_margins.example.json
PILOT_R ?= 10
PILOT_N ?= 10000
PILOT_T ?= 300
PILOT_SEED ?= 2025

.DEFAULT_GOAL := help
.PHONY: help install run test experiment analyze-experiment experiment-smoke pilot analyze-pilot pilot-smoke clean

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

pilot: install ## Executa ou retoma o piloto provisório 2T e o analisa
	MPLBACKEND=Agg $(BIN)/python -m experiments.run_pilot all --pilot-id "$(PILOT_ID)" --results-root "$(PILOT_RESULTS_ROOT)" --stability-config "$(STABILITY_CONFIG)" --n-realizations "$(PILOT_R)" --n-agents "$(PILOT_N)" --t-horizon "$(PILOT_T)" --base-seed "$(PILOT_SEED)"

analyze-pilot: install ## Recria a análise do piloto somente dos dados brutos
	MPLBACKEND=Agg $(BIN)/python -m experiments.run_pilot analyze --pilot-id "$(PILOT_ID)" --results-root "$(PILOT_RESULTS_ROOT)"

pilot-smoke: install ## Executa os nove cenários do piloto com R=2, N=20, T=8
	MPLBACKEND=Agg $(BIN)/python -m experiments.run_pilot all --pilot-id "$(PILOT_ID)" --results-root "$(PILOT_RESULTS_ROOT)" --stability-config "$(STABILITY_CONFIG)" --n-realizations 2 --n-agents 20 --t-horizon 8 --base-seed "$(PILOT_SEED)"

clean: ## Remove venv e caches
	rm -rf $(VENV) .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
