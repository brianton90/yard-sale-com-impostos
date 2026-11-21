# Estrutura do Projeto

```
yard-sale-com-impostos/
├── main.py                  # Ponto de entrada: conecta config → simulação → visualização
├── config.py                # Frozen dataclass SimulationConfig (fonte única de verdade para parâmetros)
├── requirements.txt         # Dependências Python
├── metodologia.txt          # Descrição formal do modelo e dos regimes de imposto
├── README.md
├── Dockerfile / .dockerignore
├── Results/                 # Figuras geradas (PNG); versionar apenas o .gitkeep
│
├── simulation/
│   ├── model.py             # SimulationState (riqueza, aversão, caixa) e SimulationHistory (snapshots)
│   ├── initialization.py    # initialize_state() — constrói o SimulationState inicial a partir da config
│   ├── rules.py             # Lógica de troca: select_agents, compute_stake, decide_winner, apply_trade, yard_sale_step
│   ├── taxes.py             # apply_wealth_tax (imposto patrimonial) e redistribute_cash (devolução igualitária do caixa)
│   ├── metrics.py           # SimulationMetrics, SimulationSeries + compute_gini, compute_max_share, compute_conservation_error, summarize_wealth
│   └── engine.py            # run_simulation() — orquestra o loop completo e devolve SimulationResult
│
├── visualization/
│   └── plots.py             # plot_wealth_distribution() e plot_time_series() com matplotlib
│
└── tests/
    ├── test_rules.py        # Testes unitários da mecânica de troca (conservação de riqueza, cálculo da aposta)
    └── test_metrics.py      # Testes unitários das propriedades do coeficiente de Gini
```

## Regras de Arquitetura

- **Separação de responsabilidades**: `rules.py` contém a lógica de troca; `taxes.py` contém a tributação e a redistribuição; `engine.py` é dono do loop; `metrics.py` é apenas computação sem estado.
- **Sem importações entre camadas**: `visualization` não deve importar internos de `simulation` além do que `engine.py` retorna. Módulos de `simulation` não devem importar de `visualization`.
- **Config flui para baixo**: `SimulationConfig` é criado uma única vez em `main.py` e passado para as funções — nunca instanciado dentro dos módulos de simulação. Os módulos de regras e impostos recebem apenas valores (ex.: alíquotas), não a config.
- **Funções puras são preferidas**: `compute_stake`, `decide_winner` etc. devem ser puras. Só mutam o estado `apply_trade` (e `yard_sale_step`, que o chama), `apply_wealth_tax` e `redistribute_cash`.
- **Novos mecanismos de simulação** (ex.: novas políticas fiscais) pertencem a `simulation/rules.py`, a `simulation/taxes.py` ou a um novo módulo em `simulation/` — não em `engine.py`, que apenas os orquestra. Novos regimes também exigem atualizar `REGIMES` em `config.py`.
- **Conservação**: toda mudança de regra deve preservar `soma(riqueza) + caixa = W0`. O engine mede o desvio em `max_conservation_error`.
- **Novas métricas** pertencem a `simulation/metrics.py` e devem ser adicionadas a `SimulationMetrics` (valores finais) ou `SimulationSeries` (por passo).
- **Testes espelham a estrutura de módulos**: testes de `simulation/rules.py` vão em `tests/test_rules.py`, e assim por diante (ex.: `taxes.py` → `tests/test_taxes.py`).
