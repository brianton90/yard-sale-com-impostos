# Yard-Sale com impostos

Simulação baseada em agentes do **modelo Yard-Sale** (econofísica) com **tributação e redistribuição universal**. O código é a parte operacional de um Trabalho de Conclusão de Curso de Engenharia Física. O texto da metodologia, em LaTeX, está em [`metodologia.txt`](metodologia.txt).

## Motivação

No modelo Yard-Sale, agentes que começam com a mesma riqueza fazem trocas aleatórias em que cada lado tem 50% de chance de vencer. Mesmo com trocas justas, a riqueza se concentra em poucos agentes. Este projeto adiciona dois regimes tributários, um sobre a riqueza acumulada e um sobre as transações, ambos com devolução igualitária da arrecadação. O objetivo é comparar o efeito de cada regime sobre a desigualdade (Gini), a concentração no agente mais rico e o volume transacionado.

## O modelo

**População.** `N` agentes (padrão 10 000), todos com riqueza inicial `w_0 = 1`. Cada agente `i` tem um parâmetro de aversão ao risco `β_i`, sorteado de `U[0, 1)` e mantido fixo durante a execução.

**Passo de Monte Carlo.** Um passo corresponde a `N/2` tentativas de troca, executadas em sequência. Cada tentativa:

1. Sorteia dois agentes distintos `i` e `j` de toda a população (sem rede de contatos).
2. Calcula a aposta bruta (regra *fair*), com base no estado imediatamente anterior:
   `a = min{(1 − β_i)·w_i, (1 − β_j)·w_j}`.
3. Sorteia o vencedor com probabilidade 1/2 para cada lado, independentemente de riqueza e aversão.
4. O perdedor perde `a`. O vencedor recebe `(1 − τ_X)·a`. O valor `τ_X·a` vai para o caixa `C`. Sem imposto sobre transações, `τ_X = 0`.

Depois das `N/2` tentativas, o passo continua assim:

1. No regime patrimonial, todos os agentes são tributados ao mesmo tempo: `T_i = τ_W·w_i`, e o total vai para o caixa.
2. O caixa é devolvido em partes iguais, `b = C / N`, e zerado. Esta etapa roda em todos os regimes. Sem imposto, o caixa é zero e a devolução não altera nada.
3. Os indicadores do passo são registrados, depois da redistribuição.

### Regimes tributários

Implementados em [`simulation/rules.py`](simulation/rules.py) (imposto sobre transações) e [`simulation/taxes.py`](simulation/taxes.py) (imposto patrimonial e redistribuição).

| `regime` | Imposto | Onde é aplicado |
|---|---|---|
| `none` | Nenhum. `tax_rate` é ignorada. | |
| `wealth` | `τ_W = tax_rate` sobre a riqueza de cada agente, ao fim de cada passo. | `apply_wealth_tax` |
| `transaction` | `τ_X = tax_rate` sobre o valor recebido pelo vencedor, em cada troca. | `apply_trade` |

Nos regimes `wealth` e `transaction`, `redistribute_cash` devolve toda a arrecadação igualmente a cada passo. Não há custo administrativo, retenção nem redistribuição focalizada.

## Métricas

Definidas em [`simulation/metrics.py`](simulation/metrics.py).

**Resumo final** (`SimulationMetrics`): riqueza total, média, mediana, mínima e máxima; coeficiente de Gini; participação do agente mais rico (`max(w) / Σw`); e três diagnósticos acumulados na execução: erro máximo de conservação, menor saldo individual observado e maior saldo do caixa.

**Séries por passo** (`SimulationSeries`, registradas após a redistribuição):

| Série | Significado |
|---|---|
| `gini` | Gini `G(t)`, pela fórmula com riquezas ordenadas |
| `max_share` | participação do agente mais rico |
| `volume` | volume bruto transacionado no passo, dividido por `W_0` |
| `revenue` | arrecadação do ciclo (o caixa imediatamente antes da devolução), dividida por `W_0` |

O erro de conservação é `|Σw + C − W_0| / W_0`, calculado antes e depois da redistribuição. O código apenas o registra e nunca corrige saldos.

## Estrutura do projeto

```
yard-sale-com-impostos/
├── main.py                  # ponto de entrada: config → simulação → resumo e figuras
├── config.py                # SimulationConfig (dataclass imutável com todos os parâmetros)
├── requirements.txt
├── Dockerfile
├── metodologia.txt          # capítulo de metodologia (LaTeX)
├── simulation/
│   ├── model.py             # SimulationState (mutável) e SimulationHistory
│   ├── initialization.py    # initialize_state()
│   ├── rules.py             # select_agents, compute_stake, decide_winner, apply_trade, yard_sale_step
│   ├── taxes.py             # apply_wealth_tax, redistribute_cash
│   ├── metrics.py           # SimulationMetrics, SimulationSeries, Gini e diagnósticos
│   └── engine.py            # run_simulation(): laço completo
├── visualization/
│   └── plots.py             # histograma e séries temporais (matplotlib)
├── tests/
│   ├── test_rules.py
│   └── test_metrics.py
├── Results/                 # figuras geradas
└── .kiro/                   # steering e specs (ver seção abaixo)
```

## Instalação

Requer Python 3.10 ou superior.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Dependências (`requirements.txt`): `numpy>=1.24`, `matplotlib>=3.7`, `pytest>=7.0`.

## Como executar

```bash
python main.py
```

O script executa uma simulação com a configuração padrão, imprime o resumo no terminal e salva duas figuras em `Results/`. Com os valores padrão (`N = 10 000`, 300 passos), uma execução levou cerca de 55 s em um teste local. O tempo depende da máquina.

`main.py` não tem argumentos de linha de comando. Para mudar a configuração, edite a linha `config = SimulationConfig()` em `main.py`, por exemplo:

```python
config = SimulationConfig(regime="wealth", tax_rate=0.10)
```

### Parâmetros (`SimulationConfig`)

| Parâmetro | Tipo | Padrão | Descrição |
|---|---|---|---|
| `n_agents` | `int` | `10000` | número de agentes `N` |
| `n_steps` | `int` | `300` | passos de Monte Carlo |
| `initial_wealth` | `float` | `1.0` | riqueza inicial de cada agente |
| `seed` | `int` | `42` | semente do gerador de números aleatórios |
| `hist_bins` | `int` | `50` | número de bins do histograma |
| `snapshot_interval` | `int` | `10` | intervalo, em passos, entre snapshots da riqueza (deve ser ≥ 1) |
| `results_dir` | `str` | `"Results"` | pasta de saída das figuras |
| `show_plots` | `bool` | `False` | abre as janelas dos gráficos além de salvar |
| `regime` | `str` | `"none"` | `"none"`, `"wealth"` ou `"transaction"` |
| `tax_rate` | `float` | `0.05` | alíquota `τ` do regime escolhido, em `[0, 1]` |

Um `regime` inválido ou uma `tax_rate` fora de `[0, 1]` levanta `ValueError` ao criar a configuração.

### Saídas

Os arquivos recebem o sufixo `<tag>`: `none` para o regime sem imposto, ou `<regime>_tau<alíquota>` nos demais (por exemplo, `wealth_tau0.05`).

- `Results/distribuicao_riqueza_<tag>.png`: histograma da riqueza final, com eixo y em escala logarítmica.
- `Results/series_temporais_<tag>.png`: Gini, participação do mais rico, volume transacionado e arrecadação por passo.

Os snapshots da riqueza (`result.history`) ficam apenas em memória.

Exemplo para o regime `none` com a configuração padrão:

![Distribuição final de riqueza, regime none](Results/distribuicao_riqueza_none.png)

![Séries temporais, regime none](Results/series_temporais_none.png)

## Experimento comparativo reproduzível

O fluxo experimental separado de `main.py` executa a grade canônica de nove configurações:

- `none`, com alíquota canônica zero;
- `wealth` com `τ ∈ {0.05, 0.10, 0.20, 0.30}`;
- `transaction` com as mesmas quatro alíquotas.

Por padrão, o protocolo usa `R=10`, `N=10000`, 300 passos, janela inclusiva `[150, 300]`, mobilidade com `Δt=50`, snapshots regulares a cada 10 passos e semente-base 2025. A duração, a janela e o intervalo de mobilidade ainda não foram avaliados por pilotos. Por isso, o protocolo registra `stability_status=not_assessed` e todas as saídas são identificadas como **tempo finito**, não como estimativas estacionárias.

Cada realização recebe streams independentes derivados com `SeedSequence.spawn`. Dentro de uma realização, as nove configurações compartilham exatamente os mesmos parâmetros `β`, a condição inicial igualitária, os pares e os vencedores. Uma chave de desempate própria, fixa por agente e gerada em stream separado, é usada apenas na mobilidade por quintis. Os hashes do protocolo e das entradas comuns permitem auditar o pareamento.

```bash
# Experimento científico completo: 90 execuções. Rode apenas explicitamente.
make experiment EXPERIMENT_ID=principal

# Retoma resultados válidos e executa somente o que falta.
python -m experiments.run_experiment simulate --experiment-id principal

# Recria tabelas e figuras exclusivamente dos dados brutos em disco.
python -m experiments.run_experiment analyze --experiment-id principal

# Smoke rápido com 18 execuções pequenas.
make experiment-smoke RESULTS_ROOT=/tmp/yard-sale-experiments
```

O subcomando `all` executa `simulate` e depois `analyze`. A persistência de NPZ e JSON é atômica. Uma segunda chamada pula somente saídas completas cujos IDs e hashes conferem. Um protocolo incompatível no mesmo `experiment_id` é recusado.

Cada experimento fica isolado em `Results/experiments/<experiment_id>/`:

```text
protocol.json
manifest.json
raw/realization_NNN/common_inputs.npz
raw/realization_NNN/<configuration_tag>.npz
tables/series_mean_se.csv
tables/metric_summary.csv e metric_summary.md
tables/paired_differences.csv e paired_differences.md
tables/diagnostics.csv
tables/lorenz_mean_se.csv
tables/mobility_matrices.csv
tables/boundary_ties.csv
figures/series_mean_se.png
figures/metrics_vs_tau.png
figures/lorenz_selected.png
figures/mobility_none.png
figures/mobility_wealth_tau0.05.png
figures/mobility_transaction_tau0.05.png
```

As médias de janela são calculadas primeiro em cada realização. A média final e o erro-padrão usam as realizações como unidades independentes, com `EP = std(ddof=1)/sqrt(R)`. As diferenças `wealth − transaction` são calculadas de forma pareada para cada alíquota. As curvas de Lorenz usam cinco cenários predefinidos. A mobilidade usa os três cenários predefinidos na metodologia, preserva a identidade dos agentes e registra empates nas quatro fronteiras dos quintis.

## Docker

O `Dockerfile` usa `python:3.11-slim`, instala `requirements.txt` e executa `python main.py` com o backend `Agg` do matplotlib, sem janela gráfica. Para recuperar as figuras no host, monte a pasta `Results`:

```bash
docker build -t yard-sale-com-impostos .
docker run --rm -v "$(pwd)/Results:/app/Results" yard-sale-com-impostos
```

## Testes

```bash
make test
```

A suíte cobre regras de troca e impostos, invariantes do motor, métricas, aleatoriedade comum, persistência retomável, estatística, Lorenz, mobilidade e a integração experimental pequena. Nenhum teste executa o experimento científico de 90 execuções.

## Reprodutibilidade

A execução simples mantém o gerador `np.random.default_rng(config.seed)` e a API `run_simulation(config)`. O fluxo experimental torna os números aleatórios comuns explícitos em `common_inputs.npz`, usa streams independentes por realização e valida hashes antes de retomar ou agregar. Todo NPZ é lido com `allow_pickle=False`.

## Diretório `.kiro`

A pasta [`.kiro/`](.kiro) guarda arquivos de apoio do assistente de IA Kiro:

- `.kiro/steering/`: convenções do projeto (stack, estrutura de pastas, descrição do produto).
- `.kiro/specs/end-to-end-taxed-simulation/`: requisitos, design e plano de tarefas da simulação com impostos.

Servem como documentação de design e rastreio das decisões de implementação. Partes do código e da documentação foram desenvolvidas com apoio desse assistente. Os resultados e a interpretação científica são de responsabilidade do autor.

## Licença

Distribuído sob a licença Apache 2.0. Veja o arquivo [`LICENSE`](LICENSE).

## Citação e autoria

- **Autor:** _a preencher_
- **Orientador(a):** _a preencher_
- **Instituição:** _a preencher_
- **Ano:** _a preencher_

```bibtex
@misc{yard_sale_com_impostos,
  author = {{A preencher}},
  title  = {Yard-Sale com impostos: simulação baseada em agentes com tributação e redistribuição},
  year   = {A preencher},
  url    = {https://github.com/brianton90/yard-sale-com-impostos}
}
```
