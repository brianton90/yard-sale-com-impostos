# Produto

Este projeto é uma simulação econômica baseada em agentes que implementa o **modelo Yard-Sale** — um modelo conhecido em econofísica que demonstra como trocas aleatórias de riqueza entre agentes levam a uma concentração extrema de riqueza (oligarquia), mesmo quando todos os agentes começam com riqueza igual e as trocas são perfeitamente justas. O projeto estuda como a tributação e a redistribuição afetam essa desigualdade (daí o nome "com impostos").

## Modelo

Cada agente tem riqueza e um parâmetro de aversão ao risco. A cada passo de Monte Carlo ocorrem N/2 tentativas de troca. Em cada uma, dois agentes distintos são sorteados, a aposta é o mínimo dos montantes que ambos disponibilizam (regra fair) e o vencedor é escolhido com 50% de probabilidade.

## Regimes de política (`SimulationConfig.regime`)

- `none`: sem imposto (linha de base).
- `wealth`: imposto patrimonial de alíquota `tax_rate` (τ), cobrado de todos os agentes simultaneamente ao fim de cada passo.
- `transaction`: imposto de alíquota `tax_rate` sobre o valor recebido pelo vencedor em cada troca.

Em ambos os regimes com imposto, a arrecadação vai para um **caixa** (`SimulationState.cash`) e é devolvida em partes iguais a todos os agentes ao fim do passo. A riqueza total (agentes + caixa) deve ser conservada.

## Saídas

- Histograma da distribuição final de riqueza e séries temporais dos indicadores, salvos em `Results/`.
- Estatísticas resumidas: coeficiente de Gini, participação do agente mais rico, média, mediana, mínimo e máximo.
- Diagnósticos: erro máximo de conservação, riqueza mínima observada e caixa máximo.
- Séries por passo: Gini, participação máxima, volume transacionado e arrecadação (os dois últimos normalizados por W0).

A descrição formal do modelo está em `metodologia.txt`.
