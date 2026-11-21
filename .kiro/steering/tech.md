# Stack Tecnológica

## Linguagem e Runtime
- Python 3.10+ (sem uso de `match`, mas utiliza sintaxe moderna de `tuple[...]` e `dataclass`)

## Bibliotecas
- **numpy** `>=1.24` — todos os arrays numéricos e geração de números aleatórios (`np.random.default_rng`)
- **matplotlib** `>=3.7` — histograma da distribuição de riqueza e séries temporais via `pyplot`
- **pytest** `>=7.0` — framework de testes

## Padrões Importantes
- `np.random.default_rng(seed)` é o único gerador de números aleatórios; nunca usar a API legada `np.random.*`
- Toda configuração é centralizada em `SimulationConfig` (dataclass imutável em `config.py`)
- O estado é mutável (`SimulationState`); métricas, séries, histórico, resultado e configuração são imutáveis (frozen dataclasses)
- `SimulationConfig` valida seus campos em `__post_init__` (regime e `tax_rate` em [0, 1])
- Type hints são obrigatórios em todas as assinaturas de função, incluindo o tipo de retorno

## Comandos Comuns

```bash
# Executar a simulação
python main.py

# Executar os testes (execução única, sem modo watch)
pytest

# Executar os testes com saída detalhada
pytest -v

# Executar um arquivo de teste específico
pytest tests/test_rules.py
```
