from simulation.model import SimulationState


def apply_wealth_tax(state: SimulationState, rate: float) -> float:
    """Tributa a riqueza de todos os agentes simultaneamente e envia o total ao caixa.

    Os impostos são calculados a partir do mesmo estado anterior à cobrança,
    então a ordem de processamento dos agentes não altera a arrecadação.
    Devolve o valor arrecadado.
    """
    taxes = rate * state.wealth
    state.wealth -= taxes
    collected = float(taxes.sum())
    state.cash += collected
    return collected


def redistribute_cash(state: SimulationState) -> float:
    """Devolve o caixa igualmente a todos os agentes e zera o caixa.

    Devolve o valor recebido por agente, b = C / N.
    """
    share = state.cash / len(state.wealth)
    state.wealth += share
    state.cash = 0.0
    return share
