import matplotlib.pyplot as plt
import numpy as np


def plot_wealth_distribution(wealth: np.ndarray, bins: int = 50) -> None:
    """Plot the final wealth distribution."""
    plt.figure(figsize=(10, 6))
    plt.hist(wealth, bins=bins, edgecolor="black", alpha=0.8, label="Distribuição final")
    plt.yscale("log")
    plt.xlabel("Riqueza dos agentes", fontsize=12)
    plt.ylabel("Número de agentes (escala log)", fontsize=12)
    plt.title("Distribuição de Riqueza – Modelo Yard Sale", fontsize=14, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()
