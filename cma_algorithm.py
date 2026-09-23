"""CMA-ES-Kern: Standard-(μ/μ_w, λ)-CMA-ES nach Hansens Tutorial ("The CMA Evolution Strategy: A Tutorial", 2016),
OHNE aktive (negative) Kovarianz-Updates und OHNE Restart-Strategien (IPOP/BIPOP) - bewusste Vereinfachungen,
siehe README. Kein GA-Kern kopiert - andere Mechanik (eine sich entwickelnde Gauß-Verteilung statt einer Population
mit Crossover/Mutation).

Formeln für Rekombinationsgewichte, μ_eff, c_c und c_1 stimmen exakt mit der Referenzbibliothek `pycma` überein
(geprüft in tests/test_algorithm.py). c_sigma/d_sigma und c_mu weichen bewusst von `pycma`s Standardeinstellung ab:
`pycma` verwendet dort neuere, verfeinerte Formeln (Akimoto & Hansen 2020 für c_sigma/d_sigma; einen zusätzlichen
"rankmu_offset" für c_mu) statt der in Lehrbüchern zitierten Original-Tutorial-Formeln - dieselbe Art bewusster,
dokumentierter Abweichung wie bei nsga3-demos pymoo-Achsenabschnitten."""

from dataclasses import dataclass, field

import numpy as np


def default_lambda(dim):
    """Standard-Populationsgröße λ = 4 + floor(3 ln n)."""
    return 4 + int(np.floor(3 * np.log(dim)))


def recombination_weights(mu, lam):
    """Log-lineare, positive Gewichte für die besten μ von λ Nachkommen, normiert auf Summe 1, plus μ_eff.
    Exakt geprüft gegen pycma (`cma.CMAEvolutionStrategy(...).sp.weights`)."""
    raw = np.log((lam + 1) / 2.0) - np.log(np.arange(1, mu + 1))
    weights = raw / raw.sum()
    mu_eff = 1.0 / np.sum(weights ** 2)
    return weights, float(mu_eff)


def strategy_params(dim, mu_eff):
    """Adaptionsraten nach Hansens Tutorial. c_c/c_1 exakt wie pycma; c_sigma/d_sigma/c_mu bewusst die
    Original-Tutorial-Formel statt pycmas neuerer Verfeinerung (siehe Moduldoc)."""
    n = float(dim)
    c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
    d_sigma = 1 + 2 * max(0.0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
    c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
    c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
    c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
    chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
    return dict(c_sigma=c_sigma, d_sigma=d_sigma, c_c=c_c, c_1=c_1, c_mu=c_mu, chiN=chiN)


@dataclass
class Generation:
    mean: np.ndarray        # (dim,) Mittelwert VOR der Stichprobe dieser Generation
    sigma: float             # Schrittweite VOR der Stichprobe dieser Generation
    B: np.ndarray            # (dim, dim) Eigenvektoren von C (Spalten)
    D: np.ndarray            # (dim,) Wurzel der Eigenwerte von C
    samples: np.ndarray      # (lam, dim) die λ Nachkommen dieser Generation


@dataclass
class CMAESResult:
    best_individual: np.ndarray
    best_fitness: float
    best_history: np.ndarray       # (generations + 1,) - bester bisher gefundener Wert je Generation (0 = Startpunkt)
    mean_history: np.ndarray       # (generations + 1, dim)
    sigma_history: np.ndarray      # (generations + 1,)
    generations: list = field(default_factory=list)   # nur befüllt, wenn keep_history=True


def run_cmaes(cost_fn, dim, x0, sigma0, lam, generations, seed, keep_history=False):
    """Ein CMA-ES-Lauf. `cost_fn(population) -> (lam,)`, niedriger ist besser. `x0`: Startmittelwert (dim,)."""
    rng = np.random.default_rng(seed)
    mu = lam // 2
    weights, mu_eff = recombination_weights(mu, lam)
    p = strategy_params(dim, mu_eff)
    c_sigma, d_sigma, c_c, c_1, c_mu, chiN = p["c_sigma"], p["d_sigma"], p["c_c"], p["c_1"], p["c_mu"], p["chiN"]

    m = np.array(x0, dtype=float)
    sigma = float(sigma0)
    C = np.eye(dim)
    p_sigma = np.zeros(dim)
    p_c = np.zeros(dim)

    best_x = m.copy()
    best_f = float(cost_fn(m[None, :])[0])
    best_history = [best_f]
    mean_history = [m.copy()]
    sigma_history = [sigma]
    gens_snapshots = []

    for gen in range(1, generations + 1):
        eigvals, B = np.linalg.eigh(C)
        eigvals = np.clip(eigvals, 1e-20, None)    # numerische Absicherung gegen minimal negative Eigenwerte
        D = np.sqrt(eigvals)

        Z = rng.standard_normal((lam, dim))
        Y = Z @ (B * D).T                          # Y[k] = B @ (D * Z[k]), vektorisiert über alle Nachkommen
        X = m[None, :] + sigma * Y

        f = np.asarray(cost_fn(X))
        order = np.argsort(f)
        Y_sorted = Y[order]
        f_sorted = f[order]

        if keep_history:
            gens_snapshots.append(Generation(m.copy(), sigma, B.copy(), D.copy(), X.copy()))

        if f_sorted[0] < best_f:
            best_f = float(f_sorted[0])
            best_x = X[order[0]].copy()

        y_w = weights @ Y_sorted[:mu]
        m = m + sigma * y_w

        C_inv_sqrt = (B / D) @ B.T                  # B diag(1/D) B^T
        p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ y_w)
        sigma = sigma * np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / chiN - 1))

        h_sigma_lhs = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * gen))
        h_sigma = 1.0 if h_sigma_lhs < (1.4 + 2 / (dim + 1)) * chiN else 0.0

        p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w

        delta_h = (1 - h_sigma) * c_c * (2 - c_c)
        rank_mu = np.zeros((dim, dim))
        for i in range(mu):
            rank_mu += weights[i] * np.outer(Y_sorted[i], Y_sorted[i])
        C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + delta_h * C) + c_mu * rank_mu
        C = (C + C.T) / 2.0

        best_history.append(best_f)
        mean_history.append(m.copy())
        sigma_history.append(sigma)

    return CMAESResult(best_x, best_f, np.array(best_history), np.array(mean_history), np.array(sigma_history), gens_snapshots)
