"""Auswertung der CMA-ES-Demo: ein Lauf gegen das Gitter-Optimum, Sweep über λ/σ0, und zwei Experimente -
Kopfexperiment (Trefferquote im globalen Trichter gegen die in genetic-algorithm-demo gemessenen GA-Zahlen) und
eigener Regler (Anfangsschrittweite σ0)."""

from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

import cma_algorithm as A
import cma_constants as C
import cma_scenario as S


@dataclass(frozen=True)
class Settings:
    seed: int = C.DEFAULT_SEED
    sigma0: float = C.DEFAULT_SIGMA0
    gens: int = C.DEFAULT_GEN
    lam: int = C.DEFAULT_LAMBDA
    run_seed: int = C.DEFAULT_RUN_SEED


@lru_cache(maxsize=64)
def instance(seed):
    inst = S.generate_real(seed)
    grid_xy, grid_cost = S.grid_optimum(inst)
    return inst, grid_xy, grid_cost


def cost_fn(settings):
    inst, grid_xy, grid_cost = instance(settings.seed)

    def cost(pop):
        return inst.cost(pop)
    return cost, grid_xy, grid_cost


def run(settings, keep_history=False):
    cost, grid_xy, grid_cost = cost_fn(settings)
    x0 = np.array(C.START_XY)
    return A.run_cmaes(cost, C.DIM, x0, settings.sigma0, settings.lam, settings.gens, settings.run_seed, keep_history=keep_history)


@dataclass
class Analysis:
    settings: Settings
    result: object
    inst: object
    grid_xy: np.ndarray
    grid_cost: float

    @property
    def gap(self):
        if self.grid_cost == 0:
            return float("nan")
        return 100.0 * (self.result.best_fitness - self.grid_cost) / abs(self.grid_cost)

    @property
    def found_global(self):
        return bool(np.sqrt(((self.result.best_individual - self.grid_xy) ** 2).sum()) <= C.GLOBAL_TOL_KM)


def analyse(settings, keep_history=True):
    result = run(settings, keep_history=keep_history)
    inst, grid_xy, grid_cost = instance(settings.seed)
    return Analysis(settings, result, inst, grid_xy, grid_cost)


# --- Sweep (wie die Vorgänger-Demos) ---------------------------------------------------------------------------------------------------------


def run_config(param, value, base, seeds=None):
    seeds = C.SWEEP_SEEDS if seeds is None else seeds
    s0 = replace(base, **{param: value})
    hits, gaps = [], []
    for run_seed in seeds:
        a = analyse(replace(s0, run_seed=run_seed), keep_history=False)
        hits.append(a.found_global)
        gaps.append(a.gap)
    return {"share_global": float(np.mean(hits)), "gap": float(np.mean(gaps))}


def sweep(param, base=None, values=None):
    base = Settings() if base is None else base
    values = C.SWEEP_VALUES[param] if values is None else values
    return [{"value": v, **run_config(param, v, base)} for v in values]


# --- Experiment 1: Kopfexperiment gegen GA-Zahlen (genetic-algorithm-demo) ------------------------------------------------------------------


def comparison_experiment(seed=None, seeds=None, sigma0=None):
    """Trefferquote im globalen Trichter bei zwei Budgets (Gesamtauswertungen ~GA_EVALS_SMALL/LARGE), Standard-λ,
    Standard-σ0 - direkt vergleichbar mit den in genetic-algorithm-demo gemessenen GA-Zahlen auf derselben Landschaft."""
    seed = C.DEFAULT_SEED if seed is None else seed
    seeds = C.COMPARISON_SEEDS if seeds is None else seeds
    sigma0 = C.DEFAULT_SIGMA0 if sigma0 is None else sigma0
    lam = A.default_lambda(C.DIM)

    rows = {}
    for label, evals in (("small", C.GA_EVALS_SMALL), ("large", C.GA_EVALS_LARGE)):
        gens = max(1, evals // lam)
        hits = []
        for run_seed in seeds:
            s = Settings(seed=seed, sigma0=sigma0, gens=gens, lam=lam, run_seed=run_seed)
            a = analyse(s, keep_history=False)
            hits.append(a.found_global)
        rows[label] = {"share_global": float(np.mean(hits)), "gens": gens, "lam": lam, "evals": lam * gens}
    return {
        "cma_small": rows["small"]["share_global"], "cma_large": rows["large"]["share_global"],
        "cma_small_evals": rows["small"]["evals"], "cma_large_evals": rows["large"]["evals"],
        "ga_small": C.GA_SUCCESS_SMALL, "ga_large": C.GA_SUCCESS_LARGE,
        "ga_small_evals": C.GA_EVALS_SMALL, "ga_large_evals": C.GA_EVALS_LARGE,
    }


# --- Experiment 2: eigener Regler - Anfangsschrittweite σ0 ------------------------------------------------------------------------------------


def sigma0_experiment(seed=None, values=None, seeds=None, gens=None):
    seed = C.DEFAULT_SEED if seed is None else seed
    values = C.SIGMA0_VALUES if values is None else values
    seeds = C.SIGMA0_EXPERIMENT_SEEDS if seeds is None else seeds
    gens = C.SIGMA0_EXPERIMENT_GENS if gens is None else gens

    rows = []
    for sigma0 in values:
        hits, final_sigmas = [], []
        for run_seed in seeds:
            s = Settings(seed=seed, sigma0=sigma0, gens=gens, run_seed=run_seed)
            r = run(s, keep_history=False)
            _, grid_xy, _ = instance(seed)
            hit = bool(np.sqrt(((r.best_individual - grid_xy) ** 2).sum()) <= C.GLOBAL_TOL_KM)
            hits.append(hit)
            final_sigmas.append(float(r.sigma_history[-1]))
        rows.append({"sigma0": sigma0, "share_global": float(np.mean(hits)), "sigma_end_median": float(np.median(final_sigmas))})
    return rows
