"""Kreuzproben gegen `pycma` (Hansens eigene Referenzimplementierung) wo die Formeln exakt übereinstimmen
(Rekombinationsgewichte, μ_eff, c_c, c_1, chiN), Handrechnungen für die restlichen Adaptionsraten, und
End-to-End-Konvergenz auf einfachen Testfunktionen."""

import warnings

import numpy as np
import pytest

import cma as pycma
import cma_algorithm as A

warnings.filterwarnings("ignore", category=UserWarning, module="cma")


def _pycma_sp(dim, lam, seed=1):
    es = pycma.CMAEvolutionStrategy([0.0] * dim, 15.0, {"popsize": lam, "seed": seed, "verbose": -9})
    return es


# --- default_lambda -------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("dim,expected", [(1, 4), (2, 6), (10, 10), (30, 14)])
def test_default_lambda_matches_hand_calculation(dim, expected):
    assert A.default_lambda(dim) == expected


# --- Rekombinationsgewichte / mu_eff - exakt gegen pycma -------------------------------------------------------------------------------------


def test_recombination_weights_matches_hand_calculation():
    weights, mu_eff = A.recombination_weights(mu=3, lam=6)
    raw = np.array([np.log(3.5) - np.log(i) for i in (1, 2, 3)])
    expected_weights = raw / raw.sum()
    assert weights == pytest.approx(expected_weights)
    assert mu_eff == pytest.approx(1.0 / np.sum(expected_weights ** 2))


@pytest.mark.parametrize("lam", [6, 10, 20, 40])
def test_recombination_weights_matches_pycma(lam):
    mu = lam // 2
    weights, mu_eff = A.recombination_weights(mu, lam)
    es = _pycma_sp(2, lam)
    assert mu_eff == pytest.approx(es.sp.weights.mueff)
    assert weights == pytest.approx(np.array(es.sp.weights)[:mu])


# --- Adaptionsraten: c_c, c_1, chiN exakt gegen pycma; c_sigma/d_sigma/c_mu bewusst divergent -------------------------------------------------


@pytest.mark.parametrize("dim,lam", [(2, 6), (3, 10), (5, 20)])
def test_strategy_params_cc_c1_chiN_match_pycma(dim, lam):
    mu = lam // 2
    _, mu_eff = A.recombination_weights(mu, lam)
    p = A.strategy_params(dim, mu_eff)
    es = _pycma_sp(dim, lam)
    assert p["c_c"] == pytest.approx(es.sp.cc)
    assert p["c_1"] == pytest.approx(es.sp.c1)
    assert p["chiN"] == pytest.approx(es.const.chiN)


def test_strategy_params_matches_hand_calculation_dim2_lam6():
    """dim=2, λ=6 (μ=3, μ_eff=2,0286...) - unabhängig nachgerechnete Werte, siehe Moduldoc für die Formeln."""
    _, mu_eff = A.recombination_weights(mu=3, lam=6)
    p = A.strategy_params(dim=2, mu_eff=mu_eff)
    assert p["c_sigma"] == pytest.approx(0.44620498737831715)
    assert p["d_sigma"] == pytest.approx(1.4462049873783172)
    assert p["c_c"] == pytest.approx(0.6245545390268264)
    assert p["c_1"] == pytest.approx(0.1548153998964136)
    assert p["c_mu"] == pytest.approx(0.057859085071916304)
    assert p["chiN"] == pytest.approx(1.254272742818995)


def test_c_sigma_d_sigma_c_mu_deliberately_diverge_from_pycma_defaults():
    """Dokumentiert die bewusste Abweichung (siehe Moduldoc): pycma nutzt neuere Verfeinerungen
    (Akimoto & Hansen 2020 für c_sigma/d_sigma, einen 'rankmu_offset' für c_mu) statt der
    Original-Tutorial-Formel. Kein Bug - c_c/c_1/chiN/Gewichte stimmen exakt überein (siehe Tests oben)."""
    _, mu_eff = A.recombination_weights(mu=3, lam=6)
    p = A.strategy_params(dim=2, mu_eff=mu_eff)
    es = _pycma_sp(2, 6)
    assert p["c_mu"] != pytest.approx(es.sp.cmu)
    assert 0.0 < p["c_mu"] < 1.0 - p["c_1"]


# --- Grenzen/Struktur der Adaptionsraten -----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("dim,lam", [(2, 6), (2, 40), (5, 20), (10, 50)])
def test_strategy_params_are_within_valid_ranges(dim, lam):
    mu = lam // 2
    _, mu_eff = A.recombination_weights(mu, lam)
    p = A.strategy_params(dim, mu_eff)
    assert 1.0 <= mu_eff <= mu
    assert 0.0 < p["c_sigma"] < 1.0
    assert p["d_sigma"] > 0.0
    assert 0.0 < p["c_c"] < 1.0
    assert 0.0 < p["c_1"] < 1.0
    assert 0.0 <= p["c_mu"] <= 1.0 - p["c_1"]
    assert p["chiN"] > 0.0


# --- End-to-End: Konvergenz auf einfachen Testfunktionen (Kugel, elliptisch) --------------------------------------------------------------


def sphere(pop):
    return (np.asarray(pop) ** 2).sum(axis=-1)


def ellipsoid(pop, cond=100.0):
    pop = np.asarray(pop)
    dim = pop.shape[-1]
    scales = cond ** (np.arange(dim) / max(dim - 1, 1))
    return ((pop * scales) ** 2).sum(axis=-1)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_run_cmaes_converges_on_sphere_function(seed):
    dim = 3
    lam = A.default_lambda(dim)
    r = A.run_cmaes(sphere, dim, x0=np.full(dim, 10.0), sigma0=5.0, lam=lam, generations=150, seed=seed)
    assert r.best_fitness < 1e-6
    assert r.best_history[-1] <= r.best_history[0]
    assert np.all(np.diff(r.best_history) <= 1e-12)      # best_history ist monoton fallend (Elitismus)


def test_run_cmaes_converges_on_ellipsoid_function():
    dim = 5
    lam = A.default_lambda(dim)
    r = A.run_cmaes(ellipsoid, dim, x0=np.full(dim, 5.0), sigma0=3.0, lam=lam, generations=300, seed=1)
    assert r.best_fitness < 1e-3


def test_run_cmaes_and_pycma_reach_a_comparably_good_optimum_on_the_sphere():
    """Kein Generation-für-Generation-Gleichlauf (unterschiedliche RNG-Nutzung, siehe Moduldoc) - aber beide
    Implementierungen sollen auf derselben einfachen konvexen Funktion mit demselben Budget nahe an 0 landen."""
    dim = 3
    lam = A.default_lambda(dim)
    r = A.run_cmaes(sphere, dim, x0=np.full(dim, 10.0), sigma0=5.0, lam=lam, generations=200, seed=1)

    es = pycma.CMAEvolutionStrategy([10.0] * dim, 5.0, {"popsize": lam, "seed": 1, "verbose": -9, "maxiter": 200})
    es.optimize(lambda x: float((np.asarray(x) ** 2).sum()))

    assert r.best_fitness < 1e-8
    assert es.result.fbest < 1e-8


def test_run_cmaes_history_shapes_and_generations_snapshots():
    dim = 2
    lam = 8
    gens = 12
    r = A.run_cmaes(sphere, dim, x0=np.array([5.0, 5.0]), sigma0=2.0, lam=lam, generations=gens, seed=3, keep_history=True)
    assert r.best_history.shape == (gens + 1,)
    assert r.mean_history.shape == (gens + 1, dim)
    assert r.sigma_history.shape == (gens + 1,)
    assert len(r.generations) == gens
    for g in r.generations:
        assert g.samples.shape == (lam, dim)
        assert g.B.shape == (dim, dim)
        assert g.D.shape == (dim,)


def test_run_cmaes_without_history_leaves_generations_empty():
    r = A.run_cmaes(sphere, 2, x0=np.array([1.0, 1.0]), sigma0=1.0, lam=6, generations=5, seed=1, keep_history=False)
    assert r.generations == []
