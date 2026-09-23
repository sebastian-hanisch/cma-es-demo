"""Jede im README/PRESET_HELP/App genannte Zahl wird hier nachgerechnet - keine Behauptung ohne Test.

Einzelne 80-100-Generationen-Läufe sind chaotisch empfindlich gegenüber winziger Fließkomma-Rundung (siehe
feedback_ci_platform_robust_tests.md, und die eigene Erfahrung aus nsga2-demo/nsga3-demo/moead-demo). Zahlen aus einem
EINZELNEN Lauf (Presets) bekommen deshalb nur Strukturgrenzen; Zahlen, die über mehrere Seeds mitteln (Experimente,
Sweep), sind von Natur aus robuster und dürfen engere (aber weiterhin großzügige) Bänder bekommen."""

import pytest

import cma_constants as C
import cma_evaluation as E


def _preset_analysis(name):
    p = C.PRESETS[name]
    s = E.Settings(seed=p["seed"], sigma0=p["sigma0"], gens=p["gens"], lam=p["lam"], run_seed=p["run_seed"])
    return E.analyse(s, keep_history=False)


# --- Einzelläufe (Presets) - nur Strukturgrenzen, keine Nähe zu einem Messwert ------------------------------------------------------------


def test_standardfall_preset_claims():
    """Einzelner 100-Generationen-Lauf: die konkrete Mulde (getroffen oder nicht) ist plattformabhängig
    fließkomma-empfindlich (siehe Moduldoc) - nur Strukturgrenzen, kein exaktes found_global geprüft."""
    a = _preset_analysis("Standardfall")
    assert -1.0 < a.gap < 50.0
    assert a.result.sigma_history[-1] < 1.0


def test_kleine_anfangsschrittweite_preset_claims():
    a = _preset_analysis("Kleine Anfangsschrittweite")
    assert -1.0 < a.gap < 50.0
    assert a.result.sigma_history[-1] < 1.0


def test_grosse_anfangsschrittweite_preset_claims():
    a = _preset_analysis("Große Anfangsschrittweite")
    assert -1.0 < a.gap < 50.0
    assert a.result.sigma_history[-1] < 1.0


def test_grosse_population_preset_claims():
    a = _preset_analysis("Große Population")
    assert -1.0 < a.gap < 50.0
    assert a.result.sigma_history[-1] < 1.0


# --- Headlinezahlen der beiden Experimente + Sweep (mitteln über mehrere Seeds, robuster) -----------------------------------------------------


def test_comparison_experiment_headline_claims():
    report = E.comparison_experiment()
    assert report["ga_small"] == C.GA_SUCCESS_SMALL == 0.55
    assert report["ga_large"] == C.GA_SUCCESS_LARGE == 0.95
    # Kernbefund: CMA-ES bleibt mit Standard-σ0/λ bei BEIDEN Budgets klar hinter GA zurück - mehr Budget hilft kaum,
    # weil eine einmal kollabierte Verteilung nicht mehr aus ihrer Mulde herausfindet.
    assert report["cma_small"] < report["ga_small"] - 0.2
    assert report["cma_large"] < report["ga_large"] - 0.2
    assert abs(report["cma_large"] - report["cma_small"]) < 0.3     # mehr Budget verändert die Quote nur wenig


def test_sigma0_experiment_headline_claims():
    rows = E.sigma0_experiment()
    by_sigma0 = {r["sigma0"]: r for r in rows}
    assert set(by_sigma0) == set(C.SIGMA0_VALUES)
    for r in rows:
        assert 0.0 <= r["share_global"] <= 1.0
    # Kernbefund: die größte Anfangsschrittweite schneidet klar besser ab als die kleinste.
    assert by_sigma0[C.SIGMA0_VALUES[-1]]["share_global"] > by_sigma0[C.SIGMA0_VALUES[0]]["share_global"]


def test_lambda_sweep_headline_claims():
    rows = E.sweep("lam")
    assert [r["value"] for r in rows] == list(C.SWEEP_VALUES["lam"])
    for r in rows:
        assert 0.0 <= r["share_global"] <= 1.0
    # Kernbefund: größere Populationsgröße erhöht ebenfalls die Trefferquote - ein zweiter, unabhängiger Regler.
    assert rows[-1]["share_global"] > rows[0]["share_global"]
