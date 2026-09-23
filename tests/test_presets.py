"""Presets: Vollständigkeit, gültige Werte, Grenzen/Schrittweiten - reine Datenprüfungen ohne Streamlit-Session
(Permalink-Klammern und Preset-Knöpfe werden über AppTest in test_app.py geprüft, wie im Rest des Portfolios üblich)."""

import cma_constants as C
import cma_evaluation as E
import cma_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, p in C.PRESETS.items():
        assert set(p) == set(P.PRESET_KEYS) and C.PRESET_HELP[name]


def test_preset_values_are_valid_and_match_the_setting_specs():
    for name, p in C.PRESETS.items():
        assert C.SIGMA0_MIN <= p["sigma0"] <= C.SIGMA0_MAX
        assert C.GEN_MIN <= p["gens"] <= C.GEN_MAX
        assert C.LAMBDA_MIN <= p["lam"] <= C.LAMBDA_MAX
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            spec.caster(p[key])


def test_default_preset_equals_the_default_settings():
    p = C.PRESETS["Standardfall"]
    s = E.Settings(seed=p["seed"], sigma0=p["sigma0"], gens=p["gens"], lam=p["lam"], run_seed=p["run_seed"])
    assert s == E.Settings()


def test_bounds_and_steps_constants():
    assert P.bounds("sigma0_slider") == (C.SIGMA0_MIN, C.SIGMA0_MAX)
    assert P.bounds("lam_slider") == (C.LAMBDA_MIN, C.LAMBDA_MAX)
    assert P.bounds("seed_input") == (0, C.SEED_MAX)
    assert set(P.STEPS) == {"sigma0_slider", "gens_slider", "lam_slider"}


def test_url_params_are_unique():
    assert len({spec.url_param for spec in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS)


def test_large_population_preset_uses_a_bigger_lambda_than_default():
    assert C.PRESETS["Große Population"]["lam"] > C.DEFAULT_LAMBDA
