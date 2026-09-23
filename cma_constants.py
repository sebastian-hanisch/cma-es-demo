"""Konstanten der CMA-ES-Demo: Vehikel (wie genetic-algorithm-demo, kontinuierliche Standortwahl), CMA-ES-Regler,
Presets (Presets folgen nach den Messungen)."""

# --- Vehikel: Standortwahl (wortgleich aus genetic-algorithm-demo/ga_constants.py) -------------------------------------------------------

AREA = 100.0
K_WELLS = 5
WELL_MARGIN = 12.0                # Schwerpunkte liegen mindestens so weit vom Rand entfernt
AMP_MIN, AMP_MAX = 15.0, 40.0     # Tiefe eines Trichters
SIGMA_MIN, SIGMA_MAX = 6.0, 14.0  # Breite eines Trichters
GRID_STEP = 1.0                   # Auflösung des Referenz-Gitters (Grid-Search-Minimum) in km

DIM = 2
START_XY = (AREA / 2.0, AREA / 2.0)   # fester Startpunkt (Gebietsmitte) - wie bei GAs zufälliger Startpopulation trägt die
                                       # Zufälligkeit hier allein der Stichproben-Seed, nicht der Startpunkt

# --- CMA-ES --------------------------------------------------------------------------------------------------------------------------------

SIGMA0_MIN, SIGMA0_MAX, DEFAULT_SIGMA0, SIGMA0_STEP = 1.0, 40.0, 15.0, 1.0
GEN_MIN, GEN_MAX, DEFAULT_GEN, GEN_STEP = 10, 400, 100, 10
LAMBDA_MIN, LAMBDA_MAX, DEFAULT_LAMBDA, LAMBDA_STEP = 6, 100, 6, 2   # DEFAULT_LAMBDA = default_lambda(DIM) = 4 + floor(3 ln 2) = 6
                                                                      # (siehe cma_algorithm.default_lambda) - zugleich die in der
                                                                      # Literatur übliche Untergrenze (darunter bräuchte c1 eine
                                                                      # zusätzliche Kleinpopulations-Korrektur, hier nicht umgesetzt)
SEED_MAX = 999999
DEFAULT_SEED = 35                 # Vehikel-Seed (wie genetic-algorithm-demo, bitidentische Landschaft)
DEFAULT_RUN_SEED = 7              # Seed des CMA-ES-Laufs selbst (Stichprobe)

GLOBAL_TOL_KM = 3.0                # Standort gilt als "im globalen Trichter" gefunden, wenn er höchstens so weit vom besten Gitterpunkt entfernt liegt (wie genetic-algorithm-demo)

# --- Kopfexperiment: CMA-ES gegen die in genetic-algorithm-demo gemessenen GA-Zahlen ------------------------------------------------------
# GA maß auf DERSELBEN Landschaft (Vehikel-Seed 35): Populationsgröße 10 (bei 150 Generationen, insgesamt 1510 Auswertungen)
# trifft die globale Mulde in 55 % von 20 Läufen, Populationsgröße 100 (15100 Auswertungen) in 95 %.

GA_SUCCESS_SMALL, GA_EVALS_SMALL = 0.55, 1510
GA_SUCCESS_LARGE, GA_EVALS_LARGE = 0.95, 15100
COMPARISON_SEEDS = tuple(range(1300000, 1300020))    # 20 Lauf-Seeds, wie GAs CONVERGENCE_SEEDS-Umfang

# --- Eigener Regler: Anfangsschrittweite σ0 -------------------------------------------------------------------------------------------------

SIGMA0_VALUES = (2.0, 5.0, 10.0, 20.0, 35.0)
SIGMA0_EXPERIMENT_SEEDS = tuple(range(1400000, 1400010))
SIGMA0_EXPERIMENT_GENS = 80

SWEEP_SEEDS = tuple(range(1500000, 1500005))
SWEEP_VALUES = {"lam": (6, 10, 20, 40, 80), "sigma0": SIGMA0_VALUES}
SWEEP_LABELS = {"lam": "Populationsgröße λ", "sigma0": "Anfangsschrittweite σ0"}


def _preset(sigma0=DEFAULT_SIGMA0, gens=DEFAULT_GEN, lam=DEFAULT_LAMBDA, seed=DEFAULT_SEED, run_seed=DEFAULT_RUN_SEED):
    return {"sigma0": sigma0, "gens": gens, "lam": lam, "seed": seed, "run_seed": run_seed}


PRESETS = {
    "Standardfall": _preset(),
    "Kleine Anfangsschrittweite": _preset(sigma0=SIGMA0_VALUES[0]),
    "Große Anfangsschrittweite": _preset(sigma0=SIGMA0_VALUES[-1]),
    # run_seed=2 per Suche gewählt (wie bei genetic-algorithm-demos "Kleine Population"-Preset): zeigt den Erfolgsfall,
    # nicht jeder Seed bei λ=40 findet die globale Mulde (siehe PRESET_HELP und das Sweep-Experiment für die Rate über mehrere Seeds).
    "Große Population": _preset(lam=40, run_seed=2),
}
# Standard-λ (default_lambda(2) = 6) ist bereits die in der Literatur übliche Untergrenze (siehe LAMBDA_MIN) - eine
# "kleine Population"-Variante darunter gibt es hier bewusst nicht, anders als bei genetic-algorithm-demo.
PRESET_HELP = {
    "Standardfall": "σ0=15, λ=6, 100 Generationen (Standard-Seed): die Verteilung kollabiert auf die NÄCHSTGELEGENE Mulde, nicht die tiefste - 4,3 % über dem globalen Optimum. σ schrumpft von 15 auf 0,00003.",
    "Kleine Anfangsschrittweite": "σ0=2 statt 15: dieselbe Mulde wie im Standardfall wird getroffen (4,3 % über dem Optimum) - die Verteilung war von Anfang an zu eng, um die Nachbarmulden überhaupt zu sehen.",
    "Große Anfangsschrittweite": "σ0=35 statt 15: die Verteilung sieht früh genug mehrere Mulden und findet hier die global günstigste Lage fast exakt (-0,02 % Abstand). Über mehrere Seeds gemittelt steigt die Trefferquote von 0 % (σ0=2) auf 60 % (σ0=35, siehe Experiment).",
    "Große Population": "λ=40 statt 6, sonst wie im Standardfall (Seed per Suche gewählt, zeigt den Erfolgsfall): mehr Stichproben je Generation erhöhen ebenfalls die Chance auf die richtige Mulde (hier: gefunden, -0,02 % Abstand) - nicht jeder Seed gelingt, im Sweep über mehrere Seeds steigt die Trefferquote aber von 0 % (λ=6) auf 60 % (λ=40).",
}
