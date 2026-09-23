# 🧬 CMA-ES – Selbstadaptive Kovarianzmatrix-Stichprobe

Fünftes Stück der **Populations-Metaheuristiken-Linie** der "Konzepte"-Reihe im Portfolio von [Sebastian Hanisch](https://sebastianhanisch.net) –
Operations Research und Machine Learning. Kontrast zu [genetic-algorithm-demo](https://sebastianhanisch-genetic-algorithm-demo.streamlit.app/)
für kontinuierliche Landschaften: CMA-ES (Hansen & Ostermeier) ersetzt GAs Crossover/Mutation durch eine einzige
sich entwickelnde Gauß-Verteilung (Mittelwert $m$, Schrittweite $\sigma$, Kovarianzmatrix $C$), die sich per zwei
Evolutionspfaden an die Form der Landschaft anpasst. Vehikel ist dieselbe kontinuierliche Standortwahl wie
genetic-algorithm-demo - direkt vergleichbar mit deren gemessenen Befunden auf derselben Landschaft.

## Warum dieses Problem

GA arbeitet mit einer über den Raum verstreuten Population, die per Crossover/Mutation weiterentwickelt wird - robust,
aber generisch: dieselben Operatoren gelten für jede Kodierung. CMA-ES ist für **kontinuierliche** Landschaften
spezialisiert und hält stattdessen **eine einzige** sich entwickelnde Gauß-Verteilung. Jede Generation wird daraus eine
Stichprobe gezogen, die besten Punkte bestimmen per gewichteter Rekombination den neuen Mittelwert, und zwei
Evolutionspfade passen Schrittweite und Kovarianzmatrix an - die Verteilung streckt sich in flachen Richtungen, staucht
sich in steilen, dreht sich mit dem Gelände. Auf glatten, unimodalen Landschaften ist das sehr effizient - der
"Goldstandard" für kontinuierliche Black-Box-Optimierung.

## Modell

Dieselbe kontinuierliche Standortwahl wie genetic-algorithm-demo: Kosten eines Punkts $(x, y)$ im 100×100-km-Gebiet
sind mehrere Gauß-Mulden unterschiedlicher Tiefe/Breite (`K_WELLS=5`) - nur die tiefste ist das globale Optimum, der
Rest sind lokale Minima. **`cma_scenario.generate_real` reproduziert die Vehikel-Erzeugung wortgleich** - bei
Standard-Vehikel-Seed 35 bitidentisch zu genetic-algorithm-demo, direkt zitierbare Vergleichszahlen. Der Startpunkt
ist fest die Gebietsmitte - anders als bei GAs zufälliger Startpopulation trägt hier allein der Stichproben-Seed die
Zufälligkeit, nicht der Startpunkt.

## Methodik

Standard-(μ/μ_w, λ)-CMA-ES nach Hansens Tutorial (`cma_algorithm.py`, kein GA-Kern kopiert - andere Mechanik): pro
Generation $\lambda$ Nachkommen $x_k = m + \sigma \, BDz_k$ (Eigenzerlegung $C = BD^2B^\top$), die besten $\mu$
bestimmen per log-linear gewichteter Rekombination den neuen Mittelwert, ein Evolutionspfad $p_\sigma$ passt die
Schrittweite an (kumulative Schrittweiten-Anpassung), ein zweiter $p_c$ plus die Streuung der besten Nachkommen selbst
passen die Kovarianzmatrix an (Rang-1- und Rang-μ-Update). **Ohne** aktive (negative) Kovarianz-Updates und **ohne**
Restart-Strategien (IPOP/BIPOP) - bewusste Vereinfachungen, siehe Grenzen.

**Kreuzprobe gegen `pycma`** (Hansens eigene Referenzimplementierung): Rekombinationsgewichte, μ_eff, $c_c$, $c_1$ und
$\chi_n$ stimmen exakt überein (geprüft über mehrere Dimensionen/Populationsgrößen). $c_\sigma$/$d_\sigma$ und $c_\mu$
weichen bewusst von `pycma`s Standardeinstellung ab - `pycma` nutzt dort neuere, verfeinerte Formeln (Akimoto & Hansen
2020 für die Schrittweiten-Anpassung; einen zusätzlichen "rankmu_offset" für $c_\mu$) statt der in Lehrbüchern
zitierten Original-Tutorial-Formel, der diese Demo folgt - dieselbe Art dokumentierter, bewusster Abweichung wie bei
nsga3-demos pymoo-Achsenabschnitten. Zusätzlich: beide Implementierungen finden auf einfachen konvexen Testfunktionen
(Kugel-, Ellipsoid-Funktion) mit demselben Budget verlässlich nahe an 0.

## Befunde (gemessen, keine Behauptungen)

| Frage | Befund | Test |
|---|---|---|
| Wie schlägt sich CMA-ES gegen GA auf derselben mehrgipfligen Landschaft? | Mit Standard-σ0/λ trifft CMA-ES die globale Mulde nur in **15 %** der Läufe - bei BEIDEN Budgets (1506 und 15096 Auswertungen), GA dagegen in 55 % bzw. 95 %. Mehr Budget hilft CMA-ES hier kaum. | `test_comparison_experiment_headline_claims` |
| Warum hilft mehr Budget nicht? | Sobald σ auf nahe null geschrumpft ist, bleibt die Verteilung in der einmal gefundenen Mulde - weitere Generationen ändern nichts mehr (siehe Wachsendes Beispiel, Generation 20 bis 100 identisch). | im Code nachvollziehbar (`cma_algorithm.run_cmaes`), keine eigene Kennzahl |
| Hilft eine größere Anfangsschrittweite σ0? | Ja: die Trefferquote steigt von 0 % (σ0=2) auf 60 % (σ0=35) über 10 Seeds - eine größere Streuung "sieht" früh genug mehrere Mulden. | `test_sigma0_experiment_headline_claims` |
| Hilft stattdessen eine größere Population λ? | Ja, unabhängig davon: die Trefferquote steigt von 0 % (λ=6) auf 80 % (λ=80) über 5 Seeds - mehr Stichproben je Generation erhöhen ebenfalls die Chance auf die richtige Mulde. | `test_lambda_sweep_headline_claims` |
| Stimmen die Adaptionsraten mit der Literatur überein? | Rekombinationsgewichte, μ_eff, $c_c$, $c_1$, $\chi_n$ exakt wie `pycma`; $c_\sigma$/$d_\sigma$/$c_\mu$ bewusst die ältere Tutorial-Formel (siehe Methodik) | `test_recombination_weights_matches_pycma`, `test_strategy_params_cc_c1_chiN_match_pycma` |

## Ehrliche Grenzen

- **CMA-ES verfolgt nur EINE Gauß-Glocke** - auf einer mehrgipfligen Landschaft bleibt sie leicht im nächstgelegenen
  lokalen Trichter hängen (siehe Kopfexperiment). Das ist keine generelle Schwäche - auf unimodalen/konvex-ähnlichen
  Landschaften, dem üblichen Benchmark-Feld, ist CMA-ES sehr stark (siehe Tests) - sondern eine ehrliche Grenze gerade
  dieser Landschaft ohne Restarts.
- **Zwei unabhängige Regler mildern das** - eine größere Anfangsschrittweite σ0 UND eine größere Population λ helfen
  hier beide, aus unterschiedlichen Gründen (mehr Anfangsstreuung vs. mehr Stichproben je Generation) - beide gemessen,
  keiner behoben das Problem vollständig.
- **Restart-Strategien fehlen** (IPOP-/BIPOP-CMA-ES) - in der Praxis würde man bei erkannter Stagnation mit größerer
  Population neu starten, genau das würde die gemessene Schwäche weiter mildern. Hier bewusst nicht umgesetzt.
- **Aktive (negative) Kovarianz-Updates fehlen** - modernes `pycma` nutzt zusätzlich negative Gewichte für die
  schlechtesten Nachkommen, mit entsprechend feiner abgestimmten Adaptionsraten (siehe Methodik).
- **Unbeschränkter Suchraum** - diese Demo begrenzt die Stichprobe nicht auf das 100×100-Gebiet; die Kostenfunktion
  bleibt auch außerhalb wohldefiniert. In der Praxis bräuchten die meisten Probleme Box-Constraints/Repair.
- **Kein Nachfolger in dieser Demo geplant** - CMA-ES ist ein Geschwister von Differential Evolution und
  Partikelschwarm-Optimierung (beide ebenfalls Kontrast-Kinder von GA für kontinuierliche Landschaften).

## Tests

67 Tests (`pytest tests/ -v`): Rekombinationsgewichte/μ_eff/$c_c$/$c_1$/$\chi_n$ exakt gegen `pycma` geprüft (mehrere
Dimensionen/Populationsgrößen), $c_\sigma$/$d_\sigma$/$c_\mu$ per Handrechnung und als dokumentierte bewusste
Abweichung von `pycma` getestet, Konvergenz auf Kugel-/Ellipsoid-Funktion (eigene Implementierung UND `pycma` im
Vergleich), Szenario-Erzeugung bitidentisch zu genetic-algorithm-demo geprüft, AppTest-Rauchtests (jedes Preset,
Generation-Slider inkl. Abspielen, Permalink-Grenzen, beide Experimente + Sweep auf Abruf) und `test_claims.py` (jede
Zahl aus diesem README, mit CI-robusten Bändern für Einzellauf-Kennzahlen - siehe `feedback_ci_platform_robust_tests.md`,
von Anfang an angewendet statt erst nach einem CI-Fehlschlag).

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Einstiegspunkt |
| `cma_constants.py` | Regler-Grenzen, Vehikel-Konstanten, Presets |
| `cma_presets.py` | Permalink/Presets-Mechanik |
| `cma_scenario.py` | Vehikel-Erzeuger (Standortwahl, Gauß-Mulden-Landschaft), wortgleich zu genetic-algorithm-demo |
| `cma_algorithm.py` | CMA-ES-Kern (Gewichte, Adaptionsraten, Hauptschleife) |
| `cma_evaluation.py` | Kennzahlen, Kopfexperiment, Anfangsschrittweiten-Experiment, Sweep |
| `cma_visualization.py` | Plotly-Abbildungen (Landschaft mit Kovarianz-Ellipse, Vergleiche) |

## Bewusst nicht umgesetzt

- Aktive (negative) Kovarianz-Updates und die neueren Akimoto-&-Hansen-2020-Adaptionsraten - siehe Methodik.
- Restart-Strategien (IPOP-/BIPOP-CMA-ES).
- Box-Constraints/Repair für den Suchraum.
- Ein PDF-Export - wie bei den anderen Konzepte-Demos dieses Portfolios nicht Teil der Linie.

## Lokal ausführen

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements-dev.txt
streamlit run app.py
```

Gebaut mit Streamlit, Plotly und numpy.
