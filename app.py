"""CMA-ES - Selbstadaptive Kovarianzmatrix-Stichprobe - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Fünftes Stück der Populations-Metaheuristiken-Linie der "Konzepte"-Reihe, ein KONTRAST zu GA (kein Fix) für
kontinuierliche Landschaften: CMA-ES (Hansen & Ostermeier) ersetzt Crossover/Mutation durch eine einzelne sich
entwickelnde Gauß-Verteilung (Mittelwert, Schrittweite σ, Kovarianzmatrix C), die Form und Ausrichtung an die
Landschaft anpasst. Vehikel ist dieselbe kontinuierliche Standortwahl wie genetic-algorithm-demo - direkt
vergleichbar mit deren gemessenen Befunden auf derselben Landschaft.

Lauffähig mit: streamlit run app.py
"""

import time

import numpy as np
import streamlit as st

import cma_constants as C
from cma_evaluation import Settings, analyse, comparison_experiment, sigma0_experiment, sweep
from cma_presets import apply_preset, bounds, init_session_state_defaults, load_permalink_settings, randomize_run_seed, randomize_seed, sync_query_params
from cma_visualization import build_best_curve, build_comparison, build_growing_example, build_sigma0_experiment, build_sigma_curve, build_sweep

st.set_page_config(page_title="CMA-ES – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings, keep_history=True)


@st.cache_data(show_spinner=False)
def _sweep(param, base):
    return sweep(param, base)


@st.cache_data(show_spinner=False)
def _comparison():
    return comparison_experiment()


@st.cache_data(show_spinner=False)
def _sigma0_experiment():
    return sigma0_experiment()


st.title("🧬 CMA-ES – Selbstadaptive Kovarianzmatrix-Stichprobe")
st.markdown(
    """
GA sucht mit einer Population, die per Crossover/Mutation weiterentwickelt wird. **CMA-ES** (Hansen & Ostermeier) geht
für kontinuierliche Landschaften einen anderen Weg: es hält **eine einzige, sich entwickelnde Gauß-Verteilung** - einen
Mittelwert $m$, eine Schrittweite $\\sigma$ und eine Kovarianzmatrix $C$. Jede Generation wird daraus eine Stichprobe
gezogen, die besten Punkte bestimmen den neuen Mittelwert, und zwei **Evolutionspfade** passen $\\sigma$ und $C$ so an,
dass die Verteilung sich der Form der Landschaft anschmiegt - streckt sich in flachen Richtungen, staucht sich in
steilen, dreht sich mit dem Gelände.
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren vergleichen, zeigt diese Demo - "
    "fünftes Stück der Populations-Metaheuristiken-Linie der \"Konzepte\"-Reihe, ein **Kontrast** zu "
    "[genetic-algorithm-demo](https://sebastianhanisch-genetic-algorithm-demo.streamlit.app/) statt eines Fixes - "
    "**ein** Verfahren an einem wachsenden Beispiel. Vehikel ist dieselbe kontinuierliche Standortwahl wie dort."
)

with st.expander("So funktioniert CMA-ES", expanded=True):
    st.markdown(
        r"""
1. **Stichprobe.** $\lambda$ Nachkommen $x_k = m + \sigma \cdot B D z_k$, $z_k \sim \mathcal{N}(0, I)$ - $B$ und $D$
   kommen aus der Eigenzerlegung $C = BD^2B^\top$.
2. **Rekombination.** Die besten $\mu = \lfloor\lambda/2\rfloor$ Nachkommen bestimmen per gewichtetem Mittel den neuen
   Mittelwert $m$ - schlechtere Nachkommen fließen nicht ein.
3. **Schrittweiten-Anpassung (CSA).** Ein Evolutionspfad $p_\sigma$ verfolgt, ob aufeinanderfolgende Schritte in
   dieselbe Richtung zeigen (dann größer werden) oder sich gegenseitig aufheben (dann kleiner werden).
4. **Kovarianz-Anpassung.** Ein zweiter Evolutionspfad $p_c$ (Rang-1-Update) plus die Streuung der besten Nachkommen
   selbst (Rang-μ-Update) passen Form und Ausrichtung von $C$ an - erfolgreiche Richtungen werden wahrscheinlicher.
5. **Eine einzige Verteilung.** Anders als bei GA gibt es keine verstreute Population - nur eine Gauß-Glocke, die sich
   bewegt, streckt und schrumpft. Das macht CMA-ES sehr effizient auf glatten, unimodalen Landschaften - und, wie das
   Kopfexperiment unten zeigt, verwundbar auf mehrgipfligen.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_names = list(C.PRESETS.keys())
cols = st.columns(len(preset_names))
for col, name in zip(cols, preset_names):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name], key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**CMA-ES**")
    sigma0 = st.slider("Anfangsschrittweite σ0", *bounds("sigma0_slider"), key="sigma0_slider", step=C.SIGMA0_STEP, help="Anfängliche Streuung der Gauß-Verteilung um den Startpunkt (Gebietsmitte).")
    generations = st.slider("Generationen", *bounds("gens_slider"), key="gens_slider", step=C.GEN_STEP)
    lam = st.slider("Populationsgröße λ", *bounds("lam_slider"), key="lam_slider", step=C.LAMBDA_STEP, help=f"Standardformel für 2 Dimensionen: λ = {C.DEFAULT_LAMBDA}.")
    seed = st.number_input("Zufalls-Seed des Vehikels", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neues Vehikel generieren", width="stretch", on_click=randomize_seed)
    run_seed = st.number_input("Zufalls-Seed des CMA-ES-Laufs", *bounds("run_seed_input"), key="run_seed_input", step=1)
    st.button("🎲 Neuen Lauf würfeln", width="stretch", on_click=randomize_run_seed)

sync_query_params({
    "sigma0_slider": float(sigma0), "gens_slider": int(generations), "lam_slider": int(lam),
    "seed_input": int(seed), "run_seed_input": int(run_seed),
})

settings = Settings(seed=int(seed), sigma0=float(sigma0), gens=int(generations), lam=int(lam), run_seed=int(run_seed))
with st.spinner("Rechne..."):
    a = _analysis(settings)
result = a.result
n_gens_run = len(result.generations)
data_key = settings

# --- CMA-ES in Aktion ----------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 CMA-ES in Aktion")
if "cma_gen" not in st.session_state or st.session_state.get("cma_gen_owner") != data_key:
    st.session_state["cma_gen"] = n_gens_run
    st.session_state["cma_gen_owner"] = data_key
gen_col, play_col = st.columns([5, 2])
with gen_col:
    gen = st.slider("Generation", 1, n_gens_run, key="cma_gen")
with play_col:
    auto_play = st.button("▶️ Abspielen", width="stretch")
view_slot = st.empty()


def _frames():
    return sorted({int(round(x)) for x in np.linspace(1, n_gens_run, min(n_gens_run, 40))})


def _render(g):
    gd = result.generations[g - 1]
    with view_slot.container():
        c1, c2 = st.columns([3, 2])
        c1.markdown(f"**Generation {g} von {n_gens_run} – Schrittweite σ: {gd.sigma:.3f}**")
        c1.plotly_chart(build_growing_example(a.inst, gd, a.grid_xy), width="stretch", key=f"g_map_{g}")
        c2.markdown("**Bester Fund bisher**")
        c2.plotly_chart(build_best_curve(result.best_history[:g + 1], reference=a.grid_cost), width="stretch", key=f"g_best_{g}")


if auto_play:
    for f in _frames():
        _render(f)
        time.sleep(0.15)
else:
    _render(gen)

st.markdown("---")

# --- Ergebnis --------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Was CMA-ES gefunden hat")
m1, m2, m3 = st.columns(3)
m1.metric("Im globalen Trichter gelandet?", "Ja" if a.found_global else "Nein")
m2.metric("Abstand zum Gitter-Optimum", f"{a.gap:+.1f} %")
m3.metric("Endgültige Schrittweite σ", f"{result.sigma_history[-1]:.4f}")
st.plotly_chart(build_sigma_curve(result.sigma_history), width="stretch", key="sigma_curve")

st.markdown("---")

# --- Sweep -----------------------------------------------------------------------------------------------------------------------------------

st.subheader("📐 Wie stark hängt die Trefferquote von Populationsgröße und Anfangsschrittweite ab?")
sweep_param = st.selectbox("Welcher Regler soll durchgefahren werden?", list(C.SWEEP_LABELS), format_func=lambda k: C.SWEEP_LABELS[k], key="sweep_select")
base_sweep = Settings(seed=settings.seed, gens=settings.gens, sigma0=settings.sigma0, lam=settings.lam)
if st.button("Sweep über 5 feste Vehikel berechnen (dauert etwa 10 bis 30 Sekunden)", key="sweep_start"):
    st.session_state["sweep_done"] = st.session_state.get("sweep_done", set()) | {(sweep_param, base_sweep)}
if (sweep_param, base_sweep) in st.session_state.get("sweep_done", set()):
    with st.spinner("Rechne den Sweep..."):
        rows_sweep = _sweep(sweep_param, base_sweep)
    st.plotly_chart(build_sweep(rows_sweep, C.SWEEP_LABELS[sweep_param]), width="stretch", key="sweep_chart")

st.markdown("---")

# --- Experiment 1: Kopfexperiment gegen GA ----------------------------------------------------------------------------------------------------

st.subheader("🔬 Wie schlägt sich CMA-ES gegen den genetischen Algorithmus?")
st.caption(
    f"Dieselbe Standortwahl-Landschaft wie genetic-algorithm-demo (Vehikel-Seed {C.DEFAULT_SEED}) - dort traf eine "
    f"Population von 10 die globale Mulde in {C.GA_SUCCESS_SMALL:.0%} von 20 Läufen ({C.GA_EVALS_SMALL} Auswertungen), "
    f"eine Population von 100 in {C.GA_SUCCESS_LARGE:.0%} ({C.GA_EVALS_LARGE} Auswertungen). CMA-ES bekommt dieselben "
    f"beiden Budgets, Standard-λ und Standard-σ0 - Ausgang vorab offen: CMA-ES verfolgt nur EINE Gauß-Glocke statt einer "
    f"verstreuten Population, das könnte auf dieser mehrgipfligen Landschaft auch schlechter abschneiden."
)
if st.button("CMA-ES gegen GA rechnen (dauert etwa 15 Sekunden)", key="comparison_start"):
    st.session_state["comparison_on"] = True
if st.session_state.get("comparison_on"):
    with st.spinner("Rechne 20 CMA-ES-Läufe je Budget..."):
        report = _comparison()
    st.plotly_chart(build_comparison(report), width="stretch", key="comparison_chart")
    c1, c2 = st.columns(2)
    c1.metric(f"CMA-ES, {report['cma_small_evals']} Auswertungen", f"{report['cma_small']:.0%}", delta=f"GA: {report['ga_small']:.0%}", delta_color="off")
    c2.metric(f"CMA-ES, {report['cma_large_evals']} Auswertungen", f"{report['cma_large']:.0%}", delta=f"GA: {report['ga_large']:.0%}", delta_color="off")
    st.warning(
        "**Ehrlicher Befund:** CMA-ES bleibt hier mit Standard-σ0 deutlich hinter GA zurück, und mehr Budget hilft kaum - "
        "sobald σ auf nahe null geschrumpft ist, bleibt CMA-ES in der einmal gefundenen Mulde hängen, egal wie viele "
        "Generationen noch folgen. GAs über den Raum verstreute Population hat dagegen bei jedem Individuum eine eigene "
        "Chance auf die richtige Mulde. Das ist keine generelle Schwäche von CMA-ES (auf unimodalen/konvex-ähnlichen "
        "Landschaften - dem üblichen Benchmark-Feld - ist es sehr stark, siehe Tests), sondern eine ehrliche Grenze "
        "gerade dieser mehrgipfligen Landschaft ohne Restarts (siehe Experiment und Grenzen unten)."
    )

st.markdown("---")

# --- Experiment 2: eigener Regler - Anfangsschrittweite σ0 -------------------------------------------------------------------------------------

st.subheader("🔬 Wie stark hängt die Trefferquote von der Anfangsschrittweite σ0 ab?")
st.caption("Zu klein: die Verteilung „sieht“ die Nachbarmulden nicht und schrumpft, bevor sie sie erreicht. Zu groß: verschwendet frühe Generationen auf breite, uninformative Stichproben.")
if st.button(f"Anfangsschrittweiten {C.SIGMA0_VALUES[0]:.0f} bis {C.SIGMA0_VALUES[-1]:.0f} vergleichen (dauert etwa 10 Sekunden)", key="sigma0_start"):
    st.session_state["sigma0_on"] = True
if st.session_state.get("sigma0_on"):
    with st.spinner("Rechne 5 Anfangsschrittweiten × 10 Läufe..."):
        rows_s = _sigma0_experiment()
    st.plotly_chart(build_sigma0_experiment(rows_s), width="stretch", key="sigma0_chart")

st.markdown("---")

# --- Grenzen -----------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Landschaft hat ein dominantes Optimum in Reichweite** | CMA-ES verfolgt nur EINE Gauß-Glocke - bleibt sie im nächstgelegenen lokalen Trichter hängen, kommt sie nicht mehr heraus, sobald σ geschrumpft ist (siehe Kopfexperiment). | Große Anfangsschrittweite σ0, größere Population λ (beide gemessen), oder Multistart/Restarts |
| **Restarts bei Stagnation** | Ohne sie bleibt ein einmal kollabierter Lauf für immer im gefundenen Trichter, egal wie viel Budget noch folgt. | IPOP-/BIPOP-CMA-ES (Neustart mit wachsender Population) - hier bewusst nicht umgesetzt |
| **Aktive (negative) Kovarianz-Updates** | Modernes `pycma` nutzt zusätzlich negative Gewichte für die schlechtesten Nachkommen (schnellere Anpassung, andere c_σ/d_σ/c_μ-Feinabstimmung) - diese Demo folgt der einfacheren Original-Tutorial-Formel (siehe Tests). | Kein Bug, dokumentierte bewusste Vereinfachung |
| **Unbeschränkter Suchraum** | Diese Demo begrenzt die Stichprobe nicht auf das 100×100-Gebiet - die Kostenfunktion bleibt auch außerhalb wohldefiniert. In der Praxis bräuchten die meisten Probleme Box-Constraints/Repair. | Hier bewusst nicht umgesetzt |
"""
)
st.caption(
    "CMA-ES ist ein Geschwister von Differential Evolution und Partikelschwarm-Optimierung (beide ebenfalls Kontrast-Kinder "
    "von GA für kontinuierliche Landschaften) - kein Nachfolger in dieser Demo geplant. Vorgänger: "
    "[genetic-algorithm-demo](https://sebastianhanisch-genetic-algorithm-demo.streamlit.app/), dessen Befunde hier direkt verglichen werden."
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Stichprobe.** $x_k = m + \sigma \, BDz_k$, $z_k \sim \mathcal{N}(0, I)$, $k = 1, \dots, \lambda$; $C = BD^2B^\top$
(Eigenzerlegung).

**Rekombination.** $m \leftarrow m + \sigma \sum_{i=1}^{\mu} w_i y_{i:\lambda}$, mit $y_{i:\lambda} = (x_{i:\lambda} - m)/\sigma$
der $i$-t-besten von $\lambda$ Nachkommen und log-linearen Gewichten $w_i \propto \ln\!\big(\tfrac{\lambda+1}{2}\big) - \ln i$.

**Schrittweiten-Anpassung.** $p_\sigma \leftarrow (1-c_\sigma) p_\sigma + \sqrt{c_\sigma(2-c_\sigma)\mu_{eff}}\, C^{-1/2} y_w$,
dann $\sigma \leftarrow \sigma \exp\!\Big(\tfrac{c_\sigma}{d_\sigma}\big(\tfrac{\|p_\sigma\|}{\chi_n} - 1\big)\Big)$.

**Kovarianz-Anpassung.** $p_c \leftarrow (1-c_c) p_c + h_\sigma \sqrt{c_c(2-c_c)\mu_{eff}}\, y_w$, dann
$C \leftarrow (1-c_1-c_\mu) C + c_1 \big(p_c p_c^\top + \delta_h C\big) + c_\mu \sum_{i=1}^{\mu} w_i\, y_{i:\lambda} y_{i:\lambda}^\top$.

Implementiert in `cma_algorithm.py` (Gewichte, Adaptionsraten, Hauptschleife), `cma_scenario.py` (Vehikel),
`cma_evaluation.py` (Kennzahlen, Sweep, Experimente).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
