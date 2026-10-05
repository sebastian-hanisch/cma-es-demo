"""Orakel-Tests (unabhängiger Rechenweg): Adaptionsraten/Rekombinationsgewichte als math-Formeln nach Hansens Tutorial,
und ein "teacher-forced" Nachrechnen jeder Generation: die vom Demo gezogenen Nachkommen (Snapshots) gehen in einen
unabhängigen Update-Schritt (C^(-1/2) per scipy.linalg.sqrtm statt Eigenzerlegung, explizite Schleifen für das Rang-μ-
Update); Mittelwert, σ, Kovarianz (B D² Bᵀ der Folgegeneration) und Bestwert-Verlauf müssen übereinstimmen. Ein freier
Lauf mit festem Zufallsstrom ginge nicht: bei rang-defizitärem Update ist die Eigenbasis des entarteten Raums beliebig."""

import math

import numpy as np
import pytest

import cma_algorithm as A
import cma_constants as C
import cma_scenario as S


def sphere(P):
    return (np.asarray(P) ** 2).sum(axis=-1)


def ellipsoid(P, cond=100.0):
    P = np.asarray(P)
    d = P.shape[-1]
    return ((P * (cond ** (np.arange(d) / max(d - 1, 1)))) ** 2).sum(axis=-1)


def multimodal(P):
    P = np.asarray(P)
    return (np.sin(P) ** 2).sum(axis=-1) * 5 + 0.1 * (P ** 2).sum(axis=-1)


def test_weights_and_rates_match_tutorial_formulas():
    rng = np.random.default_rng(1)
    for _ in range(100):
        n = int(rng.integers(1, 12))
        lam = int(rng.integers(4, 60))
        mu = lam // 2
        raw = [math.log((lam + 1) / 2) - math.log(i) for i in range(1, mu + 1)]
        w = [x / sum(raw) for x in raw]
        mueff = 1 / sum(x * x for x in w)
        weights, me = A.recombination_weights(mu, lam)
        assert weights == pytest.approx(w, abs=1e-12)
        assert me == pytest.approx(mueff, abs=1e-12)
        cs = (mueff + 2) / (n + mueff + 5)
        ds = 1 + 2 * max(0, math.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        p = A.strategy_params(n, mueff)
        for key, ref in (("c_sigma", cs), ("d_sigma", ds), ("c_c", cc), ("c_1", c1), ("c_mu", cmu)):
            assert p[key] == pytest.approx(ref, abs=1e-12)
        exact_chi = math.sqrt(2) * math.exp(math.lgamma((n + 1) / 2) - math.lgamma(n / 2))   # E||N(0,I)||
        assert p["chiN"] == pytest.approx(exact_chi, rel=0.03)


def test_every_generation_matches_independent_update_step():
    sl = pytest.importorskip("scipy.linalg")
    rng = np.random.default_rng(2)
    funcs = [sphere, ellipsoid, multimodal]
    for t in range(30):
        n = int(rng.integers(1, 6))
        lam = int(rng.integers(4, 20))
        gens = int(rng.integers(3, 15))
        f = funcs[t % 3]
        x0 = rng.normal(0, 3, n)
        sigma0 = float(rng.uniform(0.3, 5))
        res = A.run_cmaes(f, n, x0, sigma0, lam, gens, int(rng.integers(0, 10 ** 6)), keep_history=True)
        mu = lam // 2
        w, mueff = A.recombination_weights(mu, lam)
        cs = (mueff + 2) / (n + mueff + 5)
        ds = 1 + 2 * max(0, math.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        chi = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        Cm = np.eye(n)
        ps = np.zeros(n)
        pc = np.zeros(n)
        best = float(f(np.array(x0)[None, :])[0])
        for g, gen in enumerate(res.generations):
            m, sg, X = gen.mean, gen.sigma, gen.samples
            assert gen.B @ np.diag(gen.D ** 2) @ gen.B.T == pytest.approx(Cm, rel=1e-6, abs=1e-9)
            assert gen.B.T @ gen.B == pytest.approx(np.eye(n), abs=1e-9)
            Y = (X - m) / sg
            fx = np.asarray(f(X))
            idx = sorted(range(lam), key=lambda i: fx[i])
            best = min(best, float(fx[idx[0]]))
            yw = sum(w[i] * Y[idx[i]] for i in range(mu))
            m_new = sum(w[i] * X[idx[i]] for i in range(mu))
            ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) * np.linalg.inv(sl.sqrtm(Cm).real) @ yw
            sg_new = sg * math.exp(cs / ds * (np.linalg.norm(ps) / chi - 1))
            hs = 1.0 if np.linalg.norm(ps) / math.sqrt(1 - (1 - cs) ** (2 * (g + 1))) < (1.4 + 2 / (n + 1)) * chi else 0.0
            pc = (1 - cc) * pc + hs * math.sqrt(cc * (2 - cc) * mueff) * yw
            Cn = (1 - c1 - cmu) * Cm + c1 * (np.outer(pc, pc) + (1 - hs) * cc * (2 - cc) * Cm)
            for i in range(mu):
                Cn = Cn + cmu * w[i] * np.outer(Y[idx[i]], Y[idx[i]])
            Cm = (Cn + Cn.T) / 2
            assert res.mean_history[g + 1] == pytest.approx(m_new, rel=1e-8, abs=1e-10)
            assert res.sigma_history[g + 1] == pytest.approx(sg_new, rel=1e-8)
            assert res.best_history[g + 1] == pytest.approx(best, rel=1e-9, abs=1e-9)


def test_sample_covariance_matches_B_D_squared_Bt():
    res = A.run_cmaes(sphere, 3, np.zeros(3), 1.0, 20000, 2, 5, keep_history=True)
    for gen in res.generations:
        Y = (gen.samples - gen.mean) / gen.sigma
        assert np.cov(Y.T) == pytest.approx(gen.B @ np.diag(gen.D ** 2) @ gen.B.T, abs=0.05)


def test_landscape_cost_and_grid_optimum_match_plain_python():
    for seed in range(5):
        inst = S.generate_real(seed)
        rng = np.random.default_rng(seed)
        for pt in rng.random((5, 2)) * 100:
            ref = inst.offset - sum(a * math.exp(-((pt[0] - cx) ** 2 + (pt[1] - cy) ** 2) / (2 * s * s))
                                    for (cx, cy), a, s in zip(inst.centres, inst.amplitudes, inst.sigmas))
            assert float(inst.cost(pt)) == pytest.approx(ref, abs=1e-10)
        step = 5.0
        gxy, gc = S.grid_optimum(inst, step=step)
        pts = [(x * step, y * step) for x in range(int(C.AREA / step) + 1) for y in range(int(C.AREA / step) + 1)]
        costs = [float(inst.cost(np.array(p))) for p in pts]
        assert gc == pytest.approx(min(costs), abs=1e-10)
        assert tuple(gxy) == pts[costs.index(min(costs))]
