"""Vehikel der CMA-ES-Demo: dieselbe kontinuierliche Standortwahl wie genetic-algorithm-demo (`ga_scenario.generate_real`,
wortgleich kopiert) - x, y im 100 x 100-km-Gebiet, Kosten = mehrere Gauß-Mulden unterschiedlicher Tiefe/Breite. Nur die
tiefste Mulde ist die global günstigste Lage, die anderen sind lokale Minima. Bei Standard-Vehikel-Seed 35 bitidentisch
zu genetic-algorithm-demo - direkt zitierbare Vergleichszahlen."""

from dataclasses import dataclass

import numpy as np

import cma_constants as C


@dataclass(frozen=True)
class RealInstance:
    centres: np.ndarray        # (K, 2)
    amplitudes: np.ndarray     # (K,)
    sigmas: np.ndarray         # (K,)
    offset: float
    seed: int

    def cost(self, xy):
        """Kosten an Punkt(en) `xy` (..., 2) - niedriger ist besser. Beliebige führende Dimensionen (auch keine: ein einzelner Punkt gibt einen Skalar zurück)."""
        xy = np.asarray(xy, dtype=float)
        flat = xy.reshape(-1, 2)
        d2 = ((flat[:, None, :] - self.centres[None, :, :]) ** 2).sum(axis=-1)        # (M, K)
        wells = self.amplitudes[None, :] * np.exp(-d2 / (2.0 * self.sigmas[None, :] ** 2))
        result = self.offset - wells.sum(axis=-1)                                     # (M,)
        return result.reshape(xy.shape[:-1])


def generate_real(seed=0):
    rng = np.random.default_rng(seed)
    centres = C.WELL_MARGIN + rng.random((C.K_WELLS, 2)) * (C.AREA - 2 * C.WELL_MARGIN)
    amplitudes = rng.uniform(C.AMP_MIN, C.AMP_MAX, size=C.K_WELLS)
    sigmas = rng.uniform(C.SIGMA_MIN, C.SIGMA_MAX, size=C.K_WELLS)
    offset = float(amplitudes.sum())
    return RealInstance(centres, amplitudes, sigmas, offset, int(seed))


def grid_optimum(inst, step=C.GRID_STEP):
    """Bester Punkt eines feinen Gitters über das Gebiet - Referenz für "im globalen Trichter gefunden"."""
    xs = np.arange(0.0, C.AREA + step, step)
    gx, gy = np.meshgrid(xs, xs)
    grid = np.stack([gx.ravel(), gy.ravel()], axis=-1)
    costs = inst.cost(grid)
    k = int(np.argmin(costs))
    return grid[k], float(costs[k])
