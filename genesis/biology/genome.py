"""Genome layout and its mapping to phenotype.

Genes are floats in [0, 1]. A separate block of genes encodes the weights of
the organism's small neural decision network (see intelligence.brain).
"""
from __future__ import annotations

import numpy as np

from ..physics import laws as L

TRAITS = ["size", "speed", "sense", "metab_eff", "diet", "temp_opt", "aquatic",
          "repro_thresh", "mut_rate", "sociality", "curiosity", "aggression",
          "learn_rate", "brain", "lifespan", "parental", "sexual", "communication",
          "fear_sens", "patience", "hue"]
T = {n: i for i, n in enumerate(TRAITS)}
N_TRAITS = len(TRAITS)
FOOD_PREF0 = N_TRAITS                  # innate food preferences, one per plant type
G = N_TRAITS + L.N_PLANTS

# neural network shape
N_IN = 16
N_HID = 8
N_OUT = L.N_ACTIONS
N_W = N_IN * N_HID + N_HID * N_OUT + N_OUT


def advanced_genomes(rng: np.random.Generator, n: int) -> np.ndarray:
    """Experimental starting point: social, curious, larger-brained omnivores.
    Still no skills, words, norms or knowledge — only the capacity to learn them."""
    g = random_genomes(rng, n)
    for name, lo, hi in [("size", 0.5, 0.7), ("brain", 0.55, 0.8), ("sociality", 0.6, 0.9), ("communication", 0.5, 0.8),
                         ("curiosity", 0.5, 0.9), ("sexual", 0.85, 1.0), ("lifespan", 0.6, 0.9), ("parental", 0.6, 0.9),
                         ("diet", 0.25, 0.45), ("learn_rate", 0.5, 0.9), ("aquatic", 0.0, 0.1), ("sense", 0.4, 0.7),
                         ("mut_rate", 0.1, 0.3), ("temp_opt", 0.5, 0.65)]:
        g[:, T[name]] = rng.uniform(lo, hi, n)
    # one founding population: low within-population variance in the remaining traits
    for k in ("speed", "metab_eff", "repro_thresh", "aggression", "fear_sens", "patience", "hue"):
        g[:, T[k]] = np.clip(rng.normal(0.5, 0.08, n), 0, 1)
    return g.astype(np.float32)


def random_genomes(rng: np.random.Generator, n: int) -> np.ndarray:
    g = rng.random((n, G)).astype(np.float32)
    # primitive organisms: small, simple-brained, mostly herbivorous, asexual
    g[:, T["size"]] = rng.uniform(0.15, 0.45, n)
    g[:, T["brain"]] = rng.uniform(0.0, 0.15, n)
    g[:, T["diet"]] = rng.beta(1.2, 5.0, n)
    g[:, T["sexual"]] = rng.uniform(0.0, 0.4, n)
    g[:, T["communication"]] = rng.uniform(0.0, 0.2, n)
    g[:, T["aquatic"]] = rng.beta(1.0, 4.0, n)
    g[:, T["temp_opt"]] = rng.uniform(0.45, 0.75, n)
    g[:, T["mut_rate"]] = rng.uniform(0.2, 0.5, n)
    g[:, FOOD_PREF0:] = rng.uniform(0.45, 0.65, (n, L.N_PLANTS))
    return np.clip(g, 0, 1)


class Phenotype:
    """Vectorised genotype -> phenotype mapping."""

    @staticmethod
    def body(g):            # structural energy locked in the body
        return 0.5 + 1.5 * g[:, T["size"]]

    @staticmethod
    def capacity(g):        # max energy store
        return 2.0 + 4.0 * g[:, T["size"]]

    @staticmethod
    def speed(g):           # cells / hour
        return (0.35 + 1.3 * g[:, T["speed"]]) * (1 - 0.35 * g[:, T["metab_eff"]]) * (1.1 - 0.4 * g[:, T["size"]])

    @staticmethod
    def sense(g):           # perception radius in cells
        return 1.5 + 5.0 * g[:, T["sense"]]

    @staticmethod
    def temp_opt(g):        # degrees C
        return -8.0 + 45.0 * g[:, T["temp_opt"]]

    @staticmethod
    def max_age(g):         # hours
        return 24.0 * (12.0 + 70.0 * g[:, T["lifespan"]]) * (0.8 + 0.4 * g[:, T["size"]])

    @staticmethod
    def maturity(g):        # fraction of max_age; big brains need longer childhoods
        return 0.12 + 0.16 * g[:, T["brain"]]

    @staticmethod
    def basal(g):
        b = Phenotype.body(g)
        return (L.BASAL_COST * b ** 0.75 * (1 + L.BRAIN_COST * g[:, T["brain"]] ** 1.5)
                * (1.25 - 0.5 * g[:, T["metab_eff"]]) * (1 + 0.6 * g[:, T["sense"]] ** 2) * (1 + 0.3 * g[:, T["speed"]] ** 2))

    @staticmethod
    def mutation_sigma(g):
        return 0.01 + 0.08 * g[:, T["mut_rate"]]


def genetic_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance over trait genes only (not brain weights)."""
    return np.sqrt(((a[..., :N_TRAITS] - b[..., :N_TRAITS]) ** 2).mean(axis=-1))
