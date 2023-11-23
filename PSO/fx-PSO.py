import numpy as np
import matplotlib.pyplot as plt
import random as rd
import pandas as pd


def function(x):
    return x**3 + 3 * x**2 - 12


class PSO_4:
    def __init__(
        self,
        particle: list,
        v: list,
        c: list,
        r: list,
        w: float,
    ) -> None:
        self._particles = particle
        self._v = v
        self._c = c
        self._r = r
        self._w = w

        self.oldX = particle.copy()
        self._pBest = particle.copy()

    def decideFunction(self) -> None:
        pass

    def findGbest(self) -> None:
        pass

    def findePbest(self) -> None:
        pass

    def iterate(self, n) -> None:
        pass
