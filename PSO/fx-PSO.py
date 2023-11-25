import numpy as np
import matplotlib.pyplot as plt
import random as rd
import pandas as pd
from typing import List


def fitness_function(x):
    return x**3 + 3 * x**2 - 12


class fx_PSO:
    def __init__(
        self,
        particle: np.ndarray,
        velocity: np.ndarray,
        c: np.ndarray,
        r: np.ndarray,
        w: float,
    ) -> None:
        self.particle = particle
        self.velocity = velocity
        self.c = c
        self.r = r
        self.w = w

        self.oldParticle = np.copy(particle)
        self.pBest = np.copy(particle)
        self.gBest = self.particle

    def decideFunction(self) -> List[float]:
        fx = [fitness_function(x) for x in self.particle]
        return fx

    def findGbest(self) -> None:
        fx = self.decideFunction()
        self.gBest = self.particle[np.argmin(fx)]

    def findPbest(self) -> None:
        for i in range(len(self.particle)):
            if fitness_function(self.particle[i]) < fitness_function(self.pBest[i]):
                self.pBest[i] = self.particle[i]
            else:
                self.pBest[i] = self.oldParticle[i]

    def updateV(self) -> None:
        for i in range(len(self.particle)):
            self.velocity[i] = (
                (self.w * self.velocity[i])
                + (self.c[0] * self.r[0] * (self.pBest[i] - self.particle[i]))
                + (self.c[1] * self.r[1] * (self.gBest - self.particle[i]))
            )

    def updateX(self) -> None:
        for i in range(len(self.particle)):
            self.oldParticle[i] = self.particle[i]
            self.particle[i] = self.particle[i] + self.velocity[i]

    def iterate(self, n) -> None:
        for j in range(n):
            self.findGbest()
            self.findPbest()
            self.updateV()

            print(f"iteration {j+1}")
            print("Initialization")
            print(f"x = {self.particle}")
            print(f"fx = {fitness_function(self.pBest)}")
            print(f"fx(gBest) = {fitness_function(self.gBest)}")
            print(f"gBest = {self.gBest}")
            print(f"pBest = {self.pBest}")
            print(f"v = {self.velocity}")
            self.updateX()
            print(f"Update x = {self.particle}")
            print()
        print(f"Minimum value of f(x) = {fitness_function(self.gBest)}")

    def plot(self):
        # Generate data for visualization
        x_values = np.linspace(-5, 5, 100)
        y_values = fitness_function(x_values)

        # Plot the function
        plt.plot(x_values, y_values, label="Fungsi Objektif", color="blue")

        # Plot particle positions
        plt.scatter(
            self.particle,
            fitness_function(self.particle),
            color="red",
            label="Posisi Partikel",
        )

        # Plot pBest positions
        plt.scatter(
            self.pBest, fitness_function(self.pBest), color="green", label="pBest"
        )

        # Plot gBest position
        plt.scatter(
            self.gBest, fitness_function(self.gBest), color="blue", label="gBest"
        )

        # Configuration
        plt.title("Visualisasi PSO")
        plt.xlabel("Posisi")
        plt.ylabel("Nilai Fungsi Objektif")
        plt.legend()
        plt.grid(True)
        plt.savefig("fx-PSO.png")
        plt.show()


particle = np.random.uniform(-5, 5, 3)
velocity = np.zeros(3)
c = np.array([0.5, 1.0])
r = np.array([0.5, 0.5])
w = 1.0

pso = fx_PSO(particle, velocity, c, r, w)
pso.iterate(3)
# Visualisasi
pso.plot()
