import numpy as np
import matplotlib.pyplot as plt
import random as rd
import pandas as pd
from typing import List


def fitness_function(x,y):
    # return 20 + x**2 - (10 * np.cos(2*np.pi*x)) + y**2 - (10 * np.cos(2*np.pi*x))
    return (0.26 * (x**2 + y**2)) - (0.48 * (x * y))

class fx_PSO:
    def __init__(
        self,
        particle_x: np.ndarray,
        particle_y: np.ndarray,
        velocity_x: np.ndarray,
        velocity_y: np.ndarray,
        c: np.ndarray,
        r: np.ndarray,
        w: float,
    ) -> None:
        self.particle_x = particle_x
        self.particle_y = particle_y
        self.velocity_x = velocity_x
        self.velocity_y = velocity_y
        self.c = c
        self.r = r
        self.w = w

        self.oldParticle_x = np.copy(self.particle_x)
        self.oldParticle_y = np.copy(self.particle_y)
        self.pBest_x = np.copy(particle_x)
        self.pBest_y = np.copy(particle_y)
        self.gBest_x = self.particle_x
        self.gBest_y = self.particle_y

    def decideFunction(self) -> List[float]:
        fx = [fitness_function(x,y) for x,y in zip(self.particle_x,self.particle_y)]
        return fx

    def findGbest(self) -> None:
        fx = self.decideFunction()
        index = np.argmin(fx)
        self.gBest_x = self.particle_x[index]
        self.gBest_y = self.particle_y[index]

    def findPbest(self) -> None:
        for i in range(len(self.particle_x)):
            if fitness_function(self.particle_x[i], self.particle_y[i]) < fitness_function(self.pBest_x[i],self.pBest_y[i]):
                self.pBest_x[i] = self.particle_x[i]
                self.pBest_y[i] = self.particle_y[i]
            else:
                self.pBest_x[i] = self.oldParticle_x[i]
                self.pBest_y[i] = self.oldParticle_y[i]

    def updateV(self) -> None:
        for i in range(len(self.particle_x)):
            self.velocity_x[i] = (
                (self.w * self.velocity_x[i])
                + (self.c[0] * self.r[0] * (self.pBest_x[i] - (self.particle_x[i] * self.particle_y[i])))
                + (self.c[1] * self.r[1] * (self.gBest_x - (self.particle_x[i] * self.particle_y[i])))
            )
            self.velocity_y[i] = (
                (self.w * self.velocity_y[i])
                + (self.c[0] * self.r[0] * (self.pBest_y[i] - (self.particle_x[i] * self.particle_y[i])))
                + (self.c[1] * self.r[1] * (self.gBest_y - (self.particle_x[i] * self.particle_y[i])))
            )

    def updateXY(self) -> None:
        for i in range(len(self.particle_x)):
            self.oldParticle_x[i] = self.particle_x[i]
            self.oldParticle_y[i] = self.particle_y[i]
            self.particle_x[i] = self.particle_x[i] + self.velocity_x[i]
            self.particle_y[i] = self.particle_y[i] + self.velocity_y[i]

    def iterate(self, n) -> None:
        for j in range(n):
            self.findGbest()
            self.findPbest()
            self.updateV()

            print(f"iteration {j+1}")
            print("Initialization")
            print(f"x = {self.particle_x}")
            print(f"y = {self.particle_y}")
            print(f"fx = {self.decideFunction()}")
            print(f"fx(gBest) = {fitness_function(self.gBest_x, self.gBest_y)}")
            print(f"gBest = {self.gBest_x, self.gBest_y}")
            print(f"pBest = {self.pBest_x, self.pBest_y}")
            print(f"v_x = {self.velocity_x}")
            print(f"v_y = {self.velocity_y}")
            self.updateXY()
            print(f"Update x = {self.particle_x}")
            print(f"Update y = {self.particle_y}")
            print()

particle_x = np.array([1.0, -2.0, 2.0])
particle_y = np.array([1.0, -1.0, 2.0])
velocity_x = np.array([0.0, 0.0, 0.0])
velocity_y = np.array([0.0, 0.0, 0.0])
c = np.array([1.0, 0.5])
r = np.array([1.0, 1.0])
w = 1.0

pso = fx_PSO(particle_x, particle_y, velocity_x, velocity_y, c, r, w)
pso.iterate(3)