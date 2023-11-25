import numpy as np
import matplotlib.pyplot as plt
import random as rd
import pandas as pd
from typing import List


def fitness_function(x, y):
    return (
        20
        + x**2
        - (10 * np.cos(2 * np.pi * x))
        + y**2
        - (10 * np.cos(2 * np.pi * x))
    )


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
        fx = [fitness_function(x, y) for x, y in zip(self.particle_x, self.particle_y)]
        return fx

    def findGbest(self) -> None:
        fx = self.decideFunction()
        index = np.argmin(fx)
        self.gBest_x = self.particle_x[index]
        self.gBest_y = self.particle_y[index]

    def findPbest(self) -> None:
        for i in range(len(self.particle_x)):
            if fitness_function(
                self.particle_x[i], self.particle_y[i]
            ) < fitness_function(self.pBest_x[i], self.pBest_y[i]):
                self.pBest_x[i] = self.particle_x[i]
                self.pBest_y[i] = self.particle_y[i]
            else:
                self.pBest_x[i] = self.oldParticle_x[i]
                self.pBest_y[i] = self.oldParticle_y[i]

    def updateV(self) -> None:
        for i in range(len(self.particle_x)):
            self.velocity_x[i] = (
                (w * self.velocity_x[i])
                + c[0] * r[0] * (self.pBest_x[i] - self.particle_x[i])
                + c[1] * r[1] * (self.gBest_x - self.particle_x[i])
            )
            self.velocity_y[i] = (
                (w * self.velocity_y[i])
                + c[0] * r[0] * (self.pBest_y[i] - self.particle_y[i])
                + c[1] * r[1] * (self.gBest_y - self.particle_y[i])
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

            particle_xy = np.column_stack((self.particle_x, self.particle_y))
            pBest_xy = np.column_stack((self.pBest_x, self.pBest_y))
            velocity_xy = np.column_stack((self.velocity_x, self.velocity_y))
            print(f"iteration {j+1}")
            print("Initialization")
            print(f"Particles (x,y) = {tuple(map(tuple,particle_xy))} ")
            print(f"fx = {self.decideFunction()}")
            print(f"fx(gBest) = {fitness_function(self.gBest_x, self.gBest_y)}")
            print(f"gBest (x,y) = {self.gBest_x, self.gBest_y}")
            print(f"pBest (x,y) = {tuple(map(tuple,pBest_xy))}")
            print(f"velocity (x,y) = {tuple(map(tuple,velocity_xy))}")
            self.updateXY()
            updateXY = np.column_stack((self.particle_x, self.particle_y))

            print(f"Update (x,y) = {tuple(map(tuple,updateXY))}")
            print()

        print(f"Minimum value of f(x) = {fitness_function(self.gBest_x, self.gBest_y)}")


x = np.array([-5.12, 5.12])
dimension = 3
particle_x = np.array([1.0, 1.0, 2.0])
particle_y = np.array([1.0, -1.0, -1.0])
velocity_x = np.array([0.0, 0.0, 0.0])
velocity_y = np.array([0.0, 0.0, 0.0])
c = np.array([1.0, 0.5])
r = np.array([0.5, 0.5])
w = 1.0

pso = fx_PSO(particle_x, particle_y, velocity_x, velocity_y, c, r, w)
pso.iterate(3)


# a = np.array([1.0, 2.0, 3.0])
# b = np.array([[1.0, 2.0], [3.0, 4.0]])
# c = 2.0

# print(np.multiply(a, c))


# def debug():
#     for i in range(len(particle_x)):
#         print(particle_x)
#         velocity = (
#             (w * velocity_x)
#             + c[0] * r[0] * (particle_x - particle_x)
#             + c[1] * r[1] * (1 - particle_x)
#         )
#         print(velocity)
# debug()
