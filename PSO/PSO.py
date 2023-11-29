import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# from typing import List


# Function to be optimized
def fitness_function(x):
    return x**3 + 3 * x**2 - 12


class fx_PSO:
    # Constructor
    def __init__(
        self,
        particle: np.ndarray,
        velocity: np.ndarray,
        acceleration_coefficients: np.ndarray,
        random: np.ndarray,
        inertia_weight: float,
    ) -> None:
        self.particle = particle
        self.velocity = velocity
        self.c = acceleration_coefficients
        self.r = random
        self.w = inertia_weight

        self.oldParticle = np.copy(particle)
        self.pBest = []
        self.gBest = None

        # Flag untuk menandai iterasi pertama
        self.first_iteration = True

    # Method to decide function value of particle position (x)
    def determineFunction(self) -> list[float]:
        return [fitness_function(x) for x in self.particle]

    # Method to find fitness value of pBest
    def evaluatePbestFitness(self) -> list[float]:
        return [fitness_function(p) for p in self.pBest]

    # Method to find gBest value of particle position (x)
    def findGbest(self) -> None:
        if not self.gBest:
            self.gBest = self.particle[
                np.argmin([fitness_function(x) for x in self.particle])
            ]
        else:
            fx = self.determineFunction()
            if fitness_function(self.particle[np.argmin(fx)]) < fitness_function(
                self.gBest
            ):
                self.gBest = np.copy(self.particle[np.argmin(fx)])

    # Method to find pBest value of particle position (x)
    def findPbest(self) -> None:
        if len(self.pBest) < len(self.particle):
            self.pBest.extend([np.copy(p) for p in self.particle[len(self.pBest) :]])
        for i in range(len(self.particle)):
            if fitness_function(self.particle[i]) < fitness_function(self.pBest[i]):
                self.pBest[i] = self.particle[i]
            else:
                self.pBest[i] = self.oldParticle[i]

    # Method to update velocity of particle
    def updateV(self) -> None:
        for i in range(len(self.particle)):
            self.velocity[i] = (
                (self.w * self.velocity[i])
                + (self.c[0] * self.r[0] * (self.pBest[i] - self.particle[i]))
                + (self.c[1] * self.r[1] * (self.gBest - self.particle[i]))
            )

    # Method to update position of particle
    def updateX(self) -> None:
        for i in range(len(self.particle)):
            self.oldParticle[i] = self.particle[i]
            self.particle[i] = self.particle[i] + self.velocity[i]

    # Method to iterate PSO
    def iterate(self, n) -> None:
        print(f"Beginning Value")
        print(f"Particle (x) = {self.particle}")
        print(f"Determine fx = {self.determineFunction()}")
        print(f"fx(pBest) = {self.evaluatePbestFitness()}")
        print(
            f"fx(gBest) = {fitness_function(self.gBest) if self.gBest is not None else None}"
        )
        print(f"Global Best = {self.gBest}")
        print(f"Personal Best = {self.pBest}")
        print(f"Velocity = {self.velocity}")
        print(f"Update x = {self.particle}")
        print()
        for j in range(n):
            self.findGbest()
            self.findPbest()
            self.updateV()

            print(f"iteration {j+1}")
            print("Initialization")
            print(f"Particle (x)  = {self.particle}")
            print(f"Determine fx value = {self.determineFunction()}")
            print(f"fx(pBest) = {self.evaluatePbestFitness()}")
            print(f"fx(gBest) = {fitness_function(self.gBest)}")
            print(f"Global Best = {self.gBest}")
            print(f"Personal Best = {self.pBest}")
            print(f"Velocity = {self.velocity}")

            self.updateX()

            print(f"Update x = {self.particle}")
            print()

        print(f"Minimum value of f(x) = {fitness_function(self.gBest)}")

    # Method to display visualization of PSO
    def plot_particles(self, ax):
        # Plot particle positions
        ax.scatter(
            self.particle,
            [fitness_function(xi) for xi in self.particle],
            c="b",
            marker="o",
            label="Particles",
        )
        ax.scatter(
            self.gBest,
            fitness_function(self.gBest),
            c="r",
            marker="o",
            s=100,
            label="Global Best",
        )

    # Fungsi untuk animasi iterasi
    def animate(self, i, ax):
        if self.first_iteration:
            self.first_iteration = False
        else:
            # Print informasi untuk iterasi selanjutnya
            self.findPbest()
            self.findGbest()
            self.updateV()

            print(f"Iterasi {i+1}")
            print("Initialization")
            # Update pBest, gBest, kecepatan, dan posisi untuk iterasi selanjutnya
            print(f"Particle (x) = {[round(val, 3) for val in self.particle]}")
            print(
                f"Determine fx value = {[round(fitness_function(val), 3) for val in self.particle]}"
            )
            print(f"x = {[round(val, 3) for val in self.particle]}")
            print(f"Personal Best = {[round(val, 3) for val in self.pBest]}")
            print(f"Velocity = {[round(val, 3) for val in self.velocity]}")
            # print(f"gBest = {round(self.gBest, 3)}")
            self.updateX()
            print(
                f"f(x) = {[round(fitness_function(val), 3) for val in self.particle]}"
            )
            print()

        # Menghapus plot sebelumnya dan memplot partikel serta gunung untuk iterasi saat ini
        ax.clear()
        self.plot_particles(ax)
        self.plot_surface(ax)
        ax.set_title(f"Iteration {i+1}")
        # ax.set_xlim(-5, 5)
        # ax.set_ylim(-5, 5)
        ax.set_xlabel("X")
        ax.set_ylabel("f(X)")
        ax.legend()

    # Fungsi untuk plotting permukaan fungsi objektif sebagai gunung
    def plot_surface(self, ax):
        x = np.linspace(-100, 100, 1000)
        y = fitness_function(x)
        ax.plot(x, y, color="purple", alpha=0.5, label="Objective Function")

    # Fungsi untuk iterasi dengan animasi
    def iterate_with_animation(self, n):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        animation = FuncAnimation(
            fig, self.animate, frames=n, fargs=(ax,), interval=500, repeat=False
        )
        plt.show()


# Main program
if __name__ == "__main__":
    x = np.array([-5, 4])  # Range of particle (xMin, xMax)
    dimension = 10  # Dimension of particle
    # particle = np.array([1.0, 2.0, 3.0])  # Generate random particle
    particle = np.random.uniform(x[0], x[1], dimension)
    velocity = np.zeros(dimension)  # Initialize velocity
    c = np.array([0.5, 1.0])  # Acceleration coefficient
    r = np.array([0.5, 0.5])  # Random number (Between 0 and 1)
    w = 1.0  # Inertia weight
    pso = fx_PSO(particle, velocity, c, r, w)  # Create object
    iterate = 10
    pso.iterate(iterate)  # Iterate PSO
    pso.iterate_with_animation(iterate)
    # Visualisasi
    # pso.plot()  # Display visualization
