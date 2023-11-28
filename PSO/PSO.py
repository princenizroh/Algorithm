import numpy as np
import matplotlib.pyplot as plt

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
    def plot(self):
        # Generate data for visualization
        x_values = np.linspace(-5, 4, 3)
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
        plt.scatter(self.pBest, self.determineFunction(), color="green", label="pBest")

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


# Main program
if __name__ == "__main__":
    x = np.array([-5, 4])  # Range of particle (xMin, xMax)
    dimension = 3  # Dimension of particle
    particle = np.array([1.0, 2.0, 3.0])  # Generate random particle
    velocity = np.zeros(dimension)  # Initialize velocity
    c = np.array([0.5, 1.0])  # Acceleration coefficient
    r = np.array([0.5, 0.5])  # Random number (Between 0 and 1)
    w = 1.0  # Inertia weight
    pso = fx_PSO(particle, velocity, c, r, w)  # Create object
    pso.iterate(3)  # Iterate PSO
    # Visualisasi
    pso.plot()  # Display visualization
