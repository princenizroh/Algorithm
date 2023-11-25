import numpy as np
import matplotlib.pyplot as plt
from typing import List


# Function to be optimized
def fitness_function(x, y):
    return (
        20
        + x**2
        - (10 * np.cos(2 * np.pi * x))
        + y**2
        - (10 * np.cos(2 * np.pi * x))
    )


class fx_PSO_2D:
    # Constructor
    def __init__(
        self,
        particle_x: np.ndarray,
        particle_y: np.ndarray,
        velocity_x: np.ndarray,
        velocity_y: np.ndarray,
        accelerationn_coefficients: np.ndarray,
        random: np.ndarray,
        inertia_weight: float,
    ) -> None:
        self.particle_x = particle_x
        self.particle_y = particle_y
        self.velocity_x = velocity_x
        self.velocity_y = velocity_y
        self.c = accelerationn_coefficients
        self.r = random
        self.w = inertia_weight

        self.oldParticle_x = np.copy(self.particle_x)
        self.oldParticle_y = np.copy(self.particle_y)
        self.pBest_x = np.copy(particle_x)
        self.pBest_y = np.copy(particle_y)
        self.gBest_x = self.particle_x
        self.gBest_y = self.particle_y

    # Method to decide function value of particle position (x,y)
    def decideFunction(self) -> List[float]:
        fx = [fitness_function(x, y) for x, y in zip(self.particle_x, self.particle_y)]
        return fx

    # Method to find gBest value of particle position (x,y)
    def findGbest(self) -> None:
        fx = self.decideFunction()
        index = np.argmin(fx)
        self.gBest_x = self.particle_x[index]
        self.gBest_y = self.particle_y[index]

    # Method to find pBest value of particle position (x,y)
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

    # Method to update velocity of particle
    def updateV(self) -> None:
        for i in range(len(self.particle_x)):
            self.velocity_x[i] = (
                (w * self.velocity_x[i])
                + self.c[0] * self.r[0] * (self.pBest_x[i] - self.particle_x[i])
                + self.c[1] * self.r[1] * (self.gBest_x - self.particle_x[i])
            )
            self.velocity_y[i] = (
                (w * self.velocity_y[i])
                + self.c[0] * self.r[0] * (self.pBest_y[i] - self.particle_y[i])
                + self.c[1] * self.r[1] * (self.gBest_y - self.particle_y[i])
            )

    # Method to update position of particle
    def updateXY(self) -> None:
        for i in range(len(self.particle_x)):
            self.oldParticle_x[i] = self.particle_x[i]
            self.oldParticle_y[i] = self.particle_y[i]
            self.particle_x[i] = self.particle_x[i] + self.velocity_x[i]
            self.particle_y[i] = self.particle_y[i] + self.velocity_y[i]

    # Method to iterate PSO
    def iterate(self, n) -> None:
        self.findGbest()
        self.findPbest()
        # Matrix of particle, pBest, and velocity
        particle_xy = np.column_stack((self.particle_x, self.particle_y))
        pBest_xy = np.column_stack((self.pBest_x, self.pBest_y))
        velocity_xy = np.column_stack((self.velocity_x, self.velocity_y))

        print(f"Beginning Value")
        print(f"Particles (x,y) = {tuple(map(tuple,particle_xy))} ")
        print(f"fx = {self.decideFunction()}")
        print(f"fx(gBest) = {fitness_function(self.gBest_x, self.gBest_y)}")
        print(f"gBest (x,y) = {self.gBest_x, self.gBest_y}")
        print(f"pBest (x,y) = {tuple(map(tuple,pBest_xy))}")
        print(f"velocity (x,y) = {tuple(map(tuple,velocity_xy))}")
        print()
        for j in range(n):
            self.findGbest()
            self.findPbest()
            self.updateV()

            # Matrix of particle, pBest, and velocity
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

            # Matrix of particle position (x,y)
            updateXY = np.column_stack((self.particle_x, self.particle_y))

            print(f"Update (x,y) = {tuple(map(tuple,updateXY))}")
            print()

        print(f"Minimum value of f(x) = {fitness_function(self.gBest_x, self.gBest_y)}")

    def plot(self):
        plt.figure()
        x_values = np.linspace(-5.12, 5.12, 3)
        y_values = fitness_function(x_values, 0)

        plt.plot(x_values, y_values, label="Fungsi Objektif", color="blue")

        particle_positions = np.column_stack((self.particle_x, self.particle_y))
        plt.scatter(
            particle_positions[:, 0],
            particle_positions[:, 1],
            color="red",
            label="Posisi Partikel",
        )

        pBest_positions = np.column_stack((self.pBest_x, self.pBest_y))
        plt.scatter(
            pBest_positions[:, 0],
            pBest_positions[:, 1],
            color="green",
            label="pBest",
        )

        plt.scatter(self.gBest_x, self.gBest_y, color="blue", label="gBest")

        plt.title("Visualisasi PSO")
        plt.xlabel("Posisi X")
        plt.ylabel("Posisi Y")
        plt.legend()
        plt.grid(True)
        plt.savefig("fx-PSO-2D.png")
        plt.show()


# Main program
if __name__ == "__main__":
    xy_range = np.array([-5.12, 5.12])  # Range of particle (xMin, xMax)
    dimension = 3  # Dimension of particle
    particle_x = np.random.uniform(
        xy_range[0], xy_range[1], dimension
    )  # Generate particle x random particle
    particle_y = np.random.uniform(
        xy_range[0], xy_range[1], dimension
    )  # Generate particle y random particle
    velocity_x = np.zeros(dimension)  # Initialize velocity x
    velocity_y = np.zeros(dimension)  # Initialize velocity y
    c = np.array([1.0, 0.5])  # Acceleration coefficient
    r = np.array([np.random.rand(), np.random.rand()])  # Random number
    w = 1.0  # Inertia weight

    pso = fx_PSO_2D(
        particle_x, particle_y, velocity_x, velocity_y, c, r, w
    )  # Create object
    pso.iterate(3)  # Iterate PSO
    pso.plot()  # Display visualization of PSO
