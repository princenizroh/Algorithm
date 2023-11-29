import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate


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

    # Method to determine fitness function value of particle position (x)
    def determineFunction(self) -> list[float]:
        return [fitness_function(x) for x in self.particle]

    # Method to evaluate pBest fitness value
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
        self.print_table("Beginning")
        print()
        for j in range(n):
            self.findGbest()
            self.findPbest()
            self.updateV()

            self.print_table(j + 1)
            print()

        print(f"Minimum value of f(x) = {fitness_function(self.gBest)}")

    # Method to create table
    def print_table(self, n):
        # Rounded value
        rounded_particle = np.round(self.particle, 3)
        rounded_determine_fx = np.round(self.determineFunction(), 3)
        rounded_fx_pBest = np.round(self.evaluatePbestFitness(), 3)
        if self.gBest is not None:
            rounded_fx_gBest = np.round(fitness_function(self.gBest), 3)
            fx_gBest_str = str(rounded_fx_gBest)
        else:
            fx_gBest_str = None
        rounded_gbest = np.round(self.gBest, 3) if self.gBest is not None else None
        rounded_pBest = np.round(self.pBest, 3)
        rounded_velocity = np.round(self.velocity, 3)
        self.updateX()
        rounded_updateX = np.round(self.particle, 3)

        # Data for table
        data = [
            [f"Iteration {n}"],
            ["Particle (x)", ", ".join(map(str, rounded_particle))],
            ["Determine fx", ", ".join(map(str, rounded_determine_fx))],
            ["fx(pBest)", ", ".join(map(str, rounded_fx_pBest))],
            ["fx(gBest)", fx_gBest_str],
            [
                "Global Best",
                str(rounded_gbest) if rounded_gbest is not None else None,
            ],
            ["Personal Best", ", ".join(map(str, rounded_pBest))],
            ["Velocity", ", ".join(map(str, rounded_velocity))],
            ["Update x", ", ".join(map(str, rounded_updateX))],
        ]
        headers = ["Variabel", "Value"]
        print(tabulate(data, headers, tablefmt="grid", colalign=("left", "right")))

    def plot(self):
        # Generate data for visualization
        x_values = np.linspace(x_range[0], x_range[1], dimension)
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
            self.pBest, self.determineFunction(), color="green", label="Personal Best"
        )

        # Plot gBest position
        plt.scatter(
            self.gBest, fitness_function(self.gBest), color="blue", label="Global Best"
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
    x_range = np.array([-5, 4])  # Range of particle (xMin, xMax)
    dimension = 10  # Dimension of particle
    particle = np.random.uniform(
        x_range[0], x_range[1], dimension
    )  # Generate random particle
    velocity = np.zeros(dimension)  # Initialize velocity
    c = np.array([0.5, 1.0])  # Acceleration coefficient
    r = np.array(
        [np.random.rand(), np.random.rand()]
    )  # Random number (Between 0 and 1)
    w = 1.0  # Inertia weight
    iterate = 150  # Iterate
    pso = fx_PSO(particle, velocity, c, r, w)  # Create object
    pso.iterate(iterate)  # Iterate PSO
    pso.plot()  # Display visualization of PSO
