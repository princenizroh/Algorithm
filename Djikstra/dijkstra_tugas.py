from tabulate import tabulate
from typing import List, Dict
import networkx as nx
import matplotlib.pyplot as plt

# Define the graph type
Graph = Dict[str, Dict[str, int]]


# Create a graph class
class Dijkstra(object):
    # Constructor for the graph
    def __init__(self, nodes: List[str], graph: Graph, source: str) -> None:
        self.nodes = nodes
        self.graph = graph
        self.source = source

        self.unvisited_nodes = list(self.nodes)
        self.shortest_distance = {node: float("inf") for node in self.unvisited_nodes}
        self.shortest_distance[source] = 0
        self.previous_nodes = {source: source}
        self.history = []

    # Start the Algorithm and calculate the shortest path
    def start(self) -> None:
        while self.unvisited_nodes:
            current_min_node = self.unvisited_nodes[0]
            for node in self.unvisited_nodes:
                if (
                    self.shortest_distance[node]
                    < self.shortest_distance[current_min_node]
                ):
                    current_min_node = node

            neighbours = list(self.graph[current_min_node].keys())
            for neighbour in neighbours:
                neighbour_edge = self.graph[current_min_node][neighbour]
                distance = self.shortest_distance[current_min_node] + neighbour_edge

                if distance < self.shortest_distance[neighbour]:
                    self.shortest_distance[neighbour] = distance
                    self.previous_nodes.update({neighbour: current_min_node})

            self.history.append(
                [
                    current_min_node,
                    self.shortest_distance.copy(),
                    self.previous_nodes.copy(),
                ]
            )
            self.unvisited_nodes.remove(current_min_node)

    # Construct the path from source to destination
    def construct_path(self, destination):
        if destination not in self.shortest_distance:
            return print(f"Path from {self.source} to {destination} not found")

        path = []

        node = destination
        while node != self.source:
            path.append(node)
            if node not in self.previous_nodes:
                return print(f"Path from {self.source} to {destination} not found")
            node = self.previous_nodes[node]

        path.append(self.source)
        print(" -> ".join(reversed(path)))

        print(
            f"Found the folowing best path {self.source} to {destination} with a value of {self.shortest_distance[destination]}."
        )
        return path

    # Print the tables step-by-step
    def table_1(self) -> None:
        data = []
        for historyItem in self.history:
            row = [historyItem[0]]
            for node, dist in historyItem[1].items():
                try:
                    row.append(f"{ dist }_{ historyItem[2][node] }")
                except KeyError:
                    row.append(dist)
            data.append(row)

        print(tabulate(data, ["V", *nodes], tablefmt="fancy_grid"))

    # Print the table showing the shortest path
    def table_2(self) -> None:
        data = [[self.source, 0, "-"]]
        for node in self.nodes:
            if node == self.source:
                continue
            shortestDist = self.shortest_distance[node]
            prevNode = self.previous_nodes[node]
            data.append([node, shortestDist, prevNode])
        print(tabulate(data, ["Node", "Shortest", "Previous"], tablefmt="fancy_grid"))

    # Visualize the graph
    def visualize(self, path) -> None:
        G = nx.DiGraph()

        for node, edges in self.graph.items():
            for neighbor, value in edges.items():
                G.add_edge(node, neighbor, weight=value)

        pos = {
            "O": (0, 0),
            "A": (1, 1),
            "B": (1, -1),
            "C": (2, 0),
            "D": (3, 1),
            "E": (3, -1),
            "F": (4, 0),
            "G": (5.5, 1),
            "H": (5.5, 0),
            "I": (5.5, -1),
            "T": (7, 0),
        }

        edge_labels = nx.get_edge_attributes(G, "weight")

        path_edges = [(path[i], path[i - 1]) for i in range(1, len(path))]

        # Color nodes and edges based on whether they are in the shortest path
        node_colors = ["red" if node in path else "skyblue" for node in G.nodes]
        edge_colors = ["red" if edge in path_edges else "black" for edge in G.edges]

        # Draw the graph with specified colors and attributes
        nx.draw(
            G,
            pos,
            node_size=2000,
            node_color=node_colors,
            font_size=8,
            font_color="black",
            edge_color=edge_colors,
            with_labels=False,
        )

        # Draw labels for all nodes
        nx.draw_networkx_labels(
            G, pos, font_color="black", font_weight="bold", font_size=16
        )

        # Draw labels for nodes in the shortest path
        nx.draw_networkx_labels(
            G,
            pos,
            labels={node: node for node in path},
            font_color="white",
            font_weight="bold",
            font_size=16,
        )

        # Draw edge labels
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels, font_color="red", font_size=16
        )

        plt.title("Graph Visualization")
        plt.show()


if __name__ == "__main__":
    nodes = ["O", "A", "B", "C", "D", "E", "F", "G", "H", "I", "T"]
    init_graph = {
        "O": {"A": 4, "B": 3, "C": 6},
        "A": {"C": 5, "D": 3},
        "B": {"C": 4, "E": 6},
        "C": {"D": 2, "E": 5, "F": 2},
        "D": {"F": 2, "G": 4},
        "E": {"F": 1, "H": 2, "I": 5},
        "F": {"G": 2, "H": 5},
        "G": {"T": 7},
        "H": {"G": 2, "I": 3, "T": 8},
        "I": {"T": 4},
        "T": {},
    }
    destination = "T"
    dijkstra = Dijkstra(nodes, init_graph, "O")
    dijkstra.start()
    print("Table_1.from A to other Nodes step-by-step")
    dijkstra.table_1()
    print("Table_2.from A to other Nodes")
    dijkstra.table_2()

    path = dijkstra.construct_path(destination)
    dijkstra.visualize(path)  # Visualize the graph
