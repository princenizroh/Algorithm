from tabulate import tabulate
from typing import List, Dict
import networkx as nx
import matplotlib.pyplot as plt

Graph = Dict[str, Dict[str, int]]
# Create a graph class
class Dijkstra(object):
    # Constructor for the graph
    def __init__(self, nodes: List[str], graph: Graph, source: str) -> None:
        self.nodes = nodes
        self.graph = graph
        self.source = source

        self.unvisited_nodes = sorted(self.nodes)
        self.shortest_distance = {node: float("inf") for node in self.unvisited_nodes}
        self.shortest_distance[source] = 0
        self.previous_nodes = {source: source}
        self.history = []
    
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

    def table_2(self) -> None:
        data = [[self.source, 0, "-"]]
        for node in self.nodes:
            if node == self.source:
                continue
            shortestDist = self.shortest_distance[node]
            prevNode = self.previous_nodes[node]
            data.append([node, shortestDist, prevNode])
        print(tabulate(data, ["Node", "Shortest", "Previous"], tablefmt="fancy_grid"))

    def construct_path(self, destination) -> None:
        if destination not in self.shortest_distance:
            return print(f"Path dari {self.source} ke {destination} tidak ditemukan")

        path = []

        node = destination
        while node != self.source:
            path.append(node)
            if node not in self.previous_nodes:
                return print(f"Path dari {self.source} ke {destination} tidak ditemukan")
            node = self.previous_nodes[node]

        path.append(self.source)
        print(" -> ".join(reversed(path)))

        print(f"Found the folowing best path {self.source} to {destination} with a value of {self.shortest_distance[destination]}.")
    # Visualize the graph
    def visualize(self):
        G = nx.DiGraph()

        for node, edges in self.graph.items():
            for neighbor, value in edges.items():
                G.add_edge(node, neighbor, weight=value)

        pos = nx.spring_layout(G)
        edge_labels = nx.get_edge_attributes(G, "weight")
        node_labels = {node: node for node in G.nodes}

        nx.draw(
            G,
            pos,
            node_size=700,
            node_color="skyblue",
            font_size=8,
            font_color="black",
        )
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red")
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_color="white")

        plt.title("Graph Visualization")
        plt.show()


if __name__ == "__main__":
    nodes = ["A", "B", "C", "D", "E", "F", "G"]
    init_graph = {
        "A": {"B": 5, "C": 7, "D": 12},
        "B": {"C": 1, "E": 6},
        "C": {"D": 1, "E": 5, "F": 4},
        "D": {"F": 13},
        "E": {"F": 2, "G": 7},
        "F": {"G": 3},
        "G": {},
    }
    destination = "G"

    dijkstra = Dijkstra(nodes, init_graph, "A")

    dijkstra.start()
    dijkstra.table_1()
    dijkstra.table_2()
    dijkstra.construct_path(destination)
    dijkstra.visualize()
