from tabulate import tabulate
from typing import List, Dict
import networkx as nx
import matplotlib.pyplot as plt

Graph = Dict[str, Dict[str, int]]
# Create a graph class
class Djikstra(object):
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
    
    # Construct the graph  
    def construct_graph(self, nodes, init_graph)->dict:
        graph = {}
        for node in nodes:
            graph[node] = {}
        graph.update(init_graph)

        # Check if the graph is symmetric
        for node, edges in graph.items():
            for adjacent_node, value in edges.items():
                # If the graph is not symmetric, add the reverse edge
                if graph[adjacent_node].get(node, False) == False:
                    graph[adjacent_node][node] = value
        return graph

    # Get the nodes
    def get_nodes(self)->list:
        return self.nodes

    # Get the edges
    def get_edges(self, node)->list:
        return list(self.graph[node].keys())

    # Get the value of the edges
    def get_value(self, node1, node2)->int:
        return self.graph[node1][node2]

    # Dijkstra's algorithm
    def dijkstra(self, start_node)->None:
        # Initialize the algorithm
        unvisited_nodes = sorted(self.get_nodes())
        shortest_distance = {node:float("inf") for node in unvisited_nodes}
        previous_nodes = {}

        # Initialize the distance of the start node to 0
        for node in unvisited_nodes:
            shortest_distance[node] = float("inf")
            print(shortest_distance[node])
        shortest_distance[start_node] = 0

        iteration = 1

        # Start the algorithm
        while unvisited_nodes:
            current_min_node = None

            # Find the node with the shortest distance
            for node in unvisited_nodes:
                if (
                    current_min_node is None
                    or shortest_distance[node] < shortest_distance[current_min_node]
                ):
                    print("shortest distance",shortest_distance[node])
                    current_min_node = node

            # Find the distance from the current node to its neighbors
            neighbors = self.get_edges(current_min_node)

            # Update the distance of the neighbors
            for neighbor in neighbors:
                distance = shortest_distance[current_min_node] + self.get_value(
                    current_min_node, neighbor
                )
                if distance < shortest_distance[neighbor]:
                    shortest_distance[neighbor] = distance
                    previous_nodes[neighbor] = current_min_node

            unvisited_nodes.remove(current_min_node)

            # Print the result of the current iteration
            print(f"\nStep {iteration}:")
            print(
                f"Now we need to start checking the distance from node {current_min_node} to its adjacent nodes."
            )

            if iteration == 1:
                print(
                    f"We mark it with a red square in the list to represent that it has been 'visited' and that "
                    f"we have found the shortest path to this node:"
                )
            else:
                print(
                    f"We cross it off from the list of unvisited nodes: Unvisited Nodes: {list(unvisited_nodes)}"
                )

            result_table = [
                [node, shortest_distance[node]]
                if node in shortest_distance and shortest_distance[node] != float("inf")
                else [node, "∞"]
                for node in self.nodes
            ]
            headers = ["Node", "Shortest Distance"]
            print(
                tabulate(result_table, headers=headers, tablefmt="grid", missingval="")
            )

            iteration += 1

        # Print the final result
        self.shortest_distance = shortest_distance
        self.previous_nodes = previous_nodes

    # Print the result
    def print_result(self, target_node)->None:
        unvisited_nodes = set(self.nodes)

        start_node = nodes[0]

        if target_node not in self.shortest_distance:
            print(f"There is no path from {start_node} to {target_node}")
            return

        path = []
        node = target_node

        while node != start_node:
            path.append(node)
            if node not in self.previous_nodes:
                print(f"Error: No path found from {start_node} to {target_node}")
                return
            node = self.previous_nodes[node]

        path.append(start_node)

        print(
            f"We found the following best path with a value of {self.shortest_distance[target_node]}."
        )

        print(" -> ".join(reversed(path)))

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
            with_labels=True,
            node_size=700,
            node_color="skyblue",
            font_size=8,
            font_color="black",
        )
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red")
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_color="white")

        plt.title("Graph Visualization")
        plt.show()


nodes = ["A", "B", "C", "D", "E", "F", "G"]
init_graph = {
    "A": {"B": 5, "C": 7, "D": 12},
    "B": {"C": 1, "E": 6},
    "C": {"D": 1, "E": 5, "F": 4},
    "D": {"F": 13},
    "E": {"F": 2, "G": 7},
    "F": {"G": 3},
}
# nodes = ["Tempest", "Fittoa", "Asura", "Ranoa", "Shiron", "Milis", "Dwargon"]
# init_graph = {
#     "Tempest": {"Fittoa": 3, "Asura": 4, "Ranoa": 6},
#     "Fittoa": {"Milis": 1, "Shiron": 6},
#     "Asura": {"Fittoa": 2, "Milis": 3},
#     "Ranoa": {"Dwargon": 2},
#     "Shiron": {"Dwargon": 1},
#     "Milis": {"Dwargon": 4},
# }

my_graph = Graph(nodes, init_graph)
# my_graph.dijkstra("Tempest")
# my_graph.print_result(target_node="Dwargon")

my_graph.dijkstra("A")
my_graph.print_result(target_node="G")
# my_graph.visualize()
