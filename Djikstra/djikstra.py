import sys
import tabulate


class Graph(object):
    def __init__(self, nodes, init_graph):
        self.nodes = nodes
        self.graph = self.construct_graph(nodes, init_graph)

    def construct_graph(self, nodes, init_graph):
        graph = {}  # Membuat graf kosong

        # Membuat Node dalam graf dan setiap simpul tidak memiliki tetangga awalnya
        for node in nodes:
            graph[node] = {}
        graph.update(init_graph)

        for node, edges in graph.items():
            for adjacent_node, value in edges.items():
                if graph[adjacent_node].get(node, False) == False:
                    graph[adjacent_node][node] = value
        return graph

    def get_nodes(self):
        return self.nodes

    def get_outgoing_edges(self, node):
        connections = []
        for out_node in self.nodes:
            if self.graph[node].get(out_node, False) != False:
                connections.append(out_node)
        return connections

    def value(self, node1, node2):
        "Returns the value of an edge between two nodes."
        return self.graph[node1][node2]

    def dijkstra(self, graph, start_node):
        unvisited_node = list(graph.get_nodes())

        shortest_distance = {}
        previous_nodes = {}

        for node in unvisited_node:
            shortest_distance[node] = sys.maxsize
        shortest_distance[start_node] = 0

        while unvisited_node:
            current_min_node = None
            for node in unvisited_node:
                if current_min_node is None:
                    current_min_node = node
                elif shortest_distance[node] < shortest_distance[current_min_node]:
                    current_min_node = node
            neighbors = graph.get_outgoing_edges(current_min_node)
            for neighbor in neighbors:
                distance = shortest_distance[current_min_node] + graph.value(
                    current_min_node, neighbor
                )
                if distance < shortest_distance[neighbor]:
                    shortest_distance[neighbor] = distance
                    previous_nodes[neighbor] = current_min_node
            unvisited_node.remove(current_min_node)
        return previous_nodes, shortest_distance

    def print_result(self, previous_nodes, shortest_distance, start_node, target_node):
        path = []
        node = target_node

        while node != start_node:
            path.append(node)
            node = previous_nodes[node]

        # Add the start node manually
        path.append(start_node)

        print(
            "We found the following best path with a value of {}.".format(
                shortest_distance[target_node]
            )
        )
        print(" -> ".join(reversed(path)))


nodes = ["A", "B", "C", "D", "E", "F"]
init_graph = {
    "A": {"B": 3, "C": 4, "D": 5, "E": 9},
    "B": {"C": 5},
    "D": {"C": 2, "F": 3},
    "C": {"E": 2, "F": 2},
    "E": {"F": 3},
}
my_graph = Graph(nodes, init_graph)
previous_nodes, shortest_distance = my_graph.dijkstra(my_graph, "A")
my_graph.print_result(
    previous_nodes, shortest_distance, start_node="A", target_node="F"
)
# nodes = ["Reykjavik", "Oslo", "Moscow", "London", "Rome", "Berlin", "Belgrade", "Athens"]

# init_graph = {}
# for node in nodes:
#     init_graph[node] = {}
# init_graph["Reykjavik"]["Oslo"] = 5
# init_graph["Reykjavik"]["London"] = 4
# init_graph["Oslo"]["Berlin"] = 1
# init_graph["Oslo"]["Moscow"] = 3
# init_graph["Moscow"]["Belgrade"] = 5
# init_graph["Moscow"]["Athens"] = 4
# init_graph["Athens"]["Belgrade"] = 1
# init_graph["Rome"]["Berlin"] = 2
# init_graph["Rome"]["Athens"] = 2

# my_graph = Graph(nodes, init_graph)
# previous_nodes, shortest_distance = Graph.dijkstra(my_graph, "Reykjavik")
# print(Graph.print_result(previous_nodes, shortest_distance, start_node="Reykjavik", target_node="Belgrade"))

# Cetak representasi graf
# print(my_graph.graph)
# print(Graph.get_nodes(my_graph))
# print(Graph.get_outgoing_edges(my_graph, 'A'))
# print(Graph.value(my_graph, 'A', 'C'))
# print(Graph.dijkstra(my_graph, 'A'))
