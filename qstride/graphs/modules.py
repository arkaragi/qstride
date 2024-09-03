"""

"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union

import networkx as nx
from matplotlib import pyplot as plt

from qstride.qstride.graphs.builder import GraphBuilder

__version__ = "0.1.0"


class BipartiteGraphBuilder(GraphBuilder):
    """
    A specialized class for creating, manipulating, and visualizing bipartite
    graphs using the NetworkX library.

    Bipartite graphs are a specific type of graph where the vertex set can be
    divided into two disjoint subsets such that no two vertices within the same
    subset are adjacent. They are particularly useful in modeling relationships
    between two different classes of objects, such as in matching problems,
    network flows, and recommendation systems. This class allows for the creation
    of both weighted and unweighted bipartite graphs and can handle both directed
    and undirected variations.

    Parameters
    ----------
    set1_size: int
        The number of nodes in the first set.

    set2_size: int
        The number of nodes in the second set.

    directed: bool, default=False
        If True, initializes a directed bipartite graph.
        If False (default), initializes an undirected bipartite graph.

    name: str, default=None
        The name of the graph.

    metadata: Dict[str, Any], default=None
        A dictionary of metadata to associate with the graph.

    weights: Optional[List[Union[int, float]]], default=None
        A list of weights for the edges.
        If None (default), the edges are unweighted.
    """

    def __init__(self,
                 set1_size: int,
                 set2_size: int,
                 directed: bool = False,
                 name: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 weights: Optional[List[Union[int, float]]] = None):
        super().__init__(directed=directed,
                         name=name,
                         metadata=metadata)
        self.set1_size = set1_size
        self.set2_size = set2_size
        self.weights = weights
        self._create_bipartite_graph()

    def _create_bipartite_graph(self) -> None:
        """
        Initializes the bipartite graph by creating two disjoint sets of nodes
        and adding edges between every possible pair of nodes from these sets.
        If weights are provided, they are assigned to the edges.
        """
        # Validate the sizes of the sets
        if self.set1_size <= 0 or self.set2_size <= 0:
            raise ValueError("Both set1_size and set2_size must be positive integers.")

        # Define the nodes in each set
        set1_nodes = range(1, self.set1_size + 1)
        set2_nodes = range(self.set1_size + 1, self.set1_size + self.set2_size + 1)

        # Generate all possible edges between nodes in set1 and set2
        edges = [(u, v) for u in set1_nodes for v in set2_nodes]

        # Add the edges to the graph, considering weights if provided
        if self.weights:
            if len(self.weights) != len(edges):
                raise ValueError("The number of weights must match the number of edges.")
            self.add_edges(edges, self.weights)
        else:
            self.add_edges(edges)

    def get_bipartite_sets(self) -> Tuple[Set[int], Set[int]]:
        """
        Retrieve the two disjoint sets of nodes in the bipartite graph.

        Returns
        -------
        Tuple[Set[int], Set[int]]
            A tuple containing two sets: the first set contains the nodes in the
            first part of the bipartite graph, and the second set contains the nodes
            in the second part.
        """
        return nx.bipartite.sets(self.graph)

    def is_bipartite(self) -> bool:
        """
        Verify whether the current graph structure is bipartite.

        Returns
        -------
        bool
            True if the graph is bipartite, False otherwise.

        Raises
        ------
        ValueError
            If there is an error in checking the graph's bipartiteness.
        """
        try:
            return nx.is_bipartite(self.graph)
        except nx.NetworkXError as e:
            self.logger.error(f"NetworkXError checking bipartiteness: {e}")
            raise ValueError(f"Error checking bipartiteness: {e}")


class CompleteGraphBuilder(GraphBuilder):
    """
    A class to create, manipulate, and visualize complete graphs using the
    networkx library.

    A complete graph is a simple undirected graph in which every pair of
    distinct vertices is connected by a unique edge. This class supports
    both weighted and unweighted complete graphs.

    Parameters
    ----------
    num_nodes: int
        The number of nodes in the complete graph.

    weights: Optional[List[Union[int, float]]], optional
        A list of weights for the edges. If None (default), the edges are unweighted.

    name: str, default=None
        The name of the graph.

    metadata: Dict[str, Any], default=None
        A dictionary of metadata to associate with the graph.
    """

    def __init__(self,
                 num_nodes: int,
                 name: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 weights: Optional[List[Union[int, float]]] = None):
        super().__init__(directed=False, name=name, metadata=metadata)
        self.num_nodes = num_nodes
        self.weights = weights
        self._create_complete_graph()

    def _create_complete_graph(self) -> None:
        """
        Create a complete graph with the specified number of nodes.

        This method initializes the complete graph by connecting every pair of
        distinct vertices with a unique edge. If weights are provided, they are
        assigned to the edges.
        """
        # Validate number of nodes
        if self.num_nodes <= 1:
            raise ValueError("The number of nodes must be greater than 1 to form a complete graph.")

        # Create all possible edges between the nodes
        edges = [(u, v) for u in range(1, self.num_nodes + 1) for v in range(u + 1, self.num_nodes + 1)]

        # Add the edges to the graph, with weights if provided
        if self.weights:
            if len(self.weights) != len(edges):
                raise ValueError("The number of weights must match the number of edges.")
            self.add_edges(edges, self.weights)
        else:
            self.add_edges(edges)

    def get_complete_graph_edges(self) -> List[Tuple[int, int]]:
        """
        Get the list of edges in the complete graph.

        Returns
        -------
        List[Tuple[int, int]]
            A list of edges in the complete graph.
        """
        return list(self.graph.edges)


class CyclicGraphBuilder(GraphBuilder):
    """
    A specialized class for creating, manipulating, and visualizing cyclic
    graphs using the NetworkX library.

    A cyclic graph contains a single cycle, meaning that all nodes are
    connected in a closed loop. Cyclic graphs are useful in various applications
    such as routing, scheduling, and network analysis. This class allows for the
    creation of both weighted and unweighted cyclic graphs and can handle both
    directed and undirected variations.

    Parameters
    ----------
    num_nodes: int
        The number of nodes in the cyclic graph. A cyclic graph must have at
        least 3 nodes to form a valid cycle.

    directed: bool, default=False
        If True, initializes a directed cyclic graph.
        If False (default), initializes an undirected cyclic graph.

    name: str, default=None
        The name of the graph.

    metadata: Dict[str, Any], default=None
        A dictionary of metadata to associate with the graph.

    weights: Optional[List[Union[int, float]]], default=None
        A list of weights for the edges. If None (default), the edges are unweighted.
    """

    def __init__(self,
                 num_nodes: int,
                 directed: bool = False,
                 name: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 weights: Optional[List[Union[int, float]]] = None):
        super().__init__(directed=directed, name=name, metadata=metadata)
        self.num_nodes = num_nodes
        self.weights = weights
        self._create_cyclic_graph()

    def _create_cyclic_graph(self) -> None:
        """
        Initializes the cyclic graph by connecting all nodes in a closed loop.
        If weights are provided, they are assigned to the edges.

        Raises
        ------
        ValueError
            If the number of nodes is less than 3, as a valid cycle cannot
            exist with fewer than 3 nodes.
        """
        # Ensure a cyclic graph has at least 3 nodes
        if self.num_nodes < 3:
            raise ValueError("A cyclic graph requires at least 3 nodes.")

        # Create a list of edges that form a cycle
        edges = [(i, i + 1) for i in range(1, self.num_nodes)]
        edges.append((self.num_nodes, 1))  # Close the cycle by connecting the last node to the first

        # Add edges to the graph, considering weights if provided
        if self.weights:
            if len(self.weights) != len(edges):
                raise ValueError("The number of weights must match the number of edges.")
            self.add_edges(edges, self.weights)
        else:
            self.add_edges(edges)

    def get_cycle(self) -> List[Tuple[int, int]]:
        """
        Retrieve the cycle in the graph as a list of edges.

        Returns
        -------
        List[Tuple[int, int]]
            A list of edges representing the cycle in the graph.

        Raises
        ------
        ValueError
            If no cycle is found, indicating the graph may not be cyclic.
        """
        try:
            # Return the cycle as a list of edges
            return list(nx.find_cycle(self.graph))
        except nx.NetworkXNoCycle as e:
            self.logger.error(f"No cycle found in the graph: {e}")
            raise ValueError("The graph does not contain a cycle.")

    def is_cyclic(self) -> bool:
        """
        Verify if the current graph contains a cycle.

        Returns
        -------
        bool
            True if the graph contains a cycle, False otherwise.

        Raises
        ------
        ValueError
            If an error occurs while checking for a cycle.
        """
        try:
            return bool(nx.find_cycle(self.graph))
        except nx.NetworkXNoCycle:
            return False
        except nx.NetworkXError as e:
            self.logger.error(f"Error checking if the graph is cyclic: {e}")
            raise ValueError(f"Error checking if the graph is cyclic: {e}")


class GridGraphBuilder(GraphBuilder):
    """
    A specialized class for creating, manipulating, and visualizing grid graphs
    using the NetworkX library.

    A grid graph is a graph where the vertices are arranged in a rectangular grid,
    and edges connect adjacent vertices. Grid graphs are widely used in various
    fields such as image processing, geographic information systems (GIS), and
    simulation of physical systems. This class allows for the creation of both
    weighted and unweighted grid graphs and supports grid structures of any
    specified dimensions.

    Parameters
    ----------
    rows: int
        The number of rows in the grid.

    cols: int
        The number of columns in the grid.

    name: str, default=None
        The name of the graph.

    metadata: Dict[str, Any], default=None
        A dictionary of metadata to associate with the graph.

    weights: Optional[List[Union[int, float]]], default=None
        A list of weights for the edges. If None (default), the edges are unweighted.

    Attributes
    ----------
    node_map: Dict[Tuple[int, int], int]
        A mapping from grid coordinates (row, col) to unique integer node identifiers.
    """

    def __init__(self,
                 rows: int,
                 cols: int,
                 name: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 weights: Optional[List[Union[int, float]]] = None):
        super().__init__(directed=False, name=name, metadata=metadata)
        self.rows = rows
        self.cols = cols
        self.weights = weights
        self.node_map = {}  # To map (i, j) coordinates to integer identifiers
        self._create_grid_graph()

    def _create_grid_graph(self) -> None:
        """
        Initializes the grid graph by creating nodes for each grid position and
        adding edges between adjacent nodes. If weights are provided, they are
        assigned to the edges.

        Raises
        ------
        ValueError
            If the rows or columns are not positive integers.
        """
        # Validate the grid dimensions
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError("Both rows and columns must be positive integers.")

        # Initialize nodes in the grid and map them to unique integers
        node_id = 1
        for i in range(self.rows):
            for j in range(self.cols):
                self.node_map[(i, j)] = node_id
                node_id += 1

        # Create edges between adjacent nodes (both horizontal and vertical connections)
        edges = []
        for i in range(self.rows):
            for j in range(self.cols):
                if j + 1 < self.cols:
                    edges.append((self.node_map[(i, j)], self.node_map[(i, j + 1)]))  # Horizontal edge
                if i + 1 < self.rows:
                    edges.append((self.node_map[(i, j)], self.node_map[(i + 1, j)]))  # Vertical edge

        # Add the edges to the graph, with weights if provided
        if self.weights:
            if len(self.weights) != len(edges):
                raise ValueError("The number of weights must match the number of edges.")
            self.add_edges(edges, self.weights)
        else:
            self.add_edges(edges)

    def get_grid_graph_nodes(self) -> List[Tuple[int, int]]:
        """
        Retrieve the list of grid positions corresponding to the nodes in the graph.

        Returns
        -------
        List[Tuple[int, int]]
            A list of tuples representing the (row, col) coordinates of each node
            in the grid graph.
        """
        return list(self.node_map.keys())

    def get_grid_graph_edges(self) -> List[Tuple[int, int]]:
        """
        Retrieve the list of edges in the grid graph.

        Returns
        -------
        List[Tuple[int, int]]
            A list of tuples representing the edges between nodes in the grid graph.
        """
        return list(self.graph.edges)


class LinearGraphBuilder(GraphBuilder):
    """
    A specialized class for creating, manipulating, and visualizing linear (path)
    graphs using the NetworkX library.

    A linear graph, also known as a path graph, is a simple structure where each
    node is connected in a straight line to the next, forming a single path from
    the first node to the last. Linear graphs are useful in scenarios where you
    need to model sequences, processes, or simple connections between consecutive
    elements. This class allows for the creation of both weighted and unweighted
    linear graphs, and supports both directed and undirected versions.

    Parameters
    ----------
    num_nodes: int
        The number of nodes in the linear graph.

    directed: bool, default=False
        If True, initializes a directed linear graph.
        If False (default), initializes an undirected linear graph.

    name: str, default=None
        The name of the graph.

    metadata: Dict[str, Any], default=None
        A dictionary of metadata to associate with the graph.

    weights: Optional[List[Union[int, float]]], default=None
        A list of weights for the edges.
        If None (default), the edges are unweighted.

    Methods
    -------
    add_node_to_end(weight: Optional[Union[int, float]] = None) -> None:
        Adds a new node to the end of the linear graph, optionally with a weighted edge.

    get_path() -> List[int]:
        Returns the list of nodes in the path from the first to the last node in the graph.
    """

    def __init__(self,
                 num_nodes: int,
                 directed: bool = False,
                 name: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 weights: Optional[List[Union[int, float]]] = None):
        super().__init__(directed=directed, name=name, metadata=metadata)
        self.num_nodes = num_nodes
        self.weights = weights
        self._create_linear_graph()

    def _create_linear_graph(self) -> None:
        """
        Initializes the linear graph (path graph) by connecting nodes in a straight
        line from the first to the last. If weights are provided, they are assigned
        to the edges.

        Raises
        ------
        ValueError
            If the number of nodes is less than 2, as a linear graph requires at least
            two nodes.
        """
        if self.num_nodes < 2:
            raise ValueError("A linear graph requires at least 2 nodes.")

        # Create the linear path by connecting each node to the next
        edges = [(i, i + 1) for i in range(1, self.num_nodes)]

        # Add the edges to the graph, with weights if provided
        if self.weights:
            if len(self.weights) != len(edges):
                raise ValueError("The number of weights must match the number of edges.")
            self.add_edges(edges, self.weights)
        else:
            self.add_edges(edges)

    def add_node_to_end(self,
                        weight: Optional[Union[int, float]] = None) -> None:
        """
        Add a new node to the end of the linear graph, extending the path by one node.

        Parameters
        ----------
        weight: Optional[Union[int, float]], optional
            The weight of the edge connecting the new node to the last node in the graph.
            If None, the edge is added without a weight.
        """
        new_node = self.num_nodes + 1
        self.add_edge(self.num_nodes, new_node, weight)
        self.num_nodes += 1

    def get_path(self) -> List[int]:
        """
        Retrieve the sequence of nodes representing the path from the first to the last
        node in the graph.

        Returns
        -------
        List[int]
            A list of node identifiers representing the linear path in the graph.
        """
        return list(nx.shortest_path(self.graph, source=1, target=self.num_nodes))


class StarGraphBuilder(GraphBuilder):
    """
    A class to create, manipulate, and visualize star graphs using the
    networkx library.

    A star graph is a type of tree graph where all nodes are connected to a
    central node. This class supports both weighted and unweighted star graphs.

    Parameters
    ----------
    num_leaves: int
        The number of leaf nodes in the star graph.

    weights: Optional[List[Union[int, float]]], optional
        A list of weights for the edges. If None (default), the edges are unweighted.
    """

    def __init__(self,
                 num_leaves: int,
                 name: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 weights: Optional[List[Union[int, float]]] = None):
        super().__init__(directed=False, name=name, metadata=metadata)
        self.num_leaves = num_leaves
        self.weights = weights
        self._create_star_graph()

    def _create_star_graph(self) -> None:
        """
        Create a star graph with the specified number of leaf nodes.

        This method initializes the star graph by connecting all leaf nodes to a
        central node. If weights are provided, they are assigned to the edges.
        """
        # Validate the number of leaves
        if self.num_leaves <= 0:
            raise ValueError("The number of leaf nodes must be a positive integer.")

        # Central node is labeled as 0
        center_node = 0
        leaf_nodes = range(1, self.num_leaves + 1)

        # Create edges from the center node to all leaf nodes
        edges = [(center_node, leaf_node) for leaf_node in leaf_nodes]

        # Add the edges to the graph, with weights if provided
        if self.weights:
            if len(self.weights) != len(edges):
                raise ValueError("The number of weights must match the number of edges.")
            self.add_edges(edges, self.weights)
        else:
            self.add_edges(edges)

    def get_center_node(self) -> int:
        """
        Get the center node of the star graph.

        Returns
        -------
        int
            The center node of the star graph.
        """
        return 0

    def get_leaf_nodes(self) -> List[int]:
        """
        Get the list of leaf nodes in the star graph.

        Returns
        -------
        List[int]
            A list of leaf nodes in the star graph.
        """
        return list(range(1, self.num_leaves + 1))


class TreeGraphBuilder(GraphBuilder):
    """
    A class to create, manipulate, and visualize tree graphs using the
    networkx library.

    A tree is a connected acyclic graph, meaning it has no cycles and exactly one
    path exists between any two nodes. This class supports both directed and
    undirected tree graphs.

    Parameters
    ----------
    num_nodes: int
        The number of nodes in the tree.

    directed: bool, default=False
        If True, creates a directed tree. If False (default), creates an undirected tree.

    name: str, default=None
        The name of the graph.

    metadata: Dict[str, Any], default=None
        A dictionary of metadata to associate with the graph.
    """

    def __init__(self,
                 num_nodes: int,
                 directed: bool = False,
                 name: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        super().__init__(directed=directed, name=name, metadata=metadata)
        self.num_nodes = num_nodes
        self._create_tree_graph()

    def _create_tree_graph(self) -> None:
        """
        Create a tree graph with the specified number of nodes.

        This method initializes the tree graph by creating a simple binary tree
        structure where each node (except leaves) has two children.
        """
        # Validate the number of nodes
        if self.num_nodes <= 1:
            raise ValueError("The number of nodes must be greater than 1 to form a tree.")

        edges = []
        for i in range(1, self.num_nodes // 2 + 1):
            left_child = 2 * i
            right_child = 2 * i + 1
            if left_child <= self.num_nodes:
                edges.append((i, left_child))
            if right_child <= self.num_nodes:
                edges.append((i, right_child))

        # Add the edges to the graph
        self.add_edges(edges)

    def get_root_node(self) -> int:
        """
        Get the root node of the tree graph.

        Returns
        -------
        int
            The root node of the tree graph.
        """
        return 1

    def get_leaf_nodes(self) -> List[int]:
        """
        Get the list of leaf nodes in the tree graph.

        Returns
        -------
        List[int]
            A list of leaf nodes in the tree graph. Leaf nodes are nodes with degree 1.
        """
        return [node for node in self.graph.nodes if self.graph.degree[node] == 1]


class WeightedGraphBuilder(GraphBuilder):
    """
    A class to create, manipulate, and visualize generic weighted graphs using the
    networkx library.

    This class allows the creation of both directed and undirected graphs where edges
    can have associated weights. It provides functionality to add edges with specific
    weights and to manage the graph structure.

    Parameters
    ----------
    edges: List[Tuple[int, int]]
        A list of edges where each edge is represented by a tuple of two node identifiers.

    directed: bool, optional, default=False
        If True, creates a directed graph. If False (default), creates an undirected graph.

    name: str, default=None
        The name of the graph.

    metadata: Dict[str, Any], default=None
        A dictionary of metadata to associate with the graph.

    weights: Optional[List[Union[int, float]]], optional
        A list of weights corresponding to the edges. If None (default), the edges are unweighted.
    """

    def __init__(self,
                 edges: List[Tuple[int, int]],
                 directed: bool = False,
                 name: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 weights: Optional[List[Union[int, float]]] = None):
        super().__init__(directed=directed, name=name, metadata=metadata)
        self.edges = edges
        self.weights = weights
        self._create_weighted_graph()

    def _create_weighted_graph(self) -> None:
        """
        Create a weighted graph with the specified edges and weights.

        This method initializes the graph by adding the edges with the associated weights.
        If weights are provided, they are assigned to the edges; otherwise, the edges are
        added as unweighted.
        """
        if self.weights:
            if len(self.weights) != len(self.edges):
                raise ValueError("The number of weights must match the number of edges.")
            self.add_edges(self.edges, self.weights)
        else:
            self.add_edges(self.edges)

    def add_edge_with_weight(self,
                             node1: int,
                             node2: int,
                             weight: Optional[Union[int, float]] = None) -> None:
        """
        Add an edge with a specific weight to the graph.

        Parameters
        ----------
        node1: int
            The first node identifier.

        node2: int
            The second node identifier.

        weight: Optional[Union[int, float]], optional
            The weight of the edge. If None, the edge is added without a weight.
        """
        self.add_edge(node1, node2, weight)

    def get_edge_weight(self, node1: int, node2: int) -> Optional[Union[int, float]]:
        """
        Retrieve the weight of a specific edge.

        Parameters
        ----------
        node1: int
            The first node identifier.

        node2: int
            The second node identifier.

        Returns
        -------
        Optional[Union[int, float]]
            The weight of the edge if it exists, otherwise None.
        """
        if self.graph.has_edge(node1, node2):
            return self.graph[node1][node2].get('weight', None)
        else:
            raise ValueError(f"No edge exists between {node1} and {node2}.")


def plot_graph_examples():
    # Create a figure with subplots arranged in a 3x2 grid
    fig, axs = plt.subplots(2, 3, figsize=(16, 8))

    # 1. Bipartite Graph Example without Weights
    print("\n=== Bipartite Graph Example ===")
    bipartite_builder = BipartiteGraphBuilder(set1_size=3, set2_size=3)
    axs[0, 0].set_title("Bipartite Graph")
    bipartite_pos = nx.spring_layout(bipartite_builder.graph)
    nx.draw(bipartite_builder.graph, pos=bipartite_pos, with_labels=True, ax=axs[0, 0])
    print("Bipartite Sets:", bipartite_builder.get_bipartite_sets())
    print("Is Bipartite:", bipartite_builder.is_bipartite())

    # 2. Complete Graph Example
    print("\n=== Complete Graph Example ===")
    complete_builder = CompleteGraphBuilder(num_nodes=4, weights=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    axs[0, 1].set_title("Complete Graph")
    complete_pos = nx.spring_layout(complete_builder.graph)
    nx.draw(complete_builder.graph, pos=complete_pos, with_labels=True, ax=axs[0, 1])
    print("Complete Graph Edges:", complete_builder.get_complete_graph_edges())

    # 3. Cyclic Graph Example
    print("\n=== Cyclic Graph Example ===")
    cyclic_builder = CyclicGraphBuilder(num_nodes=5, directed=True, weights=[1, 2, 3, 4, 5])
    axs[0, 2].set_title("Cyclic Graph")
    cyclic_pos = nx.spring_layout(cyclic_builder.graph)
    nx.draw(cyclic_builder.graph, pos=cyclic_pos, with_labels=True, ax=axs[0, 2])

    # 4. Grid Graph Example
    print("\n=== Grid Graph Example ===")
    grid_builder = GridGraphBuilder(rows=3, cols=3,
                                    weights=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    axs[1, 0].set_title("Grid Graph")
    grid_pos = nx.spring_layout(grid_builder.graph)
    nx.draw(grid_builder.graph, pos=grid_pos, with_labels=True, ax=axs[1, 0])
    print("Grid Graph Nodes:", grid_builder.get_grid_graph_nodes())
    print("Grid Graph Edges:", grid_builder.get_grid_graph_edges())

    # 5. Star Graph Example
    print("\n=== Star Graph Example ===")
    star_builder = StarGraphBuilder(num_leaves=9)  # , weights=[1.0, 2.0, 3.0, 4.0, 5.0])
    axs[1, 1].set_title("Star Graph")
    star_pos = nx.spring_layout(star_builder.graph)
    nx.draw(star_builder.graph, pos=star_pos, with_labels=True, ax=axs[1, 1])
    print("Star Graph Center Node:", star_builder.get_center_node())
    print("Star Graph Leaf Nodes:", star_builder.get_leaf_nodes())

    # 6. Tree Graph Example
    print("\n=== Tree Graph Example ===")
    tree_builder = TreeGraphBuilder(num_nodes=15, directed=False)
    axs[1, 2].set_title("Tree Graph")
    tree_pos = nx.spring_layout(tree_builder.graph)
    nx.draw(tree_builder.graph, pos=tree_pos, with_labels=True, ax=axs[1, 2])
    print("Tree Graph Root Node:", tree_builder.get_root_node())
    print("Tree Graph Leaf Nodes:", tree_builder.get_leaf_nodes())

    # Adjust layout to ensure the plots do not overlap
    plt.tight_layout()
    plt.show()


def main():
    plot_graph_examples()

    # # Bipartite Graph Example
    # print("\n=== Bipartite Graph Example ===")
    # bipartite_builder = BipartiteGraphBuilder(set1_size=3, set2_size=3,
    #                                           weights=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    # bipartite_builder.visualize_graph()
    # print("Bipartite Sets:", bipartite_builder.get_bipartite_sets())
    # print("Is Bipartite:", bipartite_builder.is_bipartite())
    #
    # # Bipartite Graph Example without Weights
    # print("\n=== Bipartite Graph Example ===")
    # bipartite_builder = BipartiteGraphBuilder(set1_size=3, set2_size=3)
    # bipartite_builder.visualize_graph()
    # print("Bipartite Sets:", bipartite_builder.get_bipartite_sets())
    # print("Is Bipartite:", bipartite_builder.is_bipartite())
    #
    # # Complete Graph Example
    # print("\n=== Complete Graph Example ===")
    # complete_builder = CompleteGraphBuilder(num_nodes=4, weights=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    # complete_builder.visualize_graph()
    # print("Complete Graph Edges:", complete_builder.get_complete_graph_edges())
    #
    # # Cyclic Graph Example
    # print("\n=== Cyclic Graph Example ===")
    # cyclic_builder = CyclicGraphBuilder(num_nodes=5, directed=True, weights=[1, 2, 3, 4, 5])
    # cyclic_builder.visualize_graph()
    #
    # # Grid Graph Example
    # print("\n=== Grid Graph Example ===")
    # grid_builder = GridGraphBuilder(rows=3, cols=3,
    #                                 weights=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    # grid_builder.visualize_graph()
    # print("Grid Graph Nodes:", grid_builder.get_grid_graph_nodes())
    # print("Grid Graph Edges:", grid_builder.get_grid_graph_edges())
    #
    # # Linear Graph Example
    # print("\n=== Linear Graph Example ===")
    # linear_builder = LinearGraphBuilder(num_nodes=5, directed=True, weights=[1, 2, 3, 4])
    # linear_builder.visualize_graph()
    #
    # # Star Graph Example
    # print("\n=== Star Graph Example ===")
    # star_builder = StarGraphBuilder(num_leaves=9) #, weights=[1.0, 2.0, 3.0, 4.0, 5.0])
    # star_builder.visualize_graph()
    # print("Star Graph Center Node:", star_builder.get_center_node())
    # print("Star Graph Leaf Nodes:", star_builder.get_leaf_nodes())
    #
    # # Tree Graph Example
    # print("\n=== Tree Graph Example ===")
    # tree_builder = TreeGraphBuilder(num_nodes=15, directed=False)
    # tree_builder.visualize_graph()
    # print("Tree Graph Root Node:", tree_builder.get_root_node())
    # print("Tree Graph Leaf Nodes:", tree_builder.get_leaf_nodes())
    #
    # # Weighted Graph Example
    # print("\n=== Weighted Graph Example ===")
    # edges = [(1, 2), (2, 3), (3, 4), (4, 5)]
    # weighted_builder = WeightedGraphBuilder(edges=edges, directed=False, weights=[1.0, 2.0, 3.0, 4.0])
    # weighted_builder.visualize_graph()
    # print("Weight of edge (2, 3):", weighted_builder.get_edge_weight(2, 3))
    #
    # # Add an edge with a specific weight to the weighted graph
    # weighted_builder.add_edge_with_weight(5, 1, weight=5.0)
    # print("Added an edge with weight to the weighted graph.")
    # weighted_builder.visualize_graph()
    # print("Weight of edge (5, 1):", weighted_builder.get_edge_weight(5, 1))


if __name__ == "__main__":
    main()
