"""

"""

from typing import Dict, Any, List, Tuple, Optional, Union, Set

import matplotlib.pyplot as plt
import networkx as nx


class GraphBuilderBase:
    """
    Base class for building and manipulating graphs using the networkx library.

    It serves as the foundation for more specialized graph builders, and provides
    basic functionality for initializing a graph and managing common graph attributes.

    Parameters
    ----------
    directed: bool, default=False
        If True, initializes a directed graph.
        If False (default), initializes an undirected graph.

    name: str, optional
        The name of the graph.

    metadata: Dict[str, Any], optional
        A dictionary of metadata to associate with the graph.

    Attributes
    ----------
    graph: networkx.Graph or networkx.DiGraph
        The underlying graph object, which can be either directed or undirected.
    """

    graph_types = {
        "undirected": nx.Graph,
        "directed": nx.DiGraph
    }

    def __init__(self,
                 directed: bool = False,
                 name: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        self.directed = directed
        self.graph = None

        # Initialize the appropriate graph by type
        self._initialize_graph()

        # Set the graph name and metadata if provided
        if name is not None:
            self.name = name
        if metadata is not None:
            self.metadata = metadata

    @property
    def name(self) -> str:
        """
        Property to get or set the name of the graph.

        Returns
        -------
        str
            The name of the graph, or an empty string if no name is set.
        """
        return self.graph.graph.get('name', '')

    @name.setter
    def name(self, value: str) -> None:
        """
        Setter for the name property.

        Parameters
        ----------
        value: str
            The name to assign to the graph.
        """
        self.graph.graph['name'] = value

    @property
    def metadata(self) -> Dict[str, Any]:
        """
        Property to get or set metadata for the graph.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the metadata of the graph.
        """
        return {key: value for key, value in self.graph.graph.items() if key != 'name'}

    @metadata.setter
    def metadata(self, value: Dict[str, Any]) -> None:
        """
        Setter for the metadata property.

        Parameters
        ----------
        value: Dict[str, Any]
            A dictionary containing metadata to associate with the graph.
        """
        self.graph.graph.update(value)

    def _initialize_graph(self) -> None:
        """
        Initialize the graph based on the specified type (directed or undirected).
        """
        graph_type = "directed" if self.directed else "undirected"
        self.graph = self.graph_types[graph_type]

    def _convert_to_directed(self) -> None:
        """
        Convert the current graph to a directed graph.

        Raises
        ------
        ValueError
            If the graph is already directed or is not an undirected graph.
        """
        if isinstance(self.graph, nx.Graph) and not isinstance(self.graph, nx.DiGraph):
            self.graph = self.graph.to_directed()
        else:
            raise ValueError("The graph is already directed or is not an undirected graph.")

    def _convert_to_undirected(self) -> None:
        """
        Convert the current graph to an undirected graph.

        Raises
        ------
        ValueError
            If the graph is already undirected or is not a directed graph.
        """
        if isinstance(self.graph, nx.DiGraph):
            self.graph = self.graph.to_undirected()
        else:
            raise ValueError("The graph is already undirected or is not a directed graph.")

    def _add_nodes(self,
                   nodes: List[int]) -> None:
        """
        Add multiple nodes to the graph.

        Parameters
        ----------
        nodes: List[int]
            A list of node identifiers to add.
            Each node identifier must be an integer.

        Raises
        ------
        ValueError
            If any of the node identifiers are not integers.

        Notes
        -----
        This method allows to add multiple nodes to the graph in one operation.
        The node identifiers must all be integers, and the method will raise an
        error if any non-integer values are detected.
        """
        if not all(isinstance(node, int) for node in nodes):
            raise ValueError("All node identifiers must be integers.")

        self.graph.add_nodes_from(nodes)

    def _add_node(self,
                  node: int,
                  **attributes) -> None:
        """
        Add a node to the graph.

        Parameters
        ----------
        node: int
            The node to add.
            The node identifier must be an integer.

        **attributes: Dict[str, Any], optional
            Additional attributes to associate with the node.
            If not provided, nodes will be assigned the following default
            attributes:
                - "color": "blue"
                - "label": node

        Raises
        ------
        ValueError
            If the node identifier is not an integer.

        Notes
        -----
        The default attributes are useful for visualization and identification
        purposes. The "label" attribute is set to the node's identifier and the
        "color" attribute is set to "blue" by default.
        """
        if not isinstance(node, int):
            raise ValueError("Node identifier must be an integer.")

        # Set default attributes if none are provided
        if not attributes:
            attributes["label"] = node
            attributes["color"] = "blue"

        self.graph.add_node(node, **attributes)

    def _add_node_attribute(self,
                            node: int,
                            **attributes: Dict[str, Any]) -> None:
        """
        Add or update attributes for a specific node.

        Parameters
        ----------
        node: int
            The node to which the attributes will be added.
        attributes: Dict[str, Any]
            Key-value pairs representing the attributes to add or update.

        Raises
        ------
        ValueError
            If the node does not exist in the graph.
        """
        if self.graph.has_node(node):
            for key, value in attributes.items():
                self.graph.nodes[node][key] = value
        else:
            raise ValueError(f"Node {node} does not exist in the graph.")

    def _add_edges(self,
                   edges: List[Tuple[int, int]],
                   weights: Optional[List[float]] = None) -> None:
        """
        Add multiple edges to the graph, with optional weights.

        Parameters
        ----------
        edges: List[Tuple[int, int]]
            A list of edges, where each edge is a tuple of two node identifiers.

        weights: Optional[List[float]], default=None
            A list of weights corresponding to each edge.
            Weights can be float or integer.

        Raises
        ------
        ValueError
            If any of the node identifiers in edges are not integers.
            If the number of weights provided does not match the number of edges.
            If any weight is not a numeric value (int or float).

        Notes
        -----
        This method allows you to add multiple edges to the graph in one operation.
        If no weights are provided, the edges are added without a weight attribute.
        If weights are provided, they must match the number of edges and be numeric
        values.
        """
        if not all(isinstance(edge, tuple) and len(edge) == 2 for edge in edges):
            raise ValueError("Each edge must be a tuple of two node identifiers.")
        if not all(isinstance(node, int) for edge in edges for node in edge):
            raise ValueError("All node identifiers in edges must be integers.")

        if weights is not None:
            if len(weights) != len(edges):
                raise ValueError("The number of weights must match the number of edges.")
            if not all(isinstance(weight, (int, float)) for weight in weights):
                raise ValueError("All edge weights must be numeric values (int or float).")
            # Add weights to each provided edge
            for edge, weight in zip(edges, weights):
                self._add_edge(edge[0], edge[1], weight)
        else:
            self.graph.add_edges_from(edges)

    def _add_edge(self,
                  node1: int,
                  node2: int,
                  weight: Optional[float] = None,
                  **attributes) -> None:
        """
        Add an edge between two nodes with an optional weight.

        Parameters
        ----------
        node1: int
            The first node identifier. Must be an integer.

        node2: int
            The second node identifier. Must be an integer.

        weight: Optional[float], optional
            The weight of the edge. Can be a float or integer.

        attributes: Dict[str, Any], optional
            Additional attributes to associate with the edge.

        Raises
        ------
        ValueError
            If either node identifier is not an integer.
            If the weight is provided and is not a numeric value (int or float).

        Notes
        -----
        This method allows you to add an edge between two nodes with optional
        attributes, including a weight. If no weight is provided, the edge will
        be added without a weight attribute. If attributes are provided, they
        will be associated with the edge.
        """
        if not isinstance(node1, int) or not isinstance(node2, int):
            raise ValueError("Both node identifiers must be integers.")
        if weight is not None and not isinstance(weight, (int, float)):
            raise ValueError("Edge weight must be a numeric value (int or float).")

        if weight is not None:
            attributes['weight'] = weight

        self.graph.add_edge(node1, node2, **attributes)

    def _add_edge_attribute(self,
                            node1: int,
                            node2: int,
                            **attributes: Dict[str, Any]) -> None:
        """
        Add or update attributes for a specific edge.

        Parameters
        ----------
        node1: int
            The first node of the edge.
        node2: int
            The second node of the edge.
        attributes: Dict[str, Any]
            Key-value pairs representing the attributes to add or update.

        Raises
        ------
        ValueError
            If the edge does not exist between the specified nodes.
        """
        if self.graph.has_edge(node1, node2):
            for key, value in attributes.items():
                self.graph[node1][node2][key] = value
        else:
            raise ValueError(f"No edge exists between {node1} and {node2}.")

    def _remove_edge_weight(self,
                            node1: int,
                            node2: int) -> None:
        """
        Remove the weight attribute from an edge.

        Parameters
        ----------
        node1: int
            The first node of the edge.
        node2: int
            The second node of the edge.

        Raises
        ------
        ValueError
            If the edge does not exist or if the edge does not have
            a weight attribute.
        """
        if self.graph.has_edge(node1, node2):
            if 'weight' in self.graph[node1][node2]:
                del self.graph[node1][node2]['weight']
            else:
                raise ValueError(f"Edge between {node1} and {node2} "
                                 f"does not have a weight attribute.")
        else:
            raise ValueError(f"No edge exists between {node1} and {node2}")

    def _update_edge_weight(self,
                            node1: int,
                            node2: int,
                            new_weight: Union[int, float]) -> None:
        """
        Update the weight of an existing edge.

        Parameters
        ----------
        node1: int
            The first node of the edge.

        node2: int
            The second node of the edge.

        new_weight: Union[int, float]
            The new weight to assign to the edge.

        Raises
        ------
        ValueError
            If the edge does not exist.
        """
        if self.graph.has_edge(node1, node2):
            self.graph[node1][node2]['weight'] = new_weight
        else:
            raise ValueError(f"No edge exists between {node1} and {node2}")


class GraphBuilder(GraphBuilderBase):
    """
    A class to create and manipulate simple graphs using the networkx library.

    """

    def __init__(self,
                 directed: bool = False):
        super().__init__(directed)

    def convert_to_directed(self) -> None:
        """
        Public method to convert the current graph to a directed graph.
        """
        self._convert_to_directed()

    def convert_to_undirected(self) -> None:
        """
        Public method to convert the current graph to an undirected graph.
        """
        self._convert_to_undirected()

    def get_graph(self) -> Union[nx.Graph, nx.DiGraph]:
        """
        Return the underlying networkx graph object.

        Returns
        -------
        networkx.Graph or networkx.DiGraph
            The underlying graph object.
        """
        return self.graph

    def load_from_file(self,
                       file_path: str,
                       file_format: str = "gml") -> None:
        """
        Load a graph from a file in the specified format.

        Parameters
        ----------
        file_path: str
            The path from which the graph file will be loaded.

        file_format: str, default="gml"
            The format to load the graph from.
            Supported formats include 'gml', 'graphml', and 'json'.

        Raises
        ------
        ValueError
            If an unsupported file format is provided.

        IOError
            If there is an error loading the file.
        """

        format_to_function = {
            'gml': nx.read_gml,
            'graphml': nx.read_graphml,
            'json': nx.readwrite.json_graph.node_link_graph
        }

        if file_format not in format_to_function:
            raise ValueError(f"Unsupported file format: {file_format}")

        try:
            self.graph = format_to_function[file_format](file_path)
        except IOError as e:
            raise IOError(f"Failed to load graph from {file_path}: {e}")

    def save_to_file(self,
                     file_path: str,
                     file_format: str = "gml") -> None:
        """
        Save the graph to a file in the specified format.

        Parameters
        ----------
        file_path: str
            The path where the graph file will be saved.

        file_format: str, default="gml"
            The format to save the graph in.
            Supported formats include 'gml', 'graphml', and 'json'.

        Raises
        ------
        ValueError
            If an unsupported file format is provided.

        IOError
            If there is an error saving the file.
        """

        format_to_function = {
            'gml': nx.write_gml,
            'graphml': nx.write_graphml,
            'json': nx.readwrite.json_graph.node_link_data
        }

        if file_format not in format_to_function:
            raise ValueError(f"Unsupported file format: {file_format}")

        try:
            format_to_function[file_format](self.graph, file_path)
        except IOError as e:
            raise IOError(f"Failed to save graph to {file_path}: {e}")

    def visualize(self,
                  with_labels: bool = True,
                  node_color: Union[str, List[str]] = "lightblue",
                  edge_color: Union[str, List[str]] = "gray",
                  node_size: int = 500,
                  font_size: int = 10,
                  edge_labels: bool = True,
                  layout: str = "spring") -> None:
        """
        Visualize the graph using matplotlib.

        Parameters
        ----------
        with_labels: bool, default=True
            Whether to display the labels of the nodes.

        node_color: str or list of str, default="lightblue"
            The color(s) of the nodes.

        edge_color: str or list of str, default="gray"
            The color(s) of the edges.

        node_size: int, default=500
            The size of the nodes.

        font_size: int, default=10
            The font size of the labels.

        edge_labels: bool, default=True
            Whether to display the labels of the edges, typically the weights.

        layout: str, default="spring"
            The layout algorithm to use for positioning nodes.
            Options include 'spring' (default), 'circular', 'kamada_kawai',
            'random', 'shell', and 'spectral'.
        """
        # Select layout
        layout_funcs = {
            'spring': nx.spring_layout,
            'circular': nx.circular_layout,
            'kamada_kawai': nx.kamada_kawai_layout,
            'random': nx.random_layout,
            'shell': nx.shell_layout,
            'spectral': nx.spectral_layout,
        }

        if layout not in layout_funcs:
            raise ValueError(f"Unsupported layout type: {layout}. "
                             f"Choose from {list(layout_funcs.keys())}.")

        pos = layout_funcs[layout](self.graph)

        # Draw the graph
        nx.draw(G=self.graph,
                pos=pos,
                with_labels=with_labels,
                node_color=node_color,
                edge_color=edge_color,
                node_size=node_size,
                font_size=font_size)

        # Optionally draw edge labels (e.g., weights)
        if edge_labels:
            edge_labels_dict = nx.get_edge_attributes(self.graph, 'weight')
            nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels_dict)

        # Show the plot
        plt.show()

    def add_nodes(self,
                  nodes: List[int]) -> None:
        """
        Public method to add multiple nodes to the graph.

        Parameters
        ----------
        nodes: List[int]
            A list of node identifiers to add.
            Each node identifier must be an integer.
        """
        self._add_nodes(nodes)

    def add_node(self,
                 node: int,
                 **attributes: Dict[str, Any]) -> None:
        """
        Public method to add a node to the graph.

        Parameters
        ----------
        node: int
            The node to add.
            The node identifier must be an integer.

        **attributes: Dict[str, Any], optional
            Additional attributes to associate with the node.
        """
        self._add_node(node, **attributes)

    def add_node_attribute(self,
                           node: int,
                           **attributes: Dict[str, Any]) -> None:
        """
        Public method to add or update attributes for a specific node.

        Parameters
        ----------
        node: int
            The node to which the attributes will be added.

        attributes: Dict[str, Any]
            Key-value pairs representing the attributes to add or update.
        """
        self._add_node_attribute(node, **attributes)

    def add_edges(self,
                  edges: List[Tuple[int, int]],
                  weights: Optional[List[float]] = None) -> None:
        """
        Public method to add multiple edges to the graph, with optional weights.

        Parameters
        ----------
        edges: List[Tuple[int, int]]
            A list of edges, where each edge is a tuple of two node identifiers.

        weights: Optional[List[float]], default=None
            A list of weights corresponding to each edge.
            Weights can be float or integer.
        """
        self._add_edges(edges, weights)

    def add_edge_attribute(self,
                           node1: int,
                           node2: int,
                           **attributes: Dict[str, Any]) -> None:
        """
        Public method to add or update attributes for a specific edge.

        Parameters
        ----------
        node1: int
            The first node of the edge.

        node2: int
            The second node of the edge.

        attributes: Dict[str, Any]
            Key-value pairs representing the attributes to add or update.
        """
        self._add_edge_attribute(node1, node2, **attributes)

    def remove_edge_weight(self,
                           node1: int,
                           node2: int) -> None:
        """
        Public method to remove the weight attribute from an edge.

        Parameters
        ----------
        node1: int
            The first node of the edge.

        node2: int
            The second node of the edge.
        """
        self._remove_edge_weight(node1, node2)

    def update_edge_weight(self,
                           node1: int,
                           node2: int,
                           new_weight: Union[int, float]) -> None:
        """
        Public method to update the weight of an existing edge.

        Parameters
        ----------
        node1: int
            The first node of the edge.

        node2: int
            The second node of the edge.

        new_weight: Union[int, float]
            The new weight to assign to the edge.
        """
        self._update_edge_weight(node1, node2, new_weight)

    def add_edge(self,
                 node1: int,
                 node2: int,
                 weight: Optional[float] = None,
                 **attributes: Dict[str, Any]) -> None:
        """
        Public method to add an edge between two nodes with an optional weight.

        Parameters
        ----------
        node1: int
            The first node identifier. Must be an integer.

        node2: int
            The second node identifier. Must be an integer.

        weight: Optional[float], optional
            The weight of the edge. Can be a float or integer.

        **attributes: Dict[str, Any], optional
            Additional attributes to associate with the edge.
        """
        self._add_edge(node1, node2, weight, **attributes)

    def is_connected(self) -> bool:
        """
        Check if the graph is fully connected.

        Returns
        -------
        bool
            True if the graph is connected, False otherwise.
        """
        return nx.is_connected(self.graph)

    def get_graph_radius(self) -> int:
        """
        Return the radius of the graph.

        The radius is the shortest maximum distance from any node to all other nodes.

        Returns
        -------
        int
            The radius of the graph.

        Raises
        ------
        nx.NetworkXError
            If the graph is not connected.
        """
        return nx.radius(self.graph)

    def get_graph_diameter(self) -> int:
        """
        Return the diameter of the graph.

        The diameter is the longest shortest path between any two nodes.

        Returns
        -------
        int
            The diameter of the graph.

        Raises
        ------
        nx.NetworkXError
            If the graph is not connected.
        """
        return nx.diameter(self.graph)

    def get_graph_density(self) -> float:
        """
        Return the density of the graph.

        Returns
        -------
        float
            The density of the graph.
        """
        return nx.density(self.graph)

    def get_all_node_degrees(self) -> Dict[int, int]:
        """
        Return a dictionary mapping nodes to their degrees.

        Returns
        -------
        Dict[int, int]
            A dictionary with nodes as keys and their degrees as values.
        """
        return dict(self.graph.degree)

    def get_all_shortest_paths(self) -> Dict[Tuple[int, int], List[int]]:
        """
        Return the shortest paths between all pairs of nodes.

        Returns
        -------
        Dict[Tuple[int, int], List[int]]
            A dictionary with (node1, node2) tuples as keys and the shortest
            paths as values.
        """
        return dict(nx.all_pairs_shortest_path(self.graph))

    def get_node_degree(self,
                        node: int) -> int:
        """
        Return the degree of a specific node.

        Parameters
        ----------
        node: int
            The node identifier.

        Returns
        -------
        int
            The degree of the node.

        Raises
        ------
        ValueError
            If the node does not exist in the graph.
        """
        if not self.graph.has_node(node):
            raise ValueError(f"Node {node} does not exist in the graph.")
        return self.graph.degree[node]

    def get_shortest_path(self,
                          node1: int,
                          node2: int) -> List[int]:
        """
        Return the shortest path between two nodes.

        Parameters
        ----------
        node1: int
            The first node identifier.

        node2: int
            The second node identifier.

        Returns
        -------
        List[int]
            A list of nodes representing the shortest path.

        Raises
        ------
        nx.NetworkXNoPath
            If no path exists between node1 and node2.
        """
        return nx.shortest_path(self.graph, source=node1, target=node2)

    def get_connected_components(self) -> List[Set[int]]:
        """
        Return a list of sets, each representing a connected component of the graph.

        Returns
        -------
        List[Set[int]]
            A list of sets, each containing the nodes in a connected component.
        """
        return [component for component in nx.connected_components(self.graph)]

    def get_clustering_coefficient(self,
                                   node: int) -> float:
        """
        Return the clustering coefficient of a specific node.

        Parameters
        ----------
        node: int
            The node identifier.

        Returns
        -------
        float
            The clustering coefficient of the node.

        Raises
        ------
        ValueError
            If the node does not exist in the graph.
        """
        if not self.graph.has_node(node):
            raise ValueError(f"Node {node} does not exist in the graph.")
        return nx.clustering(self.graph, node)

    def get_average_clustering_coefficient(self) -> float:
        """
        Return the average clustering coefficient of the graph.

        Returns
        -------
        float
            The average clustering coefficient of the graph.
        """
        return nx.average_clustering(self.graph)


class WeightedGraphBuilder(GraphBuilder):
    """
    A class to create and manipulate generic weighted graphs.

    Parameters
    ----------
    edges: List[Tuple[int, int]]
        A list of edges where each edge is represented by a tuple of two node identifiers.
    directed: bool, optional
        If True, creates a directed graph. If False (default), creates an undirected graph.
    weights: Optional[List[Union[int, float]]], optional
        A list of weights corresponding to the edges. If None (default), the edges are unweighted.
    """

    def __init__(self,
                 edges: List[Tuple[int, int]],
                 directed: bool = False,
                 weights: Optional[List[Union[int, float]]] = None):
        super().__init__(directed=directed)
        self.edges = edges
        self.weights = weights
        self._create_weighted_graph()

    def _create_weighted_graph(self) -> None:
        """
        Create a weighted graph with the specified edges and weights.
        """
        self.add_edges(self.edges, self.weights)

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
            The weight of the edge. If None, the edge is unweighted.
        """
        self.add_edge(node1, node2, weight)


class LinearGraphBuilder(GraphBuilder):
    """
    A class to create and manipulate linear (path) graphs.

    Parameters
    ----------
    num_nodes: int
        The number of nodes in the linear graph.
    directed: bool, optional
        If True, creates a directed linear graph. If False (default), creates an undirected linear graph.
    weights: Optional[List[Union[int, float]]], optional
        A list of weights for the edges. If None (default), the edges are unweighted.
    """

    def __init__(self,
                 num_nodes: int,
                 directed: bool = False,
                 weights: Optional[List[Union[int, float]]] = None):
        super().__init__(directed=directed)
        self.num_nodes = num_nodes
        self.weights = weights
        self._create_linear_graph()

    def _create_linear_graph(self) -> None:
        """
        Create a linear graph (path graph) with the specified number of nodes.
        """
        edges = [(i, i + 1) for i in range(1, self.num_nodes)]
        self.add_edges(edges, self.weights)

    def add_node_to_end(self,
                        weight: Optional[Union[int, float]] = None) -> None:
        """
        Add a node to the end of the linear graph.

        Parameters
        ----------
        weight: Optional[Union[int, float]], optional
            The weight of the edge connecting the new node to the last node. If None, the edge is unweighted.
        """
        new_node = self.num_nodes + 1
        self.add_edge(self.num_nodes, new_node, weight)
        self.num_nodes += 1


class CyclicGraphBuilder(GraphBuilder):
    """
    A class to create and manipulate cyclic graphs.

    Parameters
    ----------
    num_nodes: int
        The number of nodes in the cyclic graph.
    directed: bool, optional
        If True, creates a directed cyclic graph. If False (default), creates an undirected cyclic graph.
    weights: Optional[List[Union[int, float]]], optional
        A list of weights for the edges. If None (default), the edges are unweighted.
    """

    def __init__(self,
                 num_nodes: int,
                 directed: bool = False,
                 weights: Optional[List[Union[int, float]]] = None):
        super().__init__(directed=directed)
        self.num_nodes = num_nodes
        self.weights = weights
        self._create_cyclic_graph()

    def _create_cyclic_graph(self) -> None:
        """
        Create a cyclic graph with the specified number of nodes.
        """
        edges = [(i, i + 1) for i in range(1, self.num_nodes)]
        edges.append((self.num_nodes, 1))  # Close the cycle
        self.add_edges(edges, self.weights)


# Example usage
if __name__ == "__main__":
    # Linear Graph Example
    linear_builder = LinearGraphBuilder(num_nodes=5, directed=True, weights=[1, 2, 3, 4])
    linear_builder.visualize()

    # Add a node to the end of the linear graph
    linear_builder.add_node_to_end(weight=5)
    linear_builder.visualize()

    # Cyclic Graph Example
    cyclic_builder = CyclicGraphBuilder(num_nodes=5, directed=True, weights=[1, 2, 3, 4, 5])
    cyclic_builder.visualize()
