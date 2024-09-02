"""
This module provides a comprehensive framework for creating, manipulating,
analyzing, and visualizing graphs using the NetworkX library.

It is designed to be modular, with a base class for fundamental graph operations
and extended classes that provide additional utilities and functionalities, such
as graph analysis, file I/O, and visualization.

Key Features:
-------------
- Graph Initialization:
    Create directed or undirected graphs with ease.
- Node and Edge Management:
    Add, remove, and update nodes and edges with full control over their attributes.
- Graph Analysis:
    Compute key metrics like clustering coefficients, graph density, and connectivity.
- File I/O:
    Load and save graphs in various formats, including GML, GraphML, and JSON.
- Visualization:
    Visualize graphs with customizable layouts and styles using Matplotlib.

Class Overview:
---------------
- GraphBuilderBase:
    The foundational class for initializing and managing graph attributes.
- GraphBuilderUtils:
    Extends the base class with utility methods for graph analysis and manipulation.
- GraphBuilder:
    The top-level class that incorporates file I/O and visualization capabilities,
    providing a complete framework for graph operations.

Example:
--------
>>> from qstride.qstride.graphs.builder import GraphBuilder
>>> builder = GraphBuilder(directed=False)  # Create an undirected graph
>>> builder.add_nodes([1, 2, 3])  # Add three nodes to the graph
>>> builder.add_edges([(1, 2), (2, 3)])  # Add edges between the specified nodes
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx

__version__ = "0.1.0"


class GraphBuilderBase:
    """
    Base class for building and manipulating graphs using the networkx library.

    This base class serves as the foundation for more specialized graph builders,
    and provides basic functionality for initializing a graph and managing common
    graph attributes.

    Parameters
    ----------
    directed: bool, default=False
        If True, initializes a directed graph.
        If False (default), initializes an undirected graph.

    name: str, default=None
        The name of the graph.

    metadata: Dict[str, Any], default=None
        A dictionary of metadata to associate with the graph.

    Attributes
    ----------
    graph: networkx.Graph or networkx.DiGraph
        The underlying graph object, which can be either directed or undirected.

    logger: logging.Logger
        An instance of the logging.Logger class, responsible for logging
        messages, errors, and other critical information during execution.
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

        # Initialize the logging system
        self._initialize_logger()

        # Initialize the appropriate nx.Graph object by type
        graph_type = "directed" if directed else "undirected"
        self.graph = self.graph_types[graph_type]()

        # Set the graph's name if provided
        if name is not None:
            self.name = name
        # Set the graph's metadata if provided
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
    def name(self,
             value: str) -> None:
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
    def metadata(self,
                 value: Dict[str, Any]) -> None:
        """
        Setter for the metadata property.

        Parameters
        ----------
        value: Dict[str, Any]
            A dictionary containing metadata to associate with the graph.
        """
        self.graph.graph.update(value)

    def _initialize_logger(self) -> None:
        """
        Initialize the logger for the graph-building environment.

        Raises
        ------
        RuntimeError
            If the logger cannot be initialized.
        """
        try:
            self.logger = logging.getLogger("graph_builder")
            self.logger.setLevel(logging.INFO)
            self.logger.propagate = False

            if not self.logger.handlers:
                formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(formatter)
                self.logger.addHandler(console_handler)

            self.logger.debug("Logger object initialized successfully.")

        except Exception as e:
            msg = f"Failed to initialize the logger properly: {e}"
            raise RuntimeError(msg) from e

    def _convert_to_directed(self) -> None:
        """
        Convert the current graph to a directed graph.
        """
        if isinstance(self.graph, nx.Graph) and not isinstance(self.graph, nx.DiGraph):
            self.graph = self.graph.to_directed()
        else:
            self.logger.warning("The graph is already directed or is not an undirected graph.")

    def _convert_to_undirected(self) -> None:
        """
        Convert the current graph to an undirected graph.
        """
        if isinstance(self.graph, nx.DiGraph):
            self.graph = self.graph.to_undirected()
        else:
            self.logger.warning("The graph is already undirected or is not a directed graph.")

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
        This method allows adding multiple nodes to the graph in one operation.
        The node identifiers must all be integers, and the method will raise an
        error if any non-integer values are detected.
        """
        # Validate that all nodes are integers
        if not all(isinstance(node, int) for node in nodes):
            self.logger.error("All node identifiers must be integers.")
            raise ValueError("All node identifiers must be integers.")

        # Add nodes to the graph
        self.graph.add_nodes_from(nodes)
        self.logger.info(f"Added {len(nodes)} nodes to the graph.")

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
            These can include any key-value pairs representing node properties,
            such as "color", "label", "size", or any custom attribute you wish
            to add. If not provided, default attributes "color": "blue" and
            "label": node are used.

            Examples of attributes:
            - color: The color of the node, e.g., "red", "blue".
            - label: A label to identify the node.
            - size: The size of the node in a visualization context.

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
            self.logger.error(f"Failed to add node {node}. "
                              f"Node identifier must be an integer.")
            raise ValueError("Node identifier must be an integer.")

        # Set default attributes if none are provided
        if not attributes:
            attributes["label"] = node
            attributes["color"] = "blue"

        self.graph.add_node(node, **attributes)
        self.logger.info(f"Added node {node} with attributes {attributes}.")

    def _add_node_attribute(self,
                            node: int,
                            **attributes) -> None:
        """
        Add or update attributes for a specific node.

        Parameters
        ----------
        node: int
            The node to which the attributes will be added.

        **attributes: Dict[str, Any], optional
            Key-value pairs representing the attributes to add or update.
            These can include any custom properties or visual attributes you
            wish to assign or modify for the node. Examples include "color",
            "label", "size", etc.

            Examples of attributes:
            - color: The color of the node, e.g., "green", "yellow".
            - label: A label to update the node's identifier.
            - size: The size of the node in a visualization context.

        Raises
        ------
        ValueError
            If the node does not exist in the graph.

        Notes
        -----
        This method allows you to modify or add additional properties to an
        existing node in the graph. If the node does not exist, an error is
        raised.
        """
        if self.graph.has_node(node):
            self.graph.nodes[node].update(attributes)
            self.logger.info(f"Updated node {node} with attributes {attributes}.")
        else:
            self.logger.error(f"Failed to update node {node}. "
                              f"Node does not exist in the graph.")
            raise ValueError(f"Node {node} does not exist in the graph.")

    def _add_edges(self,
                   edges: List[Tuple[int, int]],
                   weights: Optional[List[Union[int, float]]] = None) -> None:
        """
        Add multiple edges to the graph, with optional weights.

        Parameters
        ----------
        edges: List[Tuple[int, int]]
            A list of edges, where each edge is a tuple of two node identifiers.

        weights: Optional[List[Union[int, float]]], default=None
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
            self.logger.error("Each edge must be a tuple of two node identifiers.")
            raise ValueError("Each edge must be a tuple of two node identifiers.")
        if not all(isinstance(node, int) for edge in edges for node in edge):
            self.logger.error("All node identifiers in edges must be integers.")
            raise ValueError("All node identifiers in edges must be integers.")

        if weights is not None:
            if len(weights) != len(edges):
                self.logger.error("The number of weights must match the number of edges.")
                raise ValueError("The number of weights must match the number of edges.")
            if not all(isinstance(weight, (int, float)) for weight in weights):
                self.logger.error("All edge weights must be numeric values (int or float).")
                raise ValueError("All edge weights must be numeric values (int or float).")

            for edge, weight in zip(edges, weights):
                self._add_edge(edge[0], edge[1], weight)
        else:
            self.graph.add_edges_from(edges)
            self.logger.info(f"Added {len(edges)} edges to the graph.")

    def _add_edge(self,
                  node1: int,
                  node2: int,
                  weight: Optional[Union[int, float]] = None,
                  **attributes) -> None:
        """
        Add an edge between two nodes with an optional weight.

        Parameters
        ----------
        node1: int
            The first node identifier. Must be an integer.

        node2: int
            The second node identifier. Must be an integer.

        weight: Optional[Union[int, float]], default=None
            The weight of the edge. Can be a float or integer.

        **attributes: Dict[str, Any], optional
            Additional attributes to associate with the edge. These can include
            any key-value pairs representing edge properties, such as "color",
            "style", or any custom attribute you wish to add.

            Examples of attributes:
            - color: The color of the edge, e.g., "red", "blue".
            - style: The style of the edge, e.g., "dashed", "solid".
            - capacity: The capacity of the edge, useful in flow networks.

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
            self.logger.error(f"Failed to add edge from {node1} to {node2}. "
                              f"Both node identifiers must be integers.")
            raise ValueError("Both node identifiers must be integers.")
        if weight is not None and not isinstance(weight, (int, float)):
            self.logger.error(f"Failed to add edge from {node1} to {node2}. "
                              f"Edge weight must be a numeric value.")
            raise ValueError("Edge weight must be a numeric value (int or float).")

        if weight is not None:
            attributes['weight'] = weight

        self.graph.add_edge(node1, node2, **attributes)
        self.logger.info(f"Added edge from {node1} to {node2} with attributes {attributes}.")

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

        **attributes: Dict[str, Any], optional
            Key-value pairs representing the attributes to add or update.
            These can include custom properties or visual attributes you wish
            to assign or modify for the edge. Examples include "color", "style",
            "capacity", etc.

            Examples of attributes:
            - color: The color of the edge, e.g., "green", "yellow".
            - style: The style of the edge, e.g., "dotted", "bold".
            - flow: The flow on the edge, useful in flow networks.

        Raises
        ------
        ValueError
            If the edge does not exist between the specified nodes.

        Notes
        -----
        This method allows you to modify or add additional properties to an
        existing edge in the graph. If the edge does not exist, an error is
        raised.
        """
        if self.graph.has_edge(node1, node2):
            self.graph[node1][node2].update(attributes)
            self.logger.info(f"Updated edge from {node1} to {node2} with attributes {attributes}.")
        else:
            self.logger.error(f"Failed to update edge from {node1} to {node2}. "
                              f"No such edge exists.")
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
                self.logger.info(f"Removed weight from the edge between {node1} and {node2}.")
            else:
                self.logger.error(f"Failed to remove weight. "
                                  f"Edge between {node1} and {node2} does not have a weight attribute.")
                raise ValueError(f"Edge between {node1} and {node2} does not have a weight attribute.")
        else:
            self.logger.error(f"Failed to remove weight. "
                              f"No edge exists between {node1} and {node2}.")
            raise ValueError(f"No edge exists between {node1} and {node2}.")

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
            self.logger.info(f"Updated weight of the edge between {node1} and {node2} to {new_weight}.")
        else:
            self.logger.error(f"Failed to update weight. "
                              f"No edge exists between {node1} and {node2}.")
            raise ValueError(f"No edge exists between {node1} and {node2}.")


class GraphBuilderUtils(GraphBuilderBase):
    """
    Utility class for performing common graph operations and analyses.

    This class extends the base functionality provided by the base class,
    offering utility methods for analyzing and manipulating graphs.

    Parameters
    ----------
    directed: bool, default=False
        If True, initializes a directed graph.
        If False (default), initializes an undirected graph.

    name: str, default=None
        The name of the graph.

    metadata: Dict[str, Any], default=None
        A dictionary of metadata to associate with the graph.
    """

    def __init__(self,
                 directed: bool = False,
                 name: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        super().__init__(directed=directed,
                         name=name,
                         metadata=metadata)

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

    def get_connected_components(self) -> List[Set[int]]:
        """
        Return a list of sets, each representing a connected component of the graph.

        Returns
        -------
        List[Set[int]]
            A list of sets, each containing the nodes in a connected component.
        """
        return [component for component in nx.connected_components(self.graph)]

    def get_average_clustering_coefficient(self) -> float:
        """
        Return the average clustering coefficient of the graph.

        Returns
        -------
        float
            The average clustering coefficient of the graph.
        """
        return nx.average_clustering(self.graph)

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

    def get_graph(self) -> Union[nx.Graph, nx.DiGraph]:
        """
        Return the underlying networkx graph object.

        Returns
        -------
        networkx.Graph or networkx.DiGraph
            The underlying graph object.
        """
        return self.graph

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

    def get_all_node_degrees(self) -> Dict[int, int]:
        """
        Return a dictionary mapping nodes to their degrees.

        Returns
        -------
        Dict[int, int]
            A dictionary with nodes as keys and their degrees as values.
        """
        return dict(self.graph.degree)

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

    def is_connected(self) -> bool:
        """
        Check if the graph is fully connected.

        Returns
        -------
        bool
            True if the graph is connected, False otherwise.
        """
        return nx.is_connected(self.graph)


class GraphBuilder(GraphBuilderUtils):
    """
    A class to create, manipulate, and visualize graphs using the networkx library.

    This class extends the utility methods provided by GraphBuilderUtils, and adds
    functionality for file I/O operations as well as visualization. It is intended
    for building and working with both directed and undirected graphs, with additional
    features to customize and analyze graph structures.

    Parameters
    ----------
    directed: bool, default=False
        If True, initializes a directed graph.
        If False (default), initializes an undirected graph.

    name: str, default=None
        The name of the graph.

    metadata: Dict[str, Any], default=None
        A dictionary of metadata to associate with the graph.
    """

    def __init__(self,
                 directed: bool = False,
                 name: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        super().__init__(directed=directed,
                         name=name,
                         metadata=metadata)

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

    def visualize_graph(self,
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
            Options include 'spring' (default), 'circular', 'random', 'shell',
            and 'spectral'.
        """
        # Select layout
        layout_funcs = {
            'spring': nx.spring_layout,
            'circular': nx.circular_layout,
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
            These attributes can include any key-value pairs representing node properties.
            Common examples include:
            - `color` (str): The color of the node, e.g., "red", "blue".
            - `label` (str): A label to identify the node, e.g., "Node A".
            - `size` (int): The size of the node, useful for visualization purposes.
            - Any other custom attributes you wish to assign to the node.
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

        **attributes: Dict[str, Any], optional
            Key-value pairs representing the attributes to add or update.
            These attributes can include any custom properties or visual attributes
            you wish to assign or modify for the node. Examples include:
            - `color` (str): The color of the node, e.g., "green", "yellow".
            - `label` (str): A label to update the node's identifier.
            - `size` (int): The size of the node in a visualization context.
            - Any other custom attributes relevant to the node.
        """
        self._add_node_attribute(node, **attributes)

    def add_edges(self,
                  edges: List[Tuple[int, int]],
                  weights: Optional[List[Union[int, float]]] = None) -> None:
        """
        Public method to add multiple edges to the graph, with optional weights.

        Parameters
        ----------
        edges: List[Tuple[int, int]]
            A list of edges, where each edge is a tuple of two node identifiers.

        weights: Optional[List[Union[int, float]]], default=None
            A list of weights corresponding to each edge.
            Weights can be float or integer.
        """
        self._add_edges(edges, weights)

    def add_edge(self,
                 node1: int,
                 node2: int,
                 weight: Optional[Union[int, float]] = None,
                 **attributes: Dict[str, Any]) -> None:
        """
        Public method to add an edge between two nodes with an optional weight.

        Parameters
        ----------
        node1: int
            The first node identifier. Must be an integer.

        node2: int
            The second node identifier. Must be an integer.

        weight: Optional[Union[int, float]], optional
            The weight of the edge. Can be a float or integer.

        **attributes: Dict[str, Any], optional
            Additional attributes to associate with the edge.
            These attributes can include any key-value pairs representing edge properties.
            Common examples include:
            - `color` (str): The color of the edge, e.g., "red", "blue".
            - `style` (str): The style of the edge, e.g., "dashed", "solid".
            - `capacity` (int or float): The capacity of the edge, useful in flow networks.
            - Any other custom attributes you wish to assign to the edge.
        """
        self._add_edge(node1, node2, weight, **attributes)

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

        **attributes: Dict[str, Any], optional
            Key-value pairs representing the attributes to add or update.
            These attributes can include any custom properties or visual attributes
            you wish to assign or modify for the edge. Examples include:
            - `color` (str): The color of the edge, e.g., "green", "yellow".
            - `style` (str): The style of the edge, e.g., "dotted", "bold".
            - `flow` (int or float): The flow on the edge, useful in flow networks.
            - Any other custom attributes relevant to the edge.
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


def plot_graph_variants():
    """
    Plots four types of graphs: undirected/unweighted, undirected/weighted,
    directed/unweighted, and directed/weighted.
    """
    # Create figure and axes for subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # Define positions for consistent node layout across all plots
    pos = nx.spring_layout(nx.complete_graph(5))

    # 1. Undirected Unweighted Graph
    undirected_unweighted = nx.Graph()
    undirected_unweighted.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)])
    pos = nx.spring_layout(undirected_unweighted)  # Calculate positions after adding nodes
    nx.draw(undirected_unweighted, pos, with_labels=True, ax=axs[0, 0], node_color="lightblue", edge_color="gray")
    axs[0, 0].set_title("Undirected Unweighted")

    # 2. Undirected Weighted Graph
    undirected_weighted = nx.Graph()
    undirected_weighted.add_edges_from([(1, 2, {'weight': 1.5}), (2, 3, {'weight': 2.5}),
                                        (3, 4, {'weight': 1.2}), (4, 5, {'weight': 3.1}),
                                        (5, 1, {'weight': 2.8})])
    nx.draw(undirected_weighted, pos, with_labels=True, ax=axs[0, 1], node_color="lightgreen", edge_color="blue")
    edge_labels = nx.get_edge_attributes(undirected_weighted, 'weight')
    nx.draw_networkx_edge_labels(undirected_weighted, pos, edge_labels=edge_labels, ax=axs[0, 1])
    axs[0, 1].set_title("Undirected Weighted")

    # 3. Directed Unweighted Graph
    directed_unweighted = nx.DiGraph()
    directed_unweighted.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)])
    pos = nx.spring_layout(directed_unweighted)  # Recalculate positions for directed graph
    nx.draw(directed_unweighted, pos, with_labels=True, ax=axs[1, 0], node_color="lightcoral", edge_color="black")
    axs[1, 0].set_title("Directed Unweighted")

    # 4. Directed Weighted Graph
    directed_weighted = nx.DiGraph()
    directed_weighted.add_edges_from([(1, 2, {'weight': 1.5}), (2, 3, {'weight': 2.5}),
                                      (3, 4, {'weight': 1.2}), (4, 5, {'weight': 3.1}),
                                      (5, 1, {'weight': 2.8})])
    pos = nx.spring_layout(directed_weighted)  # Recalculate positions for directed weighted graph
    nx.draw(directed_weighted, pos, with_labels=True, ax=axs[1, 1], node_color="lightyellow", edge_color="purple")
    edge_labels = nx.get_edge_attributes(directed_weighted, 'weight')
    nx.draw_networkx_edge_labels(directed_weighted, pos, edge_labels=edge_labels, ax=axs[1, 1])
    axs[1, 1].set_title("Directed Weighted")

    plt.tight_layout()
    plt.show()


# Example usage
def main():
    # Initialize an undirected graph
    builder = GraphBuilder(directed=False, name="Test Graph", metadata={"author": "Tester"})

    # Add nodes to the graph
    nodes = [1, 2, 3, 4, 5]
    builder.add_nodes(nodes)

    # Add edges with weights to the graph
    edges = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)]
    weights = [1.0, 2.5, 3.0, 4.2, 5.1]
    builder.add_edges(edges, weights)

    # Add an edge with additional attributes
    builder.add_edge(2, 4, weight=3.5, color="red", style="dashed")

    # Update node and edge attributes
    builder.add_node_attribute(1, label="Start Node", color="green")
    builder.add_edge_attribute(1, 2, color="blue", style="bold")

    # Display graph information
    print("Graph Name:", builder.name)
    print("Metadata:", builder.metadata)
    print("Node Degrees:", builder.get_all_node_degrees())
    print("Graph Density:", builder.get_graph_density())

    # Check graph connectivity
    if builder.is_connected():
        print("The graph is connected.")
    else:
        print("The graph is not connected.")

    # Calculate and display the shortest paths
    shortest_path_1_to_4 = builder.get_shortest_path(1, 4)
    print(f"Shortest path from node 1 to node 4: {shortest_path_1_to_4}")

    # Visualize the graph
    builder.visualize_graph()

    # Save the graph to a file
    builder.save_to_file("test_graph.gml", file_format="gml")

    # Load the graph from the file
    builder.load_from_file("test_graph.gml", file_format="gml")

    # Verify the loaded graph
    print("Loaded Graph Name:", builder.name)
    print("Loaded Graph Metadata:", builder.metadata)
    print("Loaded Graph Node Degrees:", builder.get_all_node_degrees())


if __name__ == "__main__":

    # main()

    plot_graph_variants()
