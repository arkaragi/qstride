"""
Test suite for the graphs.builder.py module.
"""

import os
import unittest
import networkx as nx

from ..builder import GraphBuilderBase
from ..builder import GraphBuilderUtils
from ..builder import GraphBuilder

__version__ = "0.1.0"


class TestGraphBuilderBase(unittest.TestCase):
    """
    Test suite for the GraphBuilderBase class.
    """

    def setUp(self):
        """
        Set up a basic GraphBuilderBase instance for use in tests.
        """
        self.graph_builder = GraphBuilderBase()

    def test_initialization_undirected(self):
        """
        Test if the graph is correctly initialized as an undirected graph.
        """
        # Create an undirected graph
        gb = GraphBuilderBase(directed=False)

        # Assert that the graph is an instance of nx.Graph (undirected)
        self.assertIsInstance(gb.graph, nx.Graph)

    def test_initialization_directed(self):
        """
        Test if the graph is correctly initialized as a directed graph.
        """
        # Create a directed graph
        gb = GraphBuilderBase(directed=True)

        # Assert that the graph is an instance of nx.DiGraph (directed)
        self.assertIsInstance(gb.graph, nx.DiGraph)

    def test_name_property(self):
        """
        Test the name property getter and setter.
        """
        # Initialize graph with a name
        gb = GraphBuilderBase(name="Test Graph")

        # Assert that the name is correctly set
        self.assertEqual(gb.name, "Test Graph")

        # Change the name of the graph
        gb.name = "New Name"

        # Assert that the name is updated correctly
        self.assertEqual(gb.name, "New Name")

    def test_metadata_property(self):
        """
        Test the metadata property for correctly setting and updating metadata.
        """
        # Initialize graph with metadata
        metadata = {"author": "John Doe", "creation_date": "2023-01-01"}
        gb = GraphBuilderBase(metadata=metadata)

        # Assert that the metadata is correctly set
        self.assertEqual(gb.metadata, metadata)

        # Update the metadata with new values
        new_metadata = {"description": "Test graph for unit tests"}
        gb.metadata = new_metadata

        # Expected metadata after updating
        expected_metadata = {
            "author": "John Doe",
            "creation_date": "2023-01-01",
            "description": "Test graph for unit tests"
        }

        # Assert that the metadata now contains both old and new entries
        self.assertEqual(gb.metadata, expected_metadata)

    def test_convert_to_directed(self):
        """
        Test converting an undirected graph to a directed graph.
        """
        # Add an edge in an undirected graph
        self.graph_builder._add_edge(1, 2)

        # Convert the graph to directed
        self.graph_builder._convert_to_directed()

        # Verify that the graph is now a directed graph
        self.assertIsInstance(self.graph_builder.graph, nx.DiGraph)

    def test_convert_to_undirected(self):
        """
        Test converting a directed graph to an undirected graph.
        """
        # Initialize a directed graph
        self.graph_builder = GraphBuilderBase(directed=True)
        self.graph_builder._add_edge(1, 2)

        # Convert the graph to undirected
        self.graph_builder._convert_to_undirected()

        # Verify that the graph is now an undirected graph
        self.assertIsInstance(self.graph_builder.graph, nx.Graph)

    def test_convert_to_directed_invalid(self):
        """
        Test converting an already directed graph to a directed graph.
        """
        # Initialize a directed graph
        graph_builder_directed = GraphBuilderBase(directed=True)

        # Attempt to convert an already directed graph to directed
        initial_graph = graph_builder_directed.graph
        graph_builder_directed._convert_to_directed()

        # Ensure the graph remains a directed graph and is unchanged
        self.assertIs(graph_builder_directed.graph, initial_graph)
        self.assertIsInstance(graph_builder_directed.graph, nx.DiGraph)

    def test_convert_to_undirected_invalid(self):
        """
        Test converting an already undirected graph to an undirected graph.
        """
        # Initialize an undirected graph
        graph_builder_undirected = GraphBuilderBase(directed=False)

        # Attempt to convert an already undirected graph to undirected
        initial_graph = graph_builder_undirected.graph
        graph_builder_undirected._convert_to_undirected()

        # Ensure the graph remains an undirected graph and is unchanged
        self.assertIs(graph_builder_undirected.graph, initial_graph)
        self.assertIsInstance(graph_builder_undirected.graph, nx.Graph)

    def test_add_nodes(self):
        """
        Test adding multiple nodes to the graph.
        """
        # Add multiple nodes to the graph
        self.graph_builder._add_nodes([1, 2, 3])

        # Assert that each node is added to the graph
        self.assertIn(1, self.graph_builder.graph.nodes)
        self.assertIn(2, self.graph_builder.graph.nodes)
        self.assertIn(3, self.graph_builder.graph.nodes)

    def test_add_node(self):
        """
        Test adding a single node to the graph.
        """
        # Add a node to the graph
        self.graph_builder._add_node(1)

        # Assert that the node is added to the graph
        self.assertIn(1, self.graph_builder.graph.nodes)

    def test_add_node_invalid_type(self):
        """
        Test adding a node with an invalid type (non-integer).
        """
        # Attempt to add a node with a non-integer identifier
        with self.assertRaises(ValueError):
            self.graph_builder._add_node("a")

    def test_add_node_with_attributes(self):
        """
        Test adding a single node with attributes to the graph.
        """
        # Add a node with specific attributes
        self.graph_builder._add_node(1, color="red", label="Node 1")

        # Assert that the node is added to the graph
        self.assertIn(1, self.graph_builder.graph.nodes)

        # Assert that the node's attributes are set correctly
        self.assertEqual(self.graph_builder.graph.nodes[1]["color"], "red")
        self.assertEqual(self.graph_builder.graph.nodes[1]["label"], "Node 1")

    def test_add_node_attribute(self):
        """
        Test adding or updating an attribute for a specific node.
        """
        # Add a node to the graph
        self.graph_builder._add_node(1)

        # Update the node's attribute
        self.graph_builder._add_node_attribute(1, color="blue")

        # Assert that the node's attribute is set correctly
        self.assertEqual(self.graph_builder.graph.nodes[1]["color"], "blue")

    def test_add_edges(self):
        """
        Test adding multiple edges to the graph.
        """
        # Define multiple edges
        edges = [(1, 2), (2, 3), (3, 4)]

        # Add the edges to the graph
        self.graph_builder._add_edges(edges)

        # Assert that each edge is added to the graph
        self.assertTrue(self.graph_builder.graph.has_edge(1, 2))
        self.assertTrue(self.graph_builder.graph.has_edge(2, 3))
        self.assertTrue(self.graph_builder.graph.has_edge(3, 4))

    def test_add_edges_with_weights(self):
        """
        Test adding multiple edges with corresponding weights.
        """
        # Define multiple edges and corresponding weights
        edges = [(1, 2), (2, 3), (3, 4)]
        weights = [2.0, 3.0, 4.0]

        # Add the edges with weights to the graph
        self.graph_builder._add_edges(edges, weights=weights)

        # Assert that each edge's weight is set correctly
        self.assertEqual(self.graph_builder.graph[1][2]["weight"], 2.0)
        self.assertEqual(self.graph_builder.graph[2][3]["weight"], 3.0)
        self.assertEqual(self.graph_builder.graph[3][4]["weight"], 4.0)

    def test_add_edges_invalid_weights(self):
        """
        Test adding edges with a mismatched number of weights.
        """
        edges = [(1, 2), (2, 3)]
        weights = [2.0]  # Only one weight provided for two edges

        # Attempt to add edges with a mismatched number of weights
        with self.assertRaises(ValueError):
            self.graph_builder._add_edges(edges, weights=weights)

    def test_add_edge(self):
        """
        Test adding a single edge between two nodes.
        """
        # Add an edge between two nodes
        self.graph_builder._add_edge(1, 2)

        # Assert that the edge exists in the graph
        self.assertTrue(self.graph_builder.graph.has_edge(1, 2))

    def test_add_edge_invalid_type(self):
        """
        Test adding an edge with invalid node identifiers (non-integers).
        """
        # Attempt to add an edge where one of the nodes is not an integer
        with self.assertRaises(ValueError):
            self.graph_builder._add_edge("a", 2)

    def test_add_edge_with_weight(self):
        """
        Test adding an edge between two nodes with a weight attribute.
        """
        # Add an edge with a weight
        self.graph_builder._add_edge(1, 2, weight=3.5)

        # Assert that the edge exists in the graph
        self.assertTrue(self.graph_builder.graph.has_edge(1, 2))

        # Assert that the edge's weight is set correctly
        self.assertEqual(self.graph_builder.graph[1][2]["weight"], 3.5)

    def test_add_edge_with_attributes(self):
        """
        Test adding an edge between two nodes with multiple attributes.
        """
        # Add an edge with multiple attributes
        self.graph_builder._add_edge(1, 2, weight=3.5, color="green")

        # Assert that the edge exists in the graph
        self.assertTrue(self.graph_builder.graph.has_edge(1, 2))

        # Assert that the edge's attributes are set correctly
        self.assertEqual(self.graph_builder.graph[1][2]["weight"], 3.5)
        self.assertEqual(self.graph_builder.graph[1][2]["color"], "green")

    def test_add_edge_attribute(self):
        """
        Test adding or updating an attribute for a specific edge.
        """
        # Add an edge between two nodes
        self.graph_builder._add_edge(1, 2)

        # Update the edge's attribute
        self.graph_builder._add_edge_attribute(1, 2, color="blue")

        # Assert that the edge's attribute is set correctly
        self.assertEqual(self.graph_builder.graph[1][2]["color"], "blue")

    def test_remove_edge_weight(self):
        """
        Test removing the weight attribute from an edge.
        """
        # Add an edge with a weight attribute
        self.graph_builder._add_edge(1, 2, weight=3.5)

        # Remove the weight attribute
        self.graph_builder._remove_edge_weight(1, 2)

        # Ensure the weight attribute is no longer present
        self.assertNotIn("weight", self.graph_builder.graph[1][2])

    def test_remove_edge_weight_no_edge(self):
        """
        Test attempting to remove the weight of a non-existent edge.
        """
        # Attempt to remove a weight from an edge that does not exist
        with self.assertRaises(ValueError):
            self.graph_builder._remove_edge_weight(1, 2)

    def test_update_edge_weight(self):
        """
        Test updating the weight of an existing edge.
        """
        # Add an edge with an initial weight
        self.graph_builder._add_edge(1, 2, weight=3.5)

        # Update the edge's weight
        self.graph_builder._update_edge_weight(1, 2, new_weight=4.5)

        # Verify that the edge's weight has been updated
        self.assertEqual(self.graph_builder.graph[1][2]["weight"], 4.5)

    def test_update_edge_weight_no_edge(self):
        """
        Test attempting to update the weight of a non-existent edge.
        """
        # Attempt to update the weight of an edge that does not exist
        with self.assertRaises(ValueError):
            self.graph_builder._update_edge_weight(1, 2, new_weight=4.5)


class TestGraphBuilderUtils(unittest.TestCase):
    """
    Test suite for the GraphBuilderUtils class.
    """

    def setUp(self):
        """
        Set up a basic GraphBuilderUtils instance for use in tests.
        """
        self.graph_builder = GraphBuilderUtils()

    def test_convert_to_directed(self):
        """
        Test converting an undirected graph to a directed graph.
        """
        # Add an edge in an undirected graph
        self.graph_builder._add_edge(1, 2)

        # Convert the graph to directed
        self.graph_builder.convert_to_directed()

        # Verify that the graph is now a directed graph
        self.assertIsInstance(self.graph_builder.graph, nx.DiGraph)

    def test_convert_to_undirected(self):
        """
        Test converting a directed graph to an undirected graph.
        """
        # Initialize a directed graph
        self.graph_builder = GraphBuilderUtils(directed=True)
        self.graph_builder._add_edge(1, 2)

        # Convert the graph to undirected
        self.graph_builder.convert_to_undirected()

        # Verify that the graph is now an undirected graph
        self.assertIsInstance(self.graph_builder.graph, nx.Graph)

    def test_get_connected_components(self):
        """
        Test retrieving the connected components of a graph.
        """
        # Add edges to create two connected components
        self.graph_builder._add_edges([(1, 2), (2, 3), (4, 5)])

        # Get the connected components
        components = self.graph_builder.get_connected_components()

        # Verify that there are two connected components
        self.assertEqual(len(components), 2)

        # Verify the nodes in each connected component
        self.assertIn({1, 2, 3}, components)
        self.assertIn({4, 5}, components)

    def test_get_average_clustering_coefficient(self):
        """
        Test retrieving the average clustering coefficient of a graph.
        """
        # Add a triangle (fully connected subgraph)
        self.graph_builder._add_edges([(1, 2), (2, 3), (1, 3)])

        # Calculate the average clustering coefficient
        avg_clustering = self.graph_builder.get_average_clustering_coefficient()

        # The expected average clustering coefficient for a triangle is 1.0
        self.assertEqual(avg_clustering, 1.0)

    def test_get_clustering_coefficient(self):
        """
        Test retrieving the clustering coefficient of a specific node.
        """
        # Add a triangle (fully connected subgraph)
        self.graph_builder._add_edges([(1, 2), (2, 3), (1, 3)])

        # Calculate the clustering coefficient for node 1
        clustering_coefficient = self.graph_builder.get_clustering_coefficient(1)

        # The expected clustering coefficient for node 1 in a triangle is 1.0
        self.assertEqual(clustering_coefficient, 1.0)

    def test_get_clustering_coefficient_invalid_node(self):
        """
        Test retrieving the clustering coefficient of a non-existent node.
        """
        with self.assertRaises(ValueError):
            self.graph_builder.get_clustering_coefficient(1)

    def test_get_graph(self):
        """
        Test retrieving the underlying networkx graph object.
        """
        # Ensure the returned object is the same as the internal graph
        self.assertIs(self.graph_builder.get_graph(), self.graph_builder.graph)

    def test_get_graph_radius(self):
        """
        Test retrieving the radius of the graph.
        """
        # Add a linear chain of nodes
        self.graph_builder._add_edges([(1, 2), (2, 3), (3, 4)])

        # The radius should be 2 (distance from node 2 to all other nodes)
        radius = self.graph_builder.get_graph_radius()
        self.assertEqual(radius, 2)

    def test_get_graph_radius_disconnected(self):
        """
        Test retrieving the radius of a disconnected graph.
        """
        # Add two disconnected components
        self.graph_builder._add_edges([(1, 2), (3, 4)])

        with self.assertRaises(nx.NetworkXError):
            self.graph_builder.get_graph_radius()

    def test_get_graph_diameter(self):
        """
        Test retrieving the diameter of the graph.
        """
        # Add a linear chain of nodes
        self.graph_builder._add_edges([(1, 2), (2, 3), (3, 4)])

        # The diameter should be 3 (longest shortest path from node 1 to node 4)
        diameter = self.graph_builder.get_graph_diameter()
        self.assertEqual(diameter, 3)

    def test_get_graph_diameter_disconnected(self):
        """
        Test retrieving the diameter of a disconnected graph.
        """
        # Add two disconnected components
        self.graph_builder._add_edges([(1, 2), (3, 4)])

        with self.assertRaises(nx.NetworkXError):
            self.graph_builder.get_graph_diameter()

    def test_get_graph_density(self):
        """
        Test retrieving the density of the graph.
        """
        # Add edges to create a square with a diagonal (4 nodes, 5 edges)
        self.graph_builder._add_edges([(1, 2), (2, 3), (3, 4), (4, 1), (1, 3)])

        # The density should be 5/6 (5 edges, 6 possible edges in a complete graph of 4 nodes)
        density = self.graph_builder.get_graph_density()
        self.assertAlmostEqual(density, 5 / 6)

    def test_get_node_degree(self):
        """
        Test retrieving the degree of a specific node.
        """
        # Add edges to connect node 1 with two other nodes
        self.graph_builder._add_edges([(1, 2), (1, 3)])

        # The degree of node 1 should be 2
        degree = self.graph_builder.get_node_degree(1)
        self.assertEqual(degree, 2)

    def test_get_node_degree_invalid_node(self):
        """
        Test retrieving the degree of a non-existent node.
        """
        with self.assertRaises(ValueError):
            self.graph_builder.get_node_degree(1)

    def test_get_all_node_degrees(self):
        """
        Test retrieving the degrees of all nodes in the graph.
        """
        # Add edges to create a triangle
        self.graph_builder._add_edges([(1, 2), (2, 3), (1, 3)])

        # Get all node degrees
        degrees = self.graph_builder.get_all_node_degrees()

        # The degree of each node in the triangle should be 2
        self.assertEqual(degrees, {1: 2, 2: 2, 3: 2})

    def test_get_shortest_path(self):
        """
        Test retrieving the shortest path between two nodes.
        """
        # Add edges to create a path
        self.graph_builder._add_edges([(1, 2), (2, 3), (3, 4)])

        # The shortest path between node 1 and node 4 should be [1, 2, 3, 4]
        shortest_path = self.graph_builder.get_shortest_path(1, 4)
        self.assertEqual(shortest_path, [1, 2, 3, 4])

    def test_get_shortest_path_no_path(self):
        """
        Test retrieving the shortest path between two disconnected nodes.
        """
        # Add two disconnected components
        self.graph_builder._add_edges([(1, 2), (3, 4)])

        with self.assertRaises(nx.NetworkXNoPath):
            self.graph_builder.get_shortest_path(1, 3)

    def test_get_all_shortest_paths(self):
        """
        Test retrieving all shortest paths between all pairs of nodes.
        """
        # Add edges to create a square
        self.graph_builder._add_edges([(1, 2), (2, 3), (3, 4), (4, 1)])

        # Get all shortest paths
        all_shortest_paths = self.graph_builder.get_all_shortest_paths()

        # Check that the shortest paths match expected results
        expected_paths = {1: {1: [1], 2: [1, 2], 3: [1, 2, 3], 4: [1, 4]},
                          2: {1: [2, 1], 2: [2], 3: [2, 3], 4: [2, 1, 4]},
                          3: {1: [3, 2, 1], 2: [3, 2], 3: [3], 4: [3, 4]},
                          4: {1: [4, 1], 2: [4, 3, 2], 3: [4, 3], 4: [4]}}
        self.assertEqual(all_shortest_paths, expected_paths)

    def test_is_connected(self):
        """
        Test checking if the graph is connected.
        """
        # Add edges to create a connected graph
        self.graph_builder._add_edges([(1, 2), (2, 3), (3, 4)])

        # The graph should be connected
        self.assertTrue(self.graph_builder.is_connected())

    def test_is_connected_disconnected(self):
        """
        Test checking if a graph is disconnected.
        """
        # Add edges to create two disconnected components
        self.graph_builder._add_edges([(1, 2), (3, 4)])

        # The graph should be disconnected
        self.assertFalse(self.graph_builder.is_connected())


class TestGraphBuilder(unittest.TestCase):
    """
    Test suite for the GraphBuilder class.
    """

    def setUp(self):
        """
        Set up a basic GraphBuilder instance for use in tests.
        """
        # Create an instance of the GraphBuilder class to be used in all test cases
        self.graph_builder = GraphBuilder()

    def tearDown(self):
        """
        Clean up any files created during the tests.
        """
        if os.path.exists("test_graph.gml"):
            os.remove("test_graph.gml")

    def test_load_from_file_invalid_format(self):
        """
        Test loading a graph from a file with an invalid format.
        """
        # Attempt to load a graph from a file with an unsupported format and expect a ValueError
        with self.assertRaises(ValueError):
            self.graph_builder.load_from_file("test.gml", file_format="invalid_format")

    def test_save_to_file_invalid_format(self):
        """
        Test saving a graph to a file with an invalid format.
        """
        # Attempt to save a graph to a file with an unsupported format and expect a ValueError
        with self.assertRaises(ValueError):
            self.graph_builder.save_to_file("test.gml", file_format="invalid_format")

    def test_visualize_graph_invalid_layout(self):
        """
        Test visualizing the graph with an invalid layout type.
        """
        # Attempt to visualize the graph using an invalid layout type and expect a ValueError
        with self.assertRaises(ValueError):
            self.graph_builder.visualize_graph(layout="invalid_layout")

    def test_add_nodes(self):
        """
        Test adding multiple nodes to the graph.
        """
        # Add multiple nodes to the graph and verify that they are successfully added
        self.graph_builder.add_nodes([1, 2, 3])
        self.assertIn(1, self.graph_builder.graph.nodes)
        self.assertIn(2, self.graph_builder.graph.nodes)
        self.assertIn(3, self.graph_builder.graph.nodes)

    def test_add_node_with_attributes(self):
        """
        Test adding a single node with attributes to the graph.
        """
        # Add a single node with attributes to the graph and verify that the attributes are correctly set
        self.graph_builder.add_node(1, color="red", label="Node 1")
        self.assertIn(1, self.graph_builder.graph.nodes)
        self.assertEqual(self.graph_builder.graph.nodes[1]["color"], "red")
        self.assertEqual(self.graph_builder.graph.nodes[1]["label"], "Node 1")

    def test_add_node_attribute(self):
        """
        Test adding or updating an attribute for a specific node.
        """
        # Add a node to the graph and then update its attributes
        self.graph_builder.add_node(1)
        self.graph_builder.add_node_attribute(1, color="blue")
        self.assertEqual(self.graph_builder.graph.nodes[1]["color"], "blue")

    def test_add_edges_with_weights(self):
        """
        Test adding multiple edges with corresponding weights.
        """
        # Add multiple edges with weights to the graph and verify that the weights are correctly set
        edges = [(1, 2), (2, 3), (3, 4)]
        weights = [2.0, 3.0, 4.0]
        self.graph_builder.add_edges(edges, weights=weights)
        self.assertEqual(self.graph_builder.graph[1][2]["weight"], 2.0)
        self.assertEqual(self.graph_builder.graph[2][3]["weight"], 3.0)
        self.assertEqual(self.graph_builder.graph[3][4]["weight"], 4.0)

    def test_add_edge_with_attributes(self):
        """
        Test adding an edge with multiple attributes.
        """
        # Add a single edge with multiple attributes to the graph and verify that the attributes are correctly set
        self.graph_builder.add_edge(1, 2, weight=3.5, color="green")
        self.assertEqual(self.graph_builder.graph[1][2]["weight"], 3.5)
        self.assertEqual(self.graph_builder.graph[1][2]["color"], "green")

    def test_remove_edge_weight(self):
        """
        Test removing the weight attribute from an edge.
        """
        # Add an edge with a weight, then remove the weight and verify that it no longer exists
        self.graph_builder.add_edge(1, 2, weight=3.5)
        self.graph_builder.remove_edge_weight(1, 2)
        self.assertNotIn("weight", self.graph_builder.graph[1][2])

    def test_update_edge_weight(self):
        """
        Test updating the weight of an existing edge.
        """
        # Add an edge with an initial weight, then update the weight and verify the update
        self.graph_builder.add_edge(1, 2, weight=3.5)
        self.graph_builder.update_edge_weight(1, 2, new_weight=4.5)
        self.assertEqual(self.graph_builder.graph[1][2]["weight"], 4.5)

    def test_save_and_load_graph(self):
        """
        Test saving and loading a graph to and from a file.
        """
        self.graph_builder.add_nodes([1, 2, 3])
        self.graph_builder.add_edge(1, 2, weight=3.5)

        # Save the graph to a GML file
        self.graph_builder.save_to_file("test_graph.gml", file_format="gml")
        self.assertTrue(os.path.exists("test_graph.gml"))

        # Load the graph from the GML file
        new_graph_builder = GraphBuilder()
        new_graph_builder.load_from_file("test_graph.gml", file_format="gml")

        # Check that the graph structure and attributes are intact
        # GML loads node identifiers as strings, so adjust the assertions accordingly
        self.assertIn('1', new_graph_builder.graph.nodes)
        self.assertIn('2', new_graph_builder.graph.nodes)
        self.assertEqual(new_graph_builder.graph['1']['2']['weight'], 3.5)


if __name__ == "__main__":
    unittest.main()
