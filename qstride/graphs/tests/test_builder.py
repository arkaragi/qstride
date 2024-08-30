import unittest
import networkx as nx
from ..builder import GraphBuilderBase  # Assuming the class is saved in a module named graph_builder_base


class TestGraphBuilderBase(unittest.TestCase):

    def setUp(self):
        # Setup runs before each test
        self.graph_builder = GraphBuilderBase()

    def test_graph_initialization_undirected(self):
        gb = GraphBuilderBase(directed=False)
        self.assertIsInstance(gb.graph, nx.Graph)

    def test_graph_initialization_directed(self):
        gb = GraphBuilderBase(directed=True)
        self.assertIsInstance(gb.graph, nx.DiGraph)

    def test_name_property(self):
        gb = GraphBuilderBase(name="Test Graph")
        self.assertEqual(gb.name, "Test Graph")
        gb.name = "New Name"
        self.assertEqual(gb.name, "New Name")

    def test_metadata_property(self):
        metadata = {"author": "John Doe", "creation_date": "2023-01-01"}
        gb = GraphBuilderBase(metadata=metadata)
        self.assertEqual(gb.metadata, metadata)
        new_metadata = {"description": "Test graph for unit tests"}
        gb.metadata = new_metadata
        self.assertDictContainsSubset(new_metadata, gb.metadata)

    def test_add_node(self):
        self.graph_builder._add_node(1)
        self.assertIn(1, self.graph_builder.graph.nodes)

    def test_add_node_with_attributes(self):
        self.graph_builder._add_node(1, color="red", label="Node 1")
        self.assertIn(1, self.graph_builder.graph.nodes)
        self.assertEqual(self.graph_builder.graph.nodes[1]["color"], "red")
        self.assertEqual(self.graph_builder.graph.nodes[1]["label"], "Node 1")

    def test_add_multiple_nodes(self):
        self.graph_builder._add_nodes([1, 2, 3])
        self.assertIn(1, self.graph_builder.graph.nodes)
        self.assertIn(2, self.graph_builder.graph.nodes)
        self.assertIn(3, self.graph_builder.graph.nodes)

    def test_add_edge(self):
        self.graph_builder._add_edge(1, 2)
        self.assertTrue(self.graph_builder.graph.has_edge(1, 2))

    def test_add_edge_with_weight(self):
        self.graph_builder._add_edge(1, 2, weight=3.5)
        self.assertTrue(self.graph_builder.graph.has_edge(1, 2))
        self.assertEqual(self.graph_builder.graph[1][2]["weight"], 3.5)

    def test_add_edge_with_attributes(self):
        self.graph_builder._add_edge(1, 2, weight=3.5, color="green")
        self.assertTrue(self.graph_builder.graph.has_edge(1, 2))
        self.assertEqual(self.graph_builder.graph[1][2]["weight"], 3.5)
        self.assertEqual(self.graph_builder.graph[1][2]["color"], "green")

    def test_add_multiple_edges(self):
        edges = [(1, 2), (2, 3), (3, 4)]
        self.graph_builder._add_edges(edges)
        self.assertTrue(self.graph_builder.graph.has_edge(1, 2))
        self.assertTrue(self.graph_builder.graph.has_edge(2, 3))
        self.assertTrue(self.graph_builder.graph.has_edge(3, 4))

    def test_add_multiple_edges_with_weights(self):
        edges = [(1, 2), (2, 3), (3, 4)]
        weights = [2.0, 3.0, 4.0]
        self.graph_builder._add_edges(edges, weights=weights)
        self.assertEqual(self.graph_builder.graph[1][2]["weight"], 2.0)
        self.assertEqual(self.graph_builder.graph[2][3]["weight"], 3.0)
        self.assertEqual(self.graph_builder.graph[3][4]["weight"], 4.0)

    def test_add_node_attribute(self):
        self.graph_builder._add_node(1)
        self.graph_builder._add_node_attribute(1, color="blue")
        self.assertEqual(self.graph_builder.graph.nodes[1]["color"], "blue")

    def test_add_edge_attribute(self):
        self.graph_builder._add_edge(1, 2)
        self.graph_builder._add_edge_attribute(1, 2, color="blue")
        self.assertEqual(self.graph_builder.graph[1][2]["color"], "blue")

    def test_remove_edge_weight(self):
        self.graph_builder._add_edge(1, 2, weight=3.5)
        self.graph_builder._remove_edge_weight(1, 2)
        self.assertNotIn("weight", self.graph_builder.graph[1][2])

    def test_update_edge_weight(self):
        self.graph_builder._add_edge(1, 2, weight=3.5)
        self.graph_builder._update_edge_weight(1, 2, new_weight=4.5)
        self.assertEqual(self.graph_builder.graph[1][2]["weight"], 4.5)

    def test_convert_to_directed(self):
        self.graph_builder._add_edge(1, 2)
        self.graph_builder._convert_to_directed()
        self.assertIsInstance(self.graph_builder.graph, nx.DiGraph)

    def test_convert_to_undirected(self):
        self.graph_builder = GraphBuilderBase(directed=True)
        self.graph_builder._add_edge(1, 2)
        self.graph_builder._convert_to_undirected()
        self.assertIsInstance(self.graph_builder.graph, nx.Graph)

    def test_add_node_invalid_type(self):
        with self.assertRaises(ValueError):
            self.graph_builder._add_node("a")

    def test_add_edge_invalid_type(self):
        with self.assertRaises(ValueError):
            self.graph_builder._add_edge("a", 2)

    def test_add_edges_invalid_weights(self):
        edges = [(1, 2), (2, 3)]
        weights = [2.0]
        with self.assertRaises(ValueError):
            self.graph_builder._add_edges(edges, weights=weights)

    def test_remove_edge_weight_no_edge(self):
        with self.assertRaises(ValueError):
            self.graph_builder._remove_edge_weight(1, 2)

    def test_update_edge_weight_no_edge(self):
        with self.assertRaises(ValueError):
            self.graph_builder._update_edge_weight(1, 2, new_weight=4.5)

    def test_convert_to_directed_invalid(self):
        self.graph_builder = GraphBuilderBase(directed=True)
        with self.assertRaises(ValueError):
            self.graph_builder._convert_to_directed()

    def test_convert_to_undirected_invalid(self):
        self.graph_builder = GraphBuilderBase(directed=False)
        with self.assertRaises(ValueError):
            self.graph_builder._convert_to_undirected()

if __name__ == "__main__":
    unittest.main()
