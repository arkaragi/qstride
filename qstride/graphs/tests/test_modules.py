"""
Test suite for the graphs.modules.py module.
"""

import os
import unittest
import networkx as nx

from ..modules import BipartiteGraphBuilder
from ..modules import CompleteGraphBuilder
from ..modules import CyclicGraphBuilder
from ..modules import GridGraphBuilder
from ..modules import LinearGraphBuilder
from ..modules import StarGraphBuilder
from ..modules import TreeGraphBuilder
from ..modules import WeightedGraphBuilder

__version__ = "0.1.0"


class TestBipartiteGraphBuilder(unittest.TestCase):
    """
    Test suite for the BipartiteGraphBuilder class.
    """

    def setUp(self):
        """
        Set up a basic BipartiteGraphBuilder instance for use in tests.
        """
        self.builder = BipartiteGraphBuilder(set1_size=3, set2_size=3)

    def test_bipartite_graph_creation(self):
        """
        Test the creation of a bipartite graph.
        """
        # Check that the graph has the expected number of nodes and edges
        self.assertEqual(len(self.builder.graph.nodes), 6)
        self.assertEqual(len(self.builder.graph.edges), 9)

        # Check if the graph is bipartite
        self.assertTrue(self.builder.is_bipartite())

        # Check the bipartite sets
        set1, set2 = self.builder.get_bipartite_sets()
        self.assertEqual(len(set1), 3)
        self.assertEqual(len(set2), 3)

    def test_weighted_bipartite_graph_creation(self):
        """
        Test the creation of a weighted bipartite graph.
        """
        weights = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        weighted_builder = BipartiteGraphBuilder(set1_size=3, set2_size=3, weights=weights)

        # Check that the graph has the expected number of nodes and edges
        self.assertEqual(len(weighted_builder.graph.nodes), 6)
        self.assertEqual(len(weighted_builder.graph.edges), 9)

        # Check that the weights are correctly assigned
        edge_weights = nx.get_edge_attributes(weighted_builder.graph, 'weight')
        self.assertEqual(len(edge_weights), 9)
        self.assertEqual(list(edge_weights.values()), weights)

    def test_invalid_set_sizes(self):
        """
        Test creation of bipartite graph with invalid set sizes.
        """
        with self.assertRaises(ValueError):
            BipartiteGraphBuilder(set1_size=-1, set2_size=3)

        with self.assertRaises(ValueError):
            BipartiteGraphBuilder(set1_size=3, set2_size=0)

    def test_mismatched_weights(self):
        """
        Test creation of bipartite graph with mismatched number of weights and edges.
        """
        weights = [1.0, 2.0]  # Not enough weights for 9 edges
        with self.assertRaises(ValueError):
            BipartiteGraphBuilder(set1_size=3, set2_size=3, weights=weights)

    def test_bipartite_check_error(self):
        """
        Test the error handling in the is_bipartite method.
        """
        # Create an invalid bipartite graph scenario by adding an edge within one set
        self.builder.graph.add_edge(1, 2)  # Both nodes are in set1

        with self.assertRaises(ValueError):
            self.builder.is_bipartite()

    def test_bipartite_sets_retrieval(self):
        """
        Test retrieval of bipartite sets.
        """
        set1, set2 = self.builder.get_bipartite_sets()
        self.assertEqual(len(set1), 3)
        self.assertEqual(len(set2), 3)
        self.assertTrue(all(node in range(1, 4) for node in set1))
        self.assertTrue(all(node in range(4, 7) for node in set2))

    def test_directed_bipartite_graph(self):
        """
        Test creation of a directed bipartite graph.
        """
        directed_builder = BipartiteGraphBuilder(set1_size=3, set2_size=3, directed=True)
        self.assertTrue(directed_builder.directed)
        self.assertIsInstance(directed_builder.graph, nx.DiGraph)


class TestCompleteGraphBuilder(unittest.TestCase):
    """
    Test suite for the CompleteGraphBuilder class.
    """

    def setUp(self):
        self.builder = CompleteGraphBuilder(num_nodes=4)

    def test_complete_graph_creation(self):
        self.assertEqual(len(self.builder.graph.nodes), 4)
        self.assertEqual(len(self.builder.graph.edges), 6)
        expected_edges = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
        self.assertEqual(sorted(self.builder.get_complete_graph_edges()), sorted(expected_edges))

    def test_weighted_complete_graph(self):
        weights = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        builder = CompleteGraphBuilder(num_nodes=4, weights=weights)
        self.assertEqual(len(builder.graph.edges), 6)
        for (u, v) in builder.graph.edges:
            self.assertIn(builder.graph[u][v]['weight'], weights)

    def test_invalid_num_nodes(self):
        with self.assertRaises(ValueError):
            CompleteGraphBuilder(num_nodes=1)


class TestCyclicGraphBuilder(unittest.TestCase):
    """
    Test suite for the CyclicGraphBuilder class.
    """

    def setUp(self):
        self.builder = CyclicGraphBuilder(num_nodes=5)

    def test_cyclic_graph_creation(self):
        self.assertEqual(len(self.builder.graph.nodes), 5)
        self.assertEqual(len(self.builder.graph.edges), 5)
        expected_edges = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)]
        self.assertEqual(sorted(self.builder.graph.edges), sorted(expected_edges))

    def test_weighted_cyclic_graph(self):
        weights = [1.0, 2.0, 3.0, 4.0, 5.0]
        builder = CyclicGraphBuilder(num_nodes=5, weights=weights)
        self.assertEqual(len(builder.graph.edges), 5)
        for (u, v) in builder.graph.edges:
            self.assertIn(builder.graph[u][v]['weight'], weights)

    def test_invalid_num_nodes(self):
        with self.assertRaises(ValueError):
            CyclicGraphBuilder(num_nodes=2)


class TestLinearGraphBuilder(unittest.TestCase):
    """
    Test suite for the LinearGraphBuilder class.
    """

    def setUp(self):
        self.builder = LinearGraphBuilder(num_nodes=5)

    def test_linear_graph_creation(self):
        self.assertEqual(len(self.builder.graph.nodes), 5)
        self.assertEqual(len(self.builder.graph.edges), 4)
        expected_edges = [(1, 2), (2, 3), (3, 4), (4, 5)]
        self.assertEqual(sorted(self.builder.graph.edges), sorted(expected_edges))

    def test_weighted_linear_graph(self):
        weights = [1.0, 2.0, 3.0, 4.0]
        builder = LinearGraphBuilder(num_nodes=5, weights=weights)
        self.assertEqual(len(builder.graph.edges), 4)
        for (u, v) in builder.graph.edges:
            self.assertIn(builder.graph[u][v]['weight'], weights)

    def test_add_node_to_end(self):
        self.builder.add_node_to_end(weight=2.5)
        self.assertEqual(len(self.builder.graph.nodes), 6)
        self.assertEqual(self.builder.graph[5][6]['weight'], 2.5)

    def test_invalid_num_nodes(self):
        with self.assertRaises(ValueError):
            LinearGraphBuilder(num_nodes=1)


class TestGridGraphBuilder(unittest.TestCase):
    """
    Test suite for the GridGraphBuilder class.
    """

    def setUp(self):
        self.builder = GridGraphBuilder(rows=2, cols=3)

    def test_grid_graph_creation(self):
        self.assertEqual(len(self.builder.graph.nodes), 6)
        self.assertEqual(len(self.builder.graph.edges), 7)
        expected_edges = [
            ((0, 0), (0, 1)), ((0, 1), (0, 2)),
            ((0, 0), (1, 0)), ((0, 1), (1, 1)),
            ((0, 2), (1, 2)), ((1, 0), (1, 1)),
            ((1, 1), (1, 2))
        ]
        self.assertEqual(sorted(self.builder.get_grid_graph_edges()), sorted(expected_edges))

    def test_weighted_grid_graph(self):
        weights = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        builder = GridGraphBuilder(rows=2, cols=3, weights=weights)
        self.assertEqual(len(builder.graph.edges), 7)
        for (u, v) in builder.graph.edges:
            self.assertIn(builder.graph[u][v]['weight'], weights)

    def test_invalid_grid_dimensions(self):
        with self.assertRaises(ValueError):
            GridGraphBuilder(rows=0, cols=3)


class TestStarGraphBuilder(unittest.TestCase):
    """
    Test suite for the StarGraphBuilder class.
    """

    def setUp(self):
        self.builder = StarGraphBuilder(num_leaves=5)

    def test_star_graph_creation(self):
        self.assertEqual(len(self.builder.graph.nodes), 6)
        self.assertEqual(len(self.builder.graph.edges), 5)
        self.assertEqual(self.builder.get_center_node(), 0)
        self.assertEqual(self.builder.get_leaf_nodes(), [1, 2, 3, 4, 5])

    def test_weighted_star_graph(self):
        weights = [1.0, 2.0, 3.0, 4.0, 5.0]
        builder = StarGraphBuilder(num_leaves=5, weights=weights)
        self.assertEqual(len(builder.graph.edges), 5)
        for u, v in builder.graph.edges:
            self.assertIn(builder.graph[u][v]['weight'], weights)

    def test_invalid_num_leaves(self):
        with self.assertRaises(ValueError):
            StarGraphBuilder(num_leaves=0)


class TestTreeGraphBuilder(unittest.TestCase):
    """
    Test suite for the TreeGraphBuilder class.
    """

    def setUp(self):
        self.builder = TreeGraphBuilder(num_nodes=5)

    def test_tree_graph_creation(self):
        self.assertEqual(len(self.builder.graph.nodes), 5)
        self.assertEqual(len(self.builder.graph.edges), 4)
        self.assertEqual(self.builder.get_root_node(), 1)
        self.assertEqual(self.builder.get_leaf_nodes(), [5])

    def test_directed_tree_graph(self):
        builder = TreeGraphBuilder(num_nodes=5, directed=True)
        self.assertTrue(builder.directed)
        self.assertIsInstance(builder.graph, nx.DiGraph)

    def test_invalid_num_nodes(self):
        with self.assertRaises(ValueError):
            TreeGraphBuilder(num_nodes=1)


class TestWeightedGraphBuilder(unittest.TestCase):
    """
    Test suite for the WeightedGraphBuilder class.
    """

    def setUp(self):
        self.edges = [(1, 2), (2, 3), (3, 4)]
        self.weights = [1.5, 2.5, 3.5]
        self.builder = WeightedGraphBuilder(edges=self.edges, weights=self.weights)

    def test_weighted_graph_creation(self):
        """
        Test the creation of a weighted graph and verify that the weights are correctly assigned.
        """
        self.assertEqual(len(self.builder.graph.nodes), 4)
        self.assertEqual(len(self.builder.graph.edges), 3)
        for (u, v), weight in zip(self.edges, self.weights):
            self.assertEqual(self.builder.get_edge_weight(u, v), weight)

    def test_add_edge_with_weight(self):
        """
        Test adding a new edge with a specific weight to the graph.
        """
        self.builder.add_edge_with_weight(4, 5, weight=4.5)
        self.assertEqual(self.builder.get_edge_weight(4, 5), 4.5)

    def test_unweighted_graph_creation(self):
        """
        Test the creation of an unweighted graph (i.e., no weights provided).
        """
        builder = WeightedGraphBuilder(edges=self.edges)
        self.assertEqual(len(builder.graph.nodes), 4)
        self.assertEqual(len(builder.graph.edges), 3)
        for (u, v) in self.edges:
            self.assertIsNone(builder.get_edge_weight(u, v))

    def test_invalid_weights_length(self):
        """
        Test that an error is raised if the number of weights does not match the number of edges.
        """
        with self.assertRaises(ValueError):
            WeightedGraphBuilder(edges=self.edges, weights=[1.5, 2.5])

    def test_get_edge_weight_no_edge(self):
        """
        Test that an error is raised when trying to get the weight of a non-existent edge.
        """
        with self.assertRaises(ValueError):
            self.builder.get_edge_weight(1, 4)


if __name__ == "__main__":
    unittest.main()
