"""
Test suite for the walker.coin.py module.
"""

import unittest
import math

from qiskit import QuantumCircuit, circuit

from qstride.qstride.walker.coin import CoinBase
from qstride.qstride.walker.coin import CoinOperator

__version__ = "0.1.0"


class TestCoinBase(unittest.TestCase):
    """
    Test suite for the CoinBase class, ensuring that all methods function
    correctly, including validation, initialization, and circuit setup.
    """

    def setUp(self):
        """
        Set up the environment before each test.
        """
        self.default_qubits = 2
        self.default_coin_type = "hadamard"

    def test_initialization_default(self):
        """
        Test default initialization of CoinBase.
        Ensure that the quantum circuit is created with the correct number of qubits.
        """
        coin_base = CoinBase(num_coin_qubits=self.default_qubits)

        self.assertEqual(coin_base.num_coin_qubits, self.default_qubits)
        self.assertIsInstance(coin_base.coin_circuit, QuantumCircuit)
        self.assertEqual(coin_base.coin_circuit.num_qubits, self.default_qubits)

    def test_initialization_with_coin_type(self):
        """
        Test initialization of CoinBase with a specified coin type.
        Ensure the coin type is stored correctly.
        """
        coin_base = CoinBase(num_coin_qubits=self.default_qubits,
                             coin_type=self.default_coin_type)

        self.assertEqual(coin_base.coin_type, self.default_coin_type)

    def test_validate_coin_type_valid(self):
        """
        Test _validate_coin_type static method with a valid coin type.
        Ensure no exception is raised for a valid coin type.
        """
        try:
            CoinBase._validate_coin_type("hadamard")
        except ValueError:
            self.fail("_validate_coin_type() raised ValueError unexpectedly!")

    def test_validate_coin_type_invalid(self):
        """
        Test _validate_coin_type static method with an invalid coin type.
        Ensure that a ValueError is raised with the appropriate message.
        """
        with self.assertRaises(ValueError) as context:
            CoinBase._validate_coin_type("invalid_coin_type")
        self.assertIn("Unknown coin type", str(context.exception))

    def test_get_required_param_exists(self):
        """
        Test _get_required_param static method when the required parameter exists.
        Ensure the correct value is returned.
        """
        params = {"param1": 42, "param2": "test_value"}
        result = CoinBase._get_required_param("param1", params)
        self.assertEqual(result, 42)

    def test_get_required_param_missing(self):
        """
        Test _get_required_param static method when the required parameter is missing.
        Ensure that a ValueError is raised with the appropriate message.
        """
        params = {"param1": 42}
        with self.assertRaises(ValueError) as context:
            CoinBase._get_required_param("param2", params)
        self.assertIn("Missing required parameter", str(context.exception))

    def test_validate_angles_valid(self):
        """
        Test _validate_angles static method with valid angles.
        Ensure no exception is raised for angles within the valid range.
        """
        try:
            CoinBase._validate_angles(math.pi / 2, math.pi, 2 * math.pi)
        except ValueError:
            self.fail("_validate_angles() raised ValueError unexpectedly!")

    def test_validate_angles_invalid(self):
        """
        Test _validate_angles static method with an invalid angle.
        Ensure that a ValueError is raised with the appropriate message.
        """
        with self.assertRaises(ValueError) as context:
            CoinBase._validate_angles(-1, math.pi, 2 * math.pi)
        self.assertIn("Angles must be between 0 and 2π", str(context.exception))

    def test_validate_bias_valid(self):
        """
        Test _validate_bias static method with a valid bias value.
        Ensure no exception is raised for bias within the valid range.
        """
        try:
            CoinBase._validate_bias(0.5)
        except ValueError:
            self.fail("_validate_bias() raised ValueError unexpectedly!")

    def test_validate_bias_invalid(self):
        """
        Test _validate_bias static method with an invalid bias value.
        Ensure that a ValueError is raised with the appropriate message.
        """
        with self.assertRaises(ValueError) as context:
            CoinBase._validate_bias(1.5)
        self.assertIn("Bias must be between 0 and 1", str(context.exception))

    def test_validate_custom_gate_valid(self):
        """
        Test _validate_custom_gate static method with a valid custom gate.
        Ensure no exception is raised for a gate operating on a single qubit.
        """
        custom_gate = circuit.library.HGate()
        try:
            CoinBase._validate_custom_gate(custom_gate)
        except ValueError:
            self.fail("_validate_custom_gate() raised ValueError unexpectedly!")

    def test_validate_custom_gate_invalid(self):
        """
        Test _validate_custom_gate static method with an invalid custom gate.
        Ensure that a ValueError is raised for a gate operating on more than one qubit.
        """
        custom_gate = circuit.library.CXGate()
        with self.assertRaises(ValueError) as context:
            CoinBase._validate_custom_gate(custom_gate)
        self.assertIn("Custom gate must be a single-qubit gate", str(context.exception))


class TestCoinOperator(unittest.TestCase):
    """
    Test suite for the CoinOperator class, ensuring that all methods function
    correctly, including validation, initialization, coin circuit creation,
    and specific quantum coin operations.
    """

    def setUp(self):
        """
        Set up the environment before each test.
        """
        self.default_qubits = 2
        self.default_coin_type = "hadamard"

    def test_initialization_default(self):
        """
        Test default initialization of CoinOperator.
        Ensure that the quantum circuit is created with the correct number of qubits.
        """
        coin_operator = CoinOperator(num_coin_qubits=self.default_qubits)

        self.assertEqual(coin_operator.num_coin_qubits, self.default_qubits)
        self.assertIsInstance(coin_operator.coin_circuit, QuantumCircuit)
        self.assertEqual(coin_operator.coin_circuit.num_qubits, self.default_qubits)

    def test_initialization_with_coin_type(self):
        """
        Test initialization of CoinOperator with a specified coin type.
        Ensure the coin type is stored correctly.
        """
        coin_operator = CoinOperator(num_coin_qubits=self.default_qubits,
                                     coin_type=self.default_coin_type)

        self.assertEqual(coin_operator.coin_type, self.default_coin_type)

    def test_create_coin_circuit(self):
        """
        Test the creation of a complete Hadamard coin circuit via create_coin_circuit method.
        Ensure the circuit is correctly constructed.
        """
        coin_operator = CoinOperator(num_coin_qubits=self.default_qubits, coin_type="hadamard")
        coin_operator.create_coin_circuit()

        for qubit in range(self.default_qubits):
            self.assertIn(('h', qubit), [(instruction.operation.name, qubit._index)
                                         for instruction in coin_operator.coin_circuit.data
                                         for qubit in instruction.qubits])

    def test_create_coin_circuit_invalid(self):
        """
        Test the create_coin_circuit method with an invalid coin type.
        Ensure that a ValueError is raised with the appropriate message.
        """
        coin_operator = CoinOperator(num_coin_qubits=self.default_qubits, coin_type="invalid_type")

        with self.assertRaises(ValueError) as context:
            coin_operator.create_coin_circuit()

        self.assertIn("Unsupported coin type", str(context.exception))

    def test_create_hadamard_coin(self):
        """
        Test the creation of a Hadamard coin circuit.
        Ensure that a Hadamard gate is applied to each qubit.
        """
        coin_operator = CoinOperator(num_coin_qubits=self.default_qubits, coin_type="hadamard")
        coin_operator.create_hadamard_coin()

        for qubit in range(self.default_qubits):
            self.assertIn(('h', qubit), [(instruction.operation.name, qubit._index)
                                         for instruction in coin_operator.coin_circuit.data
                                         for qubit in instruction.qubits])

    def test_create_grover_coin(self):
        """
        Test the creation of a Grover coin circuit.
        Ensure that the Grover's diffusion operator is correctly applied.
        """
        coin_operator = CoinOperator(num_coin_qubits=self.default_qubits, coin_type="grover")
        coin_operator.create_grover_coin()

        expected_ops = [('h', qubit) for qubit in range(self.default_qubits)] + \
                       [('z', qubit) for qubit in range(self.default_qubits)] + \
                       [('h', qubit) for qubit in range(self.default_qubits)]

        actual_ops = [(instruction.operation.name, qubit._index)
                      for instruction in coin_operator.coin_circuit.data
                      for qubit in instruction.qubits]

        self.assertEqual(actual_ops, expected_ops)

    def test_create_pauli_x_coin(self):
        """
        Test the creation of a Pauli-X coin circuit.
        Ensure that the Pauli-X gate is applied to each qubit.
        """
        coin_operator = CoinOperator(num_coin_qubits=self.default_qubits, coin_type="pauli_x")
        coin_operator.create_pauli_x_coin()

        for qubit in range(self.default_qubits):
            self.assertIn(('x', qubit), [(instruction.operation.name, qubit._index)
                                         for instruction in coin_operator.coin_circuit.data
                                         for qubit in instruction.qubits])

    def test_create_pauli_y_coin(self):
        """
        Test the creation of a Pauli-Y coin circuit.
        Ensure that the Pauli-Y gate is applied to each qubit.
        """
        coin_operator = CoinOperator(num_coin_qubits=self.default_qubits, coin_type="pauli_y")
        coin_operator.create_pauli_y_coin()

        for qubit in range(self.default_qubits):
            self.assertIn(('y', qubit), [(instruction.operation.name, qubit._index)
                                         for instruction in coin_operator.coin_circuit.data
                                         for qubit in instruction.qubits])

    def test_create_pauli_z_coin(self):
        """
        Test the creation of a Pauli-Z coin circuit.
        Ensure that the Pauli-Z gate is applied to each qubit.
        """
        coin_operator = CoinOperator(num_coin_qubits=self.default_qubits, coin_type="pauli_z")
        coin_operator.create_pauli_z_coin()

        for qubit in range(self.default_qubits):
            self.assertIn(('z', qubit), [(instruction.operation.name, qubit._index)
                                         for instruction in coin_operator.coin_circuit.data
                                         for qubit in instruction.qubits])

    def test_create_rotation_coin(self):
        """
        Test the creation of a rotation coin circuit.
        Ensure that the rotation gates are applied with the correct angles.
        """
        theta, phi, lambda_ = math.pi / 2, math.pi / 4, math.pi / 8
        coin_operator = CoinOperator(num_coin_qubits=self.default_qubits, coin_type="rotation")
        coin_operator.create_rotation_coin(theta, phi, lambda_)

        operations = [(instruction.operation.name, qubit._index)
                      for instruction in coin_operator.coin_circuit.data
                      for qubit in instruction.qubits]

        for qubit in range(self.default_qubits):
            self.assertIn(('rz', qubit), operations)
            self.assertIn(('ry', qubit), operations)


if __name__ == "__main__":
    unittest.main()
