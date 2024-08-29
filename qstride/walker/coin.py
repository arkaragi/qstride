"""
Quantum Coin Operators for Discrete-Time Quantum Walks

This module defines the CoinOperator class, which provides a flexible framework for
creating and applying various quantum coin operators in discrete-time quantum walks.
Quantum walks are a fundamental component in quantum algorithms and the coin operator
is a key element that governs the behavior of the walker by applying specific quantum
gates to a set of qubits. It supports multiple types of coin operations, including:

- Hadamard ('balanced'):
    Applies a Hadamard gate to each qubit, generating a balanced superposition
    of states.
- Unbalanced Hadamard ('biased'):
    Applies a modified Hadamard gate that introduces bias, resulting in an
    asymmetric superposition.
- Pauli Gates ('pauli_x', 'pauli_y', 'pauli_z'):
    Applies the respective Pauli gate to each qubit, which are fundamental
    operations in quantum computing.
- Rotation ('rotation'):
    Applies a general rotation gate, parameterized by angles, allowing for
    arbitrary rotations on the Bloch sphere.
- Grover ('grover'):
    Implements Grover's diffusion operator, often used in quantum search
    algorithms for amplitude amplification.
- Custom ('custom'):
    Allows the application of a user-defined unitary gate, providing the
    flexibility to implement specialized quantum operations tailored to
    specific needs.

Usage Example
-------------
>>> from qiskit import QuantumCircuit

# Instantiate different CoinOperators for a system with 2 coin qubits
>>> hadamard_coin = CoinOperator(num_coin_qubits=2, coin_type='balanced')
>>> grover_coin = CoinOperator(num_coin_qubits=2, coin_type='grover')

# Combine with a main quantum circuit
>>> main_qc = QuantumCircuit(4)

# Attach the coin circuits to the main circuit
>>> hadamard_circuit = hadamard_coin.create_coin_circuit()
>>> grover_circuit = grover_coin.create_coin_circuit()
>>> main_qc.compose(hadamard_circuit, qubits=[2, 3], inplace=True)
>>> main_qc.compose(grover_circuit, qubits=[2, 3], inplace=True)

# Visualize the combined circuit
# main_qc.draw('mpl')
# plt.show()
"""

import math

import matplotlib.pyplot as plt
import qiskit
from qiskit import QuantumCircuit

__version__ = "0.1.0"


class CoinOperator:
    """
    A class for creating and applying quantum coin operators in discrete-time
    quantum walks.

    The CoinOperator class provides a robust framework for implementing various
    quantum coin operations, which are fundamental to discrete-time quantum walks.
    It allows for the application of different quantum coin types to a specified
    number of qubits, enabling the simulation of a wide range of quantum walk
    scenarios.

    - **Hadamard ('hadamard')**:
        Applies a Hadamard gate to each qubit, generating a balanced superposition
        of states, which is foundational for unbiased quantum walks.
    - **Unbalanced Hadamard ('unbalanced')**:
        Applies a modified Hadamard gate that introduces bias, resulting in an
        asymmetric superposition of states.
    - **Pauli-X ('pauli_x')**:
        Applies the Pauli-X (NOT) gate to each qubit, flipping the qubit state
        from |0⟩ to |1⟩ and vice versa.
    - **Pauli-Y ('pauli_y')**:
        Applies the Pauli-Y gate, which combines state flipping with a phase shift,
        altering both the amplitude and phase of the qubit state.
    - **Pauli-Z ('pauli_z')**:
        Applies the Pauli-Z gate, which flips the phase of the qubit state,
        inverting the sign of the |1⟩ component.
    - **Rotation ('rotation')**:
        Applies a general rotation gate to each qubit, parameterized by angles,
        enabling arbitrary rotations on the Bloch sphere for precise control of
        qubit states.
    - **Grover ('grover')**:
        Implements Grover's diffusion operator, used for amplitude amplification,
        which is particularly effective in quantum search algorithms.
    - **Custom ('custom')**:
        Allows the application of a user-defined unitary gate, providing the
        flexibility to implement specialized quantum operations tailored to
        specific needs.


    Parameters
    ----------
    num_coin_qubits: int
        The number of qubits used for the coin operation.
        This defines the dimension of the coin space.
        For example, 1 qubit corresponds to a 2-dimensional coin space,
        while 2 qubits correspond to a 4-dimensional coin space.

    coin_type: str, default='hadamard'
        The type of coin operator to apply.
        The available options include:
        'hadamard', 'unbalanced_hadamard', 'pauli_x', 'pauli_y', 'pauli_z',
        'rotation', 'grover', and 'custom'.
        If 'custom' is selected, the `custom_gate` parameter must be provided
        during the creation of the quantum circuit.
    """

    # Validate the coin type
    valid_coin_types = (
        "balanced",
        "grover",
        "pauli_x",
        "pauli_y",
        "pauli_z",
        "biased",
        "rotation",
        "custom"
    )

    def __init__(self,
                 num_coin_qubits: int,
                 coin_type: str = "balanced"):
        self.num_coin_qubits = num_coin_qubits
        self.coin_type = coin_type.lower()

        # Validate the coin type
        self._validate_coin_type()

        # Initialize the Coin circuit
        self.coin_circuit = QuantumCircuit(num_coin_qubits,
                                           name="coin_register")

    @staticmethod
    def _get_required_param(param_name: str,
                            params: dict):
        """
        Helper method to retrieve required parameters from kwargs.

        Parameters
        ----------
        param_name: str
            The name of the required parameter.

        params: dict
            The dictionary containing all parameters.

        Returns
        -------
        Any
            The value of the required parameter.

        Raises
        ------
        ValueError
            If the required parameter is not found in the dictionary.
        """
        if param_name not in params:
            raise ValueError(f"Missing required parameter: '{param_name}' for coin type.")
        return params[param_name]

    @staticmethod
    def _validate_bias(bias: float):
        """
        Validates that the bias parameter is within the correct range [0, 1].

        Parameters
        ----------
        bias: float
            The bias parameter to validate.

        Raises
        ------
        ValueError
            If the bias is not within the range [0, 1].
        """
        if not (0 <= bias <= 1):
            raise ValueError("Bias must be between 0 and 1.")

    @staticmethod
    def _validate_angles(theta: float,
                         phi: float,
                         lambda_: float):
        """
        Validates that the rotation angles are within a valid range (0 to 2π).

        Parameters
        ----------
        theta: float
            The angle for rotation around the X-axis.

        phi: float
            The angle for rotation around the Y-axis.

        lambda_: float
            The angle for rotation around the Z-axis.

        Raises
        ------
        ValueError
            If any of the angles are not within the range [0, 2π].
        """
        for angle in [theta, phi, lambda_]:
            if not (0 <= angle <= 2 * math.pi):
                raise ValueError("Angles must be between 0 and 2π.")

    @staticmethod
    def _validate_custom_gate(custom_gate: qiskit.circuit.Gate):
        """
        Validates that the custom gate is appropriate for the number of qubits.

        Parameters
        ----------
        custom_gate: Gate
            The custom gate to validate.

        Raises
        ------
        ValueError
            If the custom gate is not suitable for the number of qubits.
        """
        if custom_gate.num_qubits != 1:
            raise ValueError(f"Custom gate must be a single-qubit gate. "
                             f"Provided gate operates on {custom_gate.num_qubits} qubits.")

    def _validate_coin_type(self):
        """
        Validates that the provided coin_type is supported.

        Raises
        ------
        ValueError
            If the coin_type is not one of the valid types.
        """
        if self.coin_type not in self.valid_coin_types:
            raise ValueError(f"Unknown coin type: {self.coin_type}. "
                             f"Valid types are {self.valid_coin_types}.")

    def _create_balanced_coin(self):
        """
        Applies a Hadamard gate to each qubit, creating an equal superposition.

        This operation generates a uniform probability distribution across all
        qubits, setting up a balanced starting point for quantum walks.
        """
        for qubit in range(self.num_coin_qubits):
            self.coin_circuit.h(qubit)

    def _create_grover_coin(self):
        """
        Applies Grover's diffusion operator to each qubit, used in search
        algorithms for amplitude amplification.

        Grover's diffusion operator inverts the amplitudes of the qubit
        states about their average, effectively amplifying the probability
        of the desired state in quantum search algorithms.
        """
        for qubit in range(self.num_coin_qubits):
            self.coin_circuit.h(qubit)
        for qubit in range(self.num_coin_qubits):
            self.coin_circuit.z(qubit)
        for qubit in range(self.num_coin_qubits):
            self.coin_circuit.h(qubit)

    def _create_pauli_x_coin(self):
        """
        Applies the Pauli-X (NOT) gate to each qubit in the quantum circuit.

        The Pauli-X gate acts as a quantum equivalent of a classical NOT gate,
        flipping the state of each qubit from |0⟩ to |1⟩ and vice versa.
        In the context of a quantum walk, this operation effectively inverts
        the coin state for each qubit.
        """
        for qubit in range(self.num_coin_qubits):
            self.coin_circuit.x(qubit)

    def _create_pauli_y_coin(self):
        """
        Applies the Pauli-Y gate to each qubit in the quantum circuit.

        The Pauli-Y gate flips the state of each qubit similar to the Pauli-X gate,
        but with an additional phase shift, mapping |0⟩ to i|1⟩ and |1⟩ to -i|0⟩.
        This introduces a complex phase that can be useful in certain quantum walk
        scenarios, particularly those involving interference effects.
        """
        for qubit in range(self.num_coin_qubits):
            self.coin_circuit.y(qubit)

    def _create_pauli_z_coin(self):
        """
        Applies the Pauli-Z gate to each qubit in the quantum circuit.

        The Pauli-Z gate flips the phase of the qubit's state, mapping |0⟩ to |0⟩
        and |1⟩ to -|1⟩.
        This gate is often used to introduce a relative phase difference between
        the qubit states, which can be crucial in algorithms or quantum walks that
        rely on phase interference.
        """
        for qubit in range(self.num_coin_qubits):
            self.coin_circuit.z(qubit)

    def _create_biased_coin(self,
                            bias: float):
        """
        Applies an unbalanced Hadamard gate to each qubit in the quantum circuit,
        introducing asymmetry in the superposition.

        The unbalanced Hadamard gate creates an asymmetric superposition, where
        the bias parameter controls the probability amplitude of the |0⟩ and |1⟩
        states. This can be useful for creating biased quantum walks or for other
        applications where non-uniform distributions are desired.

        Parameters
        ----------
        bias: float
            The bias parameter that adjusts the asymmetry of the superposition.
            This should be a value between 0 and 1, where 0.5 corresponds to the
            balanced case (standard Hadamard gate).
        """
        self._validate_bias(bias)

        angle = math.acos(1 - 2 * bias)
        cos_half_angle = math.cos(angle / 2)
        sin_half_angle = math.sin(angle / 2)

        matrix = [
            [cos_half_angle, -sin_half_angle],
            [-sin_half_angle, cos_half_angle]
        ]

        gate = qiskit.circuit.Gate(name="unbalanced_hadamard",
                                   num_qubits=1,
                                   params=matrix)

        for qubit in range(self.num_coin_qubits):
            self.coin_circuit.append(gate, [qubit])

    def _create_rotation_coin(self,
                              theta: float,
                              phi: float,
                              lambda_: float):
        """
        Applies a general rotation gate parameterized by angles to each qubit,
        enabling arbitrary rotations.

        Parameters
        ----------
        theta: float
            The angle for rotation around the X-axis.

        phi: float
            The angle for rotation around the Y-axis.

        lambda_: float
            The angle for rotation around the Z-axis.
        """
        self._validate_angles(theta, phi, lambda_)

        for qubit in range(self.num_coin_qubits):
            self.coin_circuit.rz(lambda_, qubit)
            self.coin_circuit.ry(theta, qubit)
            self.coin_circuit.rz(phi, qubit)

    def _create_custom_coin(self,
                            custom_gate: qiskit.circuit.Gate):
        """
        Applies a user-defined custom unitary gate to each qubit.

        Parameters
        ----------
        custom_gate: Gate
            The custom unitary gate to apply to each qubit.
        """
        self._validate_custom_gate(custom_gate)

        for qubit in range(self.num_coin_qubits):
            self.coin_circuit.append(custom_gate, [qubit])

    def create_coin_circuit(self,
                            **kwargs):
        """
        Public method to create the coin circuit based on the coin_type.

        Depending on the coin_type, this method applies the appropriate
        quantum gates to the circuit. Additional parameters are passed via
        kwargs.

        Parameters
        ----------
        **kwargs
            Additional parameters required for specific coin types:
            - 'bias': float (0 to 1) - Required for 'unbalanced_hadamard'.
            - 'theta', 'phi', 'lambda_': float - Required for 'rotation'.
            - 'custom_gate': qiskit.circuit.Gate - Required for 'custom'.

        Returns
        -------
        QuantumCircuit
            The quantum circuit with the applied coin operations.

        Raises
        ------
        ValueError
            If the required parameters for the selected coin_type are not
            provided, or if an unsupported coin_type is specified.
        """

        coin_operations = {
            'balanced': self._create_balanced_coin,
            'grover': self._create_grover_coin,
            'pauli_x': self._create_pauli_x_coin,
            'pauli_y': self._create_pauli_y_coin,
            'pauli_z': self._create_pauli_z_coin,

            'biased': lambda: self._create_biased_coin(
                self._get_required_param('bias', kwargs)),

            'rotation': lambda: self._create_rotation_coin(
                self._get_required_param('theta', kwargs),
                self._get_required_param('phi', kwargs),
                self._get_required_param('lambda_', kwargs)),

            'custom': lambda: self._create_custom_coin(
                self._get_required_param('custom_gate', kwargs))
        }

        try:
            coin_operations[self.coin_type]()
        except KeyError:
            raise ValueError(f"Unsupported coin type: {self.coin_type}")

        return self.coin_circuit

    def visualize_coin_circuit(self):
        """
        Visualize the quantum circuit for the specified coin operator.
        """
        self.coin_circuit.draw('mpl')
        plt.show()


def main():
    # Instantiate different CoinOperators for a system with 2 coin qubits
    hadamard_coin = CoinOperator(num_coin_qubits=2, coin_type='balanced')
    grover_coin = CoinOperator(num_coin_qubits=2, coin_type='grover')

    # Combine with a main quantum circuit
    # Assuming 2 position qubits and 2 coin qubits
    main_qc = QuantumCircuit(4)

    # Attach the coin circuits to the main circuit
    hadamard_circuit = hadamard_coin.create_coin_circuit()
    grover_circuit = grover_coin.create_coin_circuit()
    main_qc.compose(hadamard_circuit, qubits=[2, 3], inplace=True)
    main_qc.compose(grover_circuit, qubits=[2, 3], inplace=True)

    # Visualize the combined circuit
    print("Visualizing Combined Quantum Circuit:")
    main_qc.draw('mpl')
    plt.show()


if __name__ == "__main__":
    main()
