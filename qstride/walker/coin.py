"""
Quantum Coin Operators for Discrete-Time Quantum Walks

This module defines the CoinOperator class, which provides a flexible framework for
creating and applying various quantum coin operators in discrete-time quantum walks.
Quantum walks are a fundamental component in quantum algorithms and the coin operator
is a key element that governs the behavior of the walker by applying specific quantum
gates to a set of qubits.

It supports multiple types of coin operations, including:

- Hadamard ('balanced'):
    Applies a Hadamard gate to each qubit, generating a balanced superposition
    of states.

- Grover ('grover'):
    Implements Grover's diffusion operator, often used in quantum search algorithms
    for amplitude amplification.

- Pauli Gates ('pauli_x', 'pauli_y', 'pauli_z'):
    Applies the respective Pauli gate to each qubit, which are fundamental operations
    in quantum computing.

- Rotation ('rotation'):
    Applies a general rotation gate, parameterized by angles, allowing for arbitrary
    rotations on the Bloch sphere.

Usage Example
-------------
>>> from qiskit import QuantumCircuit

# Instantiate different CoinOperators for a system with 2 coin qubits
>>> hadamard_coin = CoinOperator(num_coin_qubits=2, coin_type='hadamard')
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
from typing import Optional, Dict, Any, List

import matplotlib.pyplot as plt
import qiskit
from qiskit import QuantumCircuit

__version__ = "0.1.0"


class CoinBase:
    """
    Base class for creating and validating quantum coin operators in discrete-time
    quantum walks.

    It provides foundational methods for setting up the quantum coin circuit,
    validating input parameters, and ensuring that operations are consistent
    with the requirements of quantum computing. Subclasses should implement
    specific quantum coin operations such as Hadamard, Grover, or custom gate
    operations.

    Parameters
    ----------
    num_coin_qubits: int, default=1
        The number of qubits used for the coin operation.
        This defines the dimension of the coin space.
        For example, 1 qubit corresponds to a 2-dimensional coin space,
        while 2 qubits correspond to a 4-dimensional coin space.

    coin_type: str, default=None
        The type of coin operator to apply.
        The available options include: 'hadamard', 'grover', 'pauli_x',
        'pauli_y', 'pauli_z', and 'rotation'.
    """

    valid_coin_types = (
        "hadamard",
        "grover",
        "pauli_x",
        "pauli_y",
        "pauli_z",
        "rotation",
    )

    def __init__(self,
                 num_coin_qubits: int = 1,
                 coin_type: Optional[str] = None):
        self.num_coin_qubits = num_coin_qubits
        self.coin_type = coin_type
        
        # Initialize the Coin circuit
        self.coin_circuit = QuantumCircuit(num_coin_qubits,
                                           name="coin_circuit")

    @staticmethod
    def _get_required_param(param_name: str,
                            params: Dict[str, Any]) -> Any:
        """
        Helper method to retrieve required parameters from a dictionary.

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
    def _validate_coin_type(coin_type: str):
        """
        Validates that the provided coin_type is supported.

        Parameters
        ----------
        coin_type : str
            The type of the coin to validate.

        Raises
        ------
        ValueError
            If the coin_type is not one of the valid types.
        """
        if coin_type is not None and coin_type not in CoinBase.valid_coin_types:
            raise ValueError(f"Unknown coin type: {coin_type}. "
                             f"Valid types are {CoinBase.valid_coin_types}.")

    @staticmethod
    def _validate_custom_gate(custom_gate: qiskit.circuit.Gate):
        """
        Validates that the custom gate is appropriate for the number of qubits.

        Parameters
        ----------
        custom_gate: qiskit.circuit.Gate
            The custom gate to validate.

        Raises
        ------
        ValueError
            If the custom gate is not suitable for the number of qubits.
        """
        if custom_gate.num_qubits != 1:
            raise ValueError(f"Custom gate must be a single-qubit gate. "
                             f"Provided gate operates on {custom_gate.num_qubits} qubits.")


class CoinOperator(CoinBase):
    """
    A class for creating and applying quantum coin operators in discrete-time
    quantum walks.

    The CoinOperator class provides a robust framework for implementing various
    quantum coin operations, which are fundamental to discrete-time quantum walks.
    It allows for the application of different quantum coin types to a specified
    number of qubits, enabling the simulation of a wide range of quantum walk
    scenarios.

    - Hadamard ('hadamard'):
        Applies a Hadamard gate to each qubit, generating a balanced superposition
        of states, which is foundational for unbiased quantum walks.
    - Grover ('grover'):
        Implements Grover's diffusion operator, used for amplitude amplification,
        which is particularly effective in quantum search algorithms.
    - Pauli-X ('pauli_x'):
        Applies the Pauli-X (NOT) gate to each qubit, flipping the qubit state
        from |0⟩ to |1⟩ and vice versa.
    - Pauli-Y ('pauli_y'):
        Applies the Pauli-Y gate, which combines state flipping with a phase shift,
        altering both the amplitude and phase of the qubit state.
    - Pauli-Z ('pauli_z'):
        Applies the Pauli-Z gate, which flips the phase of the qubit state,
        inverting the sign of the |1⟩ component.
    - Rotation ('rotation'):
        Applies a general rotation gate to each qubit, parameterized by angles,
        enabling arbitrary rotations on the Bloch sphere for precise control of
        qubit states.

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

    def __init__(self,
                 num_coin_qubits: int,
                 coin_type: Optional[str] = None):
        super().__init__(num_coin_qubits=num_coin_qubits,
                         coin_type=coin_type)

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
            'hadamard': self.create_hadamard_coin,
            'grover': self.create_grover_coin,
            'pauli_x': self.create_pauli_x_coin,
            'pauli_y': self.create_pauli_y_coin,
            'pauli_z': self.create_pauli_z_coin,
            'rotation': lambda: self.create_rotation_coin(
                self._get_required_param('theta', kwargs),
                self._get_required_param('phi', kwargs),
                self._get_required_param('lambda_', kwargs)),
        }

        try:
            coin_operations[self.coin_type]()
        except KeyError:
            raise ValueError(f"Unsupported coin type: {self.coin_type}")

        return self.coin_circuit

    def create_hadamard_coin(self):
        """
        Applies a Hadamard gate to each qubit, creating a balanced superposition.

        The Hadamard gate is a fundamental quantum gate that transforms the basis
        states |0⟩ and |1⟩ into equal superpositions. For each qubit the Hadamard
        gate maps |0⟩ → (|0⟩ + |1⟩) / √2 and |1⟩ → (|0⟩ - |1⟩) / √2.
        By applying this gate to all qubits in the coin space, the quantum walker
        achieves an equal probability of moving in any direction after each step
        of the walk. This is the standard starting point for unbiased quantum walks,
        where no direction is preferred.
        """
        for qubit in range(self.num_coin_qubits):
            self.coin_circuit.h(qubit)

    def create_grover_coin(self):
        """
        Applies Grover's diffusion operator to each qubit for amplitude
        amplification.

        Grover's diffusion operator is a key component of Grover's search
        algorithm, utilized to amplify the probability amplitude of certain
        states by suppressing others. The operator is defined as: D = HZH,
        where H, Z are the Hadamard and Pauli-Z gates respectively.

        The action of Grover's operator on a qubit in a superposition state
        inverts the amplitudes about their average, leading to constructive
        interference for some states and destructive interference for others.
        In the context of quantum walks, applying Grover's diffusion operator
        to all qubits biases the walk toward certain states, depending on the
        initial configuration and the walk's dynamics. This can be utilized
        in search-related quantum algorithms or quantum walks where specific
        states need to be amplified.
        """
        for qubit in range(self.num_coin_qubits):
            self.coin_circuit.h(qubit)
        for qubit in range(self.num_coin_qubits):
            self.coin_circuit.z(qubit)
        for qubit in range(self.num_coin_qubits):
            self.coin_circuit.h(qubit)

    def create_pauli_x_coin(self):
        """
        Applies the Pauli-X (NOT) gate to each qubit in the quantum circuit.

        The Pauli-X gate acts as a quantum equivalent of a classical NOT gate,
        flipping the state of each qubit from |0⟩ to |1⟩ and vice versa.
        In the context of a quantum walk, this operation effectively inverts
        the coin state for each qubit.
        """
        for qubit in range(self.num_coin_qubits):
            self.coin_circuit.x(qubit)

    def create_pauli_y_coin(self):
        """
        Applies the Pauli-Y gate to each qubit in the quantum circuit.

        The Pauli-Y gate flips the state of each qubit similar to the Pauli-X gate,
        but with an additional phase shift, mapping |0⟩ to i|1⟩ and |1⟩ to -i|0⟩.
        This introduces a complex phase that can be useful in certain quantum walk
        scenarios, particularly those involving interference effects.
        """
        for qubit in range(self.num_coin_qubits):
            self.coin_circuit.y(qubit)

    def create_pauli_z_coin(self):
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

    def create_rotation_coin(self,
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

    def visualize_coin_circuit(self):
        """
        Visualize the quantum circuit for the specified coin operator.
        """
        self.coin_circuit.draw('mpl')
        plt.show()


class WeightedCoinOperator(CoinOperator):
    """

    """

    def __init__(self,
                 num_coin_qubits: int,
                 weights: Optional[List[float]] = None):
        super().__init__(num_coin_qubits=num_coin_qubits,
                         coin_type="weighted")
        self.weights = weights or [0.5] * (2 ** num_coin_qubits)  # Default to balanced weights

        # Validate weights
        self._validate_weights()

    def _validate_weights(self):
        if len(self.weights) != 2 ** self.num_coin_qubits:
            raise ValueError("Length of weights must match the dimension of the coin space.")
        if not math.isclose(sum(self.weights), 1.0):
            raise ValueError("Weights must sum to 1.")
        if any(w < 0 or w > 1 for w in self.weights):
            raise ValueError("Each weight must be between 0 and 1.")

    def create_weighted_coin(self):
        """
        Applies a weighted coin operation based on the provided weights.
        """
        for qubit in range(self.num_coin_qubits):
            # Custom logic to apply the weights as gates
            # This is a placeholder, and you would implement the actual weighted gate logic here
            pass

    def create_coin_circuit(self, **kwargs):
        """
        Overrides the method to create the coin circuit based on weighted probabilities.
        """
        self.create_weighted_coin()
        return self.coin_circuit


def main():
    # Instantiate different CoinOperators for a system with 2 coin qubits
    hadamard_coin = CoinOperator(num_coin_qubits=2, coin_type='hadamard')
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
