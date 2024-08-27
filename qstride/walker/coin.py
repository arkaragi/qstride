"""
This module provides the CoinOperator class, which allows for the creation and
application of various quantum coin operators for quantum walks.

The class supports multiple coin types including Hadamard and Grover diffusion
operators. The coin operator can act on a specified number of qubits, enabling
the construction of multidimensional quantum walks.


Example usage
-------------
from qiskit import QuantumCircuit
from coin_operator import CoinOperator

# Instantiate a CoinOperator for a system with 2 coin qubits
hadamard_coin = CoinOperator(num_coin_qubits=2, coin_type='hadamard')

# Create the quantum circuit for the Hadamard coin
hadamard_circuit = hadamard_coin.create_coin_circuit()

# Integrate with a main quantum circuit
main_qc = QuantumCircuit(4)
main_qc.compose(hadamard_circuit, qubits=[2, 3], inplace=True)
main_qc.draw('mpl')
"""

from typing import Optional

import matplotlib.pyplot as plt
import qiskit


class CoinOperator:
    """
    A class to create and apply quantum coin operators in a quantum walk.

    The CoinOperator class supports various types of coin operations that can be applied
    to a specified number of qubits. This allows for the simulation of quantum walks in
    different dimensions. Supported coin types include:
    - Hadamard ('hadamard'): Creates a superposition of states.
    - Grover ('grover'): Implements Grover's diffusion operator, for amplitude amplification.
    - Custom ('custom'): Allows the use of a user-defined quantum gate.

    Parameters
    ----------
    num_coin_qubits: int
        The number of qubits used for the coin operation.

    coin_type: str, default="hadamard"
        The type of coin operator to apply.
        Options include 'hadamard', 'grover', and 'custom'.
    """

    def __init__(self,
                 num_coin_qubits: int,
                 coin_type: str = 'hadamard'):
        self.num_coin_qubits = num_coin_qubits
        self.coin_type = coin_type

    def _apply_hadamard(self,
                        qc: qiskit.QuantumCircuit):
        """
        Apply a Hadamard gate to all coin qubits.
        """
        for qubit in range(self.num_coin_qubits):
            qc.h(qubit)

    def _apply_grover(self,
                      qc: qiskit.QuantumCircuit):
        """
        Apply the Grover diffusion operator as the coin across all coin qubits.
        For a multi-qubit Grover diffusion operator, apply H -> Z -> H to each qubit.
        """
        for qubit in range(self.num_coin_qubits):
            qc.h(qubit)
        for qubit in range(self.num_coin_qubits):
            qc.z(qubit)
        for qubit in range(self.num_coin_qubits):
            qc.h(qubit)

    def _apply_custom(self,
                      qc: qiskit.QuantumCircuit,
                      custom_gate: qiskit.circuit.Gate):
        """
        Apply a custom gate to all coin qubits.

        Parameters
        ----------
        qc: qiskit.QuantumCircuit
            The quantum circuit to which the custom gate is applied.

        custom_gate: qiskit.circuit.Gate
            The custom quantum gate to apply to each coin qubit.
        """
        for qubit in range(self.num_coin_qubits):
            qc.append(custom_gate, [qubit])

    def create_coin_circuit(self,
                            custom_gate: Optional[qiskit.circuit.Gate] = None) -> qiskit.QuantumCircuit:
        """
        Create a quantum circuit for the specified coin operator.

        This method constructs a quantum circuit with the given number of qubits and applies
        the specified coin operator to them.
        If 'custom' is selected as the coin type, a custom quantum gate can be provided.
        The resulting quantum circuit can then be used as part of a larger quantum walk
        or other quantum algorithm.

        Parameters
        ----------
        custom_gate: qiskit.circuit.Gate, default=None
            A custom quantum gate to apply if the 'custom' coin_type is selected.
            This should be a Qiskit Gate instance.

        Returns
        -------
        qiskit.QuantumCircuit
            A quantum circuit implementing the specified coin operator.

        Raises
        ------
        ValueError
            If an unknown coin type is specified.
        """

        qc = qiskit.QuantumCircuit(self.num_coin_qubits)

        if self.coin_type == 'hadamard':
            self._apply_hadamard(qc)

        elif self.coin_type == 'grover':
            self._apply_grover(qc)

        elif self.coin_type == 'custom' and custom_gate is not None:
            self._apply_custom(qc, custom_gate)

        else:
            raise ValueError(f"Unknown coin type: {self.coin_type}. "
                             f"Please select 'hadamard', 'grover', or 'custom'.")

        return qc


    def visualize_coin_circuit(self,
                               custom_gate: Optional[qiskit.circuit.Gate] = None):
        """
        Visualize the quantum circuit for the specified coin operator.

        This method creates the quantum circuit for the coin operator and displays
        a visualization of the circuit.

        Parameters
        ----------
        custom_gate: qiskit.circuit.Gate, default=None
            A custom quantum gate to apply if the 'custom' coin_type is selected.
            This should be a Qiskit Gate instance.

        Raises
        ------
        ValueError
            If an unknown coin type is specified.
        """

        # Create the coin circuit using the existing method
        qc = self.create_coin_circuit(custom_gate)

        # Visualize the circuit
        qc.draw('mpl')
        plt.show()


def main():
    # Instantiate different CoinOperators for a system with 2 coin qubits
    hadamard_coin = CoinOperator(num_coin_qubits=2, coin_type='hadamard')
    grover_coin = CoinOperator(num_coin_qubits=2, coin_type='grover')

    # Combine with a main quantum circuit
    main_qc = qiskit.QuantumCircuit(4)  # Assuming 2 position qubits and 2 coin qubits

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