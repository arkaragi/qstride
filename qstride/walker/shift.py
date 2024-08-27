"""
This module provides the ShiftOperator class, which allows for the creation and
application of shift operations in quantum walks.

The shift operator controls the movement of the quantum walker across the
positions (or nodes) in the quantum walk. The class supports different types of
shift operations that can be applied to a specified number of position qubits,
enabling the construction of multidimensional quantum walks.

Example usage
-------------
from qiskit import QuantumCircuit
from shift_operator import ShiftOperator

# Instantiate a ShiftOperator for a system with 2 position qubits and 2 coin qubits
shift_op = ShiftOperator(num_position_qubits=2, num_coin_qubits=2)

# Create the quantum circuit for the shift operation
shift_circuit = shift_op.create_shift_circuit()

# Integrate with a main quantum circuit
main_qc = QuantumCircuit(4)  # Assuming 2 position qubits and 2 coin qubits
main_qc.compose(shift_circuit, qubits=[0, 1, 2, 3], inplace=True)
main_qc.draw('mpl')
"""
from typing import Optional

import matplotlib.pyplot as plt
import qiskit


class ShiftOperator:
    """
    A class to create and apply shift operations in a quantum walk.

    The ShiftOperator class supports various types of shift operations that can be applied
    to a specified number of position qubits. This allows for the simulation of quantum walks
    in different dimensions.

    Parameters
    ----------
    num_position_qubits: int
        The number of qubits used for the positions in the quantum walk.

    num_coin_qubits: int
        The number of qubits used for the coin in the quantum walk.
    """

    def __init__(self,
                 num_position_qubits: int,
                 num_coin_qubits: int):
        self.num_position_qubits = num_position_qubits
        self.num_coin_qubits = num_coin_qubits

    def _apply_shift_left(self,
                          qc: qiskit.QuantumCircuit):
        """
        Apply the shift-left operation based on the coin qubits.
        """
        for pos in range(self.num_position_qubits):
            qc.cx(self.num_coin_qubits - 1, pos)

    def _apply_shift_right(self,
                           qc: qiskit.QuantumCircuit):
        """
        Apply the shift-right operation based on the coin qubits.
        """
        for pos in range(self.num_position_qubits):
            qc.cx(self.num_coin_qubits - 2, pos)

    def create_shift_circuit(self) -> qiskit.QuantumCircuit:
        """
        Create a quantum circuit for the shift operation in the quantum walk.

        This method constructs a quantum circuit with the given number of position and coin qubits,
        and applies the specified shift operations to them. The resulting quantum circuit can then
        be used as part of a larger quantum walk or other quantum algorithm.

        Returns
        -------
        qiskit.QuantumCircuit
            A quantum circuit implementing the shift operation.
        """
        qc = qiskit.QuantumCircuit(self.num_position_qubits + self.num_coin_qubits)

        self._apply_shift_left(qc)
        self._apply_shift_right(qc)

        return qc


    def visualize_shift_circuit(self):
        """
        Visualize the quantum circuit for the shift operation.

        This method creates the quantum circuit for the shift operation and displays
        a visualization of the circuit.
        """

        # Create the shift circuit using the existing method
        qc = self.create_shift_circuit()

        # Visualize the circuit
        qc.draw('mpl')
        plt.show()


def main():
    # Instantiate a ShiftOperator for a system with 2 position qubits and 2 coin qubits
    shift_op = ShiftOperator(num_position_qubits=2, num_coin_qubits=2)

    # Create the shift circuit
    shift_circuit = shift_op.create_shift_circuit()

    # Combine with a main quantum circuit
    main_qc = qiskit.QuantumCircuit(4)  # Assuming 2 position qubits and 2 coin qubits

    # Attach the shift circuit to the main circuit
    main_qc.compose(shift_circuit, qubits=[0, 1, 2, 3], inplace=True)

    # Visualize the combined circuit
    print("Visualizing Combined Quantum Circuit:")
    main_qc.draw('mpl')
    plt.show()


if __name__ == "__main__":
    main()
