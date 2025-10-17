from abc import ABC, abstractmethod
from ..bucket_counter import BucketCounter


class ScaffoldPenalty(ABC):
    def __init__(self, scaffold_memory: BucketCounter):
        """Initialize the Penalty class with a specified soft function.

        :param soft_function: The type of soft function to use for penalization.
        """
        self.scaffold_memory = scaffold_memory

    @abstractmethod
    def calculate_penalty(self, scaffold: str) -> float:
        """Calculate penalty according to the concrete scaffold penalty function.
        This penalty is multiplied with the original reward to reduce it.

        :param scaffold: string representation of the scaffold
        :return: float value of penalty
        """
