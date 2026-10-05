from abc import ABC, abstractmethod


class BasePenalty(ABC):

    def __init__(self, bucket_size: int):
        """Initialize the Penalty class with a specified soft function.

        :param soft_function: The type of soft function to use for penalization.
        :param bucket_size: maximum allowed bucket size
        """
        self.bucket_size = bucket_size

    @abstractmethod
    def calculate_penalty(self, bucket_utilization: int) -> float:
        """Calculate penalty according to a concrete penalty function.
        This penalty is multiplied with the original reward to reduce it.

        :param bucket_utilization: current number of instances in the specified bucket
        :return: float value of penalty between 0 and 1, gets multiplied with the score
        """

    def __call__(self, bucket_utilization: int) -> float:
        return self.calculate_penalty(bucket_utilization)
