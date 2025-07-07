import json
from abc import abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Literal, Optional, Union

import numpy as np
import pydantic
from typing_extensions import override

from reinvent_plugins.components.cazp.tree_edit_distance import TED


@dataclass
class Endpoint:
    """Base class for CAZP endpoints.

    Each endpoint should implement get_scores method,
    that takes a list of SMILES strings and AiZynthFinder output,
    and returns a numpy array of scores.
    """

    # This is the key in scoring component parameters that chooses the endpoint.
    # Value for this key is the name of the endpoint in AiZynthFinder output.
    score_to_extract: str = None

    @abstractmethod
    def get_scores(self, smilies: list[str], out: dict) -> np.ndarray:
        raise NotImplementedError


@dataclass
class SimpleTreeScoreEndpoint(Endpoint):
    """Base class for CAZP endpoints based on individual tree scores.

    Each endpoint should implement tree_score method,
    that takes a tree dict and returns a tree score.
    """

    @abstractmethod
    def tree_score(self, tree: dict) -> float:
        """Returns a score for a single synthesis tree."""
        raise NotImplementedError

    def best_score(self, scores: list[float]) -> float:
        """Choice of the best tree.

        Each molecule can have multiple synthesis trees.
        Some endpoints might require the tree with the highest score,
        some might require the tree with the lowest score.

        Built-in `max` behaves inconsistently depending on the first list element:
        max([np.nan, 2, 3]) is np.nan, but max([1, np.nan, 3]) is 3.
        Use np.nanmax to avoid this issue.

        Default: tree with max score.
        """
        return np.nanmax(scores)

    def keep(self, tree: dict) -> bool:
        """Filter to choose which trees to keep per endpoint and molecule.

        For some endpoints, like number of reactions,
        the value for trees that were not solved might be misleading.
        """
        return True

    def default_score(self) -> float:
        """Default score if there are no trees for a molecule."""
        return np.nan

    def get_scores(self, smilies: list[str], out: dict) -> np.ndarray:
        """Extract scores from AiZynthFinder output.

        This method takes all trees for each molecule,
        removes trees that do not pass the filter,
        calculates scores for remaining trees using tree_score method,
        and chooses the best score for each molecule.
        """

        # Change from a flat list to a dict with smiles as keys.
        moltrees = {mol["target"]: mol["trees"] for mol in out["data"]}

        ordered_scores = []
        for smi in smilies:
            trees = moltrees.get(smi, [])
            tree_scores = [self.tree_score(t) for t in trees if self.keep(t)]

            # Pick score of the best tree, as defined by endpoint aggregator.
            # If there are no trees, return NaN.
            score = self.default_score() if len(tree_scores) == 0 else self.best_score(tree_scores)

            ordered_scores.append(score)

        return np.array(ordered_scores)


@dataclass
class CazpEndpoint(SimpleTreeScoreEndpoint):
    """Endpoint for Reinvent CAZP score.

    This score is a product of three AiZynthFinder scores:
    - stock availability
    - reaction class membership
    - number of reactions

    In the future, this score could be moved to AiZynthFinder.
    """

    score_to_extract: Literal["cazp"] = "cazp"
    reaction_step_coefficient: float = 0.90

    def tree_score(self, tree: dict) -> float:

        scores = tree["scores"]
        bb_score = scores["stock availability"]
        reacticlass_score = scores["reaction class membership"]
        numsteps = scores["number of reactions"]

        # Score by number of steps.
        # Hard cut-off in aizynth on num_steps (depth of search tree).
        # We can add "soft" signal to reward fewer steps.
        # Resulting score is "ease of synthesis", not binary synthesizeability.
        # We could even increase num_steps for aizynth config,
        # and add two penalties: smaller below threshold, higher above.
        numsteps_score = self.reaction_step_coefficient**numsteps

        score = bb_score * reacticlass_score * numsteps_score
        return score


@dataclass
class RouteDistanceEndpoint(SimpleTreeScoreEndpoint):
    """Endpoint for Route Distance score.

    In the future, this score could be moved to AiZynthFinder,
    with a caveat that users could request multiple reference routes.
    """

    score_to_extract: Literal["route_distance"] = "route_distance"
    reference_route_file: Optional[str] = None

    @cached_property
    def reference_route(self) -> dict:
        if self.reference_route_file is None:
            raise ValueError("Missing reference_route_file in RouteDistanceEndpoint.")

        with open(self.reference_route_file) as f:
            reference_route = json.load(f)

        return reference_route

    @override
    def best_score(self, scores: list[float]) -> float:
        return np.nanmin(scores)

    @override
    def keep(self, tree: dict) -> bool:
        return tree.get("metadata", {}).get("is_solved", False)

    @override
    def tree_score(self, tree: dict) -> float:
        return TED(tree, self.reference_route)


@dataclass
class NumberOfReactionsEndpoint(SimpleTreeScoreEndpoint):
    """Endpoint for scores that AiZynthFinder returns directly."""

    score_to_extract: Literal["number of reactions"] = "number of reactions"

    @override
    def tree_score(self, tree: dict) -> float:
        return tree["scores"][self.score_to_extract]

    @override
    def best_score(self, scores: list[float]) -> float:
        return np.nanmin(scores)

    @override
    def keep(self, tree: dict) -> bool:
        return tree.get("metadata", {}).get("is_solved", False)


AnyEndpoint = Union[
    CazpEndpoint,
    RouteDistanceEndpoint,
    NumberOfReactionsEndpoint,
]


def endpoint_from_dict(data: dict) -> AnyEndpoint:
    """Return the right Endpoint based on score_to_extract"""

    # We can do if-elif-else based on score_to_extract to choose class,
    # but Pydantic and apischema can choose class automatically
    # (see also Discriminated Unions).

    # Minor inconvenience is that Pydantic does not have a stand-alone function
    # to parse arbitrary supported types, including Union (AnyEndpoint).
    # (Side note: apischema has 'apischema.deserialize(cls, data)').
    # Parsing in Pydantic uses model_validate method of a BaseModel subclass.
    # Let's create such a subclass.
    class Wrapper(pydantic.BaseModel):
        obj: AnyEndpoint

    # Now we can call model_validate to parse the Union.
    wrapped_endpoint = Wrapper.model_validate({"obj": data})
    return wrapped_endpoint.obj
