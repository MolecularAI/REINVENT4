import json
import logging
from abc import abstractmethod
from dataclasses import field
from functools import cached_property
from typing import Literal, Optional, Union
from typing_extensions import Annotated, override

import numpy as np
import pydantic
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from reinvent_plugins.components.synthsense.tree_edit_distance import TED, route_signature

logger = logging.getLogger("reinvent")


@dataclass
class Endpoint:
    """Base class for synthsense endpoints.

    Each endpoint should implement get_scores method,
    that takes a list of SMILES strings and AiZynthFinder output,
    and returns a numpy array of scores.
    """

    # This is the key in scoring component parameters that chooses the endpoint.
    # Value for this key is the name of the endpoint in AiZynthFinder output.
    score_to_extract: str = None

    # Whether this endpoint requires scoring cache to be disabled
    # Set to True for endpoints with batch-dependent scoring logic
    no_cache: bool = False

    @abstractmethod
    def get_scores(self, smilies: list[str], out: dict) -> np.ndarray:
        raise NotImplementedError


@dataclass
class SimpleTreeScoreEndpoint(Endpoint):
    """Base class for synthsense endpoints based on individual tree scores.

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
        """Only keep solved trees by default."""
        return tree.get("metadata", {}).get("is_solved", False)

    def default_score(self) -> float:
        """Default score if there are no trees for a molecule."""
        return 0.0

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
            # If there are no trees, return 0.
            score = self.default_score() if len(tree_scores) == 0 else self.best_score(tree_scores)

            ordered_scores.append(score)

        return np.array(ordered_scores)

@dataclass
class CazpEndpoint(SimpleTreeScoreEndpoint):
    """Endpoint for Reinvent CAZP score. (sfscore v1)

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

    @override
    def keep(self, tree: dict) -> bool:
        """Filter to choose which trees to keep per endpoint and molecule.

        For some endpoints, like number of reactions,
        the value for trees that were not solved might be misleading.
        """
        return True

    @override
    def default_score(self) -> float:
        """Default score if there are no trees for a molecule."""
        return np.nan

@dataclass
class RouteDistanceEndpoint(SimpleTreeScoreEndpoint):
    """Endpoint for Route Distance score (rrscore v1).

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
    def default_score(self) -> float:
        """Default score if there are no trees for a molecule."""
        return np.nan

    @override
    def tree_score(self, tree: dict) -> float:
        return TED(tree, self.reference_route)

@dataclass
class SFScore(SimpleTreeScoreEndpoint):
    """Endpoint for Reinvent synthsense score.

    This score is a product of three AiZynthFinder scores:
    - stock availability
    - reaction class membership
    - number of reactions

    In the future, this score could be moved to AiZynthFinder.
    """

    score_to_extract: Literal["sfscore"] = "sfscore"
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
    
    @override
    def keep(self, tree: dict) -> bool:
        """Filter to choose which trees to keep per endpoint and molecule.

        For some endpoints, like number of reactions,
        the value for trees that were not solved might be misleading.
        """
        return True
    
@dataclass
class RouteSimilarityEndpoint(SimpleTreeScoreEndpoint):
    """
    This endpoint measures how similar the synthesis routes are between molecules.
    It uses Tree Edit Distance (TED) to quantify the structural differences between synthesis trees.

    The scoring process works as follows:

    1. For each molecule's tree:
       - Calculate the minimum TED distance to each other molecule's trees
       - Take the average of these minimum distances for that tree

    2. For each molecule:
       - Take the minimum of the averages across all its trees
       - This selects the "best" tree that has the lowest average TED with other molecules

    The similarity score is calculated using: 1 / (1 + TED)
    This formula produces values in range (0, 1] where:
    - Values close to 1 mean trees are very similar (low TED)
    - Values close to 0 mean trees are very different (high TED)

    This calculation is batch-independent, allowing scores to be compared across different batches.

    Comparisons involving trees with unrecognized reactions ("0.0" in signature) are
    assigned a penalty.
    """

    score_to_extract: Literal["route_similarity"] = "route_similarity"
    no_cache: bool = True

    def __post_init__(self):
        # Cache to store trees by molecule
        self._all_moltrees = {}
        self._current_out = None

    @override
    def get_scores(self, smilies: list[str], out: dict) -> np.ndarray:
        """Extract and prepare molecule trees for scoring."""
        # Cache all molecule trees for this batch
        self._all_moltrees = {mol["target"]: mol["trees"] for mol in out["data"]}
        self._current_out = out

        return super().get_scores(smilies, out)

    @override
    def best_score(self, scores: list[float]) -> float:
        """Return the maximum tree similarity score for this molecule."""
        return np.nanmax(scores)

    @override
    def tree_score(self, tree: dict) -> float:
        """Calculate a similarity score for a single synthesis tree compared to all other molecules' trees in the batch.

        This method evaluates how similar a given synthesis tree is to all other molecules' trees in the batch.
        For each other molecule, it finds the minimum Tree Edit Distance (TED) between the current tree
        and any of other molecule's trees. It then averages these minimum distances across all molecules.

        The similarity is calculated as: 1/(1+TED)

        The final scores range from 0 to 1:
        - Values close to 1 mean the tree is very similar to other trees (low TED)
        - Values close to 0 mean the tree is very different from other trees (high TED)

        Comparisons involving trees with unrecognized reactions ("0.0" in signature) are
        assigned a penalty.
        """
        # Get molecule SMILES for this tree
        current_smiles = None
        for smi, trees in self._all_moltrees.items():
            if tree in trees:
                current_smiles = smi
                break

        if current_smiles is None:
            logger.warning("Could not find current tree in molecule trees")
            return np.nan

        # Check if current tree has "0.0" in its signature
        current_signature = route_signature(tree)
        current_has_unrecognized = "0.0" in current_signature
        # logger.debug(f"Processing tree for {current_smiles} with signature {current_signature}")
        # logger.debug(f"Current tree has unrecognized reactions: {current_has_unrecognized}")

        # For each other molecule, calculate the minimum TED to this tree
        min_teds = []
        valid_comparisons = 0

        for smi, trees in self._all_moltrees.items():
            if smi == current_smiles:
                continue  # Skip comparison with self/same molecule

            # Filter for solved trees only
            solved_trees = [t for t in trees if self.keep(t)]

            if not solved_trees:
                continue  # Skip if no solved trees for this molecule

            ted_values = []

            for other_tree in solved_trees:
                # Check if the other tree has "0.0" in its signature
                other_signature = route_signature(other_tree)
                other_has_unrecognized = "0.0" in other_signature

                # Handle trees with unrecognized reactions
                if current_has_unrecognized or other_has_unrecognized:
                    ted = TED(tree, other_tree)
                    penalty_multiplier = 3.0  # Apply 3x penalty for unrecognized reactions
                    penalized_ted = ted * penalty_multiplier
                    ted_values.append(penalized_ted)
                else:
                    ted = TED(tree, other_tree)
                    ted_values.append(ted)

            min_ted = min(ted_values)
            min_teds.append(min_ted)
            valid_comparisons += 1

        # If no other molecules with solved trees, return 0
        if not min_teds:
            logger.warning(
                f"No other molecules with solved trees found for {current_smiles}, returning 0.0"
            )
            return 0.0

        # Calculate average of all TED values
        avg_min_ted = sum(min_teds) / len(min_teds)

        # Apply an additional small penalty if this tree has unrecognized reactions
        # This ensures trees with unrecognized reactions are slightly demotivated
        if current_has_unrecognized:
            unrecognized_penalty = 1.5  # 50% additional penalty
            avg_min_ted = avg_min_ted * unrecognized_penalty

        # Calculate similarity using 1/(1+TED) formula
        similarity_score = 1.0 / (1.0 + avg_min_ted)

        return similarity_score


@dataclass
class RoutePopularityEndpoint(SimpleTreeScoreEndpoint):
    """Endpoint for Route Popularity score.

    This endpoint calculates how popular/common each molecule's synthesis routes are WITHIN the batch.
    It measures the fraction of molecules in the batch that share a specific route signature.
    The scoring process works as follows:

    1. For each unique valid route signature found across all solved trees in the batch:
       - Count how many distinct molecules possess a tree with this signature.
       - Normalize this count by the total number of molecules in the batch.
         This yields the "molecule popularity" for that route signature (range [0, 1]).

    2. For each molecule:
       - Get the (potentially penalized) molecule popularity score for each of its solved trees' signatures.
       - The final score for the molecule is the *maximum* of these scores.

    3. Popularity penalty mechanism:
       - After scoring a batch, if any route signature's molecule popularity exceeded the
         specified threshold during that batch, that route signature is permanently added to the penalized set.
       - Signatures in the penalized set will have their molecule popularity scores multiplied by the
         penalty_multiplier when scoring *subsequent* batches (see `tree_score`).
       - This ensures permanent exploration away from overused routes by applying the penalty
         from the step *after* high popularity is detected.

    This endpoint returns values in range [0,1] where:
    - 1 means the molecule's most popular route is shared by all molecules in the batch (before penalty).
    - 0 means the molecule's routes are unique or invalid/unrecognized.
    """

    score_to_extract: Literal["route_popularity"] = "route_popularity"
    no_cache: bool = True

    popularity_threshold: float = 1.0  # Threshold above which routes are penalized
    penalty_multiplier: float = 1.0  # Multiplier applied to penalize overly popular routes

    def __post_init__(self):
        # Cache to store trees and route frequencies
        self._all_moltrees = {}
        self._route_signatures = {}  # Maps molecule SMILES to its list of parsed route signatures
        self._molecule_frequencies = {}  # Maps route signature to the count of molecules having it
        self._normalized_molecule_frequencies = (
            {}
        )  # Maps route signature to its normalized molecule frequency
        self._batch_size = 0
        self._penalized_routes = set()  # Permanently track routes that exceed threshold

        # Special markers
        self.INVALID_ROUTE_MARKER = frozenset(
            ["INVALID_ROUTE_0.0"]
        )  # Marker for routes containing "0.0"
        logger.info(
            f"Initialized RoutePopularityEndpoint with threshold={self.popularity_threshold}, penalty_multiplier={self.penalty_multiplier}"
        )

    def _parse_signature(self, signature: str) -> frozenset:
        """Parse route signature string into a set of reaction classes to ignore order.

        Returns a frozenset of reaction classes, or a special marker frozenset for invalid signatures.
        """
        if "0.0" in signature:
            # logger.debug(f"Invalid signature contains '0.0': {signature}")
            return self.INVALID_ROUTE_MARKER

        # Clean and split the signature into reaction classes
        reaction_classes = signature.split(",")
        return frozenset(reaction_classes)

    def _calculate_batch_frequencies(self):
        """Calculate raw and normalized route signature frequencies for the current batch."""
        # Extract and parse all route signatures per molecule
        self._route_signatures = {}
        all_unique_parsed_signatures = set()  # Store all unique valid signatures in the batch

        for smi, trees in self._all_moltrees.items():
            solved_trees = [t for t in trees if self.keep(t)]

            molecule_signatures = []
            for tree in solved_trees:
                raw_signature = route_signature(tree)
                parsed_signature = self._parse_signature(raw_signature)

                # Only consider valid signatures (exclude marker frozensets)
                if parsed_signature not in (self.INVALID_ROUTE_MARKER):
                    molecule_signatures.append(parsed_signature)
                    all_unique_parsed_signatures.add(parsed_signature)

            self._route_signatures[smi] = molecule_signatures

        # Calculate frequency based on unique molecules having the signature
        self._molecule_frequencies = {}
        for signature in all_unique_parsed_signatures:
            count = 0
            for mol_signatures in self._route_signatures.values():
                if signature in mol_signatures:
                    count += 1
            self._molecule_frequencies[signature] = count

        # Calculate normalized molecule frequencies
        self._normalized_molecule_frequencies = {}
        if self._batch_size > 0:
            for signature, count in self._molecule_frequencies.items():
                normalized_mol_freq = count / self._batch_size
                self._normalized_molecule_frequencies[signature] = normalized_mol_freq
        else:
            # Handle empty batch case
            for signature in all_unique_parsed_signatures:
                self._normalized_molecule_frequencies[signature] = 0.0

    def _update_penalized_routes(self):
        """Update the set of penalized routes based on current batch frequencies."""
        newly_penalized = set()
        if self._batch_size > 0:
            for signature, normalized_mol_freq in self._normalized_molecule_frequencies.items():
                # Check if route popularity exceeds threshold for potential penalization next batch
                # Only add if it's not already penalized to avoid repeated logging.
                # Exclude marker sets from penalization checks
                if (
                    signature not in (self.INVALID_ROUTE_MARKER)
                    and normalized_mol_freq >= self.popularity_threshold
                    and signature not in self._penalized_routes
                ):
                    newly_penalized.add(signature)

                    # Add debug logging to see exactly what's being added to the penalized set
                    logger.info(
                        f"Route identified for penalization next batch: {signature} with molecule popularity {normalized_mol_freq:.4f} >= threshold {self.popularity_threshold}"
                    )

        # Update the main penalized set for the next batch
        if newly_penalized:
            self._penalized_routes.update(newly_penalized)
            logger.info(
                f"{len(newly_penalized)} routes added to penalized set for next batch. Total penalized: {len(self._penalized_routes)}."
            )

    @override
    def get_scores(self, smilies: list[str], out: dict) -> np.ndarray:
        """Calculate route popularity scores, applying penalties from previous batches
        and updating the penalized set for the next batch."""
        # 1. Cache trees and update batch size for this run
        self._all_moltrees = {mol["target"]: mol["trees"] for mol in out["data"]}
        self._batch_size = len(smilies)
        # logger.debug(f"RoutePopularityEndpoint: Submitted {self._batch_size} SMILES, received results for {len(self._all_moltrees)} molecules from AiZynthFinder.")

        # 2. Calculate raw and normalized frequencies for the current batch
        self._calculate_batch_frequencies()

        # 3. Calculate scores using parent method.
        #    tree_score uses the _normalized_molecule_frequencies calculated above,
        #    and applies penalties based on _penalized_routes from previous batches.
        scores = super().get_scores(smilies, out)

        # 4. Update the penalized routes set for the *next* batch based on current frequencies.
        self._update_penalized_routes()

        return scores

    @override
    def tree_score(self, tree: dict) -> float:
        """Calculate popularity score for this tree based on molecule frequency.

        The score is the frequency of this tree's route signature among molecules
        normalized by the total number of molecules in the batch.

        If the signature is invalid ("0.0"), return 0 as the worst score.

        If the route has ever exceeded the popularity_threshold (based on molecule frequency),
        it will always be penalized, even if its current popularity is below the threshold.
        """
        raw_signature = route_signature(tree)
        parsed_signature = self._parse_signature(raw_signature)

        # Return 0 if the signature is invalid
        if parsed_signature in (self.INVALID_ROUTE_MARKER):
            # logger.debug(f"Tree has invalid signature '{raw_signature}', assigning score 0.0")
            return 0.0

        # Get the pre-calculated normalized molecule frequency
        normalized_mol_freq = self._normalized_molecule_frequencies.get(parsed_signature, 0.0)

        # Apply penalty if this route is in the penalized set
        if parsed_signature in self._penalized_routes:
            normalized_mol_freq *= self.penalty_multiplier
            # logger.debug(f"Applied penalty to signature {parsed_signature}, score: {normalized_mol_freq}")

        return normalized_mol_freq


@dataclass
class RRScore(SimpleTreeScoreEndpoint):
    """Endpoint for Reference Route score.

    In the future, this score could be moved to AiZynthFinder,
    with a caveat that users could request multiple reference routes.
    """

    score_to_extract: Literal["rrscore"] = "rrscore"
    reference_route_file: Optional[str] = None

    @cached_property
    def reference_route(self) -> dict:
        if self.reference_route_file is None:
            raise ValueError("Missing reference_route_file in RRScore.")

        with open(self.reference_route_file) as f:
            reference_route = json.load(f)

        return reference_route

    @override
    def default_score(self) -> float:
        """Default score if there are no trees for a molecule (super high ted)."""
        return 1000.0

    @override
    def best_score(self, scores: list[float]) -> float:
        return np.nanmin(scores)

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
    def default_score(self) -> float:
        """Default score if there are no trees for a molecule."""
        return 1000.0

    @override
    def best_score(self, scores: list[float]) -> float:
        return np.nanmin(scores)


@dataclass
class FillaPlate(SimpleTreeScoreEndpoint):
    """Scores molecules by filling route 'buckets' (or plates).

    Routes are scored based on their cumulative unique molecule count relative
    to `bucket_threshold` (higher count = higher score, capped at 1.0).
    This incentivizes using routes until they reach their defined capacity.

    It tracks cumulative unique molecules per route signature across batches.
    If `penalization_enabled` is True, hitting the `bucket_threshold`
    triggers permanent penalization for the specific route (if eligible based
    on `min_steps_for_penalization`) and all molecules associated with it.

    Scoring Process:
      1. `get_scores` incrementally updates cumulative molecule counts per route
         signature, skipping molecules that are already in `_penalized_molecules`.
      2. If adding a molecule causes a route count to hit `bucket_threshold`,
         `_check_threshold_and_penalize` is called.
      3. `_check_threshold_and_penalize` logs the event and potentially applies
         permanent penalties if `penalization_enabled` is True.
      4. `super().get_scores` is called, which uses `tree_score`.
      5. `tree_score` calculates a score based on the ratio of cumulative unique
         molecules to `bucket_threshold` for the route. The score is 0.0 if the
         route is invalid, in stock, or if `penalization_enabled` is True and
         the route is in `_penalized_routes`.
      6. `best_score` (default `max`) selects the highest `tree_score` for the molecule.
      7. `get_scores` applies final molecule penalties: if `penalization_enabled`
         is True, the score for any molecule in `_penalized_molecules` is set
         to 0.0.

    Attributes:
        score_to_extract: Endpoint identifier literal.
        bucket_threshold: Cumulative molecule count threshold to trigger
            penalization. Must be > 0.
        min_steps_for_penalization: Minimum reaction steps (signature length)
            for a route to be eligible for penalization.
        penalization_enabled: If False, disables all route and molecule
            penalization based on cumulative counts (threshold hitting is still logged).
        _all_moltrees: Cache for molecule trees from the current AiZynthFinder output.
        _route_signatures: Cache mapping SMILES to parsed route signatures for
            the current batch.
        _batch_size: Size of the current molecule batch.
        _cumulative_molecule_counts_per_route: Tracks unique molecules seen per
            route signature across all batches. Dict[frozenset, set[str]].
        _penalized_routes: Set of route signatures permanently penalized.
        _penalized_molecules: Set of SMILES strings permanently penalized.
        _processed_threshold_routes_this_call: Temporarily tracks routes
            whose thresholds were met during the current `get_scores` call to
            prevent redundant processing.
    """

    score_to_extract: Literal["fill_a_plate"] = "fill_a_plate"
    no_cache: bool = True

    bucket_threshold: int = 500
    min_steps_for_penalization: int = 1
    penalization_enabled: bool = True

    # State
    _all_moltrees: dict = field(default_factory=dict, init=False)
    _route_signatures: dict = field(default_factory=dict, init=False)
    _batch_size: int = field(default=0, init=False)

    # Cumulative state for penalization
    _cumulative_molecule_counts_per_route: dict[frozenset, set[str]] = field(default_factory=dict)
    _penalized_routes: set[frozenset] = field(default_factory=set)
    _penalized_molecules: set[str] = field(default_factory=set)

    _processed_threshold_routes_this_call: set[frozenset] = field(default_factory=set)

    # Special markers for non-standard routes
    INVALID_ROUTE_MARKER: frozenset = frozenset(["INVALID_ROUTE_0.0"])
    # IN_STOCK_ROUTE_MARKER: frozenset = frozenset(["IN_STOCK_ROUTE"]) # when route is 0 steps

    def __post_init__(self):
        """Logs initialization parameters."""
        logger.info(
            f"Initialized {self.__class__.__name__} with: "
            f"bucket_threshold={self.bucket_threshold}, "
            f"min_steps_for_penalization={self.min_steps_for_penalization}, "
            f"penalization_enabled={self.penalization_enabled}"
        )
        if self.bucket_threshold <= 0:
            raise ValueError("bucket_threshold must be greater than 0.")

    def _parse_signature(self, signature: str) -> frozenset:
        """Parses a route signature string into a frozenset of reaction classes.

        Handles special cases for empty signatures (molecule in stock) and signatures
        containing invalid reaction markers ('0.0').

        Args:
            signature: The raw route signature string (e.g., "1.2.3,4.5.6").

        Returns:
            A frozenset representing the reaction classes, or a special marker
            frozenset (INVALID_ROUTE_MARKER or IN_STOCK_ROUTE_MARKER).
        """
        # if not signature:
        #    return self.IN_STOCK_ROUTE_MARKER
        if "0.0" in signature:
            return self.INVALID_ROUTE_MARKER
        # Use frozenset for hashability (needed for dict keys/set elements).
        return frozenset(signature.split(","))

    def _extract_batch_signatures(self):
        """Extracts unique valid route signatures per molecule for the current batch.

        Populates `_route_signatures` based on the trees stored in
        `_all_moltrees`.
        """
        self._route_signatures.clear()

        # Extract and parse unique valid signatures per molecule
        for smi, trees in self._all_moltrees.items():
            solved_trees = [t for t in trees if self.keep(t)]
            molecule_signatures = set()  # Use set for uniqueness per molecule
            for tree in solved_trees:
                parsed_signature = self._parse_signature(route_signature(tree))
                # Exclude markers
                if parsed_signature not in (
                    self.INVALID_ROUTE_MARKER,
                    #                            self.IN_STOCK_ROUTE_MARKER
                ):
                    molecule_signatures.add(parsed_signature)
            self._route_signatures[smi] = list(molecule_signatures)

    def _check_threshold_and_penalize(self, signature: frozenset):
        """Checks if threshold is met, logs, and applies penalties if enabled.

        Called after a *new*, *non-penalized* molecule is added to a
        signature's cumulative count. Determines if the count for the
        signature now equals `bucket_threshold`. If so, logs the event and, if
        `penalization_enabled` is True and the route hasn't already been processed
        this call, applies permanent penalties to the eligible route and
        associated molecules.

        Args:
            signature: The route signature whose count was just updated.
        """
        if signature in (
            self.INVALID_ROUTE_MARKER,
            #                 self.IN_STOCK_ROUTE_MARKER
        ):
            return  # Ignore special markers

        is_processed = signature in self._processed_threshold_routes_this_call
        is_penalized = self.penalization_enabled and signature in self._penalized_routes

        if is_processed or is_penalized:
            return  # Already handled or route penalized (when penalization is on)

        # Get molecules associated *only* with this specific signature
        associated_mols = self._cumulative_molecule_counts_per_route.get(signature, set())
        individual_count = len(associated_mols)

        if individual_count == self.bucket_threshold:
            log_prefix = f"Route Threshold [{signature}]"
            logger.info(
                f"{log_prefix}: Reached! " f"Count: {individual_count}/{self.bucket_threshold}"
            )
            # Log details regardless of penalization status
            self._log_bucket_filled_details(signature, associated_mols)
            # Mark this specific route to prevent re-processing this call
            self._processed_threshold_routes_this_call.add(signature)

            # Apply Penalties only if enabled
            if self.penalization_enabled:
                route_penalized_now = False
                is_eligible = len(signature) >= self.min_steps_for_penalization
                # Check if not already penalized (covered by is_penalized check above)
                if is_eligible:
                    logger.info(
                        f"{log_prefix}: Penalizing route {signature} " f"(length {len(signature)})"
                    )
                    self._penalized_routes.add(signature)
                    route_penalized_now = True
                else:
                    # Log even if not penalizing the route itself
                    logger.info(
                        f"{log_prefix}: Route {signature} (length {len(signature)}) "
                        f"met threshold but is too short to penalize "
                        f"(min_steps={self.min_steps_for_penalization}). "
                        f"Molecules will still be penalized (if penalization is enabled)."
                    )

                mols_to_penalize = associated_mols - self._penalized_molecules
                penalized_mols_now = len(mols_to_penalize)
                if mols_to_penalize:
                    logger.info(f"{log_prefix}: Penalizing {penalized_mols_now} new molecules.")
                    self._penalized_molecules.update(mols_to_penalize)
                    # Remove newly penalized molecules from all cumulative counts
                    # so they can't fill new buckets later on.
                    for penalized_smi in mols_to_penalize:
                        for signature_set in self._cumulative_molecule_counts_per_route.values():
                            signature_set.discard(penalized_smi)

                logger.info(
                    f"{log_prefix}: Applied penalties - Route Penalized: {route_penalized_now}, "
                    f"Molecules: {penalized_mols_now}"
                )
            else:
                logger.info(f"{log_prefix}: Penalization disabled, skipping penalty application.")

    @override
    def get_scores(self, smilies: list[str], out: dict) -> np.ndarray:
        """Calculates fill_a_plate scores for a batch of SMILES.

        Updates cumulative counts (ignoring already penalized molecules),
        checks thresholds incrementally (logging bucket fills and potentially
        triggering penalties), calculates batch scores based on cumulative fullness,
        and applies final molecule penalties if enabled.

        Args:
            smilies: List of SMILES strings in the current batch.
            out: The output dictionary from AiZynthFinder containing synthesis trees.

        Returns:
            A numpy array of final scores for the input SMILES list.
        """
        # 1. Setup: Cache trees, update batch size, reset processed set for call
        self._all_moltrees = {mol["target"]: mol["trees"] for mol in out["data"]}
        self._batch_size = len(smilies)
        self._processed_threshold_routes_this_call.clear()

        # 2. Extract Batch Signatures (needed for updating cumulative counts)
        self._extract_batch_signatures()

        # 3. Update Cumulative Counts & Check Thresholds Incrementally
        for smi in smilies:
            # Skip if molecule already penalized
            if self.penalization_enabled and smi in self._penalized_molecules:
                continue

            signatures = self._route_signatures.get(smi, [])
            for signature in signatures:
                # Skip markers and already penalized routes
                if signature in (
                    self.INVALID_ROUTE_MARKER,
                    #                self.IN_STOCK_ROUTE_MARKER
                ) or (self.penalization_enabled and signature in self._penalized_routes):
                    continue

                if signature not in self._cumulative_molecule_counts_per_route:
                    self._cumulative_molecule_counts_per_route[signature] = set()

                # Check if this molecule is NEW for this signature's count
                if smi not in self._cumulative_molecule_counts_per_route[signature]:
                    # Add the new molecule
                    self._cumulative_molecule_counts_per_route[signature].add(smi)
                    # Check threshold immediately after adding.
                    self._check_threshold_and_penalize(signature)

        # 4. Calculate Raw Scores using Parent Method
        #    `tree_score` handles penalized routes based on the
        #    `penalization_enabled` flag
        scores = super().get_scores(smilies, out)

        # 5. Apply Molecule Penalties (if enabled)
        final_scores = scores.copy()
        penalized_in_batch = 0
        if self.penalization_enabled:
            for i, current_smi in enumerate(smilies):
                if current_smi in self._penalized_molecules:
                    if final_scores[i] != 0.0:  # Log only if score was changed
                        penalized_in_batch += 1
                    final_scores[i] = 0.0

        if penalized_in_batch > 0:
            logger.debug(f"Applied final penalty to {penalized_in_batch} molecules in this batch.")

        return final_scores

    def _log_bucket_filled_details(
        self, trigger_signature: frozenset, molecules_in_scope: set[str]
    ):
        """Logs detailed information when a penalization threshold is met.

        Formats the output to clearly show the triggering signature and lists the
        molecules involved. Molecules already penalized at the time of logging
        are marked.

        Args:
            trigger_signature: The signature that caused the threshold check.
            molecules_in_scope: The set of unique molecules associated with the
                `trigger_signature`.
        """
        trigger_sig_str = ",".join(sorted(list(trigger_signature)))
        formatted_route = f"{{{trigger_sig_str}}}"

        # Build detailed molecule list string
        mol_details_lines = []
        for smi in sorted(list(molecules_in_scope)):
            # Check if molecule is already penalized
            penalty_marker = " (penalized)" if smi in self._penalized_molecules else ""
            mol_details_lines.append(f"  - {smi}{penalty_marker}")

        detailed_smiles_log = "\n".join(mol_details_lines)
        logger.info(
            f"BUCKET FILLED for Route [{formatted_route}]. "
            f"Molecules ({len(molecules_in_scope)} total associated):\n{detailed_smiles_log}"
        )

    @override
    def keep(self, tree: dict) -> bool:
        """Filter to choose which trees to keep per endpoint and molecule.

        For some endpoints, like number of reactions,
        the value for trees that were not solved might be misleading.
        """
        return True

    @override
    def tree_score(self, tree: dict) -> float:
        """Calculates the score for a single synthesis tree based on cumulative fullness.

        Returns 0.0 if the route is invalid, in stock, or has been penalized
        (if penalization_enabled is True).

        The score is the ratio of cumulative unique molecules associated with
        the route to the `bucket_threshold`.

        Args:
            tree: The synthesis tree dictionary.

        Returns:
            The calculated score (0.0 to 1.0).
        """
        parsed_signature = self._parse_signature(route_signature(tree))

        # 1. Check for invalid/in stock routes
        if parsed_signature in (
            self.INVALID_ROUTE_MARKER,
            #                        self.IN_STOCK_ROUTE_MARKER
        ):
            return 0.0

        # 2. Check if the specific route is penalized
        if self.penalization_enabled and parsed_signature in self._penalized_routes:
            return 0.0

        # 3. Calculate score based on cumulative fullness
        # Get unique molecules for this specific signature only
        cumulative_count = len(
            self._cumulative_molecule_counts_per_route.get(parsed_signature, set())
        )

        # Calculate fullness ratio (capped at 1.0, threshold hit means full)
        # bucket_threshold is guaranteed > 0 by __post_init__ check
        fullness = min(1.0, cumulative_count / self.bucket_threshold)
        return fullness


AnyEndpoint = Union[
    CazpEndpoint,
    RouteDistanceEndpoint,
    SFScore,
    RRScore,
    NumberOfReactionsEndpoint,
    RouteSimilarityEndpoint,
    RoutePopularityEndpoint,
    FillaPlate,
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
        model_config = ConfigDict(extra='forbid')
        obj: Annotated[AnyEndpoint, Field(discriminator='score_to_extract')]

    # Now we can call model_validate to parse the Union.
    wrapped_endpoint = Wrapper.model_validate({"obj": data})
    return wrapped_endpoint.obj
