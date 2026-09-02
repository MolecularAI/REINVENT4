from __future__ import annotations

from .scaffold_filter import ScaffoldDiversityFilter


class ScaffoldSimilarity(ScaffoldDiversityFilter):
    """Penalizes compounds based on atom pair Tanimoto similarity to previously generated Murcko Scaffolds."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, topological=False, similar=True, **kwargs)
