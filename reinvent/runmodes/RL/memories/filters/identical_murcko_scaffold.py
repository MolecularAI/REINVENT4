from __future__ import annotations

from .scaffold_filter import ScaffoldDiversityFilter


class IdenticalMurckoScaffold(ScaffoldDiversityFilter):
    """Penalizes compounds based on exact Murcko Scaffolds previously generated."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, topological=False, similar=False, **kwargs)
