from __future__ import annotations

from .scaffold_filter import ScaffoldDiversityFilter


class IdenticalTopologicalScaffold(ScaffoldDiversityFilter):
    """Penalizes compounds based on exact Topological Scaffolds previously generated."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, topological=True, similar=False, **kwargs)
