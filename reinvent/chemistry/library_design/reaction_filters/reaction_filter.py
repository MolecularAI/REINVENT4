from reinvent.chemistry.library_design.reaction_filters.defined_selective_filter import (
    DefinedSelectiveFilter,
)
from reinvent.chemistry.library_design.reaction_filters.reaction_filter_enum import (
    ReactionFiltersEnum,
)
from reinvent.chemistry.library_design.reaction_filters.base_reaction_filter import (
    BaseReactionFilter,
)
from reinvent.chemistry.library_design.reaction_filters.non_selective_filter import (
    NonSelectiveFilter,
)
from reinvent.chemistry.library_design.reaction_filters.selective_filter import SelectiveFilter
from reinvent.chemistry.library_design.reaction_filters.reaction_filter_configruation import (
    ReactionFilterConfiguration,
)


class ReactionFilter:
    def __new__(cls, configuration: ReactionFilterConfiguration) -> BaseReactionFilter:
        enum = ReactionFiltersEnum()
        if configuration.type == enum.NON_SELECTIVE:
            return NonSelectiveFilter(configuration)
        elif configuration.type == enum.SELECTIVE:
            return SelectiveFilter(configuration)
        elif configuration.type == enum.DEFINED_SELECTIVE:
            return DefinedSelectiveFilter(configuration)
        else:
            raise TypeError(f"Requested filter type: '{configuration.type}' is not implemented.")
