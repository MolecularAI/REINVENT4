"""Optional PUMAS transforms and aggregations"""

__all__ = ["aggregation_catalogue", "desirability_catalogue", "check_pumas_available"]

try:
    from pumas.aggregation import aggregation_catalogue
    from pumas.desirability import desirability_catalogue

    have_pumas = True
except ModuleNotFoundError:
    aggregation_catalogue = None
    desirability_catalogue = None
    have_pumas = False


def check_pumas_available() -> None:
    if not have_pumas:
        raise RuntimeError(
            "use_pumas is set but the optional 'pumas' package is not "
            "installed: pip install reinvent[pumas]"
        )
