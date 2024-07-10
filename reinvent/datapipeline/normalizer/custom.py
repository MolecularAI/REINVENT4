"""Custom RDKit normalization transformtions

Transformation names are optional.
"""

__all__ = ["four_valent_nitrogen"]


# FIXME: needs review
four_valent_nitrogen = """\
[#7-0D3:1]=[*:2]>>[+:1]=[*:2]
[#7-0D2H:1]=[*:2]>>[+:1]=[*:2]
[#7-0D3H:1]>>[+:1]
[#7-0D4;z0:1]>>[+:1]
"""
