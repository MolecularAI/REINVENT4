from dataclasses import dataclass


@dataclass(frozen=True)
class SamplingModesEnum:
    GREEDY = "greedy"
    MULTINOMIAL = "multinomial"
    BEAMSEARCH = "beamsearch"