from dataclasses import dataclass, fields


@dataclass(frozen=True)
class SamplingModesEnum:
    GREEDY: str = "greedy"
    MULTINOMIAL: str = "multinomial"
    BEAMSEARCH: str = "beamsearch"

    def is_supported_sampling_mode(self, sample_mode):
        return sample_mode in [getattr(self, field.name) for field in fields(self)]
