

class SliceTypeEnum:
    RECAP = "recap"
    COMPLETE = "complete"
    REACTION = "reaction"

    def __getattr__(self, name):
        if name in self:
            return name
        raise AttributeError

    def __setattr__(self, key, value):
        raise ValueError("Do not assign value to a SliceTypeEnum field.")
