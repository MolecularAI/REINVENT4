def add_tag(label: str, text: str = "True"):
    """A simple decorator to tag a class"""

    def wrapper(cls):
        setattr(cls, label, text)
        return cls

    return wrapper
