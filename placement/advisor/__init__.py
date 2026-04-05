"""In-repo LLM placement advisor."""

__all__ = ["PlacementAdvisor"]


def __getattr__(name):
    if name == "PlacementAdvisor":
        from .advisor import PlacementAdvisor
        return PlacementAdvisor
    raise AttributeError(name)
