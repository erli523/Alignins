"""Instance-local return-contract adaptation, without patching native modules."""

from types import FunctionType


class ProjectionUtilities:
    def __init__(self, original):
        self.original = original

    def __getattr__(self, name):
        return getattr(self.original, name)

    def vector_to_model_wo_load(self, vector, model):
        state = model.state_dict()
        offset = 0
        for name, value in state.items():
            state[name] = vector[offset:offset + value.numel()].reshape_as(value).to(value)
            offset += value.numel()
        if offset != vector.numel():
            raise ValueError("Projected vector does not match the model")
        return state


def bind_pgd_projection(agent):
    original = agent.local_train.__func__
    namespace = dict(original.__globals__)
    namespace["utils"] = ProjectionUtilities(namespace["utils"])
    adapted = FunctionType(original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__)
    adapted.__kwdefaults__ = original.__kwdefaults__
    agent.local_train = adapted.__get__(agent, type(agent))
