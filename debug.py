from abc import ABC, abstractmethod


class State:

    @classmethod
    def from_frame(cls, data) -> None:
        pass


class BaseMetric(ABC):

    @abstractmethod
    def evaluate(self) -> dict:
        pass


class Metric(BaseMetric):

    def __init__(self, state: State) -> None:
        self._state = state

    def evaluate(self) -> dict:
        return {}


class BaseOutput(ABC):

    @abstractmethod
    def metrics(self) -> BaseMetric:
        pass


class Output(BaseOutput):

    def __init__(self, state: State) -> None:
        self._state = state

    def metrics(self, lazy: bool = True) -> BaseMetric:
        return Metric(self._state)


if __name__ == "__main__":
    output = Output(State())
    output.metrics().evaluate()
