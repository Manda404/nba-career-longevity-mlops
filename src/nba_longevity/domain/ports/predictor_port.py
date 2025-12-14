from typing import Protocol, Sequence, Mapping


class PredictorPort(Protocol):
    def predict_proba(
        self,
        rows: Sequence[Mapping[str, object]],
        feature_columns: Sequence[str],
    ) -> Sequence[float]:
        ...
