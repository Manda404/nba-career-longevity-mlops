from typing import Protocol, Any, Sequence, Mapping


class TrainerPort(Protocol):
    """
    Entraîne un modèle à partir de features + target.
    Le Domain ne sait pas si c'est XGBoost / CatBoost.
    """

    def train(
        self,
        train_rows: Sequence[Mapping[str, object]],
        valid_rows: Sequence[Mapping[str, object]],
        feature_columns: Sequence[str],
        target_column: str,
        params: dict[str, object],
    ) -> Any:
        ...
