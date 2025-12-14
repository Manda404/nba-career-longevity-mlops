from typing import Any, Sequence, Mapping
from pandas import DataFrame
from catboost import CatBoostClassifier, Pool

from domain.ports.trainer_port import TrainerPort


class CatBoostTrainer(TrainerPort):
    def train(
        self,
        train_rows: Sequence[Mapping[str, object]],
        valid_rows: Sequence[Mapping[str, object]],
        feature_columns: Sequence[str],
        target_column: str,
        params: dict[str, object],
    ) -> Any:
        train_df = DataFrame(list(train_rows))
        valid_df = DataFrame(list(valid_rows))

        X_train = train_df[feature_columns]
        y_train = train_df[target_column]
        X_valid = valid_df[feature_columns]
        y_valid = valid_df[target_column]

        train_pool = Pool(X_train, y_train)
        valid_pool = Pool(X_valid, y_valid)

        model = CatBoostClassifier(**params)
        model.fit(
            train_pool,
            eval_set=valid_pool,
            use_best_model=True,
            verbose=False,
        )
        return model
