from typing import Any, Sequence, Mapping
from pandas import DataFrame
from xgboost import DMatrix, train

from domain.ports.trainer_port import TrainerPort


class XGBoostTrainer(TrainerPort):
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

        dtrain = DMatrix(X_train, label=y_train)
        dvalid = DMatrix(X_valid, label=y_valid)

        # params typiques : objective, eval_metric, max_depth, eta, subsample, colsample_bytree...
        num_boost_round = int(params.pop("num_boost_round", 500))
        early_stopping_rounds = int(params.pop("early_stopping_rounds", 50))

        booster = train(
            params=params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,
        )
        return booster