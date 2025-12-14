from typing import Any, Sequence, Mapping

from pandas import DataFrame
from xgboost import DMatrix, train

from nba_longevity.application.bootstrap import app_logger
from nba_longevity.domain.ports.trainer_port import TrainerPort


class XGBoostTrainer(TrainerPort):
    """
    Entraîneur XGBoost conforme au TrainerPort.

    Responsabilités :
    - préparer les données (Pandas → DMatrix)
    - lancer l'entraînement XGBoost
    - gérer l'early stopping
    - retourner le booster entraîné
    """

    def train(
        self,
        train_rows: Sequence[Mapping[str, object]],
        valid_rows: Sequence[Mapping[str, object]],
        feature_columns: Sequence[str],
        target_column: str,
        params: dict[str, object],
    ) -> Any:
        app_logger.info("Démarrage de l'entraînement XGBoost")

        # =========================
        # 1. Chargement des données
        # =========================
        train_df = DataFrame(list(train_rows))
        valid_df = DataFrame(list(valid_rows))

        app_logger.debug(
            f"Train shape: {train_df.shape} | "
            f"Valid shape: {valid_df.shape}"
        )

        # =========================
        # 2. Séparation X / y
        # =========================
        X_train = train_df[feature_columns]
        y_train = train_df[target_column]
        X_valid = valid_df[feature_columns]
        y_valid = valid_df[target_column]

        app_logger.debug(
            f"Nombre de features utilisées: {len(feature_columns)}"
        )

        # =========================
        # 3. Création des DMatrix
        # =========================
        dtrain = DMatrix(X_train, label=y_train)
        dvalid = DMatrix(X_valid, label=y_valid)

        app_logger.info("DMatrix XGBoost créées")

        # =========================
        # 4. Paramètres d'entraînement
        # =========================
        num_boost_round = int(params.pop("num_boost_round", 500))
        early_stopping_rounds = int(params.pop("early_stopping_rounds", 50))

        app_logger.info(
            f"Hyperparamètres clés | "
            f"num_boost_round={num_boost_round}, "
            f"early_stopping_rounds={early_stopping_rounds}"
        )

        app_logger.debug(f"Paramètres XGBoost complets : {params}")

        # =========================
        # 5. Entraînement
        # =========================
        booster = train(
            params=params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,
        )

        # =========================
        # 6. Résumé entraînement
        # =========================
        app_logger.info(
            f"Entraînement terminé | "
            f"best_iteration={booster.best_iteration}, "
            f"best_score={booster.best_score}"
        )

        return booster
