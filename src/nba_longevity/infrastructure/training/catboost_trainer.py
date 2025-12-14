from typing import Any, Sequence, Mapping

from pandas import DataFrame
from catboost import CatBoostClassifier, Pool

from nba_longevity.application.bootstrap import app_logger
from nba_longevity.domain.ports.trainer_port import TrainerPort


class CatBoostTrainer(TrainerPort):
    """
    Entraîneur CatBoost conforme au TrainerPort.

    Responsabilités :
    - préparer les données (Pandas → Pool)
    - lancer l'entraînement CatBoost
    - gérer la validation et le best model
    - retourner le modèle entraîné
    """

    def train(
        self,
        train_rows: Sequence[Mapping[str, object]],
        valid_rows: Sequence[Mapping[str, object]],
        feature_columns: Sequence[str],
        target_column: str,
        params: dict[str, object],
    ) -> Any:
        app_logger.info("Démarrage de l'entraînement CatBoost")

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
        # 3. Création des Pools CatBoost
        # =========================
        train_pool = Pool(X_train, y_train)
        valid_pool = Pool(X_valid, y_valid)

        app_logger.info("Pools CatBoost créés")

        # =========================
        # 4. Paramètres du modèle
        # =========================
        app_logger.info("Initialisation du CatBoostClassifier")
        app_logger.debug(f"Paramètres CatBoost : {params}")

        model = CatBoostClassifier(**params)

        # =========================
        # 5. Entraînement
        # =========================
        model.fit(
            train_pool,
            eval_set=valid_pool,
            use_best_model=True,
            verbose=False,
        )

        # =========================
        # 6. Résumé entraînement
        # =========================
        best_iteration = model.get_best_iteration()
        best_score = model.get_best_score()

        app_logger.info(
            f"Entraînement terminé | "
            f"best_iteration={best_iteration}, "
            f"best_score={best_score}"
        )

        return model
