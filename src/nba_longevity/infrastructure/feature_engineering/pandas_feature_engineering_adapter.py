from pandas import DataFrame
from nba_longevity.application.bootstrap import app_logger
from nba_longevity.domain.dataset.dataset import Dataset
from nba_longevity.infrastructure.dataset.pandas_dataset import PandasDataset
from nba_longevity.domain.ports.feature_engineering_port import FeatureEngineeringPort


class PandasFeatureEngineeringAdapter(FeatureEngineeringPort):
    """
    Feature engineering métier NBA (backend Pandas).

    Responsabilités :
    - Transformer un Dataset abstrait en DataFrame Pandas
    - Créer des features métier interprétables
    - Retourner un Dataset enrichi
    """

    def add_features(self, dataset: Dataset) -> Dataset:
        app_logger.info("Démarrage du feature engineering (Pandas)")

        # Conversion Dataset -> DataFrame
        df = DataFrame(list(dataset))
        app_logger.debug(f"Dataset chargé avec {df.shape[0]} lignes et {df.shape[1]} colonnes")

        eps = 1e-6  # Sécurité divisions par zéro

        # =========================
        # 1. Usage & efficacité
        # =========================
        app_logger.info("Création des features d'usage et d'efficacité")

        df["PointsPerMinute"] = df["PointsPerGame"] / (df["MinutesPerGame"] + eps)
        df["FieldGoalEfficiency"] = df["FieldGoalsMade"] / (df["FieldGoalsAttempted"] + eps)
        df["ThreePointRate"] = df["ThreePointersAttempted"] / (df["FieldGoalsAttempted"] + eps)
        df["FreeThrowRate"] = df["FreeThrowsAttempted"] / (df["MinutesPerGame"] + eps)

        # =========================
        # 2. Impact collectif
        # =========================
        app_logger.info("Création des features d'impact collectif")

        df["AssistToTurnoverRatio"] = df["Assists"] / (df["Turnovers"] + eps)
        df["ReboundRate"] = df["TotalRebounds"] / (df["MinutesPerGame"] + eps)
        df["DefensiveImpact"] = df["Steals"] + df["Blocks"]

        app_logger.debug(
            "Features créées : PointsPerMinute, FieldGoalEfficiency, "
            "ThreePointRate, FreeThrowRate, AssistToTurnoverRatio, "
            "ReboundRate, DefensiveImpact"
        )

        app_logger.info("Feature engineering terminé avec succès")

        return PandasDataset(df)
