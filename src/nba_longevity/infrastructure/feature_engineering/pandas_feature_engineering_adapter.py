from pandas import DataFrame
from nba_longevity.domain.dataset.dataset import Dataset
from nba_longevity.infrastructure.dataset.pandas_dataset import PandasDataset
from nba_longevity.domain.features.feature_contract import FEATURE_COLUMNS
from nba_longevity.domain.ports.feature_engineering_port import FeatureEngineeringPort

class PandasFeatureEngineeringAdapter(FeatureEngineeringPort):
    """
    Feature engineering métier NBA.
    """

    def add_features(self, dataset: Dataset) -> Dataset:
        df = DataFrame(list(dataset))

        eps = 1e-6  # sécurité divisions

        # 1. Usage & efficacité
        df["PointsPerMinute"] = df["PointsPerGame"] / (df["MinutesPerGame"] + eps)
        df["FieldGoalEfficiency"] = df["FieldGoalsMade"] / (df["FieldGoalsAttempted"] + eps)
        df["ThreePointRate"] = df["ThreePointersAttempted"] / (df["FieldGoalsAttempted"] + eps)
        df["FreeThrowRate"] = df["FreeThrowsAttempted"] / (df["MinutesPerGame"] + eps)

        # 2. Impact collectif
        df["AssistToTurnoverRatio"] = df["Assists"] / (df["Turnovers"] + eps)
        df["ReboundRate"] = df["TotalRebounds"] / (df["MinutesPerGame"] + eps)
        df["DefensiveImpact"] = df["Steals"] + df["Blocks"]

        # 3. Sélection finale
        #features_df = df[FEATURE_COLUMNS + ["Target5Years"]]

        return PandasDataset(df)
