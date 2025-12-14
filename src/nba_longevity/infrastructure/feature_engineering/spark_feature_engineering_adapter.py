from pyspark.sql.functions import col

from nba_longevity.domain.ports.feature_engineering_port import FeatureEngineeringPort
from nba_longevity.domain.features.feature_contract import FEATURE_COLUMNS
from nba_longevity.infrastructure.dataset.spark_dataset import SparkDataset


class SparkFeatureEngineeringAdapter(FeatureEngineeringPort):
    """
    Feature engineering distribué avec Spark.

    Dataset → Dataset → Dataset
    """

    def add_features(self, dataset: SparkDataset) -> SparkDataset:
        df = dataset._df

        # 🔧 Feature engineering
        df = df.withColumn(
            "PTS_PER_MIN",
            col("PTS") / col("MIN")
        )

        # 🔒 Sélection explicite
        df = df.select(*FEATURE_COLUMNS)

        return SparkDataset(df)
