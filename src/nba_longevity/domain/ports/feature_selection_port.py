from typing import Protocol
from nba_longevity.domain.dataset.dataset import Dataset


class FeatureSelectionPort(Protocol):
    """
    Sélectionne l’espace de features final du modèle.
    """

    def select_features(self, dataset: Dataset) -> Dataset:
        """
        Projette vers le feature space ML.
        """
        ...