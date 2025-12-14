from typing import Protocol
from nba_longevity.domain.dataset.dataset import Dataset


class FeatureEngineeringPort(Protocol):
    """
    Contrat de feature engineering.
    Ajoute et enrichit les features (sans en supprimer).

    Points clés
        entrée = Dataset
        sortie = Dataset
        aucune dépendance technique
        aucune notion de DataFrame
    """

    def add_features(self, dataset: Dataset) -> Dataset:
        """
        Construit de nouvelles features à partir d'un dataset préprocessé.
        Retourne un nouveau Dataset.
        """
        ...