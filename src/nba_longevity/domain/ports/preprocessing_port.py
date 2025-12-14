from typing import Protocol
from nba_longevity.domain.dataset.dataset import Dataset


class PreprocessingPort(Protocol):
    """
    Contrat de preprocessing.

    Points clés :
        entrée = Dataset
        sortie = Dataset
        aucun DataFrame
        aucune lib
    """

    def preprocess(self, dataset: Dataset) -> Dataset:
        """
        Applique les règles de preprocessing
        et retourne un nouveau Dataset.
        """
        ...
