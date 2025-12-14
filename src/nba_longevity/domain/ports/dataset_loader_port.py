from typing import Protocol
from nba_longevity.domain.dataset.dataset import Dataset


class DatasetLoaderPort(Protocol):
    """
    Contrat pour charger un dataset.
    """

    def load(self) -> Dataset:
        """
        Charge un dataset tabulaire abstrait.
        """
        ...
