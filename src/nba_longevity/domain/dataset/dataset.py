from typing import Protocol, Iterable, Mapping, Any


class Dataset(Protocol):
    """
    Abstraction d'un dataset tabulaire.

    Le Domain ne connaît :
    - ni Pandas
    - ni Spark
    - ni le format physique des données
    """

    def __iter__(self) -> Iterable[Mapping[str, Any]]:
        """
        Itère sur les lignes logiques du dataset.
        Chaque ligne représente une observation métier.
        """
        ...
