



Parfait, on va faire ça **très concrètement**, exactement dans l’ordre que tu demandes, **comme si tu étais dans ton notebook**, puis on implémente **`PreprocessingPort` correctement** (flux logique, pas format physique).

---

# 1️⃣ Que se passe-t-il quand tu fais `print(dataset)` ?

## Cas 1 — Pandas

```python
df = pd.read_csv("nba.csv")
dataset = PandasDataset(df)

print(dataset)
```

### 👉 Ce que Python affiche réellement

```text
<infrastructure.dataset.pandas_dataset.PandasDataset object at 0x10f8c2d90>
```

### 🧠 Pourquoi ?

Parce que :

* `dataset` **n’est pas un DataFrame**
* c’est **un objet métier abstrait**
* tu n’as pas défini `__repr__` ou `__str__`

👉 **Et c’est NORMAL et SAIN**

Le Domain **ne veut pas afficher les données**,
il veut **les parcourir**.

---

## Cas 2 — Spark

```python
spark_df = spark.read.csv("nba.csv", header=True)
dataset = SparkDataset(spark_df)

print(dataset)
```

### 👉 Résultat

```text
<infrastructure.dataset.spark_dataset.SparkDataset object at 0x10fa12c40>
```

👉 Exactement la même chose.

---

## ✅ LA BONNE FAÇON DE “VOIR” LES DONNÉES

Si tu veux vérifier ce qu’il y a dedans (en notebook) :

```python
for i, row in enumerate(dataset):
    print(row)
    if i == 2:
        break
```

Exemple de sortie :

```python
{'GP': 82, 'MIN': 28.4, 'PTS': 14.2, ...}
{'GP': 12, 'MIN': 6.1, 'PTS': 2.3, ...}
{'GP': 65, 'MIN': 19.8, 'PTS': 8.7, ...}
```

👉 **Voilà le flux logique**
👉 Peu importe Pandas ou Spark

---

# 2️⃣ MAINTENANT : implémentation PRO de `PreprocessingPort`

On va le faire **comme il faut** :

* le Domain définit **le contrat**
* l’Infra implémente avec Pandas
* **le flux logique est respecté**

---

## 2.1️⃣ Domain — `PreprocessingPort`

📁 `domain/ports/preprocessing_port.py`

```python
from typing import Protocol
from domain.dataset.dataset import Dataset


class PreprocessingPort(Protocol):
    """
    Contrat de preprocessing.
    """

    def preprocess(self, dataset: Dataset) -> Dataset:
        """
        Applique les règles de preprocessing
        et retourne un nouveau Dataset.
        """
        ...
```

👉 Points clés :

* entrée = `Dataset`
* sortie = `Dataset`
* **aucun DataFrame**
* **aucune lib**

---

## 2.2️⃣ Infra — Implémentation Pandas

📁 `infrastructure/preprocessing/pandas_preprocessing_adapter.py`

```python
import pandas as pd
from domain.ports.preprocessing_port import PreprocessingPort
from domain.dataset.dataset import Dataset
from infrastructure.dataset.pandas_dataset import PandasDataset


class PandasPreprocessingAdapter(PreprocessingPort):
    """
    Preprocessing avec Pandas.
    """

    def preprocess(self, dataset: Dataset) -> Dataset:
        # 🔁 flux logique → DataFrame (infra seulement)
        df = pd.DataFrame(list(dataset))

        # 🔧 règles métier (exemples)
        df = df.dropna()
        df["GP"] = df["GP"].astype(int)
        df["MIN"] = df["MIN"].astype(float)
        df["PTS"] = df["PTS"].astype(float)

        # 🔁 retour au flux logique
        return PandasDataset(df)
```

👉 Oui, ici on matérialise en DataFrame
👉 Parce que **le preprocessing a souvent besoin de Pandas**
👉 Mais **ça reste confiné à l’infrastructure**

---

## 2.3️⃣ Utilisation dans ton notebook / application

```python
loader = CsvDatasetLoader("nba.csv")
dataset = loader.load()

preprocessor = PandasPreprocessingAdapter()
clean_dataset = preprocessor.preprocess(dataset)

for row in clean_dataset:
    print(row)
    break
```

Sortie :

```python
{'GP': 82, 'MIN': 28.4, 'PTS': 14.2, ...}
```

---

# 3️⃣ POURQUOI CETTE APPROCHE EST LA BONNE

### ✔ Tu peux faire :

| Source          | Ça marche ? |
| --------------- | ----------- |
| Pandas CSV      | ✅           |
| Spark DataFrame | ✅           |
| SQL demain      | ✅           |
| Feature Store   | ✅           |

### ✔ Sans changer :

* le Domain
* les ports
* la logique métier

---

# 4️⃣ RÈGLE D’OR (À RETENIR)

> **`print(dataset)` n’est pas censé afficher des données.**
> **Le dataset est un flux, pas une table.**

Si tu veux voir les données :
👉 tu **itères**
👉 tu **échantillonnes**

---

## 🔜 SUITE LOGIQUE

Maintenant, on peut :

* ajouter `FeatureEngineeringPort` sur ce flux
* brancher un `TrainerPort`
* faire un **pipeline complet d’entraînement**
* gérer Pandas / Spark automatiquement

Si tu veux, on enchaîne.



Parfait — on va faire **exactement la même chose pour Spark**, **sans rien changer au Domain**, et tu vas voir que **tout ce qu’on a construit prend son sens**.

Je vais procéder en **3 temps** :

1. 👉 Ce que `print(dataset)` donne avec Spark (pour bien ancrer)
2. 👉 Implémentation **correcte** de `SparkPreprocessingAdapter`
3. 👉 Utilisation concrète (notebook / application)

---

# 1️⃣ `print(dataset)` avec Spark (rappel rapide)

```python
spark_df = spark.read.csv("nba.csv", header=True, inferSchema=True)
dataset = SparkDataset(spark_df)

print(dataset)
```

### 👉 Résultat

```text
<infrastructure.dataset.spark_dataset.SparkDataset object at 0x...>
```

✔ Normal
✔ Sain
✔ Identique à Pandas

👉 **Le Domain ne “voit” pas les données**,
👉 il **les parcourt**.

Pour voir les données :

```python
for i, row in enumerate(dataset):
    print(row)
    if i == 2:
        break
```

---

# 2️⃣ Implémentation PRO de `PreprocessingPort` pour Spark

⚠️ **Point très important (niveau senior)**
Avec Spark, **le preprocessing doit rester distribué**.
On **NE DOIT PAS** convertir en Pandas.

---

## 2.1️⃣ Le Domain (INCHANGÉ)

📁 `domain/ports/preprocessing_port.py`

```python
from typing import Protocol
from domain.dataset.dataset import Dataset


class PreprocessingPort(Protocol):
    def preprocess(self, dataset: Dataset) -> Dataset:
        ...
```

👉 Toujours aucune dépendance technique.

---

## 2.2️⃣ Infra — `SparkPreprocessingAdapter`

📁 `infrastructure/preprocessing/spark_preprocessing_adapter.py`

```python
from domain.ports.preprocessing_port import PreprocessingPort
from infrastructure.dataset.spark_dataset import SparkDataset


class SparkPreprocessingAdapter(PreprocessingPort):
    """
    Preprocessing distribué avec Spark.
    """

    def preprocess(self, dataset: SparkDataset) -> SparkDataset:
        # ⚠️ Ici on sait que c'est SparkDataset
        df = dataset._df

        # 🔧 règles métier (exemples)
        df = df.dropna()

        df = (
            df.withColumn("GP", df["GP"].cast("int"))
              .withColumn("MIN", df["MIN"].cast("double"))
              .withColumn("PTS", df["PTS"].cast("double"))
        )

        return SparkDataset(df)
```

### 🧠 Pourquoi c’est correct

* Le **Domain** voit `Dataset → Dataset`
* L’**infra Spark** sait que c’est Spark
* Le preprocessing reste **distribué**
* Pas de `.toPandas()` ❌
* Pas de `.collect()` ❌

---

## 2.3️⃣ Utilisation (Application / Notebook)

```python
loader = SparkDatasetLoader(
    spark_session=spark,
    path="nba.csv"
)

dataset = loader.load()

preprocessor = SparkPreprocessingAdapter()
clean_dataset = preprocessor.preprocess(dataset)

for row in clean_dataset:
    print(row)
    break
```

Sortie :

```python
{'GP': 82, 'MIN': 28.4, 'PTS': 14.2, ...}
```

👉 Toujours le **flux logique**
👉 Toujours la même interface
👉 Mais **calcul distribué**

---

# 3️⃣ COMPARAISON PANDAS vs SPARK (TRÈS CLAIRE)

| Aspect     | Pandas          | Spark          |
| ---------- | --------------- | -------------- |
| Conversion | `list(dataset)` | ❌              |
| Exécution  | en mémoire      | distribuée     |
| Drop NA    | `df.dropna()`   | `df.dropna()`  |
| Cast       | `.astype()`     | `.cast()`      |
| Retour     | `PandasDataset` | `SparkDataset` |
| Domain     | identique       | identique      |

👉 **Le Domain ne voit AUCUNE différence.**

---

# 4️⃣ POINT ARCHI IMPORTANT (QUE TU DOIS RETENIR)

> ❌ Le Domain ne choisit PAS Pandas ou Spark
> ✅ L’Application choisit l’implémentation
> ✅ L’Infra exécute

Exemple Application :

```python
if use_spark:
    loader = SparkDatasetLoader(...)
    preprocessor = SparkPreprocessingAdapter()
else:
    loader = CsvDatasetLoader(...)
    preprocessor = PandasPreprocessingAdapter()
```

👉 **Aucun `if` dans le Domain.**

---

# 🧠 PHRASE À GRAVER (TRÈS IMPORTANTE)

> **Le preprocessing est une règle métier,
> mais son exécution est un détail technique.**

---

## 🔜 PROCHAINE ÉTAPE LOGIQUE

Maintenant qu’on a :

* `Dataset` abstrait
* PandasDataset
* SparkDataset
* PreprocessingPort + 2 adapters

On peut :
👉 faire `FeatureEngineeringPort` multi-backend
👉 brancher un `TrainerPort` Spark / sklearn
👉 construire un pipeline d’entraînement complet

Dis-moi ce que tu veux faire ensuite.


Parfait.
On va faire **FeatureEngineeringPort multi-backend (Pandas + Spark)** **proprement**, **sans casser le Domain**, **sans if/else cachés**, et **comme en vraie prod ML**.

Je vais procéder **pas à pas**, avec **le raisonnement + le code**.

---

# 🎯 OBJECTIF

> Construire un **FeatureEngineeringPort** :

* indépendant de Pandas / Spark
* utilisable **train & inference**
* avec **2 implémentations infra**
* basé sur un **flux logique de données**

---

# 🧠 RAPPEL DE L’ARCHI (INCHANGÉE)

```
Domain
 ├── dataset/
 │    └── dataset.py
 └── ports/
      └── feature_engineering_port.py

Infrastructure
 ├── dataset/
 │    ├── pandas_dataset.py
 │    └── spark_dataset.py
 └── feature_engineering/
      ├── pandas_feature_engineering_adapter.py
      └── spark_feature_engineering_adapter.py
```

👉 Le Domain ne change PAS
👉 On ajoute juste des implémentations

---

# 1️⃣ DOMAIN — `FeatureEngineeringPort`

📁 `domain/ports/feature_engineering_port.py`

```python
from typing import Protocol
from domain.dataset.dataset import Dataset


class FeatureEngineeringPort(Protocol):
    """
    Contrat de feature engineering.
    """

    def build_features(self, dataset: Dataset) -> Dataset:
        """
        Construit les features à partir d'un dataset préprocessé.
        Retourne un nouveau Dataset.
        """
        ...
```

### 🔑 Points clés

* entrée = `Dataset`
* sortie = `Dataset`
* **aucune dépendance technique**
* **aucune notion de DataFrame**

---

# 2️⃣ DOMAIN — Spécification des features (TRÈS IMPORTANT)

📁 `domain/features/feature_contract.py`

```python
FEATURE_COLUMNS = [
    "GP",
    "MIN",
    "PTS",
    "FGM",
    "FGA",
    "FG_perc",
    "ThreeP_Made",
    "ThreePA",
    "PTS_PER_MIN"
]
```

👉 **Le Domain décide :**

* quelles features existent
* leur nom
* leur signification

👉 **L’Infra décide comment les calculer.**

---

# 3️⃣ INFRA — Pandas Feature Engineering

📁 `infrastructure/feature_engineering/pandas_feature_engineering_adapter.py`

```python
import pandas as pd

from domain.ports.feature_engineering_port import FeatureEngineeringPort
from domain.dataset.dataset import Dataset
from domain.features.feature_contract import FEATURE_COLUMNS
from infrastructure.dataset.pandas_dataset import PandasDataset


class PandasFeatureEngineeringAdapter(FeatureEngineeringPort):
    """
    Feature engineering avec Pandas.
    """

    def build_features(self, dataset: Dataset) -> Dataset:
        # Flux logique → Pandas
        df = pd.DataFrame(list(dataset))

        # 🔧 Feature engineering
        df["PTS_PER_MIN"] = df["PTS"] / df["MIN"]

        # 🔒 Sélection explicite
        df = df[FEATURE_COLUMNS]

        return PandasDataset(df)
```

### 🧠 Pourquoi c’est bien

* Pandas reste confiné
* règles métier respectées
* features alignées train / inference

---

# 4️⃣ INFRA — Spark Feature Engineering

📁 `infrastructure/feature_engineering/spark_feature_engineering_adapter.py`

```python
from pyspark.sql.functions import col

from domain.ports.feature_engineering_port import FeatureEngineeringPort
from domain.features.feature_contract import FEATURE_COLUMNS
from infrastructure.dataset.spark_dataset import SparkDataset


class SparkFeatureEngineeringAdapter(FeatureEngineeringPort):
    """
    Feature engineering distribué avec Spark.
    """

    def build_features(self, dataset: SparkDataset) -> SparkDataset:
        df = dataset._df

        # 🔧 Feature engineering
        df = df.withColumn(
            "PTS_PER_MIN",
            col("PTS") / col("MIN")
        )

        # 🔒 Sélection explicite
        df = df.select(*FEATURE_COLUMNS)

        return SparkDataset(df)
```

### ⚠️ POINT IMPORTANT

* on **n’utilise PAS** `list(dataset)`
* on reste distribué
* aucune collecte mémoire

---

# 5️⃣ APPLICATION — Choix du backend (1 seul endroit)

📁 `application/run_training.py`

```python
USE_SPARK = True

if USE_SPARK:
    loader = SparkDatasetLoader(...)
    preprocessor = SparkPreprocessingAdapter()
    feature_engineer = SparkFeatureEngineeringAdapter()
else:
    loader = CsvDatasetLoader(...)
    preprocessor = PandasPreprocessingAdapter()
    feature_engineer = PandasFeatureEngineeringAdapter()

dataset = loader.load()
clean_dataset = preprocessor.preprocess(dataset)
feature_dataset = feature_engineer.build_features(clean_dataset)
```

👉 **Zéro condition dans le Domain**
👉 **Zéro duplication logique**
👉 Backend interchangeable

---

# 6️⃣ CE QUE LE DOMAIN “VOIT” (MENTALEMENT)

Le Domain raisonne comme ça :

```text
Dataset → Dataset → Dataset
```

Il ne sait PAS :

* si c’est Pandas
* si c’est Spark
* où c’est stocké
* comment c’est exécuté

👉 **Il ne voit que des transformations métier.**

---

# 7️⃣ CE QUE TU AS MAINTENANT (TRÈS IMPORTANT)

✅ Feature engineering multi-backend
✅ Train / inference alignés
✅ Scalabilité (Spark ready)
✅ Clean Architecture respectée
✅ Prêt pour MLOps réel

---

# 🔑 PHRASE À RETENIR

> **Les features sont une décision métier,
> leur calcul est une décision technique.**

---

## 🔜 PROCHAINE ÉTAPE LOGIQUE

On peut maintenant :
👉 faire un `TrainerPort` Pandas / Spark
👉 gérer le split train / valid proprement
👉 construire un pipeline complet de bout en bout
👉 gérer l’inférence temps réel

Dis-moi où tu veux aller.
