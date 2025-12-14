# NBA Career Longevity – MLOps Project

End-to-end MLOps project for predicting NBA rookie career longevity.

---

# 🏀 NBA Career Longevity Prediction  
**A Clean Architecture Machine Learning Project**

---

## 📌 Project Overview

This project aims to predict whether an NBA player will have a **career lasting more than 5 years** in the league, based on early-career performance statistics.

Beyond the predictive task itself, the **core objective of this repository** is to demonstrate how to build a **robust, maintainable, and production-ready Machine Learning system** using **Clean Architecture principles**.

> 🎯 **Primary goal**  
> Build a binary classification model predicting long-term NBA career longevity.

> 🧠 **Secondary (but essential) goal**  
> Showcase a **Clean Architecture applied to Machine Learning**, from raw data ingestion to experiment tracking.

---

## 🚀 Why Clean Architecture for Machine Learning?

Most ML projects start as notebooks and quickly turn into:
- tangled pipelines
- duplicated preprocessing logic
- training/inference inconsistencies
- hard-to-maintain codebases

This project addresses these issues by applying **Clean Architecture**, ensuring:

- clear separation of concerns
- reproducibility
- testability
- scalability
- production readiness

### Key principles applied

- **Domain-first design**: business logic is independent from tools
- **Ports & adapters**: infrastructure details are replaceable
- **Train / inference consistency** through explicit contracts
- **ML-specific rigor**: dataset schema, feature contracts, experiment tracking

---

## 🧠 Problem Definition

### Business Question

> Can we predict if an NBA player will stay in the league for more than 5 years based on early performance metrics?

### Target Variable

- `TARGET_5Yrs`
  - `1` → Career longer than 5 years
  - `0` → Career shorter or equal to 5 years

### Dataset Characteristics

- Player-level statistics (per-game averages)
- No categorical variables
- Strong dependency on playing time (minutes)
- Domain-specific edge cases (e.g. 3PT% when no attempts)

---

## 🏗️ Architecture Overview

The project follows a **strict Clean Architecture layering**:

```

src/
└─ nba_longevity/
├─ domain/         # Business & ML rules (NO tech)
├─ application/    # Use cases & orchestration
├─ infrastructure/# Technical implementations
└─ presentation/  # Interfaces (API, notebooks, etc.)

```

### Dependency rule (non-negotiable)

```

Infrastructure → Application → Domain

```

The **Domain layer depends on nothing**.

---

## 🧩 Domain Layer (The Core)

The `domain/` layer represents the **heart of the ML system**.

### What the Domain contains

- Dataset contracts & schemas
- Validation and preprocessing rules
- Feature definitions & feature contracts
- Training intent & target definition
- Ports (interfaces) expressing domain needs

### Structure

```

domain/
├─ dataset/        # Raw data schema & validation
├─ preprocessing/ # Business cleaning & normalization rules
├─ features/       # Feature specification & contracts
├─ training/       # Target & training intent
└─ ports/          # Interfaces to the outside world

```

### Key idea

> The Domain defines **WHAT** the system needs, never **HOW** it is done.

---

## 🔌 Ports & Adapters (Hexagonal Architecture)

Ports define **domain expectations**.  
Adapters (infrastructure) implement them.

| Domain Port | Responsibility |
|------------|----------------|
| DatasetLoaderPort | Load raw data |
| PreprocessingPort | Clean & normalize data |
| FeatureEngineeringPort | Build model features |
| TrainerPort | Train ML models |
| ExperimentTrackerPort | Track experiments |

This enables:
- easy replacement of tools (Pandas → Spark, MLflow → others)
- simple unit testing with mocks
- long-term maintainability

---

## 🛠️ Infrastructure Layer

The `infrastructure/` layer contains **pure technical implementations**:

- Pandas for data processing
- Scikit-learn for modeling
- MLflow for experiment tracking
- CSV-based data loading (easily replaceable)

```

infrastructure/
├─ dataset/        # CSV loaders
├─ preprocessing/ # Pandas-based preprocessing
├─ features/       # Pandas feature engineering
├─ training/       # ML model implementations
├─ tracking/       # MLflow adapter
└─ config/         # Centralized configuration

```

> Infrastructure code can be changed without touching the Domain.

---

## 🔁 End-to-End ML Pipeline

The real ML pipeline implemented by this architecture is:

```

Raw Dataset
↓
Schema Validation
↓
Business Cleaning Rules
↓
Normalization (minutes-based)
↓
Feature Engineering
↓
Feature Contract Validation
↓
Model Training
↓
Experiment Tracking (MLflow)

```

Every step is:
- explicit
- traceable
- reusable
- testable

---

## 📊 Experiment Tracking

Experiment tracking is handled through an **abstract port**, implemented using MLflow.

Tracked artifacts include:
- dataset schema
- feature definitions
- feature groups
- training parameters
- evaluation metrics
- trained models

This guarantees:
- full reproducibility
- auditability
- professional MLOps practices

---

## 🧪 Notebook-Friendly by Design

Although architected for production, the project is **notebook-friendly**.

You can safely:
- import domain contracts in notebooks
- experiment interactively
- reuse the same logic as production pipelines

This avoids the classic **notebook vs production gap**.

---

## 🎯 Key Takeaways

✔ Clean Architecture applied to ML (not just software)  
✔ Domain-driven, pipeline-oriented design  
✔ Clear separation of business logic and tools  
✔ Production-ready structure  
✔ Educational & extensible  

---

## 📎 Next Steps

Possible extensions:
- Add inference pipeline
- Add model registry & versioning
- Add data drift monitoring
- Swap Pandas for Spark
- Deploy via API or batch jobs

---

## 👤 Author & Intent

This project is designed as:
- a **learning reference**
- a **professional portfolio project**
- a **template for real-world ML systems**

If you work in Data Science, ML Engineering, or MLOps, this repository aims to reflect **how ML systems should be built in production**, not just in notebooks.
```

---