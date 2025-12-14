## 🏗️ Architecture recommandée (très utilisée)

```
project/
│
├── config/
│   ├── infra.yaml
│   ├── train.yaml
│   └── model/
│       ├── catboost.yaml
│       └── xgboost.yaml
│
├── .env            # secrets ONLY
├── .env.example    # documenté
│
├── infrastructure/
│   └── config/
│       └── settings.py  # loader + pydantic
```

---

## 🔑 Règle d’or (retenir ça)

> **Si un paramètre change un modèle, il ne doit PAS être dans `.env`.**


## TL;DR (réponse courte)

👉 **La stratégie la plus utilisée en pratique** est :

> **Un fichier de configuration unique (YAML / TOML / JSON)**
> chargé **au démarrage**,
> validé par un **schema strict (Pydantic)**,
> puis **injecté partout comme un objet immuable**.

On **n’écrit jamais** les paramètres “en dur” dans le code métier.





