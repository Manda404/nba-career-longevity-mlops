"""
Définition des espaces de features utilisés pour la modélisation
de la longévité des joueurs NBA.

Ce module appartient au Domain : il exprime comment un joueur NBA
est représenté d’un point de vue métier pour le Machine Learning.

Aucune logique technique ici (Pandas, Spark, sklearn, etc.)
Les listes définissent des projections de données, pas des transformations.

# Originals features
PlayerName                 object
GamesPlayed                 int64
MinutesPerGame            float64
PointsPerGame             float64
FieldGoalsMade            float64
FieldGoalsAttempted       float64
FieldGoalPct              float64
ThreePointersMade         float64
ThreePointersAttempted    float64
ThreePointerPct           float64
FreeThrowsMade            float64
FreeThrowsAttempted       float64
FreeThrowPct              float64
OffensiveRebounds         float64
DefensiveRebounds         float64
TotalRebounds             float64
Assists                   float64
Steals                    float64
Blocks                    float64
Turnovers                 float64
Target5Years              float64
"""


FEATURE_SPACE_MINIMAL = [
    # News features
    "PointsPerMinute",          # Efficacité offensive indépendante du temps de jeu
    "FieldGoalEfficiency",      # Qualité réelle du scoring (FGM / FGA)
    "ThreePointRate",           # Part des tirs à 3 points dans le jeu offensif
    "FreeThrowRate",            # Agressivité offensive (LF tentés par minute)
    "AssistToTurnoverRatio",    # Maîtrise du jeu et prise de décision
    "ReboundRate",              # Activité globale au rebond par minute
    "DefensiveImpact",          # Impact défensif direct (interceptions + contres)
    
    # Olds features
    "MinutesPerGame",           # Confiance du coach / rôle dans l’équipe
    "GamesPlayed",              # Disponibilité et durabilité du joueur
]


FEATURE_SPACE_EXTENDED = FEATURE_SPACE_MINIMAL + [
    "PointsPerGame",            # Volume de scoring brut
    "Assists",                  # Contribution offensive collective
    "Turnovers",                # Risque et prise d’initiative
    "TotalRebounds",            # Volume brut au rebond
]

"""
Colonne cible du problème de classification.

Valeur binaire :
- 1 : carrière NBA supérieure ou égale à 5 ans
- 0 : carrière NBA inférieure à 5 ans
"""
TARGET_COLUMN = "Target5Years"


"""
Identifiant métier du joueur.

Utilisé uniquement pour :
- audit
- debug
- suivi des prédictions

⚠️ Ne doit jamais être utilisé comme feature ML.
"""
ID_COLUMN = "PlayerName"
