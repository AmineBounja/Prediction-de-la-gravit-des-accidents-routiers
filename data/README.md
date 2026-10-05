# Dataset RTA - Road Traffic Accidents

## 📊 Source des Données

**Dataset**: Road Traffic Severity Classification
- **Source**: [Kaggle - Road Traffic Severity Classification](https://www.kaggle.com/datasets/avikumart/road-traffic-severity-classification)
- **Fichier**: `RTA Dataset.csv`
- **Taille**: ~12,000 accidents
- **Période**: Données d'accidents routiers au Royaume-Uni

## 📁 Structure du Dossier

```
data/
├── raw/                          # Données brutes
│   └── RTA Dataset.csv          # Dataset original
└── processed/                    # Données prétraitées
    ├── X_train_processed.csv    # Features d'entraînement
    ├── y_train_processed.csv    # Labels d'entraînement
    ├── X_val_processed.csv      # Features de validation
    ├── y_val_processed.csv      # Labels de validation
    ├── X_test_processed.csv     # Features de test
    ├── y_test_processed.csv     # Labels de test
    ├── preprocessor.pkl         # Pipeline de preprocessing
    ├── label_encoders.pkl       # Encodeurs pour variables catégorielles
    ├── target_encoder.pkl       # Encodeur pour la variable cible
    └── feature_names.pkl        # Noms des features
```

## 🎯 Variable Cible

**`Accident_severity`** : Gravité de l'accident (3 classes)
- **Fatal** : Accident mortel
- **Serious** : Blessures graves
- **Slight** : Blessures légères

## 📋 Variables Principales

### Variables Temporelles
- `Time` / `Hour_of_Day` : Heure de l'accident
- `Day_of_week` : Jour de la semaine

### Variables de Localisation
- `Area_accident_occured` : Zone de l'accident
- `Types_of_Junction` : Type de jonction routière
- `Road_allignment` : Alignement de la route

### Variables de Véhicules
- `Number_of_vehicles_involved` : Nombre de véhicules impliqués
- `Type_of_vehicle` : Type de véhicule
- `Vehicle_driver_relation` : Relation conducteur-véhicule

### Variables de Victimes
- `Number_of_casualties` : Nombre de victimes
- `Age_band_of_driver` : Tranche d'âge du conducteur
- `Sex_of_driver` : Sexe du conducteur
- `Educational_level` : Niveau d'éducation

### Variables Contextuelles
- `Driving_experience` : Expérience de conduite
- `Service_year_of_vehicle` : Années de service du véhicule
- `Cause_of_accident` : Cause de l'accident

## 🔄 Prétraitement Appliqué

1. **Nettoyage** : Suppression des duplicatas et valeurs manquantes
2. **Outliers** : Détection et traitement par méthode IQR
3. **Feature Engineering** : Extraction de l'heure depuis Time
4. **Encodage** : LabelEncoder pour variables catégorielles
5. **Scaling** : RobustScaler pour normalisation
6. **Rééchantillonnage** : SMOTE-Tomek pour équilibrer les classes
7. **Split** : 70% train, 15% validation, 15% test

## 📊 Distribution des Classes (Après SMOTE)

Les classes sont équilibrées après application de SMOTE-Tomek pour améliorer la détection des accidents graves et mortels.

## 🔒 Reproductibilité

- **Random State**: 42 (fixé dans `config.py`)
- Tous les splits et transformations utilisent la même graine aléatoire

## 📝 Notes

- Les données brutes ne doivent **jamais être modifiées**
- Les données processed sont générées par `preprocessing_pipeline.py`
- Pour régénérer les données processed, exécutez le pipeline de prétraitement
