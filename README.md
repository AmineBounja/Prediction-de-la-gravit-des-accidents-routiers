# 🚗 Projet ML: Prédiction de la Gravité des Accidents de la Route (RTA)

## 📊 Vue d'ensemble

Ce projet utilise le Machine Learning pour prédire la gravité des accidents de la route au Royaume-Uni. Il implémente 7 algorithmes de classification, des techniques d'optimisation d'hyperparamètres (GridSearchCV), une analyse non-supervisée (PCA, K-Means), et une application Streamlit interactive pour les prédictions en temps réel.

## 🎯 Objectifs

- Prédire la gravité des accidents (classification multi-classe: Fatal, Serious, Slight)
- Comparer 7 modèles de ML différents
- Optimiser les hyperparamètres avec GridSearchCV
- Analyser les données avec PCA et K-Means
- Gérer le déséquilibre des classes avec SMOTE-Tomek
- Déployer une application de démonstration interactive

## 📁 Structure du Projet

```
PROJET_ML_RTA/
├── data/
│   ├── raw/                    # Dataset brut (RTA Dataset.csv - UK Accidents)
│   ├── processed/              # Données preprocessées (train/val/test)
│   └── README.md               # Documentation des données
├── src/
│   ├── config.py               # Configuration centralisée
│   ├── preprocessing_pipeline.py # Pipeline de preprocessing
│   ├── training_pipeline.py    # Entraînement des modèles
│   ├── models.py               # Définitions des 7 modèles
│   ├── evaluation.py           # Métriques et évaluation
│   ├── hpo_pipeline.py         # Optimisation Hyperparamètres (GridSearchCV)
│   ├── unsupervised_analysis.py # Analyse Non-Supervisée (PCA, K-Means)
│   └── utils.py                # Fonctions utilitaires
├── models/
│   ├── *.pkl                   # Modèles de base entraînés (7 modèles)
│   └── optimized/              # Modèles optimisés (Random Forest, XGBoost)
├── results/
│   ├── model_comparison.csv    # Comparaison des performances
│   ├── final_report.pkl        # Rapport final
│   ├── confusion_matrices/     # Matrices de confusion
│   ├── visualizations/         # Graphiques et visualisations
│   └── unsupervised/           # Résultats PCA et K-Means
├── app/
│   └── streamlit_app.py        # Application de démonstration
├── docs/
│   ├── Article_RTA_Project.md  # Article du projet
│   └── Presentation_Structure.md # Structure de présentation
├── requirements.txt            # Dépendances Python
├── .gitignore
├── README.md                   # Ce fichier
└── START_HERE.md               # Guide de démarrage rapide
```

## 🔧 Installation

### Prérequis
- Python 3.8+
- pip

### Installation des dépendances

```bash
pip install -r requirements.txt
```

**Dépendances principales:**
- pandas, numpy
- scikit-learn
- xgboost
- imbalanced-learn (SMOTE-Tomek)
- matplotlib, seaborn
- streamlit

## 🚀 Utilisation

### 1️⃣ Preprocessing des données

```bash
python src/preprocessing_pipeline.py
```

**Opérations effectuées:**
- Nettoyage des données (duplicatas, valeurs manquantes)
- Détection et traitement des outliers (méthode IQR)
- Feature engineering (extraction de l'heure, encodage cyclique)
- Encodage des variables catégorielles (LabelEncoder)
- Normalisation (RobustScaler)
- Rééquilibrage des classes (SMOTE-Tomek)
- Split: 70% train, 15% validation, 15% test

### 2️⃣ Entraînement des modèles

```bash
python src/training_pipeline.py
```

**Entraîne les 7 modèles** et génère les métriques de performance.

### 3️⃣ Analyse Non-Supervisée (PCA/K-Means)

```bash
python src/unsupervised_analysis.py
```

**Effectue:**
- Réduction de dimensionnalité avec PCA
- Clustering avec K-Means
- Visualisations des résultats

### 4️⃣ Optimisation des Hyperparamètres (HPO)

```bash
python src/hpo_pipeline.py
```

**Optimise** Random Forest et XGBoost avec GridSearchCV.

### 5️⃣ Lancer l'application de démonstration

```bash
streamlit run app/streamlit_app.py
```

**Fonctionnalités:**
- Sélection du modèle (standard ou optimisé)
- Prédictions en temps réel
- Visualisation des métriques
- Matrices de confusion
- Importance des features

## 📊 Pipeline ML

```
┌─────────────────────┐
│  Dataset UK RTA     │
│  (~12,000 accidents)│
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Preprocessing      │
│  - Cleaning         │
│  - Outlier handling │
│  - Feature eng.     │
│  - Encoding         │
│  - Scaling          │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  SMOTE-Tomek        │
│  (Class balancing)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Split Data         │
│  70% / 15% / 15%    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Training           │
│  7 ML Models        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  HPO (GridSearchCV) │
│  RF + XGBoost       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Evaluation         │
│  - Metrics          │
│  - Comparison       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Deployment         │
│  Streamlit App      │
└─────────────────────┘
```

## 🤖 Modèles Implémentés

| Modèle | Type | Statut |
|--------|------|--------|
| **Logistic Regression** | Linéaire | ✅ Entraîné |
| **Random Forest** | Ensemble (Bagging) | ✅ Entraîné + Optimisé |
| **SVM** | Kernel-based | ✅ Entraîné |
| **K-NN** | Instance-based | ✅ Entraîné |
| **MLP** | Neural Network | ✅ Entraîné |
| **Naive Bayes** | Probabiliste | ✅ Entraîné |
| **XGBoost** | Ensemble (Boosting) | ✅ Entraîné + Optimisé |

**Modèles optimisés** (avec GridSearchCV):
- Random Forest Optimized
- XGBoost Optimized

## 📈 Métriques d'Évaluation

Pour chaque modèle, les métriques suivantes sont calculées:

- **Accuracy** - Précision globale du modèle
- **Precision** - Précision par classe (macro et weighted)
- **Recall** - Rappel par classe (macro et weighted)
- **F1-Score** - Score F1 (macro et weighted)
- **ROC-AUC** - Aire sous la courbe ROC (multi-classe)
- **Confusion Matrix** - Matrice de confusion détaillée

**Résultats disponibles dans:**
- `results/model_comparison.csv` - Tableau comparatif
- `results/confusion_matrices/` - Matrices de confusion
- `results/final_report.pkl` - Rapport complet

## 🔬 Techniques Avancées

### 📊 Analyse Non-Supervisée
- **PCA (Principal Component Analysis)** - Réduction de dimensionnalité pour visualisation
- **K-Means Clustering** - Identification de groupes d'accidents similaires
- Visualisations dans `results/unsupervised/`

### 🎯 Optimisation des Hyperparamètres
- **GridSearchCV** - Recherche exhaustive des meilleurs hyperparamètres
- Optimisation de Random Forest et XGBoost
- Validation croisée 5-fold

### ⚖️ Gestion du Déséquilibre
- **SMOTE-Tomek** - Combinaison de sur-échantillonnage (SMOTE) et sous-échantillonnage (Tomek Links)
- Amélioration de la détection des classes minoritaires (Fatal, Serious)

## 📚 Sources de Données

**Dataset**: Road Traffic Severity Classification (UK Accidents)
- **Source**: [Kaggle - Road Traffic Severity Classification](https://www.kaggle.com/datasets/avikumart/road-traffic-severity-classification)
- **Taille**: ~12,000 accidents
- **Période**: Accidents routiers au Royaume-Uni
- **Classes**: Fatal, Serious, Slight

**Variables principales:**
- Temporelles: Heure, Jour de la semaine
- Localisation: Zone, Type de jonction, Alignement de la route
- Véhicules: Nombre, Type, Années de service
- Conducteur: Âge, Sexe, Expérience, Niveau d'éducation
- Contexte: Cause de l'accident, Nombre de victimes

Voir [`data/README.md`](data/README.md) pour plus de détails.

## 📄 Documentation

- **Article**: [`docs/Article_RTA_Project.md`](docs/Article_RTA_Project.md) - Article détaillé du projet
- **Présentation**: [`docs/Presentation_Structure.md`](docs/Presentation_Structure.md) - Structure de présentation
- **Guide de démarrage**: [`START_HERE.md`](START_HERE.md) - Guide de démarrage rapide

## 🎨 Application de Démonstration

L'application Streamlit (`app/streamlit_app.py`) permet de:

- **Sélectionner un modèle** - Choisir parmi les modèles standards ou optimisés
- **Faire des prédictions** - Prédire la gravité d'un accident en temps réel
- **Visualiser les performances** - Afficher les métriques et matrices de confusion
- **Explorer les features** - Voir l'importance des variables
- **Comparer les modèles** - Tableau comparatif des performances

**Lancement:**
```bash
streamlit run app/streamlit_app.py
```

## 🎓 Contraintes Académiques Respectées

✅ **7 Modèles ML** (≥3 requis)
- Logistic Regression, Random Forest, SVM, K-NN, MLP, Naive Bayes, XGBoost

✅ **Modèle Probabiliste** - Naive Bayes

✅ **Modèle Ensemble** - Random Forest (Bagging) + XGBoost (Boosting)

✅ **Analyse Non-Supervisée** - PCA + K-Means

✅ **Optimisation Hyperparamètres** - GridSearchCV

✅ **Application de Démonstration** - Streamlit

✅ **Documentation Complète** - Article, présentation, README

## 👥 Auteurs

**Développé par :**
- Echaaouer Zakariae
- Fadwa Errami
- Bounja Mohammed Amine

**Institution :**
🏫 École Nationale des Sciences Appliquées de Fès (ENSAF)

**Projet :** Machine Learning - Prédiction de la Gravité des Accidents de la Route

## 🙏 Remerciements

- **Dataset**: Kaggle - Road Traffic Severity Classification
- **Bibliothèques**: Scikit-learn, XGBoost, Imbalanced-learn, Streamlit
- **Inspiration**: Recherche en sécurité routière et ML appliqué

---

*Dernière mise à jour: Décembre 2025*
#   P r e d i c t i o n - d e - l a - g r a v i t - d e s - a c c i d e n t s - r o u t i e r s  
 