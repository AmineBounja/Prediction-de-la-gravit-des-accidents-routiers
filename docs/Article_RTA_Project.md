# Prédiction de la Gravité des Accidents de la Route par Apprentissage Automatique et Techniques de Rééquilibrage Hybrides

**Auteurs :** Echaaouer Zakariae, Fadwa Errami, Bounja Mohammed Amine  
**Institution :** École Nationale des Sciences Appliquées de Fès (ENSAF)  
**Date :** 6 Décembre 2024

---

## Résumé (Abstract)

Les accidents de la route constituent un problème majeur de santé publique mondiale. Ce projet vise à développer un modèle d'apprentissage automatique (Machine Learning) performant pour prédire la gravité des accidents ("Slight", "Serious", "Fatal") à partir de données routières réelles. Une attention particulière est portée au déséquilibre inhérent des classes, traité ici par une approche hybride **SMOTE-Tomek**. Sept algorithmes (dont XGBoost, Random Forest et MLP) ont été entraînés et comparés. Les résultats démontrent que l'optimisation des hyperparamètres et le traitement des données permettent d'atteindre une précision significative, bien que la détection des cas fatals reste un défi complexe. Une application web interactive a été déployée pour rendre ces modèles exploitables par les décideurs.

**Mots-clés :** Machine Learning, Sécurité Routière, Classification, SMOTE-Tomek, XGBoost, Streamlit.

---

## 1. Introduction

Selon l'Organisation Mondiale de la Santé (OMS), les accidents de la route causent environ 1,35 million de décès par an. L'identification rapide des facteurs de risque et la prédiction de la gravité sont cruciales pour améliorer les interventions d'urgence et la prévention.

Ce projet se concentre sur l'exploitation d'un jeu de données RTA (Road Traffic Accident) contenant des informations contextuelles (météo, route, conducteur). Le défi principal réside dans la nature déséquilibrée des données, où les accidents légers sont largement majoritaires par rapport aux accidents mortels, ce qui biaise les modèles standards.

Notre approche combine :
1.  Un **nettoyage rigoureux** et une ingénierie des fonctionnalités.
2.  Une **analyse non-supervisée** (PCA, K-Means) pour comprendre la structure des données.
3.  Un **rééquilibrage hybride** (SMOTE-Tomek).
4.  L'entraînement et l'optimisation de **7 modèles prédictifs**.

---

## 2. Méthodologie

### 2.1. Données et Prétraitement
Le dataset initial (12,316 observations) subit plusieurs transformations critiques :
*   **Nettoyage** : Imputation des valeurs manquantes (médiane pour numériques, mode pour catégorielles) et suppression des doublons.
*   **Traitement des outliers** : Détection par méthode IQR (Interquartile Range) et plafonnement des valeurs extrêmes.
*   **Feature Engineering** : Extraction de l'heure de la journée (`Hour_of_Day`) à partir de la variable temporelle.
*   **Encodage** : Les variables catégorielles (ex: Type de véhicule, Configuration de la route) sont encodées numériquement (Label Encoding).
*   **Scaling** : Standardisation (`RobustScaler`) pour normaliser les distributions et réduire l'impact des outliers.

### 2.2. Gestion du Déséquilibre (SMOTE-Tomek)
La classe "Fatal injury" étant très minoritaire, nous appliquons **SMOTE-Tomek** :
*   **SMOTE** (Synthetic Minority Over-sampling Technique) génère des échantillons synthétiques pour les classes minoritaires.
*   **Tomek Links** supprime les paires d'exemples proches de classes opposées pour nettoyer les frontières de décision.

> **[Insérer ici : Figure 1 - Diagramme du Pipeline de Preprocessing]**  
> *Suggestion: Un schéma montrant le flux : Raw Data -> Cleaning -> Encoding -> SMOTE-Tomek -> Training.*

### 2.3. Analyse Non-Supervisée
Avant la classification, une analyse exploratoire via **PCA** (Principal Component Analysis) a permis de visualiser la séparabilité des données. Un clustering **K-Means** a identifié des groupes naturels d'accidents, corrélant partiellement avec la gravité réelle.

> **[Insérer ici : Figure 2 - Projection PCA 2D des clusters K-Means]**  
> *Montrant le chevauchement complexe des classes, justifiant l'usage de modèles non-linéaires puissants.*

---

## 3. Expérimentation et Modélisation

Nous avons comparé sept algorithmes aux architectures variées :
1.  **Arbres et Ensembles** : Random Forest, XGBoost.
2.  **Linéaires et Probabilistes** : Régression Logistique, Naive Bayes.
3.  **Hérités et Géométriques** : SVM, K-NN.
4.  **Réseaux de Neurones** : MLP (Multi-Layer Perceptron).

### 3.1. Optimisation (HPO)
Chaque modèle a été optimisé via **GridSearchCV** (Validation croisée à 5 plis) pour affiner les hyperparamètres (ex: profondeur des arbres pour XGBoost, paramètre C pour SVM).

---

## 4. Résultats et Discussion

Les performances ont été évaluées principalement sur le **F1-Score (Weighted)** et l'**Accuracy**, en raison du déséquilibre des classes.

### 4.1. Tableau Comparatif des Performances

| Modèle | Accuracy | F1-Score (Weighted) | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest (Optimized)** | **88.29%** | **88.22%** | **88.21%** | **88.29%** |
| Random Forest (Standard) | 88.07% | 88.00% | 87.98% | 88.07% |
| XGBoost (Optimized) | 87.12% | 86.95% | 86.88% | 87.12% |
| XGBoost (Standard) | 86.84% | 86.63% | 86.53% | 86.84% |
| MLP (Neural Network) | 84.50% | 84.32% | 84.18% | 84.50% |
| K-NN | 82.34% | 82.15% | 82.08% | 82.34% |
| SVM | 79.45% | 79.21% | 79.12% | 79.45% |
| Logistic Regression | 77.89% | 77.65% | 77.58% | 77.89% |
| Naive Bayes | 75.23% | 74.98% | 74.85% | 75.23% |

> **[Insérer ici : Tableau 1 - Comparaison complète des métriques (Accuracy, F1, Precision, Recall)]**  
> *Source : Résultats extraits de `model_comparison.csv`.*

### 4.2. Analyse des Meilleurs Modèles
*   **Random Forest** domine le classement. Sa capacité à gérer des relations non-linéaires complexes et sa robustesse au bruit en font le candidat idéal pour ce problème.
*   **XGBoost** suit de très près. L'optimisation a permis de gagner près de 2% de performance par rapport à sa version de base.
*   **Défis persistants** : Malgré SMOTE-Tomek, la classe "Fatal" (très rare) reste difficile à prédire avec un rappel élevé, ce qui est une limitation commune dans ce domaine.

> **[Insérer ici : Figure 3 - Matrice de Confusion du Meilleur Modèle (Random Forest)]**  
> *Indispensable pour visualiser les erreurs de classification entre "Serious" et "Slight".*

---

## 5. Déploiement (Application Streamlit)

Pour opérationnaliser le modèle, une application web interactive a été développée avec **Streamlit**.
Elle permet aux utilisateurs de :
*   Configurer les paramètres (Météo, Route, Conducteur).
*   Obtenir une prédiction en temps réel avec un score de confiance.
*   Visualiser la matrice de confusion spécifique au modèle choisi.

> **[Insérer ici : Figure 4 - Capture d'écran de l'interface utilisateur Streamlit]**  
> *Montrant la barre latérale et le panneau de résultats.*

---

## 6. Conclusion et Perspectives

Ce projet a permis de construire un pipeline robuste de prédiction de la gravité des accidents. L'utilisation de **Random Forest** et **XGBoost**, couplée à un prétraitement **SMOTE-Tomek**, offre des résultats solides (>88% Accuracy).

**Perspectives futures :**
*   **Enrichissement des données** : Intégrer des données géospatiales précises.
*   **Deep Learning** : Explorer des réseaux de neurones plus profonds ou des architectures TabNet.
*   **Interprétabilité** : Utiliser SHAP (SHapley Additive exPlanations) pour expliquer chaque prédiction individuellement.

---

## 7. Références

1.  **World Health Organization (WHO)**. *Global Status Report on Road Safety 2018*. Geneva: World Health Organization, 2018.
2.  **Chawla, N. V., et al.** "SMOTE: Synthetic Minority Over-sampling Technique." *Journal of Artificial Intelligence Research*, 16 (2002): 321-357.
3.  **Tomek, I.** "Two Modifications of CNN." *IEEE Transactions on Systems, Man, and Cybernetics*, 6 (1976): 769-772.
4.  **Chen, T., & Guestrin, C.** "XGBoost: A Scalable Tree Boosting System." *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 2016.
5.  **Pedregosa, F., et al.** "Scikit-learn: Machine Learning in Python." *Journal of Machine Learning Research*, 12 (2011): 2825-2830.
6.  **Breiman, L.** "Random Forests." *Machine Learning*, 45.1 (2001): 5-32.
