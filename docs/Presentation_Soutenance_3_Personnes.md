# 🎓 Structure de Présentation "Soutenance RTA" (3 Intervenants)

Ce document propose une structure complète : **Visuels + Diagrammes + Texte à écrire (Puces) + Discours**.

---

## 👥 Répartition des Rôles

*   **👤 Intervenant 1 :** Introduction, Contexte, Pipeline, Prétraitement, Analyse Non-Supervisée.
*   **👤 Intervenant 2 :** Modélisation (Choix des algos), Stratégie d'Optimisation (HPO), Métriques.
*   **👤 Intervenant 3 :** Résultats détaillés, Analyse critique, Démonstration (Streamlit), Conclusion & Perspectives.

---

# 🟢 PARTIE 1 : Contexte & Exploration (Intervenant 1)

### 📌 Slide 1 : Titre & Introduction
*   **📝 Texte à écrire sur la slide :**
    *   **Titre :** Prédiction de la Gravité des Accidents de la Route
    *   **Sous-titre :** Approche Machine Learning & Gestion du Déséquilibre
    *   **Réalisé par :** Echaaouer Zakariae, Fadwa Errami, Bounja Mohammed Amine
    *   **Encadré par :** [Nom Encadrant]
    *   **Année :** 2024-2025
*   **Visuel :** Logos ENSAF / Université. Fond tech/route.
*   **Speech :** "Bonjour. Notre projet vise à utiliser le Machine Learning pour prédire la gravité des accidents et sauver des vies."

### 📌 Slide 2 : Problématique & Données
*   **📝 Texte à écrire sur la slide :**
    *   **Contexte :** 1.35 Million de morts/an (OMS).
    *   **Dataset :** RTA Dataset (12k accidents).
    *   **Défi Majeur :** Déséquilibre Extrême des classes.
        *   *Slight* : ~85% (Majoritaire)
        *   *Fatal* : ~1.4% (Critique mais rare)
    *   **Objectif :** Ne pas rater les cas graves.
*   **Diagramme Mermaid (Pie) :**
    ```mermaid
    pie title Répartition des Classes
        "Slight (85%)" : 85
        "Serious (13.5%)" : 13.5
        "Fatal (1.5%)" : 1.5
    ```
*   **Speech :** "Le problème est l'invisibilité statistique des accidents mortels. Un modèle standard les ignorerait."

### 📌 Slide 3 : Architecture Technique (Pipeline)
*   **📝 Texte à écrire sur la slide :**
    *   **Approche :** Pipeline Modulaire & Reproductible.
    *   **Workflow :**
        1.  Préparation des données (Cleaning/Encoding).
        2.  Rééquilibrage (SMOTE-Tomek).
        3.  Entraînement (7 modèles).
        4.  Optimisation (GridSearch).
        5.  Déploiement (Streamlit).
*   **Diagramme Mermaid (Flowchart) :**
    ```mermaid
    flowchart LR
        Data --> Preprocessing
        Preprocessing --> SMOTE_Tomek
        SMOTE_Tomek --> Models
        Models --> Optimization
        Optimization --> App
    ```
*   **Speech :** "Voici notre architecture industrielle, de la donnée brute à l'application web."

### 📌 Slide 4 : Prétraitement (Data Cleaning)
*   **📝 Texte à écrire sur la slide :**
    *   **Nettoyage :** Imputation (Valeurs manquantes) & Suppression doublons.
    *   **Outliers :** Traitement via méthode IQR (Interquartile Range).
    *   **Feature Engineering :** Extraction `Hour_of_Day`.
    *   **Encodage :** Label Encoding (Var. Catégorielles).
    *   **Scaling :** RobustScaler (Gestion des valeurs extrêmes).
*   **Diagramme Mermaid (Process) :**
    ```mermaid
    graph TD
        Raw -->|Clean| Imputed
        Imputed -->|IQR| No_Outliers
        No_Outliers -->|LabelEnc| Numeric_Data
    ```
*   **Speech :** "Qualité avant quantité. Nous avons purifié les données pour maximiser l'apprentissage."

### 📌 Slide 5 : Stratégie de Rééquilibrage (SMOTE-Tomek)
*   **📝 Texte à écrire sur la slide :**
    *   **Pourquoi ?** Éviter le biais vers la classe majoritaire "Slight".
    *   **Solution Hybride :**
        *   ➕ **SMOTE :** Création d'exemples synthétiques (Oversampling).
        *   ➖ **Tomek Links :** Suppression du bruit aux frontières (Undersampling).
    *   **Résultat :** Meilleure séparation des classes.
*   **Diagramme Mermaid (Concept) :**
    ```mermaid
    flowchart TD
       Minority -->|SMOTE| Synthetic_Samples
       Majority -->|Tomek| Clean_Borders
       Synthetic_Samples & Clean_Borders --> Balanced_Set
    ```
*   **Speech :** "Nous enrichissons artificiellement les accidents graves pour que le modèle 'apprenne' à les reconnaître."

### 📌 Slide 6 : Analyse Non-Supervisée
*   **📝 Texte à écrire sur la slide :**
    *   **Méthode :** PCA (Analyse en Composantes Principales).
    *   **Constat :** Chevauchement fort des classes.
    *   **Clusters :** K-Means identifie 3 groupes, mais mélangés.
    *   **Conclusion :** Problème non-linéaire complexe → Nécessite des modèles robustes (RF, XGB).
*   **Visuel :** Image PCA 2D (Nuage de points).
*   **Speech :** "L'analyse visuelle prouve qu'il n'y a pas de solution simple séparant les gravités."

---

# 🟠 PARTIE 2 : Modélisation (Intervenant 2)

### 📌 Slide 7 : Benchmark des Modèles
*   **📝 Texte à écrire sur la slide :**
    *   **7 Algorithmes Testés :**
        *   *Linéaire :* Logistic Regression.
        *   *Probabiliste :* Naive Bayes.
        *   *Géométrique :* SVM, KNN.
        *   *Ensemble :* **Random Forest**, **XGBoost**.
        *   *Neural :* MLP.
    *   **Stratégie :** Comparaison large pour trouver le meilleur candidat.
*   **Speech :** "Nous avons exploré tout l'éventail du Machine Learning classique."

### 📌 Slide 8 : Importance des Variables
*   **📝 Texte à écrire sur la slide :**
    *   **Qu'est-ce qui joue ?** (Feature Importance).
    *   **Top 3 Facteurs :**
        1.  `Hour_of_Day` (Moment de l'accident).
        2.  `Age_of_Driver` (Expérience du conducteur).
        3.  `Vehicle_Type` & `Engine_Capacity`.
*   **Diagramme Mermaid (Bar Chart) :**
    ```mermaid
    xychart-beta
        title "Facteurs Clés"
        x-axis [Heure, Age, Moteur]
        bar [30, 25, 15]
    ```
*   **Speech :** "L'analyse montre que l'heure et l'âge sont déterminants dans la gravité."

### 📌 Slide 9 : Optimisation (Hyperparamètres)
*   **📝 Texte à écrire sur la slide :**
    *   **Objectif :** Maximiser le F1-Score.
    *   **Méthode :** **GridSearchCV**.
        *   Validation Croisée (5-folds).
        *   Test exhaustif des paramètres.
    *   **Exemples réglés :**
        *   *RF :* `n_estimators`, `max_depth`.
        *   *XGB :* `learning_rate`, `subsample`.
*   **Speech :** "Nous avons réglé mathématiquement chaque bouton de nos algorithmes pour la performance maximale."

---

# 🔵 PARTIE 3 : Résultats & Conclusion (Intervenant 3)

### 📌 Slide 10 : Résultats Comparatifs
*   **📝 Texte à écrire sur la slide :**
    *   **Métrique Clé :** F1-Score Weighted (Équilibre Précision/Rappel).
    *   **Podium :**
        1.  🥇 **Random Forest (Optimisé) : ~88.3%**
        2.  🥈 XGBoost : ~87.1%
        3.  🥉 MLP : ~84.5%
    *   **Validation :** Modèles robustes et généralisables.
*   **Diagramme Mermaid (Bar Chart) :**
    ```mermaid
    xychart-beta
        title "Performance (Accuracy/F1)"
        x-axis [LogReg, SVM, MLP, XGB, RF]
        bar [77, 79, 84, 87, 88]
    ```
*   **Speech :** "Random Forest est le grand gagnant, alliant précision et robustesse."

### 📌 Slide 11 : Analyse Critique (La Réalité)
*   **📝 Texte à écrire sur la slide :**
    *   **Forces :** Excellente détection des accidents "Slight" et "Serious".
    *   **Limites :** Classe "Fatal" reste difficile (trop peu d'exemples réels).
    *   **Matrice de Confusion :**
        *   Peu de Faux Positifs.
        *   Encore quelques Faux Négatifs sur les cas critiques.
*   **Speech :** "Malgré nos bons scores, la rareté extrême des décès reste un défi statistique."

### 📌 Slide 12 : Démonstration (Streamlit)
*   **📝 Texte à écrire sur la slide :**
    *   **Application Web Interactive.**
    *   **Fonctionnalités :**
        *   Saisie des paramètres (Formulaire).
        *   Prédiction Temps Réel.
        *   Visualisation de Confiance (Probabilités).
    *   **Usage :** Outil d'aide à la décision pour la sécurité routière.
*   **Visuel :** Capture d'écran de l'App.
*   **Speech :** (Pendant la démo) "Démonstration de l'outil en situation réelle."

### 📌 Slide 13 : Conclusion & Perspectives
*   **📝 Texte à écrire sur la slide :**
    *   **Bilan :** Pipeline Complet ✅ | Modèle Performant (88%) ✅ | App Déployée ✅
    *   **Perspectives (Futur) :**
        *   📡 **Données :** Intégration GPS & Météo temps réel.
        *   🧠 **Modèles :** Deep Learning (TabNet), Ensembles avancés.
        *   🔍 **Explicabilité :** SHAP Values pour comprendre chaque décision.
*   **Diagramme Mermaid (Mindmap) :**
    ```mermaid
    mindmap
      root((Futur))
        GPS Data
        Deep Learning
        SHAP Explainability
    ```
*   **Speech :** "Ce projet est une base solide pour le futur de la prévention routière assistée par IA."

### � Slide 14 : Fin
*   **📝 Texte à écrire s*ur la slide :**
    *   *Merci de votre attention.**
    *   Questions ?
*   **Speech :** "Nous sommes à votre disposition pour vos questions."
