"""
Pipeline de prétraitement des données RTA.

Ce module exécute toutes les étapes de préparation des données :
nettoyage, traitement des outliers, feature engineering, encodage,
scaling et rééchantillonnage SMOTE-Tomek.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from src import config
from src.utils import save_pickle, save_dataframe, print_section_header, print_subsection_header, ensure_dir

warnings.filterwarnings('ignore')


def load_data():
    """Charge le dataset RTA depuis le fichier CSV."""
    print_section_header("1️⃣ CHARGEMENT DES DONNÉES")
    
    if not config.DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset non trouvé: {config.DATASET_PATH}")
        
    df = pd.read_csv(config.DATASET_PATH)
    print(f"  ▸ Dataset chargé: {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df

def clean_data(df):
    """
    Nettoie les données : suppression des duplicatas, standardisation des valeurs manquantes,
    et suppression des colonnes avec plus de 50% de valeurs manquantes.
    """
    print_section_header("2️⃣ NETTOYAGE DES DONNÉES")
    
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"  • Suppression de {duplicates} lignes dupliquées")
        df = df.drop_duplicates()
        
    print("  • Nettoyage des chaînes et standardisation des valeurs manquantes")
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace(['Unknown', 'unknown', 'na', 'Na', 'NA', ''], np.nan)
        
    missing_pct = (df.isnull().sum() / len(df)) * 100
    cols_to_drop = missing_pct[missing_pct > 50].index.tolist()
    
    if cols_to_drop:
        print(f"  • Suppression de {len(cols_to_drop)} colonnes (>50% missing): {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
        
    print(f"  ▸ Dataset après nettoyage: {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df

def drop_unnecessary_columns(df):
    print_section_header("3️⃣ SUPPRESSION DES COLONNES INUTILES")
    cols_to_drop = [
        'Casualty_class',           # Redondant avec d'autres variables
        'Casualty_severity',        # Risque de data leakage (trop corrélé à la cible)
        'Defect_of_vehicle',
        'Fitness_of_casuality',
        'Light_conditions',
        'Owner_of_vehicle',
        'Pedestrian_movement',
        'Road_surface_conditions',
        'Road_surface_type',
        'Vehicle_movement',
        'Weather_conditions',
        'Work_of_casuality',
        'Lanes_or_Medians',
    ]
    
    cols_to_drop_existing = [col for col in cols_to_drop if col in df.columns]
    
    if cols_to_drop_existing:
        print(f"  • Suppression de {len(cols_to_drop_existing)} colonnes inutiles:")
        for col in cols_to_drop_existing:
            print(f"    - {col}")
        df = df.drop(columns=cols_to_drop_existing)
    else:
        print("  • Aucune colonne à supprimer.")
    
    print(f"  ▸ Dataset après suppression: {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df

def treat_outliers(df):
    """
    Détecte et traite les outliers avec la méthode IQR (Interquartile Range).
    Les valeurs sont écrêtées aux bornes [Q1-1.5*IQR, Q3+1.5*IQR].
    Génère des visualisations avant/après pour chaque colonne traitée.
    """
    print_section_header("4️⃣ TRAITEMENT DES OUTLIERS (IQR)")
    
    num_cols_outliers = [
        'Number_of_casualties',
        'Number_of_vehicles_involved'
    ]
    
    num_cols_existing = [
        col for col in num_cols_outliers 
        if col in df.columns and df[col].dtype in ['int64', 'float64']
    ]
    
    if not num_cols_existing:
        print("  • Aucune colonne numérique à traiter (liste vide ou colonnes inexistantes)")
        print(f"  ▸ Dataset inchangé: {df.shape[0]} lignes, {df.shape[1]} colonnes")
        return df
    
    print(f"  • Traitement des outliers pour {len(num_cols_existing)} colonnes numériques:")
    
    outliers_summary = []
    
    for col in num_cols_existing:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Borne inférieure à 0 pour les variables de comptage
        if lower_bound < 0:
            lower_bound = 0
        
        outliers_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        
        if outliers_count > 0:
            viz_dir = config.RESULTS_DIR / 'visualizations' / 'outliers'
            ensure_dir(viz_dir)
            
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=df[col], color='skyblue')
            plt.title(f"Distribution de {col} (Avant traitement)")
            plt.savefig(viz_dir / f"{col}_before.png", dpi=300)
            plt.close()
            
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=df[col], color='lightgreen')
            plt.title(f"Distribution de {col} (Après traitement)")
            plt.savefig(viz_dir / f"{col}_after.png", dpi=300)
            plt.close()
            
            print(f"    📊 Graphiques sauvegardés dans: {viz_dir}")

            outliers_summary.append({
                'Colonne': col,
                'Q1': Q1,
                'Q3': Q3,
                'IQR': IQR,
                'Borne_inf': lower_bound,
                'Borne_sup': upper_bound,
                'Outliers_traités': outliers_count
            })
            print(f"    - {col}: {outliers_count} outliers traités (bornes: [{lower_bound:.2f}, {upper_bound:.2f}])")
        else:
            print(f"    - {col}: Aucun outlier détecté")
    
    if outliers_summary:
        print(f"\n  • Résumé: {sum(s['Outliers_traités'] for s in outliers_summary)} valeurs outliers écrêtées au total")
    
    print(f"  ▸ Traitement des outliers terminé: {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df

def feature_engineering(df):
    print_section_header("5️⃣ FEATURE ENGINEERING")
    
    if 'Time' in df.columns:
        print("  • Création de Hour_of_Day à partir de Time")
        df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S', errors='coerce')
        df['Hour_of_Day'] = df['Time'].dt.hour
        
        df = df.drop(columns=['Time'])
        
    print(f"  ▸ Feature engineering terminé: {df.shape[1]} colonnes")
    return df

def prepare_splits(df):
    print_section_header("6️⃣ PRÉPARATION DES SPLITS")
    
    target_col = 'Accident_severity'
    if target_col not in df.columns:
        raise ValueError(f"Colonne cible '{target_col}' non trouvée")
        
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    print("  • Encodage de la variable cible")
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)
    
    print(f"   Classes: {le_target.classes_}")
    
    print(f"  • Split Train/Val/Test ({config.TRAIN_SIZE}/{config.VAL_SIZE}/{config.TEST_SIZE})")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_encoded, 
        test_size=(1 - config.TRAIN_SIZE), 
        random_state=config.RANDOM_STATE, 
        stratify=y_encoded
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, 
        test_size=0.5, 
        random_state=config.RANDOM_STATE, 
        stratify=y_temp
    )
    
    print(f"   Train: {X_train.shape[0]} samples")
    print(f"   Val:   {X_val.shape[0]} samples")
    print(f"   Test:  {X_test.shape[0]} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, le_target

def handle_rare_categories(X_train, X_val, X_test):
    """
    Regroupe les catégories rares (< 5% de fréquence) en une catégorie 'Other'.
    Prévient les problèmes d'encodage avec des catégories peu représentées.
    """
    print_section_header("7️⃣ GESTION DES CATÉGORIES RARES")
    
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    print(f"  ▸ Traitement de {len(categorical_cols)} colonnes catégorielles")
    
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()
    
    for col in categorical_cols:
        value_counts = X_train[col].value_counts(normalize=True)
        rare_cats = value_counts[value_counts < 0.05].index.tolist()
        
        valid_cats = value_counts[value_counts >= 0.05].index.tolist()
        
        if rare_cats:
            X_train[col] = X_train[col].replace(rare_cats, 'Other')
            
            X_val.loc[~X_val[col].isin(valid_cats), col] = 'Other'
            X_test.loc[~X_test[col].isin(valid_cats), col] = 'Other'
            
    print("  ▸ Regroupement des catégories rares terminé")
    return X_train, X_val, X_test

def process_features(X_train, X_val, X_test):
    """
    Encode les variables catégorielles et applique le pipeline de preprocessing.
    
    Pipeline:
    1. LabelEncoder pour les catégorielles (fit sur train uniquement)
    2. SimpleImputer (médiane) pour les valeurs manquantes
    3. RobustScaler pour la normalisation (résistant aux outliers)
    """
    print_section_header("8️⃣ PREPROCESSING DES FEATURES")
    
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"  • {len(categorical_cols)} colonnes catégorielles")
    print(f"  • {len(numeric_cols)} colonnes numériques")
    
    print("  • Encodage des variables catégorielles (Fit sur Train uniquement)")
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        
        X_train[col] = X_train[col].fillna('Missing')
        X_val[col] = X_val[col].fillna('Missing')
        X_test[col] = X_test[col].fillna('Missing')
        
        le.fit(X_train[col])
        
        X_train[col] = le.transform(X_train[col])
        
        # Fonction pour gérer les catégories inconnues dans val/test
        def safe_transform(encoder, data):
            known_classes = set(encoder.classes_)
            data_set = set(data)
            unknown_classes = data_set - known_classes
            
            if unknown_classes:
                replacement = encoder.classes_[0] 
                data = data.replace(list(unknown_classes), replacement)
            
            return encoder.transform(data)
            
        X_val[col] = safe_transform(le, X_val[col])
        X_test[col] = safe_transform(le, X_test[col])
        
        label_encoders[col] = le
        
    print("  • Application du Pipeline (Imputer + RobustScaler)")
    preprocessor = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])
    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    
    feature_names = X_train.columns.tolist()
    
    return X_train_processed, X_val_processed, X_test_processed, preprocessor, label_encoders, feature_names

def apply_smote(X_train, y_train):
    """
    Applique SMOTE-Tomek pour rééquilibrer les classes.
    
    SMOTE: Génère des exemples synthétiques pour les classes minoritaires
    Tomek Links: Nettoie les frontières de décision en supprimant les paires ambiguës
    
    k_neighbors réduit à 3 car la classe Fatal a très peu d'exemples.
    """
    print_section_header("9️⃣ GESTION DU DÉSÉQUILIBRE (SMOTE + TOMEK)")
    
    print("Distribution avant SMOTE:")
    unique, counts = np.unique(y_train, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"   Classe {u}: {c}")
        
    if config.USE_TOMEK:
        from imblearn.combine import SMOTETomek
        from imblearn.under_sampling import TomekLinks
        
        print("\n  • Utilisation de SMOTE + Tomek Links")
        print(f"  • k_neighbors = {config.SMOTE_K_NEIGHBORS} (adapté aux classes rares)")
        
        smote_tomek = SMOTETomek(
            smote=SMOTE(
                sampling_strategy=config.SMOTE_SAMPLING_STRATEGY,
                k_neighbors=config.SMOTE_K_NEIGHBORS,
                random_state=config.RANDOM_STATE
            ),
            tomek=TomekLinks(sampling_strategy='majority')
        )
        X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)
    else:
        smote = SMOTE(
            random_state=config.RANDOM_STATE, 
            k_neighbors=config.SMOTE_K_NEIGHBORS
        )
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    print("\nDistribution après SMOTE" + (" + Tomek" if config.USE_TOMEK else "") + ":")
    unique, counts = np.unique(y_resampled, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"   Classe {u}: {c}")
    
    print(f"\n  • Exemples totaux: {len(X_train)} → {len(X_resampled)}")
    print(f"  • Exemples ajoutés: {len(X_resampled) - len(X_train)}")
    
    return X_resampled, y_resampled

def save_processed_data(X_train, y_train, X_val, y_val, X_test, y_test, 
                       feature_names, preprocessor, label_encoders, target_encoder):
    print_section_header("🔟 SAUVEGARDE DES DONNÉES")
    
    output_dir = config.PROCESSED_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    y_train_df = pd.DataFrame(y_train, columns=['Accident_severity'])
    
    X_val_df = pd.DataFrame(X_val, columns=feature_names)
    y_val_df = pd.DataFrame(y_val, columns=['Accident_severity'])
    
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    y_test_df = pd.DataFrame(y_test, columns=['Accident_severity'])
    
    save_dataframe(X_train_df, output_dir / 'X_train_processed.csv')
    save_dataframe(y_train_df, output_dir / 'y_train_processed.csv')
    save_dataframe(X_val_df, output_dir / 'X_val_processed.csv')
    save_dataframe(y_val_df, output_dir / 'y_val_processed.csv')
    save_dataframe(X_test_df, output_dir / 'X_test_processed.csv')
    save_dataframe(y_test_df, output_dir / 'y_test_processed.csv')
    
    save_pickle(preprocessor, output_dir / 'preprocessor.pkl')
    save_pickle(label_encoders, output_dir / 'label_encoders.pkl')
    save_pickle(target_encoder, output_dir / 'target_encoder.pkl')
    save_pickle(feature_names, output_dir / 'feature_names.pkl')
    
    print(f"\n  ▸ Tout a été sauvegardé dans: {output_dir}")

def main():
    """Exécute le pipeline complet de prétraitement."""
    print("🚀 DÉMARRAGE DU PIPELINE DE PREPROCESSING\n")
    
    df = load_data()
    
    df = clean_data(df)
    
    df = drop_unnecessary_columns(df)
    
    df = treat_outliers(df)
    
    df = feature_engineering(df)
    
    X_train, X_val, X_test, y_train, y_val, y_test, le_target = prepare_splits(df)
    
    X_train, X_val, X_test = handle_rare_categories(X_train, X_val, X_test)
    
    X_train_proc, X_val_proc, X_test_proc, preprocessor, label_encoders, feature_names = process_features(
        X_train, X_val, X_test
    )
    
    X_train_resampled, y_train_resampled = apply_smote(X_train_proc, y_train)
    
    save_processed_data(
        X_train_resampled, y_train_resampled,
        X_val_proc, y_val,
        X_test_proc, y_test,
        feature_names, preprocessor, label_encoders, le_target
    )
    
    print("\n✅ PREPROCESSING TERMINÉ AVEC SUCCÈS!")

if __name__ == "__main__":
    main()
