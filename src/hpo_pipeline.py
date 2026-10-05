"""
Pipeline d'optimisation des hyperparamètres (HPO) avec GridSearchCV.

Ce module permet d'optimiser les hyperparamètres des modèles sélectionnés
en utilisant GridSearchCV avec validation croisée et métrique F1 weighted.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import time
import warnings
from sklearn.model_selection import GridSearchCV

from src import config
from src.models import ModelFactory
from src.utils import load_pickle, save_pickle, print_section_header, ensure_dir

warnings.filterwarnings('ignore')

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from src.utils import plot_confusion_matrix


def load_data():
    """Charge les données d'entraînement et de test pour l'optimisation."""
    print_section_header("1️⃣ CHARGEMENT DES DONNÉES")
    
    data_dir = config.PROCESSED_DATA_DIR
    X_train_path = data_dir / 'X_train_processed.csv'
    y_train_path = data_dir / 'y_train_processed.csv'
    X_test_path = data_dir / 'X_test_processed.csv'
    y_test_path = data_dir / 'y_test_processed.csv'
    
    if not X_train_path.exists():
        raise FileNotFoundError(f"Fichier non trouvé: {X_train_path}")
        
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).values.ravel()
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).values.ravel()
    
    target_encoder = load_pickle(data_dir / 'target_encoder.pkl')
    
    print(f"  ▸ Données chargées: Train {X_train.shape}, Test {X_test.shape}")
    return X_train, y_train, X_test, y_test, target_encoder

def evaluate_and_save_metrics(model, X_test, y_test, target_encoder, model_name, save_dir):
    """
    Évalue le modèle optimisé sur le set de test et sauvegarde les métriques.
    Génère également la matrice de confusion.
    """
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1_score': f1_score(y_test, y_pred, average='weighted')
    }
    
    metrics_path = save_dir / f"{model_name.replace(' ', '_').lower()}_metrics.pkl"
    save_pickle(metrics, metrics_path)
    
    cm = confusion_matrix(y_test, y_pred)
    cm_path = save_dir / f"{model_name.replace(' ', '_').lower()}_confusion_matrix.png"
    
    plot_confusion_matrix(
        cm, 
        target_encoder.classes_, 
        f"Confusion Matrix - {model_name} (Optimized)", 
        save_path=cm_path
    )
    
    return metrics

def optimize_model(model_name, X, y):
    """
    Optimise les hyperparamètres d'un modèle avec GridSearchCV.
    
    Utilise la validation croisée 5-fold et optimise pour le F1-Score weighted,
    qui est adapté aux problèmes de classification déséquilibrée.
    """
    print(f"\n  ▸ Optimisation de {model_name}...")
    
    try:
        base_model = ModelFactory.get_model(model_name)
        param_grid = ModelFactory.get_param_grid(model_name)
    except ValueError as e:
        print(f"  ! Erreur: {e}")
        return None, None
        
    if not param_grid:
        print(f"  ! Pas de grille de paramètres définie pour {model_name}. Optimisation ignorée.")
        return base_model, None
        
    print(f"  • Grille de paramètres: {param_grid}")
    
    # GridSearchCV avec F1 weighted pour gérer le déséquilibre des classes
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=config.GRIDSEARCH_CV,
        n_jobs=config.GRIDSEARCH_N_JOBS,
        verbose=config.GRIDSEARCH_VERBOSE,
        scoring='f1_weighted'
    )
    
    start_time = time.time()
    grid_search.fit(X, y)
    duration = time.time() - start_time
    
    print(f"  ▸ Optimisation terminée en {duration:.2f}s")
    print(f"  • Meilleurs paramètres: {grid_search.best_params_}")
    print(f"  • Meilleur score CV (F1 Weighted): {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_

def main():
    print("🎯 DÉMARRAGE DE L'OPTIMISATION DES HYPERPARAMÈTRES (HPO)\n")
    
    X_train, y_train, X_test, y_test, target_encoder = load_data()
    
    models_available = list(ModelFactory.get_all_models().keys())
    
    print("\nModèles disponibles:")
    for i, m in enumerate(models_available):
        print(f"{i+1}. {m}")
        
    print(f"{len(models_available)+1}. TOUS LES MODÈLES")
    
    try:
        user_input = input("\n👉 Choisissez un numéro de modèle à optimiser : ")
        selection = int(user_input)
        
        if 1 <= selection <= len(models_available):
            choice = models_available[selection-1]
        elif selection == len(models_available) + 1:
            choice = "ALL"
        else:
            print("❌ Choix invalide. Utilisation par défaut : XGBoost")
            choice = "XGBoost"
    except ValueError:
        print("❌ Entrée invalide. Utilisation par défaut : XGBoost")
        choice = "XGBoost"
    
    print_section_header(f"2️⃣ OPTIMISATION : {choice}")
    
    optimized_models_dir = config.MODELS_DIR / 'optimized'
    ensure_dir(optimized_models_dir)
    
    if choice == "ALL":
        models_to_run = models_available
    else:
        models_to_run = [choice]
        
    results = []
    
    for name in models_to_run:
        best_model, best_params = optimize_model(name, X_train, y_train)
        
        if best_model:
            model_save_path = optimized_models_dir / f"{name.replace(' ', '_').lower()}_optimized.pkl"
            save_pickle(best_model, model_save_path)
            
            metrics = evaluate_and_save_metrics(
                best_model, X_test, y_test, target_encoder, name, optimized_models_dir
            )
            
            results.append({
                'Model': name,
                'Best Params': best_params,
                'Test F1': metrics['f1_score'],
                'Saved Path': str(model_save_path)
            })
            
            try:
                comparison_csv_path = config.RESULTS_DIR / 'model_comparison.csv'
                if comparison_csv_path.exists():
                    df_comparison = pd.read_csv(comparison_csv_path)
                    
                    new_row = {
                        'Modèle': f"{name} (Optimized)",
                        'Accuracy': metrics['accuracy'],
                        'Precision': metrics['precision'],
                        'Recall': metrics['recall'],
                        'F1-Score': metrics['f1_score'],
                        'Temps (s)': 0, # On ne track pas le temps d'inférence ici
                        'ROC-AUC': 0 # Pas calculé ici pour simplifier
                    }
                    
                    df_comparison = df_comparison[df_comparison['Modèle'] != new_row['Modèle']]
                    
                    df_comparison = pd.concat([df_comparison, pd.DataFrame([new_row])], ignore_index=True)
                    
                    df_comparison = df_comparison.sort_values('F1-Score', ascending=False)
                    
                    df_comparison.to_csv(comparison_csv_path, index=False)
                    print(f"  ▸ Ajouté à {comparison_csv_path}")
            except Exception as e:
                print(f"  ! Impossible de mettre à jour le CSV de comparaison: {e}")
            
    print_section_header("3️⃣ RÉSULTATS HPO")
    for res in results:
        print(f"\n  ▸ {res['Model']}")
        print(f"   Params: {res['Best Params']}")
        print(f"   Test F1: {res['Test F1']:.4f}")
        print(f"   Sauvegardé: {res['Saved Path']}")

if __name__ == "__main__":
    main()
