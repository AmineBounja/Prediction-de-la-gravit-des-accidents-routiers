"""
Pipeline d'entraînement des modèles de classification RTA.

Ce module entraîne les 7 modèles, évalue leurs performances sur le set de validation,
et sélectionne le meilleur modèle pour l'évaluation finale sur le set de test.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import time
import warnings

from src import config
from src.models import ModelFactory
from src.evaluation import ModelEvaluator
from src.utils import load_pickle, save_pickle, save_dataframe, print_section_header, print_subsection_header, plot_confusion_matrix

warnings.filterwarnings('ignore')


def load_processed_data():
    """Charge les données prétraitées depuis le répertoire processed."""
    print_section_header("1️⃣ CHARGEMENT DES DONNÉES")
    
    data_dir = config.PROCESSED_DATA_DIR
    
    if not (data_dir / 'X_train_processed.csv').exists():
        raise FileNotFoundError(f"Données non trouvées dans {data_dir}. Exécutez d'abord preprocessing_pipeline.py")
    
    X_train = pd.read_csv(data_dir / 'X_train_processed.csv')
    y_train = pd.read_csv(data_dir / 'y_train_processed.csv').values.ravel()
    
    X_val = pd.read_csv(data_dir / 'X_val_processed.csv')
    y_val = pd.read_csv(data_dir / 'y_val_processed.csv').values.ravel()
    
    X_test = pd.read_csv(data_dir / 'X_test_processed.csv')
    y_test = pd.read_csv(data_dir / 'y_test_processed.csv').values.ravel()
    
    target_encoder = load_pickle(data_dir / 'target_encoder.pkl')
    feature_names = load_pickle(data_dir / 'feature_names.pkl')
    
    print(f"\n  ▸ Données chargées:")
    print(f"  • Train: {X_train.shape}")
    print(f"  • Val:   {X_val.shape}")
    print(f"  • Test:  {X_test.shape}")
    print(f"  • Classes: {target_encoder.classes_}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, target_encoder, feature_names

def train_models(X_train, y_train, X_val, y_val):
    """
    Entraîne tous les modèles et mesure leurs temps d'exécution.
    Sauvegarde chaque modèle entraîné dans le répertoire models.
    """
    print_section_header("2️⃣ ENTRAÎNEMENT DES MODÈLES")
    
    models = ModelFactory.get_all_models()
    trained_models = {}
    training_times = {}
    
    print(f"  ▸ Modèles à entraîner: {', '.join(models.keys())}\n")
    
    for name, model in models.items():
        print_subsection_header(f"Entraînement: {name}")
        
        start_time = time.time()
        
        try:
            model.fit(X_train, y_train)
            
            end_time = time.time()
            duration = end_time - start_time
            
            trained_models[name] = model
            training_times[name] = duration
            
            train_score = model.score(X_train, y_train)
            val_score = model.score(X_val, y_val)
            
            print(f"  ▸ Terminé en {duration:.2f}s")
            print(f"  • Train Accuracy: {train_score:.4f}")
            print(f"  • Val Accuracy:   {val_score:.4f}")
            
            model_path = config.MODELS_DIR / f"{name.replace(' ', '_').lower()}.pkl"
            save_pickle(model, model_path)
            
        except Exception as e:
            print(f"  ! Erreur lors de l'entraînement de {name}: {e}")
            
    return trained_models, training_times

def evaluate_models(trained_models, training_times, X_val, y_val, target_encoder):
    """
    Évalue tous les modèles sur le set de validation et génère un tableau comparatif.
    Crée les matrices de confusion pour chaque modèle.
    """
    print_section_header("3️⃣ ÉVALUATION ET COMPARAISON")
    
    evaluator = ModelEvaluator(target_encoder)
    results = {}
    
    for name, model in trained_models.items():
        print(f"\nEvaluation de {name}...")
        metrics = evaluator.evaluate_model(model, X_val, y_val, name, "Validation")
        results[name] = metrics
        
    comparison_df = evaluator.compare_models(results, training_times)
    
    print("\n  ▸ TABLEAU COMPARATIF:")
    print(comparison_df.to_string(index=False))
    
    save_dataframe(comparison_df, config.RESULTS_DIR / 'model_comparison.csv')
    
    print("\n  Genereration des matrices de confusion individuelles...")
    for name, metrics in results.items():
        cm_path = config.CONFUSION_MATRICES_DIR / f"confusion_matrix_{name.replace(' ', '_').lower()}.png"
        plot_confusion_matrix(
            metrics['confusion_matrix'],
            target_encoder.classes_,
            f"Confusion Matrix - {name}",
            save_path=cm_path
        )
    
    
    return results, comparison_df, evaluator

def final_evaluation(best_model_name, trained_models, X_test, y_test, evaluator):
    """
    Évalue le meilleur modèle sur le set de test (données jamais vues).
    Sauvegarde le rapport final avec les métriques de test.
    """
    print_section_header("4️⃣ ÉVALUATION FINALE (TEST SET)")
    
    best_model = trained_models[best_model_name]
    print(f"  ▸ Meilleur modèle sélectionné: {best_model_name}")
    
    test_metrics = evaluator.evaluate_model(best_model, X_test, y_test, best_model_name, "Test")
    
    report = {
        'best_model': best_model_name,
        'test_metrics': test_metrics
    }
    save_pickle(report, config.RESULTS_DIR / 'final_report.pkl')
    
    print("\n  ▸ Évaluation finale terminée et sauvegardée")

def main():
    print("🚀 DÉMARRAGE DU PIPELINE D'ENTRAÎNEMENT\n")
    
    X_train, y_train, X_val, y_val, X_test, y_test, target_encoder, feature_names = load_processed_data()
    
    trained_models, training_times = train_models(X_train, y_train, X_val, y_val)
    
    if not trained_models:
        print("  ! Aucun modèle n'a été entraîné avec succès.")
        return
    
    results, comparison_df, evaluator = evaluate_models(
        trained_models, training_times, X_val, y_val, target_encoder
    )
    
    # Sélection du meilleur modèle basé sur le F1-Score
    best_model_name = comparison_df.iloc[0]['Modèle']
    
    final_evaluation(best_model_name, trained_models, X_test, y_test, evaluator)
    
    print("\n✅ PIPELINE D'ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")

if __name__ == "__main__":
    main()
