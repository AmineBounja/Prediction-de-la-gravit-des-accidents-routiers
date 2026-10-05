"""
Module d'évaluation des modèles de classification.

Fournit la classe ModelEvaluator pour calculer les métriques de performance,
générer des rapports de classification et comparer les modèles.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

from src.utils import plot_confusion_matrix


class ModelEvaluator:
    """
    Évaluateur de modèles avec calcul de métriques et génération de rapports.
    
    Utilise des métriques 'weighted' pour tenir compte du déséquilibre des classes.
    """
    
    def __init__(self, target_encoder):
        """Initialise l'évaluateur avec l'encodeur de la variable cible."""
        self.target_encoder = target_encoder
        self.class_names = target_encoder.classes_
    
    def evaluate_model(self, model, X, y, model_name, dataset_name="Validation"):
        """
        Évalue un modèle et affiche un rapport détaillé.
        
        Calcule les métriques avec average='weighted' pour gérer le déséquilibre.
        """
        y_pred = model.predict(X)
        
        # Métriques weighted pour tenir compte du déséquilibre des classes
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
        
        try:
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X)
                roc_auc = roc_auc_score(y, y_pred_proba, multi_class='ovr', average='weighted')
            else:
                roc_auc = None
        except:
            roc_auc = None
        
        cm = confusion_matrix(y, y_pred)
        
        print(f"\n{'='*80}")
        print(f"  📊 ÉVALUATION: {model_name} - {dataset_name}")
        print(f"{'='*80}")
        print(f"\n  📈 Métriques:")
        print(f"   • Accuracy:  {accuracy:.4f}")
        print(f"   • Precision: {precision:.4f}")
        print(f"   • Recall:    {recall:.4f}")
        print(f"   • F1-Score:  {f1:.4f}")
        if roc_auc is not None:
            print(f"   • ROC-AUC:   {roc_auc:.4f}")
        
        print(f"\n  📋 Rapport de Classification:")
        print(classification_report(y, y_pred, target_names=self.class_names, zero_division=0))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'y_pred': y_pred
        }
    
    def compare_models(self, results_dict, training_times):
        """
        Crée un DataFrame comparatif des performances de tous les modèles.
        Trié par F1-Score décroissant.
        """
        comparison_data = []
        
        for name, results in results_dict.items():
            row = {
                'Modèle': name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score'],
                'Temps (s)': training_times.get(name, 0)
            }
            
            if results['roc_auc'] is not None:
                row['ROC-AUC'] = results['roc_auc']
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
        
        return comparison_df
    
    def get_best_model(self, results_dict, metric='f1_score'):
        """Retourne le nom du meilleur modèle selon la métrique spécifiée."""
        best_model = max(results_dict.items(), key=lambda x: x[1][metric])
        return best_model[0]
    
    def plot_all_confusion_matrices(self, results_dict, save_dir=None):
        """Génère un graphique avec toutes les matrices de confusion côte à côte."""
        import matplotlib.pyplot as plt
        
        n_models = len(results_dict)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (name, results) in enumerate(results_dict.items()):
            cm = results['confusion_matrix']
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            import seaborn as sns
            sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                       xticklabels=self.class_names,
                       yticklabels=self.class_names,
                       ax=axes[idx], cbar_kws={'label': 'Proportion'})
            
            axes[idx].set_title(f'{name}\nAccuracy: {results["accuracy"]:.4f}',
                              fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Prédiction', fontsize=10)
            axes[idx].set_ylabel('Vérité', fontsize=10)
        
        plt.tight_layout()
        
        if save_dir:
            from pathlib import Path
            save_path = Path(save_dir) / 'confusion_matrices.png'
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"▸ Matrices de confusion sauvegardées: {save_path}")
        
        plt.show()
