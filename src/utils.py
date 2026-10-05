"""
Fonctions utilitaires pour le projet ML RTA.

Ce module fournit des fonctions pour la sauvegarde/chargement de fichiers,
la visualisation des résultats et le formatage de l'affichage console.
"""

import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path


def save_pickle(obj, filepath):
    """
    Sauvegarde un objet Python en format pickle.
    
    Args:
        obj: Objet à sauvegarder (modèle, encodeur, etc.)
        filepath: Chemin de destination
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    print(f"  💾 Sauvegardé: {filepath}")


def load_pickle(filepath):
    """
    Charge un objet Python depuis un fichier pickle.
    
    Args:
        filepath: Chemin du fichier pickle
        
    Returns:
        L'objet chargé
    """
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    print(f"  📂 Chargé: {filepath}")
    return obj


def save_dataframe(df, filepath):
    """
    Sauvegarde un DataFrame en format CSV.
    
    Args:
        df: DataFrame à sauvegarder
        filepath: Chemin de destination
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(filepath, index=False)
    print(f"  💾 Sauvegardé: {filepath}")


def plot_confusion_matrix(cm, class_names, title, save_path=None):
    """
    Affiche et sauvegarde une matrice de confusion normalisée.
    
    Args:
        cm: Matrice de confusion brute
        class_names: Noms des classes
        title: Titre du graphique
        save_path: Chemin de sauvegarde (optionnel)
    """
    plt.figure(figsize=(8, 6))
    
    # Normalisation par ligne pour obtenir des proportions
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Proportion'})
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Prédiction', fontsize=12)
    plt.ylabel('Vérité', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  📊 Graphique sauvegardé: {save_path}")
    
    plt.show()


def plot_metrics_comparison(comparison_df, save_path=None):
    """
    Crée un graphique comparatif des métriques et temps d'entraînement.
    
    Args:
        comparison_df: DataFrame contenant les résultats de comparaison
        save_path: Chemin de sauvegarde (optionnel)
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    metrics_df = comparison_df[['Modèle', 'Accuracy', 'Precision', 'Recall', 'F1-Score']].set_index('Modèle')
    metrics_df.plot(kind='bar', ax=axes[0], width=0.8)
    axes[0].set_title('Comparaison des Métriques de Performance', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Modèle', fontsize=12)
    axes[0].set_ylabel('Score', fontsize=12)
    axes[0].set_ylim([0, 1])
    axes[0].legend(loc='lower right')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(axis='y', alpha=0.3)
    
    axes[1].bar(comparison_df['Modèle'], comparison_df['Temps (s)'], color='coral')
    axes[1].set_title("Temps d'Entraînement", fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Modèle', fontsize=12)
    axes[1].set_ylabel('Temps (secondes)', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(comparison_df['Temps (s)']):
        axes[1].text(i, v + 0.5, f'{v:.2f}s', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  📊 Graphique sauvegardé: {save_path}")
    
    plt.show()


def print_section_header(title):
    """Affiche un en-tête de section formaté."""
    print("\n" + "="*80)
    print(f"{title}")
    print("="*80 + "\n")


def print_subsection_header(title):
    """Affiche un en-tête de sous-section formaté."""
    print("\n" + "-"*80)
    print(f"{title}")
    print("-"*80 + "\n")


def ensure_dir(directory):
    """Crée un répertoire s'il n'existe pas déjà."""
    Path(directory).mkdir(parents=True, exist_ok=True)
