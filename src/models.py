"""
Factory de modèles pour le projet ML RTA.

Ce module fournit une interface centralisée pour créer et configurer
les 7 modèles de classification utilisés dans le projet.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

from src.config import (
    LR_PARAMS, RF_PARAMS, SVM_PARAMS, KNN_PARAMS,
    MLP_PARAMS, NB_PARAMS, XGB_PARAMS,
    LR_PARAM_GRID, RF_PARAM_GRID, SVM_PARAM_GRID, KNN_PARAM_GRID,
    MLP_PARAM_GRID, XGB_PARAM_GRID
)


class ModelFactory:
    """Factory pour créer et configurer les modèles de classification."""
    
    @staticmethod
    def get_all_models():
        """
        Retourne tous les modèles configurés avec leurs hyperparamètres par défaut.
        
        Returns:
            dict: Dictionnaire {nom_modèle: instance_modèle}
        """
        models = {
            'Logistic Regression': LogisticRegression(**LR_PARAMS),
            'Random Forest': RandomForestClassifier(**RF_PARAMS),
            'SVM': SVC(**SVM_PARAMS),
            'K-NN': KNeighborsClassifier(**KNN_PARAMS),
            'MLP': MLPClassifier(**MLP_PARAMS),
            'Naive Bayes': GaussianNB(**NB_PARAMS),
            'XGBoost': XGBClassifier(**XGB_PARAMS)
        }
        return models
    
    @staticmethod
    def get_model(model_name):
        """
        Retourne une instance d'un modèle spécifique.
        
        Args:
            model_name: Nom du modèle à créer
            
        Returns:
            Instance du modèle configuré
            
        Raises:
            ValueError: Si le nom du modèle n'est pas reconnu
        """
        models = ModelFactory.get_all_models()
        if model_name not in models:
            raise ValueError(f"Modèle '{model_name}' non trouvé. Modèles disponibles: {list(models.keys())}")
        return models[model_name]
    
    @staticmethod
    def get_param_grid(model_name):
        """
        Retourne la grille de paramètres pour l'optimisation d'un modèle.
        
        Args:
            model_name: Nom du modèle
            
        Returns:
            dict: Grille de paramètres pour GridSearchCV
        """
        param_grids = {
            'Logistic Regression': LR_PARAM_GRID,
            'Random Forest': RF_PARAM_GRID,
            'SVM': SVM_PARAM_GRID,
            'K-NN': KNN_PARAM_GRID,
            'MLP': MLP_PARAM_GRID,
            'XGBoost': XGB_PARAM_GRID,
            'Naive Bayes': {}  # Pas d'hyperparamètres à optimiser
        }
        return param_grids.get(model_name, {})
    
    @staticmethod
    def get_model_info():
        """
        Retourne les métadonnées descriptives de chaque modèle.
        
        Returns:
            dict: Informations sur chaque modèle (type, catégorie, description)
        """
        info = {
            'Logistic Regression': {
                'type': 'Linear',
                'category': 'Classification',
                'description': 'Modèle linéaire de classification'
            },
            'Random Forest': {
                'type': 'Ensemble',
                'category': 'Tree-based',
                'description': 'Ensemble d\'arbres de décision'
            },
            'SVM': {
                'type': 'Kernel-based',
                'category': 'Classification',
                'description': 'Support Vector Machine avec kernel RBF'
            },
            'K-NN': {
                'type': 'Instance-based',
                'category': 'Classification',
                'description': 'K plus proches voisins'
            },
            'MLP': {
                'type': 'Neural Network',
                'category': 'Deep Learning',
                'description': 'Réseau de neurones multi-couches'
            },
            'Naive Bayes': {
                'type': 'Probabilistic',
                'category': 'Bayesian',
                'description': 'Modèle probabiliste bayésien'
            },
            'XGBoost': {
                'type': 'Ensemble',
                'category': 'Boosting',
                'description': 'Gradient Boosting optimisé'
            }
        }
        return info
