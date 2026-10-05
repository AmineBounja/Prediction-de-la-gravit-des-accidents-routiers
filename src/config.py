"""
Configuration centralisée du projet ML RTA.

Ce module contient tous les paramètres de configuration pour le pipeline de machine learning,
incluant les chemins, paramètres de prétraitement, hyperparamètres des modèles et grilles d'optimisation.
"""

import os
from pathlib import Path


# ============================================================================
# CHEMINS DU PROJET
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"

RESULTS_DIR = PROJECT_ROOT / "results"
CONFUSION_MATRICES_DIR = RESULTS_DIR / "confusion_matrices"
PERFORMANCE_METRICS_DIR = RESULTS_DIR / "performance_metrics"

DATASET_PATH = RAW_DATA_DIR / "RTA Dataset.csv"


# ============================================================================
# PARAMÈTRES DE SPLIT ET REPRODUCTIBILITÉ
# ============================================================================

TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15
RANDOM_STATE = 42


# ============================================================================
# PARAMÈTRES DE RÉÉCHANTILLONNAGE (SMOTE-TOMEK)
# ============================================================================

SMOTE_K_NEIGHBORS = 3
SMOTE_SAMPLING_STRATEGY = 'auto'
USE_TOMEK = True


# ============================================================================
# PARAMÈTRES D'ANALYSE NON-SUPERVISÉE
# ============================================================================

PCA_N_COMPONENTS = 0.95 
KMEANS_N_CLUSTERS = 3


# ============================================================================
# HYPERPARAMÈTRES DES MODÈLES
# ============================================================================

LR_PARAMS = {
    'max_iter': 1000,
    'random_state': RANDOM_STATE,
    'class_weight': 'balanced',
    'solver': 'lbfgs',
    'multi_class': 'multinomial'
}

RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 20,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': RANDOM_STATE,
    'class_weight': 'balanced',
    'n_jobs': -1
}

SVM_PARAMS = {
    'kernel': 'rbf',
    'C': 1.0,
    'gamma': 'scale',
    'random_state': RANDOM_STATE,
    'class_weight': 'balanced',
    'probability': True
}

KNN_PARAMS = {
    'n_neighbors': 5,
    'weights': 'distance',
    'n_jobs': -1
}

MLP_PARAMS = {
    'hidden_layer_sizes': (100, 50),
    'max_iter': 500,
    'random_state': RANDOM_STATE,
    'early_stopping': True,
    'validation_fraction': 0.1
}

NB_PARAMS = {}

XGB_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'eval_metric': 'mlogloss',
    'scale_pos_weight': 10
}


# ============================================================================
# GRILLES DE PARAMÈTRES POUR L'OPTIMISATION (HPO)
# ============================================================================

LR_PARAM_GRID = {
    'C': [0.01, 0.1, 1.0, 10.0],
    'solver': ['lbfgs', 'liblinear']
}

RF_PARAM_GRID = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10]
}

SVM_PARAM_GRID = {
    'C': [0.1, 1.0, 10.0],
    'gamma': ['scale', 'auto', 0.01, 0.1]
}

KNN_PARAM_GRID = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance']
}

MLP_PARAM_GRID = {
    'hidden_layer_sizes': [(50,), (100,), (100, 50)],
    'alpha': [0.0001, 0.001, 0.01]
}

XGB_PARAM_GRID = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.3]
}


# ============================================================================
# PARAMÈTRES DE GRIDSEARCH
# ============================================================================

GRIDSEARCH_CV = 5
GRIDSEARCH_N_JOBS = -1 
GRIDSEARCH_VERBOSE = 1


# ============================================================================
# PARAMÈTRES DE VISUALISATION
# ============================================================================

FIGURE_DPI = 300
FIGURE_FORMAT = 'png'


# ============================================================================
# LOGGING
# ============================================================================

LOG_LEVEL = 'INFO'
