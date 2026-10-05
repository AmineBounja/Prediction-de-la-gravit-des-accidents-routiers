"""
Application Streamlit pour la prédiction de la gravité des accidents routiers (RTA).

Cette application permet de charger un modèle ML entraîné et de faire des prédictions
en temps réel sur la gravité des accidents (Fatal, Serious, Slight).
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Ajouter le répertoire parent au path pour importer les modules src
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src import config

# Configuration de la page
st.set_page_config(
    page_title="RTA Prediction | AI Safety",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour un design moderne
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Couleurs Principales */
    :root {
        --primary-color: #4F46E5;
        --secondary-color: #10B981;
        --bg-color: #F3F4F6;
        --card-bg: #FFFFFF;
        --text-color: #1F2937;
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 {
        color: white !important;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Card Styling */
    .stCard {
        background-color: var(--card-bg);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        border: 1px solid #E5E7EB;
    }
    
    /* Input Section Headers */
    .input-section-header {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left-width: 5px;
        border-left-style: solid;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s;
    }
    .input-section-header:hover {
        transform: translateX(5px);
    }
    .input-section-header h4 {
        margin: 0;
        font-weight: 700;
        font-size: 1.1rem;
    }
    .input-section-header p {
        margin: 0.5rem 0 0 0;
        font-size: 0.9rem;
        opacity: 0.8;
    }

    /* Button Styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        color: white;
        font-weight: 600;
        padding: 1rem;
        border-radius: 10px;
        border: none;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 15px rgba(79, 70, 229, 0.4);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(79, 70, 229, 0.6);
    }
    
    /* Metric Cards */
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
        color: var(--primary-color);
        font-weight: 700;
    }
    
    /* Prediction Result */
    .prediction-box {
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        color: white;
        margin: 2rem 0;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        animation: slideUp 0.6s cubic-bezier(0.16, 1, 0.3, 1);
        position: relative;
        overflow: hidden;
    }
    
    .prediction-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(rgba(255,255,255,0.1), rgba(255,255,255,0));
        pointer-events: none;
    }

    /* Progress Bar */
    .prob-container {
        background-color: #F3F4F6;
        border-radius: 8px;
        overflow: hidden;
        margin-top: 5px;
        height: 10px;
        width: 100%;
    }
    .prob-bar {
        height: 100%;
        border-radius: 8px;
        transition: width 1s ease-out;
    }
    
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
""", unsafe_allow_html=True)

def load_resources(model_path):
    """
    Charge le modèle et toutes les ressources nécessaires pour le preprocessing.
    
    Args:
        model_path: Chemin vers le fichier .pkl du modèle
        
    Returns:
        dict: Dictionnaire contenant le modèle et les ressources de preprocessing
    """
    resources = {}
    try:
        data_dir = config.PROCESSED_DATA_DIR
        
        # Charger le modèle
        if model_path:
            with open(model_path, 'rb') as f:
                resources['model'] = pickle.load(f)
            
            # Déterminer le nom du modèle
            if 'optimized' in model_path.name:
                resources['model_name'] = f"Optimized: {model_path.stem.replace('_optimized', '').replace('_', ' ').title()}"
            elif model_path.name == 'best_model.pkl':
                resources['model_name'] = 'Best Model (Standard)'
            else:
                resources['model_name'] = model_path.stem.replace('_', ' ').title()
        else:
            return None
        
        # Charger les ressources de preprocessing
        with open(data_dir / 'preprocessor.pkl', 'rb') as f:
            resources['preprocessor'] = pickle.load(f)
        with open(data_dir / 'label_encoders.pkl', 'rb') as f:
            resources['label_encoders'] = pickle.load(f)
        with open(data_dir / 'target_encoder.pkl', 'rb') as f:
            resources['target_encoder'] = pickle.load(f)
        with open(data_dir / 'feature_names.pkl', 'rb') as f:
            resources['feature_names'] = pickle.load(f)
            
        return resources
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement des ressources: {e}")
        return None

def get_available_models():
    """Récupère la liste de tous les modèles disponibles."""
    models = {}
    
    # Modèles optimisés
    optimized_dir = config.MODELS_DIR / 'optimized'
    if optimized_dir.exists():
        for p in optimized_dir.glob('*_optimized.pkl'):
            name = f"Optimized: {p.stem.replace('_optimized', '').replace('_', ' ').title()}"
            models[name] = p
            
    # Best model
    best_model_path = config.MODELS_DIR / 'best_model.pkl'
    if best_model_path.exists():
        models["Best Model (Standard)"] = best_model_path
        
    # Modèles standards
    for p in config.MODELS_DIR.glob('*.pkl'):
        if p.name != 'best_model.pkl' and 'optimized' not in p.name:
            name = f"{p.stem.replace('_', ' ').title()}"
            models[name] = p
            
    return models

def main():
    """Fonction principale de l'application Streamlit."""
    
    # Header
    st.markdown("""
        <div class="main-header">
            <h1>🚗 RTA AI Safety System</h1>
            <p>Système intelligent de prédiction de la gravité des accidents routiers basé sur le Machine Learning.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar - Sélection du modèle
    st.sidebar.title("⚙️ Configuration")
    
    available_models = get_available_models()
    
    if not available_models:
        st.error("❌ Aucun modèle trouvé. Veuillez d'abord entraîner les modèles.")
        return
    
    selected_model_name = st.sidebar.selectbox(
        "Choisir un modèle",
        options=list(available_models.keys()),
        index=0
    )
    
    model_path = available_models[selected_model_name]
    
    # Charger les ressources
    with st.spinner("🔄 Chargement du modèle..."):
        resources = load_resources(model_path)
    
    if not resources:
        st.error("❌ Impossible de charger les ressources.")
        return
    
    st.sidebar.success(f"✅ Modèle chargé: {resources['model_name']}")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Prédiction", "📈 Performance", "ℹ️ À propos"])
    
    with tab1:
        st.header("🔮 Faire une Prédiction")
        
        # Message d'introduction avec style
        st.markdown("""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;">
                <h3 style="color: white; margin: 0;">💡 Guide de Prédiction</h3>
                <p style="color: white; opacity: 0.9; margin: 0.5rem 0 0 0;">
                    Remplissez les informations ci-dessous pour obtenir une prédiction instantanée 
                    de la gravité d'un accident routier basée sur l'intelligence artificielle.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Formulaire de prédiction avec design amélioré
        st.markdown("### 📋 Informations de l'Accident")
        
        col1, col2, col3 = st.columns(3, gap="large")
        
        with col1:
            st.markdown("""
                <div class="input-section-header" style="background-color: #F0F9FF; border-left-color: #3B82F6;">
                    <h4 style="color: #1E40AF;">🕐 Informations Temporelles</h4>
                    <p style="color: #64748B;">Quand l'accident s'est-il produit ?</p>
                </div>
            """, unsafe_allow_html=True)
            
            hour = st.slider("⏰ Heure de l'accident", 0, 23, 12, help="Heure de la journée (0-23h)")
            day_of_week = st.selectbox("📅 Jour de la semaine", 
                                       ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                                       help="Jour où l'accident s'est produit")
            
            st.markdown("---")
            
            st.markdown("""
                <div class="input-section-header" style="background-color: #FEF3C7; border-left-color: #F59E0B;">
                    <h4 style="color: #92400E;">🚗 Informations Véhicule</h4>
                    <p style="color: #64748B;">Caractéristiques du véhicule impliqué</p>
                </div>
            """, unsafe_allow_html=True)
            
            num_vehicles = st.number_input("🚙 Nombre de véhicules", 1, 10, 2, 
                                          help="Nombre total de véhicules impliqués")
            vehicle_type = st.selectbox("🚛 Type de véhicule",
                                       ["Automobile", "Public (> 45 seats)", "Lorry (41?100Q)", "Other", 
                                        "Lorry (11?40Q)", "Long lorry", "Motorcycle", "Stationwagen", 
                                        "Ridden horse", "Turbo", "Bajaj", "Bicycle"],
                                       help="Type de véhicule principal")
            service_year = st.selectbox("📆 Années de service", 
                                       ["Below 1yr", "1-2yr", "2-5yrs", "5-10yrs", "Above 10yr"],
                                       help="Âge du véhicule")
        
        with col2:
            st.markdown("""
                <div class="input-section-header" style="background-color: #F0FDF4; border-left-color: #10B981;">
                    <h4 style="color: #065F46;">👤 Profil du Conducteur</h4>
                    <p style="color: #64748B;">Informations sur le conducteur</p>
                </div>
            """, unsafe_allow_html=True)
            
            driver_age = st.selectbox("🎂 Tranche d'âge",
                                     ["18-30", "31-50", "Over 51", "Under 18", "Unknown"],
                                     help="Âge du conducteur")
            driver_sex = st.selectbox("⚧ Sexe", ["Male", "Female", "Unknown"],
                                     help="Sexe du conducteur")
            driver_experience = st.selectbox("🎓 Expérience de conduite",
                                            ["Below 1yr", "1-2yr", "2-5yr", "5-10yr", "Above 10yr", "No Licence", "unknown"],
                                            help="Années d'expérience de conduite")
            education = st.selectbox("📚 Niveau d'éducation",
                                    ["Illiterate", "Writing & reading", "Elementary school", 
                                     "Junior high school", "High school", "Above high school", "Unknown"],
                                    help="Niveau d'éducation du conducteur")
            
        with col3:
            st.markdown("""
                <div class="input-section-header" style="background-color: #FEE2E2; border-left-color: #EF4444;">
                    <h4 style="color: #991B1B;">📍 Contexte de l'Accident</h4>
                    <p style="color: #64748B;">Conditions et circonstances</p>
                </div>
            """, unsafe_allow_html=True)
            
            num_casualties = st.number_input("🚑 Nombre de victimes", 1, 10, 1,
                                            help="Nombre de personnes blessées")
            casualty_sex = st.selectbox("⚧ Sexe de la victime", ["Male", "Female", "Unknown"],
                                       help="Sexe de la victime principale")
            casualty_age = st.selectbox("🎂 Âge de la victime",
                                       ["18-30", "31-50", "Over 51", "Under 18", "Unknown"],
                                       help="Tranche d'âge de la victime")
            
            area = st.selectbox("🏙️ Zone de l'accident", ["Urban", "Rural", "Other"],
                               help="Type de zone géographique")
            road_alignment = st.selectbox("🛣️ Configuration de la route", 
                                         ["Tangent road with flat terrain", 
                                          "Tangent road with mild grade and flat terrain",
                                          "Tangent road with rolling terrain",
                                          "Gentle horizontal curve",
                                          "Escarpments", 
                                          "Steep grade downward with mountainous terrain", 
                                          "Sharp reverse curve"],
                                         help="Type et configuration de la route")
            junction_type = st.selectbox("🚦 Type de jonction",
                                        ["No junction", "Y Shape", "T Shape", "X Shape", 
                                         "Crossing", "O Shape", "Other", "Unknown"],
                                        help="Type d'intersection routière")
            collision_type = st.selectbox("💥 Type de collision",
                                         ["Collision with roadside-parked vehicles",
                                          "Vehicle with vehicle collision",
                                          "Collision with roadside objects",
                                          "Collision with animals",
                                          "Other", "Rollover", "Fall from vehicles",
                                          "With Train", "Unknown"],
                                         help="Type de collision")
            
            with st.expander("⚠️ Cause de l'accident", expanded=False):
                cause = st.selectbox("Sélectionnez la cause principale",
                                    ["No distancing", "Changing lane to the left", "Changing lane to the right",
                                     "Driving carelessly", "Driving at high speed", "Driving to the left",
                                     "Overtaking", "Moving Backward", "Overspeed",
                                     "No priority to vehicle", "No priority to pedestrian",
                                     "Overloading", "Improper parking", "Getting off the vehicle improperly",
                                     "Overturning", "Turnover", 
                                     "Drunk driving", "Driving under the influence of drugs",
                                     "Other", "Unknown"],
                                    label_visibility="collapsed",
                                    help="Cause principale de l'accident")
            
            vehicle_relation = st.selectbox("🔑 Relation avec le véhicule",
                                           ["Owner", "Employee", "Other", "Unknown"],
                                           help="Relation du conducteur avec le véhicule")
        
        # Bouton de prédiction avec style amélioré
        st.markdown("<br>", unsafe_allow_html=True)
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            if st.button("🎯 Prédire la Gravité", use_container_width=True):
                try:
                    # Mapper les jours en anglais
                    day_mapping = {
                        "Monday": "Monday", "Tuesday": "Tuesday", "Wednesday": "Wednesday",
                        "Thursday": "Thursday", "Friday": "Friday", "Saturday": "Saturday", "Sunday": "Sunday"
                    }
                    
                    # Créer un DataFrame avec les données d'entrée (18 colonnes dans l'ordre exact)
                    input_data = pd.DataFrame({
                        'Day_of_week': [day_mapping[day_of_week]],
                        'Age_band_of_driver': [driver_age],
                        'Sex_of_driver': [driver_sex],
                        'Educational_level': [education],
                        'Vehicle_driver_relation': [vehicle_relation],
                        'Driving_experience': [driver_experience],
                        'Type_of_vehicle': [vehicle_type],
                        'Service_year_of_vehicle': [service_year],
                        'Area_accident_occured': [area],
                        'Road_allignment': [road_alignment],
                        'Types_of_Junction': [junction_type],
                        'Type_of_collision': [collision_type],
                        'Number_of_vehicles_involved': [int(num_vehicles)],
                        'Number_of_casualties': [int(num_casualties)],
                        'Sex_of_casualty': [casualty_sex],
                        'Age_band_of_casualty': [casualty_age],
                        'Cause_of_accident': [cause],
                        'Hour_of_Day': [int(hour)]
                    })
                    
                    # Encoder les variables catégorielles
                    label_encoders = resources['label_encoders']
                    for col in input_data.select_dtypes(include=['object']).columns:
                        if col in label_encoders:
                            le = label_encoders[col]
                            # Gérer les valeurs inconnues
                            if input_data[col].iloc[0] in le.classes_:
                                input_data[col] = le.transform(input_data[col])
                            else:
                                # Utiliser la classe la plus fréquente si inconnue
                                input_data[col] = 0
                    
                    # Appliquer le preprocessor
                    preprocessor = resources['preprocessor']
                    X_processed = preprocessor.transform(input_data)
                    
                    # Faire la prédiction
                    model = resources['model']
                    prediction = model.predict(X_processed)[0]
                    probabilities = model.predict_proba(X_processed)[0] if hasattr(model, 'predict_proba') else None
                    
                    # Décoder la prédiction
                    target_encoder = resources['target_encoder']
                    predicted_class = target_encoder.inverse_transform([prediction])[0]
                    
                    # Afficher le résultat avec style
                    if predicted_class == "Fatal":
                        bg_color = "#DC2626"
                        icon = "💀"
                        msg = "ACCIDENT MORTEL"
                    elif predicted_class == "Serious":
                        bg_color = "#F59E0B"
                        icon = "⚠️"
                        msg = "BLESSURES GRAVES"
                    else:
                        bg_color = "#10B981"
                        icon = "✅"
                        msg = "BLESSURES LÉGÈRES"
                    
                    st.markdown(f"""
                        <div class="prediction-box" style="background-color: {bg_color};">
                            <h3 style="margin:0; opacity:0.8; text-transform: uppercase; letter-spacing: 1px;">{msg}</h3>
                            <h1 style="font-size: 4rem; margin: 15px 0; font-weight: 800; text-shadow: 0 2px 5px rgba(0,0,0,0.2);">{icon} {predicted_class}</h1>
                            <div style="background: rgba(255,255,255,0.2); padding: 10px 20px; border-radius: 30px; display: inline-block;">
                                <p style="margin:0; font-weight: 600;">Confiance: <strong>{max(probabilities):.1%}</strong></p>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Afficher les probabilités
                    if probabilities is not None:
                        st.markdown("### 📊 Analyse Détaillée")
                        prob_df = pd.DataFrame({
                            'Classe': target_encoder.classes_,
                            'Probabilité': probabilities
                        }).sort_values('Probabilité', ascending=False)
                        
                        col1, col2, col3 = st.columns(3)
                        
                        # Fonction helper pour les couleurs
                        def get_color(classe):
                            if classe == "Fatal": return "#DC2626"
                            if classe == "Serious": return "#F59E0B"
                            return "#10B981"
                        
                        for idx, (_, row) in enumerate(prob_df.iterrows()):
                            with [col1, col2, col3][idx]:
                                st.markdown(f"""
                                    <div style="margin-bottom: 15px;">
                                        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                                            <span style="font-weight: 600; color:{get_color(row['Classe'])}">{row['Classe']}</span>
                                            <span style="font-weight: 700;">{row['Probabilité']:.1%}</span>
                                        </div>
                                        <div class="prob-container">
                                            <div class="prob-bar" style="width: {row['Probabilité']*100}%; background-color: {get_color(row['Classe'])}"></div>
                                        </div>
                                    </div>
                                """, unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"❌ Erreur lors de la prédiction: {e}")
                    st.info("💡 Assurez-vous que toutes les valeurs sont correctement sélectionnées.")
    
    with tab2:
        st.header("📈 Performance du Modèle")
        
        # Charger les métriques
        metrics = {}
        
        # 1. Essayer de charger depuis le pickle (pour modèles optimisés)
        metrics_path = None
        if 'optimized' in str(model_path):
            metrics_path = model_path.parent / f"{model_path.stem.replace('_optimized', '')}_metrics.pkl"
            if metrics_path.exists():
                with open(metrics_path, 'rb') as f:
                    metrics = pickle.load(f)
        
        # 2. Si pas de métriques pickle, chercher dans le CSV de comparaison (pour modèles standards)
        if not metrics:
            comparison_path = config.RESULTS_DIR / 'model_comparison.csv'
            if comparison_path.exists():
                try:
                    df_comp = pd.read_csv(comparison_path)
                    # Normaliser la première colonne 'Model'
                    df_comp.rename(columns={df_comp.columns[0]: 'Model'}, inplace=True)
                    
                    # Fonction de normalisation pour la comparaison
                    def normalize_name(name):
                        return name.lower().replace(' ', '').replace('_', '').replace('-', '').replace(':', '')

                    # Préparer le nom cible
                    target_name = selected_model_name
                    if "Optimized: " in target_name:
                        target_name = target_name.replace("Optimized: ", "") + " (Optimized)"
                    
                    target_normalized = normalize_name(target_name)
                    
                    # Chercher la ligne correspondante avec normalisation
                    df_comp['Model_Normalized'] = df_comp['Model'].apply(normalize_name)
                    row = df_comp[df_comp['Model_Normalized'] == target_normalized]
                    
                    if not row.empty:
                        # Extraire les métriques dynamiquement
                        row = row.iloc[0]
                        for col in df_comp.columns:
                            if col == 'Model_Normalized': continue
                            
                            c_lower = col.lower()
                            val = row[col]
                            
                            if 'acc' in c_lower: metrics['accuracy'] = val
                            if 'prec' in c_lower: metrics['precision'] = val
                            if 'recall' in c_lower: metrics['recall'] = val
                            if 'f1' in c_lower: metrics['f1'] = val
                except Exception as e:
                    # En cas d'erreur de lecture, on continue sans métriques
                    print(f"Erreur lecture CSV: {e}")
                    pass

        if metrics:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{metrics.get('accuracy', 0):.2%}")
            with col2:
                st.metric("Precision", f"{metrics.get('precision', 0):.2%}")
            with col3:
                st.metric("Recall", f"{metrics.get('recall', 0):.2%}")
            with col4:
                st.metric("F1-Score", f"{metrics.get('f1', 0):.2%}")
        else:
            st.info("ℹ️ Métriques non disponibles pour ce modèle.")
        
        # Matrice de confusion - Chercher dans les deux emplacements
        cm_path = None
        
        # D'abord chercher pour les modèles optimisés
        if 'optimized' in str(model_path):
            cm_path = model_path.parent / f"{model_path.stem.replace('_optimized', '')}_confusion_matrix.png"
        
        # Si pas trouvé, chercher dans results/confusion_matrices/
        if not cm_path or not cm_path.exists():
            # Extraire le nom du modèle
            model_name = model_path.stem.replace('_optimized', '')
            cm_path = config.RESULTS_DIR / 'confusion_matrices' / f"confusion_matrix_{model_name}.png"
        
        if cm_path and cm_path.exists():
            st.subheader("🎯 Matrice de Confusion")
            st.image(str(cm_path), use_container_width=True)
        else:
            st.warning(f"📊 Matrice de confusion non trouvée pour ce modèle.")
            st.info(f"Chemin recherché: {cm_path}")
        
        # Comparaison des modèles
        comparison_path = config.RESULTS_DIR / 'model_comparison.csv'
        if comparison_path.exists():
            st.subheader("📊 Comparaison de Tous les Modèles")
            df_comparison = pd.read_csv(comparison_path)
            st.dataframe(df_comparison, use_container_width=True)
    
    with tab3:
        st.header("ℹ️ À propos du Projet")
        
        st.markdown("""
        ### 🎯 Objectif
        Ce système est une solution complète d'intelligence artificielle pour la prédiction de la gravité des accidents routiers.
        
        ### 📊 Pipeline de Données
        - **Nettoyage & Imputation** : Traitement robuste des valeurs manquantes et aberrantes.
        - **Équilibrage** : Utilisation de **SMOTE-Tomek** pour corriger le déséquilibre des classes (Fatal/Serious/Slight).
        - **Preprocessing** : Standard Scaling et Encodage adaptés aux variables catégorielles complexes.
        
        ### 🤖 Modèles ML
        - **7 Algorithmes Comparés** : XGBoost, Random Forest, SVM, MLP (Neural Network), K-NN, Naive Bayes, Logistic Regression.
        - **Optimisation (HPO)** : Recherche des meilleurs hyperparamètres via **GridSearchCV**.
        - **Exploration (Non-Supervisé)** : Analyse par **PCA** (Réduction de dimension) et **K-Means** (Clustering) pour révéler les structures cachées.
        
        ### 🎨 Fonctionnalités
        - **Prédiction Live** : Estimation instantanée de la gravité avec probabilités.
        - **Transparence** : Visualisation des performances (Matrices de confusion spécifiques par modèle).
        
        ---
        
        **Développé par :**
        - Echaaouer Zakariae
        - Fadwa Errami
        - Bounja Mohammed Amine
        
        **Institution :**
        🏫 École Nationale des Sciences Appliquées de Fès (ENSAF)
        
        **Projet :** Machine Learning - Prédiction de la Gravité des Accidents de la Route
        """)

if __name__ == "__main__":
    main()