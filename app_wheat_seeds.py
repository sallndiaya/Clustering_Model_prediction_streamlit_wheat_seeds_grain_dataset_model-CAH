import streamlit as st
import numpy as np
import joblib

# Configuration de la page
st.set_page_config(
    page_title=" Prediction des graines de blé Model CAH ",
    page_icon="🌿",
    layout="centered"
)

# Charger le modèle
artefacts = joblib.load("modele_cah.joblib")

centres_clusters = artefacts["centres_clusters"]
colonnes = artefacts["colonnes"]
valeurs_defaut = artefacts["valeurs_defaut"]
noms_classes = artefacts["noms_classes"]


# Fonction de prédiction
def predire_classe(nouveau_grain):

    grain = np.array(nouveau_grain, dtype=float)

    distances = np.linalg.norm(
        centres_clusters - grain, 
        axis=1
    )

    classe = distances.argmin()

    return classe


# Titre
st.title("🌿 Segmentation des grains")
st.write("Application de classification basée sur un modèle CAH.")


# Saisie des caractéristiques
st.subheader("Caractéristiques du grain")

valeurs = []

for i, col in enumerate(colonnes):

    valeur = st.number_input(
        col,
        value=float(valeurs_defaut[i])
    )

    valeurs.append(valeur)


# Bouton de prédiction
if st.button("Prédire la classe"):

    classe = predire_classe(valeurs)

    nom = noms_classes[classe]

    st.success(f"Ce client appartient à la classe : {nom}")
