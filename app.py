import streamlit as st

# Importation de la bibliothèque Streamlit.

st.title("SPHEREx Explorer")

# Affichage du titre principal.

st.write("Explore the evolution of the sky over time.")

st.header("Search an astronomical region")

# Création d'un formulaire regroupant les coordonnées.

with st.form("search_form"):

    ra = st.number_input(
        "Right Ascension (RA) in degrees",
        min_value=0.0,
        max_value=360.0,
        value=0.0
    )

    dec = st.number_input(
        "Declination (Dec) in degrees",
        min_value=-90.0,
        max_value=90.0,
        value=0.0
    )

    search = st.form_submit_button("Search")

# Affichage des coordonnées après validation du formulaire.

if search:
    st.write("Selected coordinates:")
    st.write("Right Ascension:", ra)
    st.write("Declination:", dec)
