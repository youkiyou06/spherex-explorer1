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

# Association de l'image à ses coordonnées célestes.

images = [
    {
        "file": "images/Test.jpg",
        "ra": 10.0,
        "dec": 20.0
    }
]

# Recherche de l'image correspondant aux coordonnées saisies.

if search:

    image_found = False

    for image in images:

        if ra == image["ra"] and dec == image["dec"]:

            st.header("Sky image")

            st.image(
                image["file"],
                caption="SPHEREx observation",
                use_container_width=True
            )

            image_found = True

    if not image_found:
        st.warning("No image found for these coordinates.")
