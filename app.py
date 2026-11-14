import streamlit as st
import math

# Configuration de la page.
st.set_page_config(
    page_title="SPHEREx Explorer",
    page_icon="🌌",
    layout="wide"
)

# Initialisation des variables de session.
if "mission_launched" not in st.session_state:
    st.session_state.mission_launched = False

if "matching_images" not in st.session_state:
    st.session_state.matching_images = []

if "active_index" not in st.session_state:
    st.session_state.active_index = 0


# Personnalisation graphique.
st.markdown("""
<style>

.stApp {
    background: linear-gradient(140deg, #050D20, #0B1930);
    color: white;
}

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
}

h1, h2, h3, p, label {
    color: white !important;
}

.hero {
    background: linear-gradient(110deg, #102C50, #10203B);
    border: 1px solid #28517B;
    border-radius: 18px;
    padding: 30px;
    margin-bottom: 25px;
}

.hero-title {
    font-size: 38px;
    font-weight: 800;
    color: white;
}

.hero-subtitle {
    color: #A9C8E8;
    font-size: 16px;
    margin-top: 8px;
}

.mission-card {
    background: #10223D;
    border: 1px solid #29466A;
    border-radius: 16px;
    padding: 22px;
    margin-bottom: 20px;
}

.status {
    color: #64D5FF;
    font-weight: bold;
    font-size: 12px;
    letter-spacing: 2px;
}

.stForm {
    background: #10223D;
    border: 1px solid #29466A;
    border-radius: 15px;
    padding: 22px;
}

.stNumberInput input {
    background-color: #193657;
    color: white;
    border: 1px solid #365B80;
    border-radius: 8px;
}

[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(90deg, #087FCB, #20B8ED);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: bold;
    min-height: 48px;
    width: 100%;
}

.stButton button {
    background: #163454;
    color: white;
    border: 1px solid #3476A8;
    border-radius: 10px;
    min-height: 42px;
}

.stButton button:hover {
    background: #1769A3;
    color: white;
    border-color: #55C8FF;
}

[data-testid="stImage"] {
    border: 1px solid #315577;
    border-radius: 14px;
    overflow: hidden;
}

hr {
    border-color: #29466A;
}

</style>
""", unsafe_allow_html=True)


# En-tête du jeu.
st.markdown("""
<div class="hero">
    <div class="status">SPACE EXPLORATION SYSTEM</div>
    <div class="hero-title">🌌 SPHEREx Explorer</div>
    <div class="hero-subtitle">
        Your mission: explore the universe through time.
    </div>
</div>
""", unsafe_allow_html=True)


# Informations sur la mission.
col1, col2 = st.columns(2)

with col1:
    st.markdown(
        '<div class="mission-card">'
        '<div class="status">MISSION STATUS</div>'
        '<h3>Ready for exploration</h3>'
        '<p>Choose your destination in the universe.</p>'
        '</div>',
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        '<div class="mission-card">'
        '<div class="status">DATA SOURCE</div>'
        '<h3>SPHEREx Archive</h3>'
        '<p>Currently using a local test image.</p>'
        '</div>',
        unsafe_allow_html=True
    )


# Formulaire de sélection de destination.
st.markdown("### 🎯 Choose your destination")

with st.form("search_form"):

    col1, col2 = st.columns(2)

    with col1:
        ra = st.number_input(
            "Right Ascension (RA) in degrees",
            min_value=0.0,
            max_value=360.0,
            value=0.0
        )

    with col2:
        dec = st.number_input(
            "Declination (Dec) in degrees",
            min_value=-90.0,
            max_value=90.0,
            value=0.0
        )

    search = st.form_submit_button(
        "🚀 LAUNCH EXPLORATION",
        use_container_width=True
    )


# Images de démonstration et coordonnées fictives.
images = [
    {
        "file": "images/Test.jpg",
        "ra": 10.0,
        "dec": 20.0
    }
]


# Recherche des images correspondantes.
if search:

    st.session_state.mission_launched = True
    st.session_state.active_index = 0

    st.session_state.matching_images = [
        image for image in images
        if math.isclose(ra, image["ra"], abs_tol=0.000001)
        and math.isclose(dec, image["dec"], abs_tol=0.000001)
    ]


# Affichage des résultats de la mission.
if st.session_state.mission_launched:

    results = st.session_state.matching_images

    st.markdown("---")
    st.markdown("### 🪐 Exploration results")

    if results:

        st.success(f"{len(results)} observation(s) found.")

        # Navigation entre les observations.
        if len(results) > 1:

            col1, col2, col3 = st.columns([1, 2, 1])

            with col1:
                if st.button("◀ PREVIOUS", use_container_width=True):
                    st.session_state.active_index = (
                        st.session_state.active_index - 1
                    ) % len(results)

            with col3:
                if st.button("NEXT ▶", use_container_width=True):
                    st.session_state.active_index = (
                        st.session_state.active_index + 1
                    ) % len(results)

            # Sélection d'une observation dans la timeline.
            selected_index = st.slider(
                "Observation timeline",
                min_value=0,
                max_value=len(results) - 1,
                value=st.session_state.active_index,
                format="Observation %d"
            )

            st.session_state.active_index = selected_index

        else:
            st.caption("Observation 1 / 1")

        # Affichage de l'image sélectionnée.
        current_image = results[st.session_state.active_index]

        st.markdown("#### 🔭 Current observation")

        st.image(
            current_image["file"],
            caption="Test image — demonstration only",
            use_container_width=True
        )

        st.caption(
            f"RA: {current_image['ra']}° | "
            f"Dec: {current_image['dec']}°"
        )

    else:
        st.warning(
            "No observation found for these coordinates. "
            "Try RA = 10 and Dec = 20 to test the application."
        )


# Pied de page.
st.markdown("---")
st.caption(
    "SPHEREx Explorer | NASA Space Apps Challenge | Demo version"
)
