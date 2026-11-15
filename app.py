
from pathlib import Path
import sqlite3
import json
import hashlib
import warnings

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import s3fs

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from reproject import reproject_interp


# ============================================================
# CONFIGURATION
# ============================================================

BASE = Path(__file__).resolve().parent
DB_PATH = BASE / "spherex_catalog_test.sqlite"
CACHE_DIR = BASE / "spherex_cache"
CACHE_DIR.mkdir(exist_ok=True)

BUCKET = "nasa-irsa-spherex"

st.set_page_config(
    page_title="SPHEREx Explorer",
    page_icon="🌌",
    layout="wide"
)


# ============================================================
# DESIGN
# ============================================================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(140deg, #041329, #0b2445);
    color: #edf3ff;
}
h1, h2, h3 {
    color: #f4c542 !important;
}
p, label, span {
    color: #e5edff;
}
section[data-testid="stSidebar"] {
    background: #07182f;
}
div.stButton > button {
    background: #f4c542;
    color: #07152b;
    border: 0;
    border-radius: 9px;
    font-weight: 700;
}
div.stButton > button:hover {
    background: #ffdc70;
    color: #07152b;
}
.panel {
    background: #102c4c;
    border: 1px solid #cba936;
    border-radius: 13px;
    padding: 19px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATABASE
# ============================================================

def load_database():

    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found: {DB_PATH}"
        )

    with sqlite3.connect(DB_PATH) as connection:
        df = pd.read_sql_query(
            "SELECT * FROM observations",
            connection
        )

    required = {"s3_key", "footprint"}

    if not required.issubset(df.columns):
        raise ValueError(
            f"Missing columns: {required - set(df.columns)}"
        )

    if "status" in df.columns:
        df = df[
            df["status"].fillna("").astype(str).str.upper()
            .isin(["OK", "SUCCESS"])
        ]

    if "mjd_obs" in df.columns:
        df["mjd_obs"] = pd.to_numeric(
            df["mjd_obs"], errors="coerce"
        )
        df = df.sort_values(
            "mjd_obs",
            na_position="last"
        )
    elif "date_obs" in df.columns:
        df = df.sort_values(
            "date_obs",
            na_position="last"
        )

    return df.reset_index(drop=True)


# ============================================================
# FOOTPRINTS
# ============================================================

def parse_footprint(value):

    try:
        polygon = (
            json.loads(value)
            if isinstance(value, str)
            else value
        )

        polygon = np.asarray(polygon, dtype=float)

        if polygon.ndim != 2 or polygon.shape[1] < 2:
            return None

        if len(polygon) < 3:
            return None

        return polygon[:, :2]

    except Exception:
        return None


def get_center(polygon):

    if polygon is None:
        return None

    coordinates = SkyCoord(
        ra=polygon[:, 0] * u.deg,
        dec=polygon[:, 1] * u.deg,
        frame="icrs"
    )

    xyz = np.array([
        np.mean(coordinates.cartesian.x.value),
        np.mean(coordinates.cartesian.y.value),
        np.mean(coordinates.cartesian.z.value)
    ])

    norm = np.linalg.norm(xyz)

    if norm == 0:
        return None

    xyz /= norm

    center = SkyCoord(
        x=xyz[0],
        y=xyz[1],
        z=xyz[2],
        representation_type="cartesian",
        frame="icrs"
    )

    return (
        float(center.spherical.lon.deg),
        float(center.spherical.lat.deg)
    )


def inside_footprint(ra, dec, polygon):

    if polygon is None:
        return False

    x = (polygon[:, 0] - ra + 180) % 360 - 180
    y = polygon[:, 1]

    px = 0.0
    py = float(dec)

    inside = False
    j = len(x) - 1

    for i in range(len(x)):

        xi, yi = x[i], y[i]
        xj, yj = x[j], y[j]

        if (yi > py) != (yj > py):

            crossing = (
                (xj - xi) * (py - yi) /
                (yj - yi) + xi
            )

            if px < crossing:
                inside = not inside

        j = i

    return inside


def get_constellation(ra, dec):

    try:
        coordinate = SkyCoord(
            ra=ra * u.deg,
            dec=dec * u.deg,
            frame="icrs"
        )
        return coordinate.get_constellation()

    except Exception:
        return "Unknown constellation"


# ============================================================
# ASTRONOMICAL DIRECTORY
# ============================================================

KNOWN_OBJECTS = [
    ("Andromeda Galaxy (M31)", 10.6847, 41.269),
    ("Orion Nebula (M42)", 83.8221, -5.3911),
    ("Pleiades (M45)", 56.75, 24.1167),
    ("Crab Nebula (M1)", 83.6331, 22.0145),
    ("Whirlpool Galaxy (M51)", 202.4696, 47.1953),
    ("Bode's Galaxy (M81)", 148.8882, 69.0653),
    ("Cigar Galaxy (M82)", 148.9685, 69.6797),
    ("Pinwheel Galaxy (M101)", 210.8023, 54.3489),
    ("Virgo A (M87)", 187.7059, 12.3911),
    ("Sombrero Galaxy (M104)", 189.9976, -11.623),
    ("Hercules Cluster (M13)", 250.4235, 36.4613),
    ("Ring Nebula (M57)", 283.396, 33.029),
    ("Dumbbell Nebula (M27)", 299.901, 22.721),
    ("Lagoon Nebula (M8)", 270.925, -24.375),
    ("Eagle Nebula (M16)", 274.7, -13.806),
    ("Omega Nebula (M17)", 275.196, -16.172),
    ("Sirius", 101.287, -16.716),
    ("Vega", 279.2347, 38.7837),
    ("Betelgeuse", 88.7929, 7.4071),
    ("Rigel", 78.634, -8.202),
    ("Arcturus", 213.915, 19.182),
    ("Antares", 247.351, -26.432),
    ("Deneb", 310.358, 45.28),
    ("Polaris", 37.9546, 89.2641),
    ("Galactic Centre", 266.4168, -29.0078),
]


def build_catalogue():

    df = load_database()
    observations = []

    for index, row in df.iterrows():

        polygon = parse_footprint(row["footprint"])
        center = get_center(polygon)

        if center is None:
            continue

        ra, dec = center

        observations.append({
            "id": int(index),
            "key": str(row["s3_key"]),
            "date": str(row.get("date_obs", "Unknown date")),
            "ra": ra,
            "dec": dec,
            "constellation": get_constellation(ra, dec),
            "polygon": polygon
        })

    directory = []

    for name, ra, dec in KNOWN_OBJECTS:

        covering = [
            obs for obs in observations
            if inside_footprint(ra, dec, obs["polygon"])
        ]

        if covering:

            directory.append({
                "id": "object_" + name,
                "name": name,
                "kind": "Astronomical object",
                "ra": ra,
                "dec": dec,
                "observations": covering
            })

    for obs in observations:

        directory.append({
            "id": "region_" + str(obs["id"]),
            "name": (
                f'Region {obs["id"] + 1:02d} — '
                f'{obs["constellation"]}'
            ),
            "kind": "SPHEREx sky region",
            "ra": obs["ra"],
            "dec": obs["dec"],
            "observations": [obs]
        })

    return observations, directory


# ============================================================
# AWS S3
# ============================================================

@st.cache_resource
def get_s3():
    return s3fs.S3FileSystem(anon=True)


def download_fits(key):

    key = str(key).strip()

    if key.startswith("s3://"):
        key = key[5:]

    if key.startswith(BUCKET + "/"):
        key = key[len(BUCKET) + 1:]

    filename = hashlib.sha256(
        key.encode("utf-8")
    ).hexdigest() + ".fits"

    local_path = CACHE_DIR / filename

    if local_path.exists() and local_path.stat().st_size > 0:
        return local_path

    filesystem = get_s3()

    with filesystem.open(
        f"{BUCKET}/{key}", "rb"
    ) as remote:

        content = remote.read()

    local_path.write_bytes(content)

    return local_path


# ============================================================
# FITS READER
# ============================================================

@st.cache_data(show_spinner=False)
def read_fits(key):

    path = download_fits(key)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        with fits.open(path, memmap=False) as hdul:

            image_hdu = (
                hdul["IMAGE"]
                if "IMAGE" in hdul
                else None
            )

            if image_hdu is None:

                for hdu in hdul:

                    if (
                        hdu.data is not None
                        and np.asarray(hdu.data).ndim >= 2
                    ):
                        image_hdu = hdu
                        break

            if image_hdu is None:
                raise ValueError("No image extension found.")

            data = np.asarray(
                image_hdu.data,
                dtype=np.float32
            ).squeeze()

            header = image_hdu.header.copy()

    if data.ndim != 2:
        raise ValueError(
            f"Expected a 2D image, received {data.shape}"
        )

    wcs = WCS(header).celestial

    if not wcs.has_celestial:
        raise ValueError("No celestial WCS found.")

    header_string = header.tostring(
        sep="\n",
        endcard=True,
        padding=False
    )

    return data, header_string


# ============================================================
# IMAGE DISPLAY
# ============================================================

def draw_image(data, title):

    valid = data[np.isfinite(data)]

    if valid.size == 0:
        st.warning("No valid pixels in this image.")
        return

    vmin, vmax = np.nanpercentile(valid, [2, 98])

    if vmin == vmax:
        vmax = vmin + 1e-8

    fig, ax = plt.subplots(figsize=(8, 7))

    ax.imshow(
        data,
        origin="lower",
        cmap="inferno",
        vmin=vmin,
        vmax=vmax
    )

    ax.set_title(title, color="white")
    ax.set_xlabel("Pixel X", color="white")
    ax.set_ylabel("Pixel Y", color="white")
    ax.tick_params(colors="white")

    fig.patch.set_facecolor("#07182f")
    ax.set_facecolor("#07182f")

    st.pyplot(fig)
    plt.close(fig)


# ============================================================
# TEMPORAL COMPARISON
# ============================================================

def compare_images(first, second):

    data1, header1 = read_fits(first["key"])
    data2, header2 = read_fits(second["key"])

    fits_header1 = fits.Header.fromstring(
        header1, sep="\n"
    )

    fits_header2 = fits.Header.fromstring(
        header2, sep="\n"
    )

    wcs1 = WCS(fits_header1).celestial
    wcs2 = WCS(fits_header2).celestial

    aligned, footprint = reproject_interp(
        (data2, wcs2),
        wcs1,
        shape_out=data1.shape
    )

    valid = (
        np.isfinite(data1)
        & np.isfinite(aligned)
        & (footprint > 0)
    )

    if not np.any(valid):
        raise ValueError(
            "No overlapping valid area between these observations."
        )

    difference = np.full(
        data1.shape,
        np.nan,
        dtype=np.float32
    )

    difference[valid] = aligned[valid] - data1[valid]

    return data1, aligned, difference


# ============================================================
# SESSION STATE
# ============================================================

if "active_observations" not in st.session_state:
    st.session_state.active_observations = []

if "current_index" not in st.session_state:
    st.session_state.current_index = 0

if "ra_input" not in st.session_state:
    st.session_state.ra_input = 160.615238

if "dec_input" not in st.session_state:
    st.session_state.dec_input = 30.669077


# ============================================================
# LOAD CATALOGUE
# ============================================================

try:
    observations, directory = build_catalogue()

except Exception as error:
    st.title("🌌 SPHEREx Explorer")
    st.error(f"Unable to load the catalogue: {error}")
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🌌 SPHEREx Explorer")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "COORDINATE SEARCH",
        "FREE EXPLORATION"
    ]
)


# ============================================================
# HOME
# ============================================================

if page == "Home":

    st.title("🌌 SPHEREx Explorer")

    st.markdown("""
    <div class="panel">
        <h2>Explore the infrared universe</h2>
        Discover real SPHEREx observations, navigate between
        different observation dates and compare astronomical images.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📍 COORDINATE SEARCH")
        st.write(
            "Find observed celestial regions using the astronomical "
            "directory or your own coordinates."
        )

    with col2:
        st.markdown("### 🔭 FREE EXPLORATION")
        st.write(
            "Explore the sky freely using an interactive sky map."
        )

    st.info(
        f"{len(observations)} usable SPHEREx observations "
        "in the local catalogue."
    )


# ============================================================
# COORDINATE SEARCH
# ============================================================

elif page == "COORDINATE SEARCH":

    st.title("🌌 SPHEREx Coordinate Search")

    st.markdown("""
    <div class="panel">
        <b>How this mode works</b><br>
        Select an astronomical object or an observed sky region.
        The application finds indexed SPHEREx images covering
        the selected coordinates.
    </div>
    """, unsafe_allow_html=True)

    # --------------------------------------------------------
    # DIRECTORY
    # --------------------------------------------------------

    st.subheader("📖 Astronomical directory")

    st.caption(
        "Named objects are listed only when an indexed FITS "
        "observation covers their coordinates. All indexed "
        "observation regions are also available."
    )

    directory_by_id = {
        item["id"]: item for item in directory
    }

    if not directory:
        st.warning("No usable regions in the catalogue.")
        st.stop()

    def update_coordinates():

        chosen = directory_by_id[
            st.session_state.directory_choice
        ]

        st.session_state.ra_input = chosen["ra"]
        st.session_state.dec_input = chosen["dec"]

    selected_id = st.selectbox(
        "Search for a celestial object or sky region",
        options=list(directory_by_id.keys()),
        format_func=lambda value: (
            f'{directory_by_id[value]["kind"]} — '
            f'{directory_by_id[value]["name"]}'
        ),
        key="directory_choice",
        on_change=update_coordinates
    )

    directory_item = directory_by_id[selected_id]

    st.write(f'**Selected:** {directory_item["name"]}')

    st.write(
        f'**Coordinates:** RA {directory_item["ra"]:.6f}° · '
        f'Dec {directory_item["dec"]:+.6f}°'
    )

    # --------------------------------------------------------
    # COORDINATES
    # --------------------------------------------------------

    st.divider()
    st.subheader("📍 Coordinate search")

    col1, col2 = st.columns(2)

    with col1:
        ra = st.number_input(
            "Right Ascension (degrees)",
            min_value=0.0,
            max_value=360.0,
            format="%.6f",
            key="ra_input"
        )

    with col2:
        dec = st.number_input(
            "Declination (degrees)",
            min_value=-90.0,
            max_value=90.0,
            format="%.6f",
            key="dec_input"
        )

    if st.button(
        "🔍 Search SPHEREx observations",
        type="primary"
    ):

        matches = [
            obs for obs in observations
            if inside_footprint(
                ra, dec, obs["polygon"]
            )
        ]

        st.session_state.active_observations = matches
        st.session_state.current_index = 0

        if matches:
            st.success(
                f"{len(matches)} observation(s) found."
            )
        else:
            st.warning(
                "No indexed FITS observation covers this position."
            )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    active = st.session_state.active_observations

    if active:

        st.success(
            f"{len(active)} observation(s) available."
        )

        st.subheader("🗂️ Available observations")

        current = min(
            st.session_state.current_index,
            len(active) - 1
        )

        selected_position = st.selectbox(
            "Choose an observation",
            options=list(range(len(active))),
            index=current,
            format_func=lambda i: (
                f'Image {i + 1} — '
                f'{active[i]["date"]} — '
                f'{active[i]["constellation"]}'
            ),
            key="observation_selector"
        )

        st.session_state.current_index = selected_position

        # ----------------------------------------------------
        # IMAGE NAVIGATION
        # ----------------------------------------------------

        st.divider()
        st.subheader("🖼️ Image explorer")

        current = st.session_state.current_index
        selected = active[current]

        col1, col2, col3 = st.columns([1, 2, 1])

        with col1:

            if st.button(
                "⬅ Previous image",
                disabled=current == 0,
                use_container_width=True
            ):
                st.session_state.current_index -= 1
                st.rerun()

        with col2:

            st.markdown(
                f"<center><b>Image {current + 1} "
                f"/ {len(active)}</b></center>",
                unsafe_allow_html=True
            )

        with col3:

            if st.button(
                "Next image ➡",
                disabled=current == len(active) - 1,
                use_container_width=True
            ):
                st.session_state.current_index += 1
                st.rerun()

        st.write(
            f'**Region:** {selected["constellation"]}  \n'
            f'**Date:** {selected["date"]}  \n'
            f'**RA:** {selected["ra"]:.6f}°  \n'
            f'**Dec:** {selected["dec"]:+.6f}°'
        )

        # ----------------------------------------------------
        # DISPLAY IMAGE
        # ----------------------------------------------------

        try:

            with st.spinner("Loading FITS image from AWS S3..."):
                image, header = read_fits(selected["key"])

            draw_image(
                image,
                f'Observation {current + 1} — {selected["date"]}'
            )

        except Exception as error:

            st.error(f"Image loading failed: {error}")

        # ----------------------------------------------------
        # TEMPORAL COMPARISON
        # ----------------------------------------------------

        st.divider()
        st.subheader("⏱️ Compare observations")

        if len(active) >= 2:

            other_index = st.selectbox(
                "Compare with another observation",
                options=[
                    i for i in range(len(active))
                    if i != current
                ],
                format_func=lambda i: (
                    f'Image {i + 1} — {active[i]["date"]}'
                ),
                key="comparison_selector"
            )

            other = active[other_index]

            if st.button(
                "⚡ Compare these observations",
                type="primary"
            ):

                try:

                    with st.spinner(
                        "Aligning images and calculating differences..."
                    ):

                        image_a, image_b, difference = compare_images(
                            selected,
                            other
                        )

                    col1, col2 = st.columns(2)

                    with col1:
                        draw_image(
                            image_a,
                            f'Image A — {selected["date"]}'
                        )

                    with col2:
                        draw_image(
                            image_b,
                            f'Image B — {other["date"]}'
                        )

                    st.subheader("Difference image")

                    st.caption(
                        "Image B minus image A, after alignment "
                        "using their celestial coordinates."
                    )

                    draw_image(
                        difference,
                        "Temporal difference"
                    )

                except Exception as error:
                    st.error(f"Comparison failed: {error}")

        else:

            st.info(
                "Two or more observations are required "
                "for temporal comparison."
            )

        # ----------------------------------------------------
        # METADATA
        # ----------------------------------------------------

        with st.expander("🔧 FITS technical information"):

            st.write("S3 key:")
            st.code(selected["key"])

            st.code(header)

    else:

        st.info(
            "Select a directory entry or enter coordinates, "
            "then click Search SPHEREx observations."
        )

    # --------------------------------------------------------
    # COMPLETE CATALOGUE
    # --------------------------------------------------------

    with st.expander("📚 Complete observation catalogue"):

        table = pd.DataFrame([
            {
                "Region": i + 1,
                "Constellation": obs["constellation"],
                "RA (degrees)": round(obs["ra"], 6),
                "Dec (degrees)": round(obs["dec"], 6),
                "Date": obs["date"]
            }
            for i, obs in enumerate(observations)
        ])

        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# FREE EXPLORATION
# ============================================================

elif page == "FREE EXPLORATION":

    st.title("🔭 Free Exploration")

    st.markdown("""
    <div class="panel">
        <b>Explore the sky freely</b><br>
        Navigate the sky interactively without entering
        coordinates in the SPHEREx search.
    </div>
    """, unsafe_allow_html=True)

    st.components.v1.iframe(
        "https://stellarium-web.org/",
        height=720,
        scrolling=True
    )
