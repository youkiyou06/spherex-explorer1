from __future__ import annotations
import base64
import hashlib
import json
import os
import sqlite3
import tempfile
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse
import matplotlib.pyplot as plt
import numpy as np
import requests
import streamlit as st
import streamlit.components.v1 as components
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astroquery.ipac.irsa import Irsa
from reproject import reproject_interp
from scipy import ndimage

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data_spherex"
CACHE_DIR = DATA_DIR / "fits_cache"
DB_PATH = DATA_DIR / "observations.sqlite3"
STEL_DIR = APP_DIR / "stellarium_component"
MAX_FITS_SIZE = 500 * 1024 * 1024
MAX_CACHE_SIZE = 5 * 1024 * 1024 * 1024
DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
STEL_DIR.mkdir(exist_ok=True)

COLLECTIONS = {
    "DR1 — Wide": "spherex_dr1",
    "DR1 — Deep": "spherex_dr1_deep",
    "QR3 — Wide": "spherex_qr3",
    "QR3 — Deep": "spherex_qr3_deep",
}
BACKGROUND = "#07152f"
PANEL = "#10264b"
YELLOW = "#e8c547"

st.set_page_config(page_title="SPHEREx Explorer", page_icon="🌌", layout="wide", initial_sidebar_state="collapsed")

STELLARIUM_HTML = r"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="https://unpkg.com/streamlit-component-lib@1.4.0/dist/streamlit.js"></script>
<style>
body{margin:0;background:#07152f;color:white;font-family:Arial;overflow:hidden}
#toolbar{padding:12px;background:#10264b;display:flex;flex-wrap:wrap;gap:12px;align-items:center}
button{background:#e8c547;color:#07152f;font-weight:bold;border:none;border-radius:8px;padding:11px 16px;cursor:pointer}
#coordinates{color:#e8c547;font-size:14px}
#stel{height:580px;width:100%;background:#000}
iframe{width:100%;height:580px;border:0}
#error{color:#ffaaaa;padding:8px}
</style>
</head>
<body>
<div id="toolbar">
    <span id="coordinates">Centre : RA 83.633° Dec 22.014°</span>
    <button id="select">Utiliser le centre de Stellarium</button>
    <button id="orion">→ Orion / M42</button>
    <button id="gal">→ Centre Galactique</button>
</div>
<iframe id="stel-frame" src="https://stellarium-web.org/"></iframe>
<div id="error"></div>
<script>
(function(){
    const Streamlit = window.Streamlit;
    let current = {ra: 83.633, dec: 22.014};
    
    function updateLabel(){
        document.getElementById("coordinates").textContent = "Centre estimé : RA " + current.ra.toFixed(4) + "° | Dec " + current.dec.toFixed(4) + "° | Utilisez le champ RA/Dec en dessous pour affiner";
    }

    function sendCoords(){
        Streamlit.setComponentValue({ra: current.ra, dec: current.dec});
    }

    function parseHash(){
        try{
            let hash = document.getElementById("stel-frame").contentWindow.location.hash;
            if(hash && hash.includes("ra=")){
                let m = hash.match(/ra=([0-9.\-]+)/);
                let m2 = hash.match(/dec=([0-9.\-]+)/);
                if(m) current.ra = parseFloat(m[1]);
                if(m2) current.dec = parseFloat(m2[1]);
                if(current.ra < 0) current.ra += 360;
                updateLabel();
            }
        }catch(e){}
    }

    document.getElementById("select").onclick = function(){ sendCoords(); };
    document.getElementById("orion").onclick = function(){
        current = {ra: 83.633, dec: -5.391};
        document.getElementById("stel-frame").src = "https://stellarium-web.org/#ra=83.633&dec=-5.391&fov=10";
        updateLabel();
    };
    document.getElementById("gal").onclick = function(){
        current = {ra: 266.416, dec: -29.007};
        document.getElementById("stel-frame").src = "https://stellarium-web.org/#ra=266.416&dec=-29.007&fov=10";
        updateLabel();
    };

    Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, function(event){
        let args = event.detail.args || {};
        if(args.ra !== undefined) current.ra = Number(args.ra);
        if(args.dec !== undefined) current.dec = Number(args.dec);
        updateLabel();
    });

    Streamlit.setComponentReady();
    Streamlit.setFrameHeight(650);
    updateLabel();
    setInterval(parseHash, 1500);
})();
</script>
</body>
</html>
"""

(STEL_DIR / "index.html").write_text(STELLARIUM_HTML, encoding="utf-8")
stellarium_component = components.declare_component("spherex_stellarium", path=str(STEL_DIR))

def database():
    connection = sqlite3.connect(DB_PATH, timeout=30)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA busy_timeout=30000")
    return connection

def init_database():
    with database() as db:
        db.execute("""
        CREATE TABLE IF NOT EXISTS observations (
            observation_id TEXT PRIMARY KEY,
            collection TEXT,
            epoch TEXT,
            band TEXT,
            access_url TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            saved_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """)
        db.execute("CREATE INDEX IF NOT EXISTS idx_band ON observations(band)")

def database_count():
    with database() as db:
        result = db.execute("SELECT COUNT(*) FROM observations").fetchone()
    return result[0]

def save_observations(records):
    count = 0
    with database() as db:
        for record in records:
            url = record.get("access_url")
            if not url:
                continue
            identifier = str(record.get("observation_id") or hashlib.sha256(url.encode()).hexdigest())
            metadata = json.dumps(record, ensure_ascii=False, default=str)
            db.execute("""
            INSERT INTO observations (observation_id, collection, epoch, band, access_url, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(observation_id) DO UPDATE SET
                collection=excluded.collection, epoch=excluded.epoch, band=excluded.band,
                access_url=excluded.access_url, metadata_json=excluded.metadata_json, saved_at=CURRENT_TIMESTAMP
            """, (identifier, str(record.get("collection", "")), get_epoch(record), get_band(record), url, metadata))
            count += 1
    return count

def plain_value(value):
    if value is None:
        return None
    try:
        if np.ma.is_masked(value):
            return None
    except Exception:
        pass
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)

def get_field(record, *names):
    lowered = {str(key).lower(): value for key, value in record.items()}
    for name in names:
        value = lowered.get(name.lower())
        if value is None:
            continue
        if str(value).strip().lower() not in {"", "--", "nan", "none", "masked"}:
            return value
    return None

def get_band(record):
    value = get_field(record, "band_value", "energy_bandpassname", "band", "filter", "filter_name", "bandpass", "bandnum")
    return str(value) if value is not None else "Bande inconnue"

def get_epoch(record):
    value = get_field(record, "epoch", "mjd", "mjd_obs", "t_min", "time", "date_obs")
    return str(value) if value is not None else "Époque inconnue"

def sort_key(record):
    value = get_field(record, "mjd", "mjd_obs", "t_min", "epoch")
    try:
        return (0, float(value), str(record.get("observation_id", "")))
    except (TypeError, ValueError):
        return (1, get_epoch(record), str(record.get("observation_id", "")))

def observation_label(record):
    identifier = get_field(record, "observation_id", "obs_id", "obsid", "dp_id") or "Observation"
    return f"{get_epoch(record)} · {get_band(record)} · {identifier}"

def validate_coordinates(ra, dec, radius):
    if not 0 <= ra < 360:
        raise ValueError("RA 0-360")
    if not -90 <= dec <= 90:
        raise ValueError("Dec -90 +90")
    if not 0.001 <= radius <= 10:
        raise ValueError("Rayon invalide")

@st.cache_data(ttl=900, show_spinner=False)
def search_irsa(ra, dec, radius, collections, limit):
    validate_coordinates(float(ra), float(dec), float(radius))
    if not collections:
        raise ValueError("Collection vide")
    position = SkyCoord(ra=float(ra) * u.deg, dec=float(dec) * u.deg, frame="icrs")
    results = []
    seen = set()
    errors = []
    for collection in collections:
        try:
            table = Irsa.query_sia(pos=(position, float(radius) * u.deg), collection=collection)
        except Exception as error:
            errors.append(f"{collection}: {error}")
            continue
        url_column = next((name for name in table.colnames if name.lower() == "access_url"), None)
        if url_column is None:
            errors.append(f"{collection}: access_url absente")
            continue
        for row in table:
            record = {str(name): plain_value(row[name]) for name in table.colnames}
            url = record.get(url_column)
            if not isinstance(url, str):
                continue
            url = url.strip()
            parsed = urlparse(url)
            if parsed.scheme != "https" or parsed.hostname != "irsa.ipac.caltech.edu":
                continue
            if url in seen:
                continue
            seen.add(url)
            record["access_url"] = url
            record["collection"] = collection
            raw_band = get_field(record, "energy_bandpassname", "band", "filter", "filter_name", "bandpass", "bandnum")
            record["band_value"] = str(raw_band) if raw_band is not None else None
            raw_epoch = get_field(record, "mjd", "mjd_obs", "t_min", "time", "date_obs")
            record["epoch"] = plain_value(raw_epoch)
            raw_id = get_field(record, "obs_id", "obsid", "dp_id", "obs_publisher_did")
            record["observation_id"] = str(raw_id or hashlib.sha256(url.encode()).hexdigest())
            results.append(record)
            if len(results) >= limit:
                break
        if len(results) >= limit:
            break
    if not results and errors:
        raise RuntimeError("IRSA: " + " | ".join(errors[:3]))
    results.sort(key=sort_key)
    return results[:limit]

def check_cache_size():
    files = [path for path in CACHE_DIR.iterdir() if path.is_file()]
    total = sum(path.stat().st_size for path in files)
    if total <= MAX_CACHE_SIZE:
        return
    for path in sorted(files, key=lambda item: item.stat().st_atime):
        try:
            size = path.stat().st_size
            path.unlink(missing_ok=True)
            total -= size
        except OSError:
            continue
        if total <= MAX_CACHE_SIZE * 0.8:
            break

def valid_fits(path):
    try:
        with fits.open(path, memmap=False) as hdus:
            hdus.verify("exception")
            return any(hdu.data is not None for hdu in hdus)
    except Exception:
        return False

def cached_download(record):
    url = record.get("access_url")
    if not isinstance(url, str):
        raise ValueError("URL absente")
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "irsa.ipac.caltech.edu":
        raise ValueError("Adresse non autorisée")
    digest = hashlib.sha256(url.encode()).hexdigest()
    suffix = ".fits.gz" if parsed.path.lower().endswith(".fits.gz") else ".fits"
    destination = CACHE_DIR / (digest + suffix)
    if destination.exists():
        if valid_fits(destination):
            return destination
        destination.unlink(missing_ok=True)
    temporary = None
    try:
        with requests.get(url, stream=True, timeout=(20, 240), headers={"User-Agent": "SPHEREx-Explorer/2.0"}) as response:
            response.raise_for_status()
            content_length = int(response.headers.get("content-length", 0))
            if content_length > MAX_FITS_SIZE:
                raise ValueError("FITS trop gros")
            with tempfile.NamedTemporaryFile(dir=CACHE_DIR, suffix=".download", delete=False) as handle:
                temporary = Path(handle.name)
                downloaded = 0
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    downloaded += len(chunk)
                    if downloaded > MAX_FITS_SIZE:
                        raise ValueError("FITS dépasse limite")
                    handle.write(chunk)
        if not valid_fits(temporary):
            raise ValueError("FITS invalide")
        check_cache_size()
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination

def read_fits_image(path):
    with fits.open(path, memmap=False) as hdus:
        candidates = sorted(enumerate(hdus), key=lambda item: item[1].name.upper() != "IMAGE")
        for index, hdu in candidates:
            data = hdu.data
            if data is None or np.ndim(data) != 2:
                continue
            if not np.issubdtype(np.asarray(data).dtype, np.number):
                continue
            try:
                wcs = WCS(hdu.header).celestial
                if not wcs.has_celestial:
                    continue
            except Exception:
                continue
            image = np.asarray(data, dtype=np.float32).copy()
            image[~np.isfinite(image)] = np.nan
            if np.isfinite(image).any():
                return image, wcs
    raise ValueError("Aucune image FITS 2D avec WCS")

def stretch_image(image):
    image = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(image)
    if not finite.any():
        raise ValueError("Image vide")
    low, high = np.nanpercentile(image[finite], [1, 99])
    if high <= low:
        low = float(np.nanmin(image[finite]))
        high = float(np.nanmax(image[finite]))
        if high <= low:
            high = low + 1
    normalized = np.clip((image - low) / (high - low), 0, 1)
    return np.nan_to_num(normalized, nan=0, posinf=1, neginf=0)

def make_png(image, title):
    normalized = stretch_image(image)
    figure, axis = plt.subplots(figsize=(5, 5), dpi=100)
    try:
        axis.imshow(normalized, cmap="gray", origin="lower", interpolation="nearest")
        axis.set_title(title, fontsize=9)
        axis.set_axis_off()
        figure.tight_layout()
        output = BytesIO()
        figure.savefig(output, format="png", bbox_inches="tight")
        return output.getvalue()
    finally:
        plt.close(figure)

@st.cache_data(max_entries=20, show_spinner=False)
def cached_preview(path, modified, label):
    image, _ = read_fits_image(path)
    return make_png(image, label)

def compare_fits(reference_record, comparison_record, threshold):
    if reference_record["access_url"] == comparison_record["access_url"]:
        raise ValueError("Deux observations différentes requises")
    first_band = get_band(reference_record)
    second_band = get_band(comparison_record)
    if first_band != "Bande inconnue" and second_band != "Bande inconnue" and first_band != second_band:
        raise ValueError("Même bande requise")
    first_path = cached_download(reference_record)
    second_path = cached_download(comparison_record)
    reference, reference_wcs = read_fits_image(first_path)
    comparison, comparison_wcs = read_fits_image(second_path)
    aligned, footprint = reproject_interp((comparison, comparison_wcs), reference_wcs, shape_out=reference.shape, return_footprint=True)
    valid = np.isfinite(reference) & np.isfinite(aligned) & np.isfinite(footprint) & (footprint > 0.5)
    if np.count_nonzero(valid) < 100:
        raise ValueError("Couverture commune insuffisante")
    difference = np.full(reference.shape, np.nan, dtype=np.float32)
    difference[valid] = aligned[valid] - reference[valid]
    median = float(np.nanmedian(difference[valid]))
    residual = difference[valid] - median
    sigma = float(1.4826 * np.nanmedian(np.abs(residual)))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.nanstd(residual))
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Dispersion nulle")
    significance = np.full(reference.shape, np.nan, dtype=np.float32)
    significance[valid] = (difference[valid] - median) / sigma
    mask = valid & (np.abs(significance) >= threshold)
    labels, count = ndimage.label(mask)
    regions = ndimage.find_objects(labels)
    candidates = []
    for label_id, slices in enumerate(regions, start=1):
        if slices is None:
            continue
        region = labels[slices] == label_id
        area = int(np.count_nonzero(region))
        if area < 2:
            continue
        local = significance[slices]
        local_abs = np.where(region, np.abs(local), -np.inf)
        y_local, x_local = np.unravel_index(np.argmax(local_abs), local_abs.shape)
        y = slices[0].start + y_local
        x = slices[1].start + x_local
        candidates.append({"X": int(x), "Y": int(y), "Surface_pixels": area, "Significativite_sigma": float(significance[y, x]), "Difference": float(difference[y, x])})
    candidates.sort(key=lambda item: abs(item["Significativite_sigma"]), reverse=True)
    return {"difference": difference, "valid_pixels": int(np.count_nonzero(valid)), "sigma": sigma, "candidates": candidates[:100], "total": len(candidates)}

st.markdown(f"""
    <style>
    .stApp{{background:linear-gradient(140deg,#050d20,{BACKGROUND});color:white;}}
    .block-container{{max-width:1250px;padding-top:1.5rem;}}
    h1,h2,h3,p,label{{color:white !important;}}
    .hero{{background:linear-gradient(110deg,{PANEL},#10203b);border:1px solid #344b70;border-radius:18px;padding:25px;margin-bottom:18px;}}
    .hero-title{{color:{YELLOW};font-size:38px;font-weight:800;}}
    .info-box{{background:{PANEL};border:1px solid #344b70;border-left:4px solid {YELLOW};border-radius:12px;padding:16px;margin:12px 0;}}
    div.stButton > button{{background:{YELLOW};color:{BACKGROUND};border:none;border-radius:10px;font-weight:700;min-height:44px;}}
    </style>
    """, unsafe_allow_html=True)

if "page" not in st.session_state:
    st.session_state.page = "home"
if "observations" not in st.session_state:
    st.session_state.observations = []
if "search_signature" not in st.session_state:
    st.session_state.search_signature = None
if "selected_ra" not in st.session_state:
    st.session_state.selected_ra = 83.633
if "selected_dec" not in st.session_state:
    st.session_state.selected_dec = 22.014
if "comparison_result" not in st.session_state:
    st.session_state.comparison_result = None

st.markdown("""<div class="hero"><div class="hero-title">🌌 SPHEREx Explorer — Stellarium Edition</div><p>Explore le ciel avec Stellarium et compare les observations SPHEREx IRSA.</p></div>""", unsafe_allow_html=True)

def back_button():
    if st.button("← Retour à l'accueil"):
        st.session_state.page = "home"
        st.rerun()

def show_home():
    st.markdown("""<div class="info-box"><b>Objectif</b><br>Recherche IRSA, visualisation FITS, carte Stellarium, comparaison temporelle.</div>""", unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        st.subheader("⏳ Exploration temporelle")
        st.write("Recherche dans IRSA, visualisation FITS et comparaison temporelle.")
        if st.button("Ouvrir l'exploration temporelle", use_container_width=True):
            st.session_state.page = "temporal"
            st.rerun()
    with right:
        st.subheader("🌠 Exploration Stellarium")
        st.write("Explore avec Stellarium Web et envoie les coordonnées vers SPHEREx.")
        if st.button("Ouvrir Stellarium", use_container_width=True):
            st.session_state.page = "map"
            st.rerun()
    st.metric("Observations enregistrées", database_count())

def show_map():
    back_button()
    st.title("🌠 Stellarium — Exploration libre")
    st.write("Stellarium Web en dessous. Centre la cible sur un objet, clique sur Utiliser le centre, puis va en exploration temporelle.")
    selected = stellarium_component(ra=float(st.session_state.selected_ra), dec=float(st.session_state.selected_dec), key="stel_selector")
    if isinstance(selected, dict):
        try:
            ra = float(selected["ra"])
            dec = float(selected["dec"])
            validate_coordinates(ra, dec, 0.1)
            st.session_state.selected_ra = ra
            st.session_state.selected_dec = dec
            st.toast(f"Position Stellarium : RA {ra:.4f} Dec {dec:.4f}")
        except Exception:
            pass
    c1, c2, c3 = st.columns(3)
    c1.metric("RA", f"{st.session_state.selected_ra:.6f}°")
    c2.metric("Dec", f"{st.session_state.selected_dec:.6f}°")
    with c3:
        ra_in = st.number_input("RA manuel", value=float(st.session_state.selected_ra), format="%.6f")
        dec_in = st.number_input("Dec manuel", value=float(st.session_state.selected_dec), format="%.6f")
        if st.button("Mettre à jour"):
            st.session_state.selected_ra = float(ra_in)
            st.session_state.selected_dec = float(dec_in)
            st.rerun()
    if st.button("Utiliser cette position dans SPHEREx →", type="primary", use_container_width=True):
        st.session_state.page = "temporal"
        st.rerun()

def blink_comparator(records):
    if len(records) < 2:
        st.info("Deux observations minimum")
        return
    frame_count = st.slider("Nombre d'images", min_value=2, max_value=min(6, len(records)), value=min(4, len(records)))
    indexes = np.linspace(0, len(records) - 1, num=frame_count, dtype=int)
    selected_records = [records[index] for index in np.unique(indexes)]
    if not st.button("Créer le comparateur"):
        return
    frames = []
    progress = st.progress(0)
    try:
        for index, record in enumerate(selected_records):
            path = cached_download(record)
            png = cached_preview(str(path), path.stat().st_mtime_ns, observation_label(record))
            encoded = base64.b64encode(png).decode("ascii")
            frames.append({"src": "data:image/png;base64," + encoded, "label": observation_label(record)})
            progress.progress((index + 1) / len(selected_records))
    except Exception as error:
        st.error(f"Comparateur impossible : {error}")
        return
    finally:
        progress.empty()
    frames_json = json.dumps(frames, ensure_ascii=False)
    html = f"""
    <html><body style="background:#07152f;color:white;text-align:center;font-family:Arial">
    <div id="label"></div><img id="frame" style="max-width:100%;max-height:450px"><br>
    <button onclick="previous()">←</button><button onclick="toggle()">▶/⏸</button><button onclick="next()">→</button>
    <script>
    const frames = {frames_json};
    let index=0; let timer=null;
    function show(){{document.getElementById("frame").src=frames[index].src; document.getElementById("label").textContent=frames[index].label;}}
    function next(){{index=(index+1)%frames.length; show();}}
    function previous(){{index=(index-1+frames.length)%frames.length; show();}}
    function toggle(){{if(timer){{clearInterval(timer); timer=null;}} else {{timer=setInterval(next,800);}}}}
    show();
    </script></body></html>
    """
    components.html(html, height=600, scrolling=False)

def show_temporal():
    back_button()
    st.title("⏳ Exploration temporelle")
    labels = list(COLLECTIONS)
    with st.form("search_form"):
        st.subheader("1. Région céleste")
        c1, c2, c3 = st.columns(3)
        with c1:
            ra = st.number_input("RA (degrés)", min_value=0.0, max_value=359.999999, value=float(st.session_state.selected_ra), format="%.6f")
        with c2:
            dec = st.number_input("Dec (degrés)", min_value=-90.0, max_value=90.0, value=float(st.session_state.selected_dec), format="%.6f")
        with c3:
            radius = st.number_input("Rayon (degrés)", min_value=0.001, max_value=10.0, value=0.2, step=0.05)
        selected_labels = st.multiselect("Collections", options=labels, default=["DR1 — Wide"])
        limit = st.slider("Nombre maximal", min_value=2, max_value=100, value=30)
        submitted = st.form_submit_button("🔎 Rechercher dans IRSA", use_container_width=True)
    collections = tuple(COLLECTIONS[label] for label in selected_labels)
    signature = (round(float(ra), 6), round(float(dec), 6), round(float(radius), 6), collections, int(limit))
    if submitted:
        st.session_state.observations = []
        st.session_state.comparison_result = None
        st.session_state.search_signature = signature
        try:
            with st.spinner("Recherche IRSA..."):
                results = search_irsa(float(ra), float(dec), float(radius), collections, int(limit))
            st.session_state.observations = results
            if results:
                saved = save_observations(results)
                st.success(f"{len(results)} trouvée(s), {saved} enregistrée(s).")
            else:
                st.info("Aucune observation.")
        except Exception as error:
            st.error(f"Échec : {error}")
    records = st.session_state.observations
    if not records:
        st.info("Lance une recherche.")
        return
    if st.session_state.search_signature != signature:
        st.warning("Paramètres changés. Relance.")
        return
    records = sorted(records, key=sort_key)
    st.subheader("2. Observations disponibles")
    st.dataframe([{"Époque": get_epoch(r), "Bande": get_band(r), "Collection": r.get("collection", "—"), "Identifiant": r.get("observation_id", "—")} for r in records], use_container_width=True, hide_index=True)
    bands = sorted({get_band(record) for record in records})
    selected_band = st.selectbox("Bande spectrale", options=bands)
    band_records = [record for record in records if get_band(record) == selected_band]
    band_records.sort(key=sort_key)
    if not band_records:
        return
    st.subheader("3. Visualiser")
    selected_index = st.select_slider("Époque", options=list(range(len(band_records))), value=0, format_func=lambda i: observation_label(band_records[i]))
    selected_record = band_records[selected_index]
    if st.button("Charger / afficher l'image"):
        try:
            with st.spinner("Chargement FITS..."):
                path = cached_download(selected_record)
                png = cached_preview(str(path), path.stat().st_mtime_ns, observation_label(selected_record))
                st.session_state.current_image_png = png
                st.session_state.current_image_url = selected_record["access_url"]
                st.session_state.current_image_label = observation_label(selected_record)
        except Exception as error:
            st.error(f"Chargement impossible : {error}")
    if st.session_state.get("current_image_png") is not None and st.session_state.get("current_image_url") == selected_record.get("access_url"):
        st.image(st.session_state.current_image_png, caption=st.session_state.current_image_label, use_container_width=True)
    with st.expander("Métadonnées"):
        st.json(dict(selected_record))
    st.subheader("4. Comparateur temporel")
    blink_comparator(band_records)
    st.subheader("5. Comparer deux époques")
    if len(band_records) < 2:
        st.info("Deux observations minimum")
        return
    c1, c2 = st.columns(2)
    indexes = list(range(len(band_records)))
    with c1:
        first_index = st.selectbox("Référence", indexes, index=0, format_func=lambda i: observation_label(band_records[i]), key="first_image")
    with c2:
        second_index = st.selectbox("À comparer", indexes, index=1, format_func=lambda i: observation_label(band_records[i]), key="second_image")
    threshold = st.slider("Seuil sigma", min_value=2.0, max_value=10.0, value=5.0, step=0.5)
    if st.button("Aligner et comparer", type="primary", use_container_width=True):
        if first_index == second_index:
            st.warning("Deux observations différentes")
        else:
            try:
                with st.spinner("Alignement WCS..."):
                    result = compare_fits(band_records[first_index], band_records[second_index], float(threshold))
                st.session_state.comparison_result = result
                st.session_state.comparison_labels = (observation_label(band_records[first_index]), observation_label(band_records[second_index]))
            except Exception as error:
                st.session_state.comparison_result = None
                st.error(f"Échec : {error}")
    result = st.session_state.comparison_result
    if result is not None:
        labels_used = st.session_state.comparison_labels
        st.caption(f"Référence : {labels_used[0]} | Comparée : {labels_used[1]}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Pixels valides", result["valid_pixels"])
        c2.metric("Dispersion", f'{result["sigma"]:.4g}')
        c3.metric("Régions", result["total"])
        difference = result["difference"]
        finite = np.isfinite(difference)
        if finite.any():
            scale = float(np.nanpercentile(np.abs(difference[finite]), 99))
            if not np.isfinite(scale) or scale <= 0:
                scale = 1.0
            figure, axis = plt.subplots(figsize=(9, 6))
            try:
                plotted = axis.imshow(difference, origin="lower", cmap="RdBu_r", vmin=-scale, vmax=scale)
                axis.set_title("Image comparée − référence")
                figure.colorbar(plotted, ax=axis, label="Différence")
                figure.tight_layout()
                st.pyplot(figure, use_container_width=True)
            finally:
                plt.close(figure)
        if result["candidates"]:
            st.dataframe(result["candidates"], use_container_width=True, hide_index=True)
        else:
            st.info("Aucune région au-dessus du seuil")
        st.warning("Résultats exploratoires, pas de découverte confirmée")

init_database()
if st.session_state.page == "home":
    show_home()
elif st.session_state.page == "map":
    show_map()
elif st.session_state.page == "temporal":
    show_temporal()
else:
    st.session_state.page = "home"
    st.rerun()
st.divider()
st.caption("SPHEREx : NASA/IPAC IRSA · Carte : Stellarium Web · FITS · SQLite")
