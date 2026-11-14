from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from spherex_service import COLLECTIONS, download_observation, read_fits_image, search_observations

st.set_page_config(page_title="SPHEREx Explorer", page_icon="🌌", layout="wide")
st.markdown("""
<style>
.stApp { background: linear-gradient(140deg, #050D20, #0B1930); color: white; }
.block-container { max-width: 1150px; padding-top: 2rem; }
h1, h2, h3, p, label { color: white !important; }
.hero { background: linear-gradient(110deg, #102C50, #10203B); border: 1px solid #28517B;
        border-radius: 18px; padding: 28px; margin-bottom: 20px; }
.status { color: #64D5FF; font-weight: bold; font-size: 12px; letter-spacing: 2px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="status">NASA/IPAC INFRARED SCIENCE ARCHIVE · SIA2</div>
  <h1>🌌 SPHEREx Explorer</h1>
  <p>Search real SPHEREx observations by sky position and browse the available epochs.</p>
</div>
""", unsafe_allow_html=True)

st.info(
    "This app queries IRSA's SIA2 service. QR3 and DR1 are searchable; QR2 is no longer "
    "searchable via SIA2. Observations are shown in time order when epoch metadata is available."
)

with st.form("search_form"):
    left, right = st.columns(2)
    with left:
        ra = st.number_input("Right Ascension (degrees)", 0.0, 360.0, 10.0, format="%.6f")
        radius = st.number_input("Search radius (degrees)", 0.001, 5.0, 0.1, format="%.3f")
    with right:
        dec = st.number_input("Declination (degrees)", -90.0, 90.0, 20.0, format="%.6f")
        limit = st.number_input("Maximum observations", 1, 200, 50)
    selected_labels = st.multiselect(
        "SPHEREx data collections", list(COLLECTIONS),
        default=["QR3 — Wide survey", "DR1 — Wide survey"],
    )
    submitted = st.form_submit_button("🔭 Search IRSA", use_container_width=True)

if submitted:
    try:
        collections = [COLLECTIONS[label] for label in selected_labels]
        with st.spinner("Searching the IRSA SPHEREx archive…"):
            st.session_state.observations = search_observations(
                ra, dec, radius, collections, int(limit)
            )
        st.session_state.search_position = (ra, dec, radius)
    except Exception as exc:
        st.session_state.observations = []
        st.error(f"IRSA search failed: {exc}")

observations = st.session_state.get("observations", [])
if observations:
    st.success(f"Found {len(observations)} observation(s).")
    summary_rows = []
    for index, observation in enumerate(observations):
        summary_rows.append({
            "#": index + 1,
            "Epoch": observation.get("epoch", "—"),
            "Collection": observation.get("collection", "—"),
            "Band": observation.get("band", observation.get("filter", "—")),
            "Observation ID": observation.get("obs_id", observation.get("dp_id", "—")),
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    labels = [
        f"{i + 1}. {item.get('epoch', 'epoch unknown')} · {item.get('collection', '')}"
        for i, item in enumerate(observations)
    ]
    selected_index = st.selectbox(
        "Observation timeline", range(len(observations)),
        format_func=lambda index: labels[index],
    )
    current = observations[selected_index]
    try:
        with st.spinner("Downloading and reading the selected FITS observation…"):
            fits_path = download_observation(current)
            image, wcs = read_fits_image(fits_path)
        finite = image[np.isfinite(image)]
        if finite.size == 0:
            st.warning("This FITS image contains no finite pixels to display.")
        else:
            low, high = np.nanpercentile(finite, [1, 99])
            if not np.isfinite(low) or not np.isfinite(high) or low == high:
                low, high = float(np.nanmin(finite)), float(np.nanmax(finite))
            if low == high:
                high = low + 1
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(image, origin="lower", cmap="magma", vmin=low, vmax=high)
            ax.set_title(f"{current.get('collection', 'SPHEREx')} · {current.get('epoch', 'epoch unknown')}")
            ax.set_xlabel("Image pixel X (WCS available in FITS)")
            ax.set_ylabel("Image pixel Y")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.caption(
                f"FITS: {fits_path.name} · Image shape: {image.shape[1]} × {image.shape[0]} · "
                f"Celestial WCS detected: {wcs.has_celestial}"
            )
            st.caption("Image values are displayed with a 1st–99th percentile stretch.")
    except Exception as exc:
        st.error(f"Could not display this observation: {exc}")
        st.caption("The archive result remains listed; its FITS extension may not be a 2-D celestial image.")
elif st.session_state.get("search_position") and not observations:
    st.warning("No observations were returned for this position and collection selection.")

st.markdown("---")
st.caption("SPHEREx Explorer · Data provided by NASA/IPAC IRSA · Candidates require scientific validation.")
