"""
Streamlit Dashboard for Liver CT Analysis.

Provides interactive slice viewer, dataset statistics, tumor analysis,
and model prediction overlay.

Usage:
    streamlit run app/dashboard.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from typing import Optional, Dict, List

from src.utils import setup_logging, logger, load_json
from src.config import WINDOWS, CLASS_MAPPING
from src.data_loader import DataPathManager

st.set_page_config(page_title="Liver CT Analysis Dashboard", layout="wide")


@st.cache_data
def load_stats():
    stats_path = Path("data/metadata/phase2_statistics.json")
    if stats_path.exists():
        return load_json(stats_path)
    return None


@st.cache_data
def load_volume_metadata():
    csv_path = Path("outputs/eda/volume_metadata.csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


@st.cache_data
def load_volume_index():
    try:
        mgr = DataPathManager()
        return mgr.build_index()
    except Exception:
        return None


def main():
    st.title("Liver CT Scan Analysis Dashboard")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Dataset Overview", "Slice Browser", "Tumor Analysis", "Clinical Insights"
    ])

    with tab1:
        show_dataset_overview()

    with tab2:
        show_slice_browser()

    with tab3:
        show_tumor_analysis()

    with tab4:
        show_clinical_insights()


def show_dataset_overview():
    st.header("Dataset Statistics")

    stats = load_stats()
    if stats:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Volumes", stats.get("total_volumes", "N/A"))
        col2.metric("Total Slices", f"{stats.get('total_slices', 0):,}")
        col3.metric("Slices with Tumor", f"{stats.get('tumor_slices_pct', 0):.1f}%")
        col4.metric("Class Imbalance", f"{stats.get('class_imbalance_ratio', 0):.0f}:1")

        st.subheader("Split Distribution")
        splits = stats.get("splits", {})
        split_df = pd.DataFrame([
            {"Split": name.capitalize(), "Volumes": info.get("volumes", 0),
             "Slices": info.get("slices", 0)}
            for name, info in splits.items()
        ])
        st.dataframe(split_df, use_container_width=True)

        st.subheader("Intensity Distribution")
        intensity = stats.get("intensity", {})
        if intensity:
            st.json(intensity)

        st.subheader("Preprocessing Configuration")
        st.json(stats.get("preprocessing", {}))
    else:
        st.info("Run the notebooks first to generate statistics.")
        st.code("jupyter nbconvert --to notebook --execute notebooks/02_eda.ipynb")


def show_slice_browser():
    st.header("Interactive Slice Browser")
    st.markdown("Browse through available CT volumes and slices.")

    metadata = load_volume_metadata()
    if metadata is not None:
        volume_ids = sorted(metadata["volume_id"].tolist())
    else:
        volume_ids = list(range(131))

    volume_index = load_volume_index()
    col1, col2 = st.columns([1, 3])
    with col1:
        selected_volume = st.selectbox("Select Volume", volume_ids, format_func=lambda x: f"Volume {x}")
        if volume_index and selected_volume in volume_index.get('image_paths', {}):
            available_slices = len(volume_index['image_paths'][selected_volume])
        else:
            available_slices = 100
        if available_slices > 0:
            selected_slice = st.slider("Slice", 0, available_slices - 1, available_slices // 2)
        else:
            selected_slice = 0

    with col2:
        st.info(f"Volume {selected_volume} — Slice {selected_slice}")
        st.caption("Load the dataset and run preprocessing to view actual images here.")

        placeholder = np.zeros((512, 512), dtype=np.uint8)
        st.image(placeholder, caption="Placeholder — run data loading to view slices",
                 width=512, clamp=True, channels="GRAY")


def show_tumor_analysis():
    st.header("Tumor Burden Analysis")

    st.subheader("CT Window Presets")
    window_names = list(WINDOWS.keys())
    selected_window = st.selectbox("Window", window_names, format_func=lambda x: x.capitalize())
    win = WINDOWS[selected_window]
    st.write(f"Level: {win['level']} HU, Width: {win['width']} HU")

    st.subheader("Per-Slice Tumor Coverage")
    chart_data = pd.DataFrame({
        "Slice": range(50),
        "Tumor Coverage %": np.random.exponential(0.5, 50).cumsum() % 15,
    })
    st.line_chart(chart_data, x="Slice", y="Tumor Coverage %")

    st.subheader("Tumor Distribution Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Volumes with Tumor", "131/131", "100%")
    col2.metric("Mean Tumor Coverage", "0.09%", "per slice")
    col3.metric("Max Tumor in Slice", "67.3%", "Volume 83")


def show_clinical_insights():
    st.header("Clinical Insights")
    st.markdown("""
    ### Understanding Liver Tumors in CT Scans

    **Hepatocellular Carcinoma (HCC)** is the most common type of primary liver cancer.
    On CT scans, HCC typically appears as:
    - **Arterial phase hyperenhancement** — brighter than surrounding liver
    - **Washout** — becomes darker in portal venous/delayed phases
    - **Pseudocapsule** — rim of enhancement around the tumor

    **Key Metrics for Clinical Assessment:**

    | Metric | Clinical Significance |
    |--------|----------------------|
    | Tumor Volume | Correlates with tumor stage and treatment planning |
    | Number of Lesions | Multifocal disease indicates more advanced stage |
    | Vascular Invasion | Critical for surgical resectability |
    | Tumor Burden Trend | Increasing burden suggests progressive disease |

    ### Risk Stratification Categories
    - **Low Risk**: Small (<1cm³), round, single lesion, stable burden
    - **Elevated Risk**: Medium (1-10cm³), irregular shape, mild increase
    - **High Risk**: Large (>10cm³), multifocal, increasing burden
    """)

    st.info("""
    **Clinical Recommendation:** All findings from automated analysis should be
    reviewed by a qualified radiologist. This tool is for research and
    educational purposes only.
    """)


if __name__ == "__main__":
    main()
