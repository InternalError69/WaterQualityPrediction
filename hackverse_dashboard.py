# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
import json

# ---------------------------------------
# CONFIGURATION
# ---------------------------------------
DATA_PATH = "data/synthetic_water_quality_3months.csv"
MODEL_METRICS_PATH = "data/model_metrics.json"
CNN_HISTORY_PATH = "data/cnn_training_history.json"
CNN_SUMMARY_PATH = "data/cnn_model_summary.json"

LAKE_COORDS = [12.8530, 77.5902]  # Lake coordinates (Bengaluru)
TIME_COLUMN = "timestamp"

st.set_page_config(page_title="💧 Water Quality AI Dashboard", layout="wide")

# ---------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------
def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading {path}: {e}")
        return {}

def load_csv(path, time_col: str):
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
        df.columns = df.columns.str.strip().str.lower()
        # Ensure timestamp column name is consistent
        if time_col not in df.columns:
            for c in df.columns:
                if "time" in c:
                    df.rename(columns={c: time_col}, inplace=True)
                    break
        if time_col in df.columns:
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
            df = df.dropna(subset=[time_col])
        return df
    except Exception as e:
        st.error(f"Error loading {path}: {e}")
        return pd.DataFrame()

def predict_next_dump(spike_times: np.ndarray):
    """Estimate when next dumping might occur based on pattern intervals."""
    if spike_times is None or len(spike_times) < 2:
        return "Not enough data"
    spike_times = np.sort(spike_times.astype("datetime64[ns]"))
    intervals = np.diff(spike_times) / np.timedelta64(1, "D")
    if len(intervals) == 0 or np.any(~np.isfinite(intervals)):
        return "Not enough data"
    avg_interval = np.mean(intervals)
    last_time = spike_times[-1]
    next_pred = last_time + np.timedelta64(int(avg_interval * 24 * 60), "m")
    return pd.to_datetime(next_pred).strftime("%Y-%m-%d %H:%M:%S")

# ---------------------------------------
# SIDEBAR: PATHS
# ---------------------------------------
st.sidebar.header("⚙️ Configuration Paths")
st.sidebar.text(f"Data: {DATA_PATH}")
st.sidebar.text(f"Metrics: {MODEL_METRICS_PATH}")
st.sidebar.text(f"CNN History: {CNN_HISTORY_PATH}")
st.sidebar.text(f"CNN Summary: {CNN_SUMMARY_PATH}")

# ---------------------------------------
# LOAD DATA
# ---------------------------------------
data = load_csv(DATA_PATH, TIME_COLUMN)
metrics = load_json(MODEL_METRICS_PATH)
cnn_history = load_json(CNN_HISTORY_PATH)
cnn_summary = load_json(CNN_SUMMARY_PATH)

# ---------------------------------------
# QUICK OVERVIEW PLOT
# ---------------------------------------
with st.expander("Quick overview plot", expanded=False):
    if not data.empty:
        plot_cols = []
        color_map = {}
        for col, color in [("ph", "blue"), ("tds", "red"), ("turbidity", "green"), ("temperature", "orange")]:
            if col in data.columns:
                plot_cols.append(col)
                color_map[col] = color
        if len(plot_cols) >= 1 and TIME_COLUMN in data.columns:
            fig, ax = plt.subplots(figsize=(12, 4))
            for col in plot_cols:
                label = "pH" if col == "ph" else col.upper()
                ax.plot(data[TIME_COLUMN], data[col], label=label, color=color_map[col], linewidth=1)
            ax.set_title("Water Quality Over Time")
            ax.set_xlabel("Timestamp")
            ax.set_ylabel("Measured Values")
            ax.legend()
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.info("No plottable columns (ph/tds/turbidity/temperature) found.")
    else:
        st.warning("No data loaded yet. Check your CSV path.")

# ---------------------------------------
# TABS
# ---------------------------------------
tab1, tab2, tab3 = st.tabs(["🌊 Lake & Spike Detection", "🧠 CNN Metrics", "📊 Model Comparison"])

# ---------------------------------------
# TAB 1: Spike Detection + Heatmap
# ---------------------------------------
with tab1:
    st.header("🌊 Spike Detection & Lake Heatmap")

    if not data.empty:
        st.subheader("Detected Dumping Events Over Time")

        ph_col = "ph" if "ph" in data.columns else None
        dump_flag_col = "dump_detected" if "dump_detected" in data.columns else None

        if ph_col and dump_flag_col and TIME_COLUMN in data.columns:
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(data[TIME_COLUMN], data[ph_col], label="pH Level", color="blue", linewidth=1)

            # dump_detected is 0.0/1.0 in CSV; equality to 1 works
            spikes = data[data[dump_flag_col] == 1]
            if not spikes.empty:
                ax.scatter(spikes[TIME_COLUMN], spikes[ph_col], color="red", label="Detected Dump", zorder=5)

            ax.set_xlabel("Timestamp")
            ax.set_ylabel("pH Level")
            ax.set_title("pH Level Spikes (Dumping Events)")
            ax.legend()
            st.pyplot(fig)

            # One-hot dump type summary from *_dump_detected columns
            type_cols = [
                "mining_dump_detected",
                "paper_dump_detected",
                "chemical_dump_detected",
                "sewage_dump_detected",
            ]
            present = [c for c in type_cols if c in data.columns]
            if present:
                counts = {
                    c.replace("_dump_detected", "").replace("_", " ").title(): int(pd.to_numeric(data[c], errors="coerce").fillna(0).sum())
                    for c in present
                }
                dump_summary = pd.DataFrame(list(counts.items()), columns=["Dump Type", "Occurrences"])
                st.subheader("🧾 Summary of Detected Dumpings")
                st.dataframe(dump_summary, use_container_width=True)

            # Predict next dumping
            dump_times = data.loc[data[dump_flag_col] == 1, TIME_COLUMN].sort_values().to_numpy(dtype="datetime64[ns]")
            next_pred = predict_next_dump(dump_times)
            st.success(f"🕒 Predicted Next Dumping Event: {next_pred}")
        else:
            missing = []
            if not ph_col: missing.append("ph")
            if not dump_flag_col: missing.append("dump_detected")
            if TIME_COLUMN not in data.columns: missing.append(TIME_COLUMN)
            st.warning(f"Cannot render spike detection. Missing columns: {', '.join(missing)}")
    else:
        st.warning("No data loaded yet. Check your CSV path.")

    st.subheader("🗺️ Lake Node Heatmap (Bengaluru)")
    m = folium.Map(location=LAKE_COORDS, zoom_start=14)
    folium.CircleMarker(
        location=LAKE_COORDS,
        radius=15,
        popup="Node 1: pH=7.2, TDS=320, Turbidity=2.1 NTU, Temp=27°C",
        color="blue",
        fill=True,
        fill_color="cyan",
        fill_opacity=0.6,
    ).add_to(m)
    st_folium(m, width=800, height=400)

# ---------------------------------------
# TAB 2: CNN Metrics
# ---------------------------------------
with tab2:
    st.header("🧠 CNN Model Training Metrics")

    if cnn_history:
        train_acc = cnn_history.get("accuracy") or cnn_history.get("train_accuracy") or []
        val_acc = cnn_history.get("val_accuracy") or []
        train_loss = cnn_history.get("loss") or cnn_history.get("train_loss") or []
        val_loss = cnn_history.get("val_loss") or []

        has_acc = len(train_acc) > 0 or len(val_acc) > 0
        has_loss = len(train_loss) > 0 or len(val_loss) > 0

        if has_acc or has_loss:
            ncols = 2 if has_acc and has_loss else 1
            fig, ax = plt.subplots(1, ncols, figsize=(12, 4))
            if ncols == 1:
                ax = [ax]

            idx = 0
            if has_acc:
                ax[idx].plot(train_acc, label="Train Accuracy")
                if len(val_acc) > 0:
                    ax[idx].plot(val_acc, label="Validation Accuracy")
                ax[idx].set_title("CNN Accuracy")
                ax[idx].legend()
                idx += 1

            if has_loss:
                ax[idx].plot(train_loss, label="Train Loss")
                if len(val_loss) > 0:
                    ax[idx].plot(val_loss, label="Validation Loss")
                ax[idx].set_title("CNN Loss")
                ax[idx].legend()

            st.pyplot(fig)
        else:
            st.info("CNN history loaded, but no accuracy/loss arrays found.")
    else:
        st.warning("CNN metrics JSON not found or empty.")

    if cnn_summary:
        st.subheader("🧩 CNN Model Architecture")
        if isinstance(cnn_summary, dict):
            df_summary = pd.DataFrame([cnn_summary])
        else:
            df_summary = pd.DataFrame(cnn_summary)
        st.dataframe(df_summary, use_container_width=True)
    else:
        st.warning("CNN model summary not found.")

# ---------------------------------------
# TAB 3: Model Comparison
# ---------------------------------------
with tab3:
    st.header("📊 Model Accuracy Comparison")

    if metrics:
        try:
            df_metrics = pd.DataFrame(list(metrics.items()), columns=["Model", "Accuracy"])
            df_metrics["Accuracy"] = pd.to_numeric(df_metrics["Accuracy"], errors="coerce")
            st.bar_chart(df_metrics.set_index("Model"))
            st.dataframe(df_metrics, use_container_width=True)
        except Exception as e:
            st.error(f"Failed to render metrics: {e}")
            st.json(metrics)
    else:
        st.warning("Model metrics JSON not found or empty.")
