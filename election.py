import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Kenya Election Forecast · LSTM",
    page_icon="🗳",
    layout="wide",
)

st.title("🗳 Kenya Election Voting Trend Forecasting (LSTM)")
st.caption(
    "Upload `kenya_voting_patterns.csv` · Aggregates voter-level records into "
    "party vote-share time series (2013 → 2017 → 2022) then forecasts future elections."
)

# ── Helpers ────────────────────────────────────────────────────────────────────
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i : i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)


def build_timeseries(df, vote_col, filter_col=None, filter_val=None):
    """Aggregate voter rows → yearly party vote-share % table."""
    sub = df[df["Voting Turnout"] == "Yes"].copy()
    if filter_col and filter_val and filter_val != "All":
        sub = sub[sub[filter_col] == filter_val]
    if sub.empty:
        return pd.DataFrame()
    counts = sub.groupby(["Year", vote_col]).size().unstack(fill_value=0)
    pct = counts.div(counts.sum(axis=1), axis=0) * 100
    pct.index = pd.to_datetime(pct.index, format="%Y")
    return pct.sort_index()


def run_lstm(series, seq_len, forecast_steps, epochs=80):
    """Train a tiny LSTM on a 1-D series and return historical fit + future forecast."""
    vals = series.values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(vals)

    X, y = create_sequences(scaled, seq_len)
    if len(X) == 0:
        return None, None

    model = Sequential([
        LSTM(64, input_shape=(seq_len, 1), return_sequences=False),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=epochs, batch_size=4, verbose=0)

    # Fitted values on training window
    fitted_scaled = model.predict(X, verbose=0)
    fitted = scaler.inverse_transform(fitted_scaled).flatten()

    # Future forecast
    last_seq = scaled[-seq_len:]
    future = []
    for _ in range(forecast_steps):
        pred = model.predict(last_seq.reshape(1, seq_len, 1), verbose=0)
        future.append(pred[0, 0])
        last_seq = np.append(last_seq[1:], pred, axis=0)
    future = scaler.inverse_transform(np.array(future).reshape(-1, 1)).flatten()

    return fitted, future


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    uploaded = st.file_uploader("Upload kenya_voting_patterns.csv", type="csv")
    st.markdown("---")
    vote_race = st.selectbox(
        "Election Race",
        ["Presidential Vote", "MP Vote", "MCA Vote"],
    )
    filter_dim = st.selectbox(
        "Filter / Segment by",
        ["None", "Region", "County", "Gender"],
    )
    st.markdown("---")
    seq_len = st.slider("LSTM Sequence Length", 1, 2, 1,
                        help="How many past elections to look back. Max 2 given only 3 years of data.")
    forecast_steps = st.slider("Future Elections to Forecast", 1, 4, 2)
    epochs = st.slider("Training Epochs", 20, 200, 80, step=20)
    st.markdown("---")
    st.info("ℹ️ Only voters who turned out (`Voting Turnout = Yes`) are counted in vote shares.")

# ── Main ───────────────────────────────────────────────────────────────────────
if uploaded is None:
    st.info("👈 Upload `kenya_voting_patterns.csv` in the sidebar to begin.")
    st.stop()

# Load
df = pd.read_csv(uploaded)

# Validate
required = {"Voting Turnout", "Year", "Presidential Vote", "MP Vote", "MCA Vote", "Region", "County", "Gender"}
missing_cols = required - set(df.columns)
if missing_cols:
    st.error(f"CSV is missing expected columns: {missing_cols}")
    st.stop()

# ── Raw data preview ───────────────────────────────────────────────────────────
with st.expander("📋 Raw data preview", expanded=False):
    st.dataframe(df.head(50), use_container_width=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Voters", f"{len(df):,}")
    c2.metric("Turnout Rate", f"{(df['Voting Turnout']=='Yes').mean()*100:.1f}%")
    c3.metric("Election Years", ", ".join(map(str, sorted(df['Year'].unique()))))
    c4.metric("Counties", df['County'].nunique())

st.markdown("---")

# ── Build time-series ──────────────────────────────────────────────────────────
filter_col = None if filter_dim == "None" else filter_dim
filter_options = ["All"]
if filter_col:
    filter_options += sorted(df[filter_col].dropna().unique().tolist())

filter_val = "All"
if filter_col:
    filter_val = st.selectbox(f"Select {filter_col}", filter_options)

ts = build_timeseries(df, vote_col=vote_race,
                      filter_col=filter_col, filter_val=filter_val)

if ts.empty:
    st.warning("No data for the selected filters.")
    st.stop()

parties = ts.columns.tolist()

st.subheader(f"📊 Historical Vote Shares — {vote_race}")
label = f"  ·  {filter_col}: {filter_val}" if filter_col and filter_val != "All" else ""
st.caption(f"Turnout-weighted · 2013 → 2017 → 2022{label}")
st.dataframe(
    ts.style.format("{:.2f}%").background_gradient(cmap="YlOrRd", axis=0),
    use_container_width=True,
)

st.markdown("---")

# ── Party selector ─────────────────────────────────────────────────────────────
selected_parties = st.multiselect(
    "Select Parties to Forecast",
    options=parties,
    default=[p for p in ["UDA", "ODM", "Jubilee", "Wiper", "Ford Kenya"] if p in parties],
)

if not selected_parties:
    st.warning("Select at least one party.")
    st.stop()

st.markdown("---")

# ── LSTM per party ─────────────────────────────────────────────────────────────
st.subheader("🤖 LSTM Training & Forecast")

# Future year labels
last_year = ts.index[-1].year
future_years = [last_year + 5 * i for i in range(1, forecast_steps + 1)]  # ~5-yr election cycles
future_index = pd.to_datetime(future_years, format="%Y")

results = {}
col_pairs = st.columns(min(len(selected_parties), 3))

for idx, party in enumerate(selected_parties):
    series = ts[party]

    if series.nunique() <= 1:
        with col_pairs[idx % 3]:
            st.warning(f"{party}: Constant series — cannot forecast.")
        continue

    with col_pairs[idx % 3]:
        with st.spinner(f"Training LSTM for {party}…"):
            fitted, future = run_lstm(series, seq_len, forecast_steps, epochs)

        if fitted is None:
            st.warning(f"{party}: Not enough data points.")
            continue

        results[party] = future

        # Plot
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        fig.patch.set_facecolor("#0f1117")
        ax.set_facecolor("#0f1117")

        hist_years = [d.year for d in ts.index]
        fitted_years = hist_years[seq_len:]  # fitted covers training window

        # Historical bars
        ax.bar(hist_years, series.values, color="#1f3a5f", width=1.2, label="Historical", zorder=2)

        # Fitted line
        ax.plot(fitted_years, fitted, color="#60a5fa", lw=1.5,
                linestyle="--", label="LSTM fit", zorder=3)

        # Forecast
        ax.plot([hist_years[-1]] + future_years,
                [series.values[-1]] + list(future),
                color="#f97316", lw=2, marker="o", markersize=5,
                label="Forecast", zorder=4)

        # Shade forecast region
        ax.axvspan(hist_years[-1], future_years[-1], alpha=0.07, color="#f97316")

        ax.set_title(party, color="#e8e6e0", fontsize=13, pad=8)
        ax.set_ylabel("Vote Share (%)", color="#9ca3af", fontsize=9)
        ax.tick_params(colors="#6b7280", labelsize=8)
        ax.spines[:].set_color("#1e1e2e")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
        ax.legend(fontsize=7, facecolor="#1a1a2e", labelcolor="#9ca3af",
                  edgecolor="#1e1e2e", loc="best")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ── Forecast summary table ─────────────────────────────────────────────────────
if results:
    st.markdown("---")
    st.subheader("📈 Forecast Summary")

    forecast_df = pd.DataFrame(results, index=[str(y) for y in future_years])
    forecast_df.index.name = "Election Year"

    # Normalise rows so they sum to 100 (only across selected parties — others exist too)
    st.dataframe(
        forecast_df.style.format("{:.2f}%").background_gradient(cmap="RdYlGn", axis=1),
        use_container_width=True,
    )

    # Bar chart
    st.subheader("📉 Forecast Trend — All Selected Parties")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    fig2.patch.set_facecolor("#0f1117")
    ax2.set_facecolor("#0f1117")

    palette = ["#3b82f6", "#ef4444", "#22c55e", "#f97316", "#a855f7"]
    x = np.arange(forecast_steps)
    width = 0.8 / len(results)

    for i, (party, preds) in enumerate(results.items()):
        offset = (i - len(results) / 2 + 0.5) * width
        ax2.bar(x + offset, preds, width=width * 0.9,
                color=palette[i % len(palette)], label=party, zorder=2)

    ax2.set_xticks(x)
    ax2.set_xticklabels([str(y) for y in future_years], color="#9ca3af")
    ax2.set_ylabel("Forecast Vote Share (%)", color="#9ca3af")
    ax2.tick_params(colors="#6b7280")
    ax2.spines[:].set_color("#1e1e2e")
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax2.legend(facecolor="#1a1a2e", labelcolor="#e8e6e0", edgecolor="#1e1e2e")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    # Download
    st.download_button(
        "⬇️ Download Forecast CSV",
        data=forecast_df.to_csv(),
        file_name="kenya_election_forecast.csv",
        mime="text/csv",
    )

# ── Regional breakdown ─────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🗺️ Regional Vote Share Breakdown (Historical)")

region_ts = {}
for region in sorted(df["Region"].dropna().unique()):
    rt = build_timeseries(df, vote_col=vote_race, filter_col="Region", filter_val=region)
    if not rt.empty:
        region_ts[region] = rt

if region_ts:
    tabs = st.tabs(list(region_ts.keys()))
    for tab, (region, rdf) in zip(tabs, region_ts.items()):
        with tab:
            st.dataframe(
                rdf.style.format("{:.2f}%").background_gradient(cmap="Blues", axis=0),
                use_container_width=True,
            )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Kenya Election Forecast · LSTM · Data: `kenya_voting_patterns.csv` · "
    "20,000 synthetic voter records across 3 election cycles (2013, 2017, 2022)"
)
