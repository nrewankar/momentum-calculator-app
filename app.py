# app.py
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import date

# ----------------------  GLOBAL CONFIG  ----------------------
START_DATE = "2023-01-01"
END_DATE   = date.today().isoformat()
CHUNK_SIZE = 100
MIN_OBS    = 504

DATASETS = {
    "S&P 1500": "stocks_sp1500_current.csv",
    "S&P 500":  "stocks_sp500_current.csv",
}

# ----------------------  CORE CALC ---------------------------
def _clean_symbols(df: pd.DataFrame) -> list[str]:
    """Return sorted ticker list with a few hard-coded fixes."""
    return sorted(
        sym for sym in (
            df["Symbol"]
            .replace({"BRK.B": "BRK-B"})
            .unique()
            .tolist()
        )
        if sym != "ATVI"           # example exclusion
    )

def run_momentum(symbols: list[str], label: str):
    total_syms = len(symbols)
    st.write(f"### 🔄 Loading {label} ticker list")
    st.write(f"Found **{total_syms}** symbols.")

    progress = st.progress(0, text="Downloading prices…")
    status   = st.empty()

    chunks = [symbols[i:i + CHUNK_SIZE] for i in range(0, total_syms, CHUNK_SIZE)]
    close_list = []
    for idx, chunk in enumerate(chunks, start=1):
        status.write(f"Batch {idx}/{len(chunks)} – {len(chunk)} tickers")
        df = yf.download(
            tickers=chunk,
            start=START_DATE,
            end=END_DATE,
            progress=False,
            actions=False,
        )["Close"]
        close_list.append(df)
        progress.progress(idx / len(chunks))

    progress.empty()
    status.write("✅ Price download complete")

    df_close   = pd.concat(close_list, axis=1).sort_index()
    valid_cols = [c for c in df_close.columns if df_close[c].count() >= MIN_OBS]
    excluded   = sorted(set(df_close.columns) - set(valid_cols))

    st.write(
        f"""
        **History check (≥ {MIN_OBS} trading days)**  
        • OK: **{len(valid_cols)}**  
        • Excluded: **{len(excluded)}**
        """
    )

    df_close = df_close[valid_cols]

    ret_252   = df_close.pct_change(252)
    ret_21    = df_close.pct_change(21)
    stdev_126 = df_close.pct_change().rolling(126).std()
    momentum  = (ret_252 - ret_21) / stdev_126

    last_date = momentum.dropna(how="all").index.max()
    if last_date is None:
        st.error("No valid momentum data found.")
        st.stop()

    # Get latest prices for those tickers
    latest_prices = df_close.loc[last_date]

    df_out = pd.DataFrame({
        "momentum":   momentum.loc[last_date],
        "1yr_return (%)": (ret_252.loc[last_date] * 100).round(2),
        "1m_return (%)":  (ret_21.loc[last_date] * 100).round(2),
        "6m_std (%)":     (stdev_126.loc[last_date] * 100).round(2),
        "last_price":     latest_prices
    }).dropna().sort_values("momentum", ascending=False)

    # Add Rank column
    df_out.insert(5, "Rank", range(1, len(df_out) + 1))

    return last_date.date(), df_out

# ----------------------  STREAMLIT PAGE ----------------------
st.set_page_config(page_title="Stock Momentum Scanner", layout="wide")

st.title("📈 Stock Momentum Scanner")
st.caption("Calculate momentum factors for S&P 500, S&P 1500, or your own list.")

st.markdown(" ")

# ---------- left-aligned action row ----------
col1, col2, col3 = st.columns([2, 2, 3], gap="medium")

with col1:
    run_1500 = st.button("Run on S&P 1500", use_container_width=True)

with col2:
    run_500  = st.button("Run on S&P 500",  use_container_width=True)

with col3:
    st.markdown(
        "**Upload CSV**\n*(file **must** contain a column named `Symbol`)*",
        help="Header must literally be Symbol – case-sensitive."
    )
    uploaded_file = st.file_uploader(
        label="Drag & drop or browse",
        type="csv",
        label_visibility="collapsed"
    )
    run_custom = st.button("Run on my CSV",
                           disabled=uploaded_file is None,
                           use_container_width=True)

st.divider()

# ---------- logic ----------
if run_1500 or run_500 or run_custom:
    if run_custom:
        try:
            df_user = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()

        if "Symbol" not in df_user.columns:
            st.error("Uploaded CSV is missing a **Symbol** column.")
            st.stop()

        symbols = _clean_symbols(df_user)
        label   = "Your list"
    else:
        label_key = "S&P 1500" if run_1500 else "S&P 500"
        df_pres   = pd.read_csv(DATASETS[label_key])
        symbols   = _clean_symbols(df_pres)
        label     = label_key

    with st.spinner(f"Starting analysis for {label} …"):
        last_dt, results = run_momentum(symbols, label)

    st.success(f"Finished! Metrics as of **{last_dt}**")

    st.dataframe(results, use_container_width=True)

    st.markdown(" ")

    st.download_button(
        label="⬇️ Download CSV",
        data=results.to_csv().encode(),
        file_name=f"{label.replace(' ', '').lower()}_momentum_{last_dt}.csv",
        mime="text/csv",
        use_container_width=True,
    )
else:
    st.info("Choose or upload a list to analyze.", icon="ℹ️")