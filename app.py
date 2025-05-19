# app.py
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import date
import numpy as np

# -------- Streamlit session defaults -----------
if "results" not in st.session_state:
    st.session_state["results"] = None
    st.session_state["label"]   = ""
    st.session_state["last_dt"] = ""

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
    return sorted(
        sym for sym in df["Symbol"].replace({"BRK.B": "BRK-B"}).unique()
        if sym != "ATVI"
    )

def run_momentum(symbols: list[str], label: str):
    total_syms = len(symbols)
    st.write(f"### 🔄 Loading {label} ticker list  —  {total_syms} symbols")

    progress = st.progress(0, text="Downloading prices…")
    status   = st.empty()

    chunks = [symbols[i:i+CHUNK_SIZE] for i in range(0, total_syms, CHUNK_SIZE)]
    close_list = []
    for idx, chunk in enumerate(chunks, 1):
        status.write(f"Batch {idx}/{len(chunks)} – {len(chunk)} tickers")
        close_list.append(
            yf.download(chunk, start=START_DATE, end=END_DATE,
                        progress=False, actions=False)["Close"]
        )
        progress.progress(idx/len(chunks))
    progress.empty(); status.write("✅ Price download complete")

    df_close = pd.concat(close_list, axis=1).sort_index()
    valid_cols = [c for c in df_close.columns if df_close[c].count() >= MIN_OBS]
    df_close = df_close[valid_cols]

    ret_252   = df_close.pct_change(252)
    ret_21    = df_close.pct_change(21)
    stdev_126 = df_close.pct_change().rolling(126).std()
    momentum  = (ret_252 - ret_21) / stdev_126

    last_date = momentum.dropna(how="all").index.max()
    latest_px = df_close.loc[last_date]

    df_out = pd.DataFrame({
        "momentum":        momentum.loc[last_date],
        "1yr_return (%)": (ret_252.loc[last_date]*100).round(2),
        "1m_return (%)":  (ret_21.loc[last_date]*100).round(2),
        "6m_std (%)":     (stdev_126.loc[last_date]*100).round(2),
        "last_price":      latest_px
    }).dropna().sort_values("momentum", ascending=False)

    df_out.insert(0, "Rank", range(1, len(df_out)+1))
    return last_date.date(), df_out

# ----------------------  STREAMLIT PAGE ----------------------
st.set_page_config(page_title="Stock Momentum Scanner", layout="wide")
st.title("📈 Stock Momentum Scanner")
st.caption("Momentum factors for S&P 500 / S&P 1500 or your own list.")

# -------- sidebar: sizing inputs --------------------------------
with st.sidebar:
    st.header("Position‑Sizing")
    port_val = st.number_input(
        "Total portfolio value",
        min_value=1_000,
        max_value=100_000_000,
        value=1_000_000,
        step=1_000
    )

# -------- action row --------------------------------------------
col1, col2, col3 = st.columns([2,2,3], gap="medium")
run_1500 = col1.button("Run on S&P 1500", use_container_width=True)
run_500  = col2.button("Run on S&P 500",  use_container_width=True)

with col3:
    st.markdown("**Upload CSV**  \n*(must contain column `Symbol`)*")
    upload = st.file_uploader(" ", type="csv", label_visibility="collapsed")
    run_custom = st.button("Run on my CSV", disabled=upload is None, use_container_width=True)

st.divider()

# -------- main logic --------------------------------------------
if run_1500 or run_500 or run_custom:
    if run_custom:
        df_raw = pd.read_csv(upload)
        if "Symbol" not in df_raw.columns:
            st.error("CSV missing `Symbol` header."); st.stop()
        symbols, label = _clean_symbols(df_raw), "Your list"
    else:
        key = "S&P 1500" if run_1500 else "S&P 500"
        symbols, label = _clean_symbols(pd.read_csv(DATASETS[key])), key

    with st.spinner(f"Running momentum on {label} …"):
        last_dt, tbl = run_momentum(symbols, label)

    # store results so they persist across widget interactions
    st.session_state["results"] = tbl
    st.session_state["label"]   = label
    st.session_state["last_dt"] = last_dt

# -------- display & interactive sizing if we have saved results ------------
if st.session_state["results"] is not None:
    tbl   = st.session_state["results"]
    label = st.session_state["label"]
    last_dt = st.session_state["last_dt"]

    st.success(f"Metrics as of {last_dt}. Select stocks below to size your portfolio.")
    st.dataframe(tbl, use_container_width=True)

    # multiselect in original momentum order but labelled with rank
    opts = [f"{i+1:02d} | {t}" for i, t in enumerate(tbl.index)]
    default = opts[:10]

    chosen_labels = st.multiselect(
        "Select stocks to include in portfolio 👉",
        options=opts,
        default=default
    )

    if chosen_labels:
        # sort selections back to momentum rank order
        chosen_labels_sorted = sorted(
            chosen_labels,
            key=lambda x: int(x.split('|')[0])  # rank is before the pipe
        )
        tickers = [c.split('|')[1].strip() for c in chosen_labels_sorted]
        tbl_top = tbl.loc[tickers].copy()
        positive_mom = tbl_top["momentum"].clip(lower=0)
        weights = positive_mom / positive_mom.sum()
        cash_alloc = weights * port_val
        shares = np.floor(cash_alloc / tbl_top["last_price"])
        tbl_top["shares"] = shares.astype(int)
        tbl_top = tbl_top.sort_values("Rank")

        st.subheader("📊 Portfolio allocation")
        st.dataframe(tbl_top, use_container_width=True)

        csv = tbl_top.to_csv().encode()
        st.download_button(
            "⬇️ Download portfolio CSV",
            csv,
            file_name=f"{label.replace(' ','').lower()}_{last_dt}_portfolio.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("Select at least one stock to build a portfolio.", icon="ℹ️")