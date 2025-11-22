# ============================================================
# 🚀 DECO-ROBO ADVISOR (Wealthsimple-Style Edition)
# ============================================================

import os
import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import minimize
import re

# ============================================================
# 🎨 UI CONFIGURATION & CSS STYLING
# ============================================================

st.set_page_config(
    page_title="ETF Robo Advisor", 
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# (CSS unchanged)
# ...

# ============================================================
# ⚡ DATA LOADING (CACHED)
# ============================================================

FILE_PATH = "robo_advisor_data.xlsx"

@st.cache_data(show_spinner=False)
def load_data():
    if not os.path.exists(FILE_PATH):
        return None
    return pd.read_excel(FILE_PATH, sheet_name=None)

with st.spinner("Loading market data..."):
    sheets_all = load_data()

if sheets_all is None:
    st.error(f"File not found: {FILE_PATH}")
    st.stop()

def normalize(name: str) -> str:
    return (
        str(name).lower()
        .replace(" price", "")
        .replace("_price", "")
        .replace(" info", "")
        .replace("_info", "")
        .replace(" ", "_")
        .strip()
    )

raw_price_sheets = {k: v for k, v in sheets_all.items() if "price" in k.lower()}
raw_info_sheets  = {k: v for k, v in sheets_all.items() if "price" not in k.lower()}

price_sheets = {normalize(k): v for k, v in raw_price_sheets.items()}
info_sheets  = {normalize(k): v for k, v in raw_info_sheets.items()}

# (Helper functions unchanged)
# ...

# ============================================================
# 🛠️ TAB 1: THE TOOL
# ============================================================

with tab_tool:
    
    with st.sidebar:
        st.header("1. Screen Assets")
        
        region_mode = st.radio("Region Focus", ["Canadian", "Global / US"], horizontal=True)
        
        if region_mode == "Canadian":
            info_key = "canadian"
            price_key = "canadian"
            asset_type = "All"
        
        else:
            asset_type = st.selectbox(
                "Asset Class",
                ["Equity (Sector)", "Equity (Thematic)", "Bond (Gov/Corp)", "High Yield"]
            )

            # ============================================================
            # NEW: Correct Dual-Key Mapping System
            # ============================================================

            info_key_map = {
                "Equity (Sector)": "sector_etfs",
                "Equity (Thematic)": "thematic",
                "Bond (Gov/Corp)": "bond_etfs",
                "High Yield": "high_yield"
            }

            price_key_map = {
                "Equity (Sector)": "sector_",
                "Equity (Thematic)": "thematic",
                "Bond (Gov/Corp)": "bond_etf",
                "High Yield": "high_yield"
            }

            info_key  = info_key_map[asset_type]
            price_key = price_key_map[asset_type]

        # ============================================================
        # LOAD INFO DATA (Corrected)
        # ============================================================

        df_info = info_sheets.get(info_key, pd.DataFrame()).copy()
        
        if df_info.empty:
            st.error("Data not available for this selection.")
            st.stop()

        # (Filter logic unchanged)
        # ...

    # ============================================================
    # MAIN CONTENT
    # ============================================================

    # (Filter application unchanged)
    # ...

    if st.button("🚀 Generate Optimized Portfolio", type="primary"):
        
        with st.spinner("Crunching numbers..."):

            ticker_col = get_ticker_col(filtered_df)
            # (sorting unchanged)

            tickers = top_liquid[ticker_col].astype(str).tolist()

            # ============================================================
            # PRICE DATA LOOKUP (Corrected)
            # ============================================================

            prices, found_tickers, missing_tickers = get_prices_for(price_key, tickers)

            # (Diagnostics, BL, optimization unchanged)
            # ...

# Footer unchanged
