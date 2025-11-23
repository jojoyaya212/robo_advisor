# ============================================================
# 🌟 PRETTIER ROBO ADVISOR (UX/UI Enhanced Edition)
# ============================================================
# Based on the robust logic of 'robo.py', but with a focus on:
# 1. Cleaner, modern aesthetic (Custom CSS)
# 2. Improved User Flow
# 3. Clearer Explanations

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
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a cleaner, more professional look
st.markdown("""
    <style>
    /* Main Font */
    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', Helvetica, Arial, sans-serif;
        color: #333333;
        background-color: #ffffff;
    }
    
    /* Headings */
    h1, h2, h3 {
        font-weight: 700;
        color: #111111;
        letter-spacing: -0.5px;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #f7f9fc;
        border-right: 1px solid #e0e0e0;
    }
    
    /* Cards for Stats (Metrics) */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e6e6e6;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Primary Buttons */
    div.stButton > button {
        background-color: #000000;
        color: white;
        border-radius: 8px;
        font-weight: 600;
        border: none;
        padding: 0.6rem 1.2rem;
        transition: background-color 0.2s;
    }
    div.stButton > button:hover {
        background-color: #333333;
        color: white;
    }
    
    /* DataFrame Styling */
    div[data-testid="stDataFrame"] {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        font-weight: 600;
        background-color: #ffffff;
        border-radius: 5px;
    }
    
    /* Custom Header for Sections */
    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #555;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        border-bottom: 2px solid #eee;
        padding-bottom: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# ⚡ DATA LOADING (CACHED & EXPLICITLY MAPPED)
# ============================================================

FILE_PATH = "robo_advisor_data.xlsx"

# EXPLICIT MAPPING: Info Sheet Name -> Price Sheet Name
SHEET_PAIRS = {
    "sector_etfs": "sector_price",
    "bond_etfs": "bond_etf_price",
    "high_yield": "high yield price",
    "thematic": "thematic price",
    "canadian": "canadian_price",
    "em_bonds": "em_bonds_price"
}

@st.cache_data(show_spinner=False)
def load_data():
    if not os.path.exists(FILE_PATH):
        return None
    return pd.read_excel(FILE_PATH, sheet_name=None)

# Graceful loading with a clean spinner
with st.spinner("Initializing Deco-Robo... Loading market data..."):
    sheets_all = load_data()

if sheets_all is None:
    st.error(f"❌ Critical Error: Data file `{FILE_PATH}` not found. Please ensure it is in the application folder.")
    st.stop()

# Separate sheets based on explicit pairs
info_sheets = {}
price_sheets = {}

for info_name, price_name in SHEET_PAIRS.items():
    actual_keys = {k.lower(): k for k in sheets_all.keys()}
    
    if info_name.lower() in actual_keys:
        real_info_name = actual_keys[info_name.lower()]
        info_sheets[info_name] = sheets_all[real_info_name]
    
    if price_name.lower() in actual_keys:
        real_price_name = actual_keys[price_name.lower()]
        price_sheets[info_name] = sheets_all[real_price_name]

# ============================================================
# 🧮 HELPER FUNCTIONS
# ============================================================

def get_ticker_col(df: pd.DataFrame) -> str | None:
    """Find the ticker column robustly."""
    for c in ["ticker", "Ticker", "TICKER", "symbol", "Symbol", "SYMBOL"]:
        if c in df.columns: return c
    return None

def get_volume_col(df: pd.DataFrame) -> str | None:
    """Find the volume column robustly."""
    candidates = ["1d volume", "1D Volume", "1d_volume", "Volume", "volume", "1D volume"]
    for c in candidates:
        if c in df.columns: return c
    for c in df.columns:
        if "volume" in str(c).lower() and "30" not in str(c): 
            return c
    return None

def ultra_clean_ticker(t):
    """Aggressive cleaner: removes ALL spaces and common suffixes."""
    t = str(t).upper()
    t = "".join(t.split())
    for suffix in ["US", "CN", ".TO", "CH", "JT", "TT", "LN", "GR", "JP", "AU", "SW"]:
        if t.endswith(suffix):
            t = t[:-len(suffix)]
            break 
    return t

def get_prices_for(base_key: str, tickers: list[str]) -> tuple[pd.DataFrame, list, list]:
    """Returns: (Price DataFrame, List of Found Tickers, List of Missing Tickers)"""
    price_df = price_sheets.get(base_key)
    
    if price_df is None or price_df.empty: 
        return pd.DataFrame(), [], tickers
    
    col_map = {}
    for c in price_df.columns:
        col_map[str(c).strip()] = c
        col_map[ultra_clean_ticker(c)] = c
    
    found_tickers = []
    missing_tickers = []
    matched_cols = []

    for t in tickers:
        t_original = str(t).strip()
        t_clean = ultra_clean_ticker(t)
        
        if t_original in col_map:
            matched_cols.append(col_map[t_original])
            found_tickers.append(t)
        elif t_clean in col_map:
            matched_cols.append(col_map[t_clean])
            found_tickers.append(t)
        else:
            missing_tickers.append(t)
    
    if not matched_cols: return pd.DataFrame(), [], tickers
    
    out = price_df[list(set(matched_cols))].copy()
    
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"])
        out.set_index("Date", inplace=True)
    
    out = out.ffill() 
    out = out.dropna(axis=0, how="any") 
    
    return out, found_tickers, missing_tickers

def calculate_metrics(prices: pd.DataFrame, freq: int = 252):
    """Returns annualized mean returns and covariance matrix."""
    prices = prices.apply(pd.to_numeric, errors='coerce')
    rets = np.log(prices / prices.shift(1)).dropna()
    if rets.empty: return None, None
    
    mu = rets.mean() * freq
    cov = rets.cov() * freq
    cov = cov + np.eye(cov.shape[0]) * 1e-6 
    return mu, cov

def black_litterman_adjustment(mu_prior, cov, views, ticker_info):
    tau = 0.025 
    n_assets = len(mu_prior)
    tickers = mu_prior.index.tolist()
    active_views = [] 
    
    def get_indices(condition_col, condition_val):
        if condition_col not in ticker_info.columns: return []
        matches = ticker_info[ticker_info[condition_col] == condition_val]
        ticker_col = get_ticker_col(ticker_info)
        target_clean = set(matches[ticker_col].apply(ultra_clean_ticker).tolist())
        indices = []
        for i, t_price in enumerate(tickers):
            if ultra_clean_ticker(t_price) in target_clean:
                indices.append(i)
        return indices

    if "Tech" in views:
        idx = get_indices('Sector_Focus', 'Technology')
        if idx:
            row = np.zeros(n_assets)
            row[idx] = 1 / len(idx)
            active_views.append((row, 0.25)) 

    if "Energy" in views:
        idx = get_indices('Sector_Focus', 'Energy')
        if idx:
            row = np.zeros(n_assets)
            row[idx] = 1 / len(idx)
            active_views.append((row, -0.05)) 

    if "NorthAmerica" in views:
        idx = get_indices('Geographic_Focus', 'North America')
        if idx:
            row = np.zeros(n_assets)
            row[idx] = 1 / len(idx)
            active_views.append((row, 0.15)) 

    if "EmergingMarkets" in views:
        idx = get_indices('Geographic_Focus', 'Emerging Markets')
        if idx:
            row = np.zeros(n_assets)
            row[idx] = 1 / len(idx)
            active_views.append((row, 0.18)) 

    if "Stability" in views:
        idx_u = get_indices('Sector_Focus', 'Utilities')
        idx_c = get_indices('Sector_Focus', 'Consumer Staples')
        idx = idx_u + idx_c
        if idx:
            row = np.zeros(n_assets)
            row[idx] = 1 / len(idx)
            active_views.append((row, 0.10)) 

    if "HighYield" in views:
        idx = get_indices('ETF_General_Type', 'Bond')
        if idx:
            row = np.zeros(n_assets)
            row[idx] = 1 / len(idx)
            active_views.append((row, 0.08)) 

    if not active_views:
        return mu_prior

    P = np.array([v[0] for v in active_views])
    Q = np.array([v[1] for v in active_views]).reshape(-1, 1)
    omega = np.diag(np.diag(P @ (tau * cov.values) @ P.T))
    
    try:
        sigma_inv = np.linalg.inv(tau * cov.values)
        omega_inv = np.linalg.inv(omega)
        term1 = np.linalg.inv(sigma_inv + P.T @ omega_inv @ P)
        term2 = (sigma_inv @ mu_prior.values.reshape(-1, 1)) + (P.T @ omega_inv @ Q)
        mu_bl = term1 @ term2
        return pd.Series(mu_bl.flatten(), index=tickers)
    except:
        return mu_prior

def optimize_portfolio(mu, cov, lambda_risk):
    n = len(mu)
    w0 = np.ones(n) / n
    
    def objective(w):
        ret = w @ mu
        var = w @ cov @ w
        return -(ret - (lambda_risk * var))

    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    
    if n >= 5:
        max_weight = 0.20
    elif n >= 3:
        max_weight = 0.35
    else:
        max_weight = 1.0
        
    bounds = [(0.0, max_weight) for _ in range(n)]

    res = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=cons)
    
    status = {
        "success": res.success,
        "message": res.message,
        "weights": pd.Series(res.x, index=mu.index) if res.success else pd.Series(np.ones(n)/n, index=mu.index)
    }
    return status

# ============================================================
# 📄 MAIN LAYOUT
# ============================================================

# Header Area
st.title("ETF Robo Advisor")
st.markdown("#### Empowering Canadian Investors with Institutional-Grade Tools")
st.markdown("---")

# Create Tabs
tab_tool, tab_edu = st.tabs(["🛠️ Robo-Advisor Tool", "📚 Education Center"])

# ============================================================
# 📚 TAB 2: EDUCATION CENTER
# ============================================================
with tab_edu:
    st.header("ETF Education Center")
    st.caption("Master the basics before you invest.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        with st.expander("1. What is an ETF?", expanded=True):
            st.markdown("""
            **Exchange-Traded Fund (ETF)** is a basket of securities that trades on an exchange just like a stock.
            - **Diversification:** One ETF can hold thousands of stocks or bonds.
            - **Liquidity:** Trade anytime during market hours.
            - **Transparency:** Holdings are disclosed daily.
            """)
            
        with st.expander("2. How are ETFs Created? (The Mechanism)"):
            st.markdown("""
            ETFs rely on **Authorized Participants (APs)** to keep prices fair.
            1. **Creation:** AP buys underlying stocks → Delivers to Issuer → Gets ETF shares.
            2. **Redemption:** AP returns ETF shares → Gets underlying stocks.
            
            *This arbitrage mechanism ensures the ETF price stays close to its Net Asset Value (NAV).*
            """)
            if os.path.exists("image_a63b46.png"):
                st.image("image_a63b46.png", caption="Creation/Redemption Process")

        with st.expander("3. Canadian ETF Landscape"):
            st.info("Did you know? The Canadian ETF market exceeds **$600 Billion** in assets.")
            st.markdown("""
            - **Regulation:** Governed by *National Instrument 81-102*.
            - **Taxation:** ETFs are tax-efficient but don't have the exact same capital gains deferral structure as US ETFs.
            - **Major Players:** RBC iShares, BMO, Vanguard Canada.
            """)

    with col2:
        st.markdown("#### Quick Glossary")
        st.info("""
        **AUM (Assets Under Management):**
        The total market value of the investments that a person or entity handles on behalf of investors.
        
        **MER (Management Expense Ratio):**
        The total annual fee charged to manage the fund, expressed as a percentage.
        
        **NAV (Net Asset Value):**
        The total value of the fund's assets minus its liabilities.
        
        **Yield:**
        The income returned on an investment, such as the interest received or dividends earned.
        """)

# ============================================================
# 🛠️ TAB 1: THE TOOL
# ============================================================
with tab_tool:
    
    # --- SIDEBAR FILTERS ---
    with st.sidebar:
        st.header("1. Asset Selection")
        
        # Debug info toggle (hidden by default for cleaner UI)
        if st.checkbox("Show Data Diagnostics"):
            with st.expander("Debug Info"):
                st.write(f"Info Sheets Loaded: {len(info_sheets)}")
                st.write(f"Price Sheets Loaded: {len(price_sheets)}")
                st.write("Active Pairs:", list(info_sheets.keys()))

        # 1. Region
        st.markdown('<div class="section-header">Region</div>', unsafe_allow_html=True)
        region_mode = st.radio("Select Focus", ["Canadian", "Global / US"], horizontal=True, label_visibility="collapsed")
        
        # 2. Asset Class & Dataset Selection
        st.markdown('<div class="section-header">Category</div>', unsafe_allow_html=True)
        if region_mode == "Canadian":
            base_key = "canadian"
            asset_type = "All" 
            st.caption("Accessing full Canadian ETF universe.")
        else:
            asset_type = st.selectbox("Select Category", ["Equity (Sector)", "Equity (Thematic)", "Bond (Gov/Corp)", "High Yield"], label_visibility="collapsed")
            # Use keys from SHEET_PAIRS
            key_map = {
                "Equity (Sector)": "sector_etfs",
                "Equity (Thematic)": "thematic",
                "Bond (Gov/Corp)": "bond_etfs",
                "High Yield": "high_yield"
            }
            base_key = key_map[asset_type]

        # 3. Load Data
        df_info = info_sheets.get(base_key, pd.DataFrame()).copy()
        
        if df_info.empty:
            st.error(f"Data not found for key: {base_key}")
            st.stop()

        # 4. MASTER FILTER LOGIC
        st.markdown('<div class="section-header">Refine Selection</div>', unsafe_allow_html=True)
        active_filters = {}
        
        potential_filters = [
            "Management_Style", "ESG_Focus", "Issuer", "structure", 
            "Use_Derivative", "ETF_General_Type", "Strategic_Focus", 
            "Sector_Focus", "Theme_Focus", "Geographic_Focus",
            "Exchange_Region", "Leverage_Type"
        ]
        
        # Create filters without cluttering the UI
        count_filters = 0
        for col in potential_filters:
            col_match = next((c for c in df_info.columns if c.lower() == col.lower()), None)
            if col_match:
                unique_vals = sorted([str(x).strip() for x in df_info[col_match].dropna().unique() if str(x).strip() != ""])
                if unique_vals:
                    count_filters += 1
                    unique_vals = ["All"] + unique_vals
                    label = col_match.replace("_", " ").replace("exposure", "").title()
                    sel = st.selectbox(label, unique_vals, key=f"filt_{base_key}_{col_match}")
                    if sel != "All":
                        active_filters[col_match] = sel
        
        if count_filters == 0:
            st.info("No additional filters available for this category.")

    # --- MAIN CONTENT AREA ---
    
    # 1. Apply Filters
    filtered_df = df_info.copy()
    for col, val in active_filters.items():
        filtered_df = filtered_df[filtered_df[col].astype(str).str.strip() == val]

    # 2. Display Screening Results
    st.subheader("2. Screening Results")
    
    col_count, col_msg = st.columns([1, 3])
    with col_count:
        st.metric("ETFs Found", len(filtered_df))
    
    if filtered_df.empty:
        st.warning("No ETFs match your filters. Please adjust your selection in the sidebar.")
    else:
        # Dynamic Column Display
        vol_col_display = get_volume_col(filtered_df)
        disp_cols = ["ticker", "name", "Expense_Ratio", "YTD_Return"]
        if vol_col_display: disp_cols.append(vol_col_display)
        
        final_disp_cols = [c for c in disp_cols if c in filtered_df.columns]
        
        with st.expander("View ETF Table", expanded=True):
            st.dataframe(filtered_df[final_disp_cols].head(50), use_container_width=True, hide_index=True)
            st.caption(f"Showing Top 50 by liquidity (if volume data available). Total matches: {len(filtered_df)}")

        # 3. Black-Litterman & Optimization
        st.markdown("---")
        st.subheader("3. Portfolio Construction")
        st.caption("We use the Black-Litterman model to combine historical data with your personal market views.")
        
        # Layout for inputs
        col_risk, col_views = st.columns([1, 2])
        
        with col_risk:
            st.markdown("#### Your Risk Profile")
            risk_level = st.select_slider(
                "Select your comfort level:",
                options=["Conservative", "Moderate", "Aggressive"],
                value="Moderate"
            )
            lambdas = {"Conservative": 5.0, "Moderate": 2.5, "Aggressive": 1.0}
            
            st.info(f"**Strategy:** {risk_level} optimization focuses on {'capital preservation' if risk_level=='Conservative' else 'growth' if risk_level=='Aggressive' else 'balanced returns'}.")
            
        with col_views:
            st.markdown("#### Market Views (Black-Litterman)")
            st.caption("Select any specific outlooks you have for the market:")
            
            views = []
            c1, c2 = st.columns(2)
            
            if c1.checkbox("Tech Boom (+25%)"): views.append("Tech")
            if c1.checkbox("Energy Slump (-5%)"): views.append("Energy")
            if c1.checkbox("NA Strength (+15%)"): views.append("NorthAmerica")
            if c2.checkbox("EM Rally (+18%)"): views.append("EmergingMarkets")
            if c2.checkbox("Stability (+10%)"): views.append("Stability")
            if c2.checkbox("High Yield (+8%)"): views.append("High Yield")

        st.write("") # Spacer
        
        # 4. Run Optimization Button
        # Centered button for better UX
        col_spacer1, col_btn, col_spacer2 = st.columns([1, 2, 1])
        with col_btn:
            run_btn = st.button("🚀 Generate Optimized Portfolio", type="primary", use_container_width=True)

        if run_btn:
            with st.spinner("Crunching numbers... (Calculating Covariance, BL Posteriors, Efficient Frontier)"):
                # A. Get Data
                ticker_col = get_ticker_col(filtered_df)
                if not ticker_col:
                    st.error("Could not find ticker column.")
                    st.stop()
                
                sort_col = get_volume_col(filtered_df)
                if sort_col:
                    filtered_df[sort_col] = pd.to_numeric(filtered_df[sort_col], errors='coerce').fillna(0)
                    # Take top 30 to increase chance of finding 5 valid price histories
                    top_liquid = filtered_df.sort_values(by=sort_col, ascending=False).head(30)
                else:
                    st.warning("Volume column not found, optimizing first 30 ETFs.")
                    top_liquid = filtered_df.head(30)

                tickers = top_liquid[ticker_col].astype(str).tolist()
                
                # --- SMART DATA MATCHING ---
                prices, found_tickers, missing_tickers = get_prices_for(base_key, tickers)
                
                # Check logic: Why did we fall below 5 assets?
                if len(found_tickers) < 5:
                    st.warning(f"⚠️ Note: Only {len(found_tickers)} valid price histories were found for the top candidates. Optimization constraints have been relaxed.")
                    with st.expander("View Missing Data Details"):
                        st.write(f"**Missing Tickers:** {missing_tickers}")
                        st.info("This usually happens because the tickers in your 'Info' sheet don't exactly match the column headers in your 'Price' sheet (e.g. 'AAPL' vs 'AAPL US').")

                if len(prices) < 10: 
                    st.error("Optimization Failed: Insufficient overlapping price history.")
                    st.stop()

                # B. Calculate Stats
                mu_hist, cov = calculate_metrics(prices)
                
                if mu_hist is None or mu_hist.isnull().values.any():
                    st.error("Optimization Failed: Covariance matrix contains NaNs.")
                    st.stop()
                
                # C. Apply Black-Litterman
                mu_bl = black_litterman_adjustment(mu_hist, cov, views, top_liquid)
                
                # D. Optimize
                result = optimize_portfolio(mu_bl, cov, lambdas[risk_level])
                weights_full = result["weights"]
                
                if not result["success"]:
                    st.warning(f"Optimization Warning: Solver failed to converge. (Reason: {result['message']})")
                
                # E. Display Results
                weights_display = weights_full[weights_full > 0.01].sort_values(ascending=False)
                
                st.success("✅ Optimization Complete!")
                
                r_col, m_col = st.columns([1, 2])
                
                with r_col:
                    st.markdown("#### Your Portfolio")
                    w_df = pd.DataFrame({"Ticker": weights_display.index, "Weight": weights_display.values})
                    w_df["Weight"] = w_df["Weight"].apply(lambda x: f"{x:.1%}")
                    st.dataframe(w_df, hide_index=True, use_container_width=True)
                    
                with m_col:
                    st.markdown("#### Portfolio Stats")
                    port_ret = weights_full @ mu_bl
                    port_vol = np.sqrt(weights_full @ cov @ weights_full)
                    sharpe = port_ret / port_vol
                    
                    k1, k2, k3 = st.columns(3)
                    k1.metric("Exp. Return", f"{port_ret:.1%}", delta="Annualized")
                    k2.metric("Volatility", f"{port_vol:.1%}", delta_color="inverse")
                    k3.metric("Sharpe Ratio", f"{sharpe:.2f}")
                    
                    st.bar_chart(weights_display)

# Footer
st.markdown("---")
st.caption("© 2025 Deco-Robo. Built for MFIN 706. Powered by Python & Streamlit.")
