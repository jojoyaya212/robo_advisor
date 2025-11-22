# ============================================================
# 🚀 DECO-ROBO ADVISOR (Wealthsimple-Style Edition)
# ============================================================
# Features:
# 1. Caching for speed
# 2. Clean, emoji-free UI (CSS injection)
# 3. Education Center
# 4. Black-Litterman Model (Investor Friendly + Quantified)
# 5. Dynamic "Master Filters" (Restores all your data work)
# 6. Enhanced Constraints (Min 5 ETFs) & SMART Data Matching

import os
import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import minimize

# ============================================================
# 🎨 UI CONFIGURATION & CSS STYLING
# ============================================================

st.set_page_config(
    page_title="ETF Robo Advisor", 
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Custom CSS for "Fintech" Look
st.markdown("""
    <style>
    /* Main Font */
    html, body, [class*="css"] {
        font-family: 'Inter', 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: #2e2e2e;
    }
    /* Headings */
    h1, h2, h3 {
        font-weight: 700;
        color: #1a1a1a;
    }
    /* Cards for Stats */
    div.stMetric {
        background-color: #f9f9f9;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 10px;
    }
    /* Buttons */
    div.stButton > button {
        background-color: #1a1a1a;
        color: white;
        border-radius: 8px;
        font-weight: 600;
        border: none;
        padding: 0.5rem 1rem;
    }
    div.stButton > button:hover {
        background-color: #333333;
        color: white;
    }
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #f4f4f4;
    }
    /* Expanders */
    .streamlit-expanderHeader {
        font-weight: 600;
        background-color: #ffffff;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

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

# Normalize sheet keys
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

# ============================================================
# 🧮 HELPER FUNCTIONS
# ============================================================

def get_ticker_col(df: pd.DataFrame) -> str | None:
    """Find the ticker column robustly."""
    for c in ["ticker", "Ticker", "TICKER", "symbol", "Symbol", "SYMBOL"]:
        if c in df.columns: return c
    return None

def get_volume_col(df: pd.DataFrame) -> str | None:
    """Find the volume column robustly (handles typos/spaces)."""
    candidates = ["1d volume", "1D Volume", "1d_volume", "Volume", "volume", "1D volume"]
    for c in candidates:
        if c in df.columns: return c
    for c in df.columns:
        if "volume" in str(c).lower() and "30" not in str(c): 
            return c
    return None

def clean_ticker(t):
    """Normalize ticker for matching (remove suffix, spaces)."""
    t = str(t).upper().strip()
    # Remove common suffixes like .TO, US, etc. for looser matching
    for suffix in [" US", " CN", ".TO", " CH", " JT", " TT"]:
        if t.endswith(suffix):
            t = t.replace(suffix, "").strip()
    return t

def get_prices_for(base_key: str, tickers: list[str]) -> tuple[pd.DataFrame, list, list]:
    """
    Returns: (Price DataFrame, List of Found Tickers, List of Missing Tickers)
    Uses smart matching to handle 'AAPL' vs 'AAPL US'.
    """
    price_df = price_sheets.get(base_key)
    if price_df is None or price_df.empty: return pd.DataFrame(), [], tickers
    
    # Create Smart Map: Clean Ticker -> Actual Column Name
    col_map = {}
    for c in price_df.columns:
        col_map[str(c).strip()] = c          # Exact match
        col_map[clean_ticker(c)] = c         # Clean match (no suffix)
    
    found_tickers = []
    missing_tickers = []
    matched_cols = []

    for t in tickers:
        t_clean = clean_ticker(t)
        t_exact = str(t).strip()
        
        # Try exact match first, then clean match
        if t_exact in col_map:
            matched_cols.append(col_map[t_exact])
            found_tickers.append(t)
        elif t_clean in col_map:
            matched_cols.append(col_map[t_clean])
            found_tickers.append(t)
        else:
            missing_tickers.append(t)
    
    if not matched_cols: return pd.DataFrame(), [], tickers
    
    # Select columns and remove duplicates (if any ticker mapped to same col)
    out = price_df[list(set(matched_cols))].copy()
    
    # FIX: Handle Data Gaps Robustly (Forward Fill)
    out = out.ffill() 
    out = out.dropna(axis=0, how="any") 
    
    return out, found_tickers, missing_tickers

def calculate_metrics(prices: pd.DataFrame, freq: int = 252):
    """Returns annualized mean returns and covariance matrix."""
    rets = np.log(prices / prices.shift(1)).dropna()
    if rets.empty: return None, None
    
    mu = rets.mean() * freq
    cov = rets.cov() * freq
    cov = cov + np.eye(cov.shape[0]) * 1e-6 # Regularization
    return mu, cov

def black_litterman_adjustment(mu_prior, cov, views, ticker_info):
    tau = 0.025 
    n_assets = len(mu_prior)
    tickers = mu_prior.index.tolist()
    
    active_views = [] 
    
    def get_indices(condition_col, condition_val):
        if condition_col not in ticker_info.columns: return []
        
        # 1. Find tickers in Info Sheet that match condition
        matches = ticker_info[ticker_info[condition_col] == condition_val]
        ticker_col = get_ticker_col(ticker_info)
        target_tickers = set(matches[ticker_col].apply(clean_ticker).tolist())
        
        # 2. Find indices in Price Matrix that match these tickers
        indices = []
        for i, t_price in enumerate(tickers):
            if clean_ticker(t_price) in target_tickers:
                indices.append(i)
        return indices

    # --- VIEW LOGIC ---
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
    
    # --- DIVERSIFICATION CONSTRAINTS (Dynamic) ---
    # If we have 5+ assets, cap at 20%.
    # If we have 3-4 assets, cap at 35%.
    # If we have < 3 assets, let it ride (100%).
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
# 📄 TABS LAYOUT
# ============================================================

st.title("ETF Robo Advisor") 
st.markdown("### The Deconstructed DIY Investment Kit")

tab_tool, tab_edu = st.tabs(["🛠️ Robo-Advisor Tool", "📚 Education Center"])

# ============================================================
# 📚 TAB 2: EDUCATION CENTER
# ============================================================
with tab_edu:
    st.header("ETF Education Center")
    st.caption("Understanding the building blocks of your portfolio.")
    
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
        st.markdown("""
        - **AUM:** Assets Under Management (Total size).
        - **MER:** Management Expense Ratio (Annual fee).
        - **NAV:** Net Asset Value (Fair value of holdings).
        - **Yield:** Income generated (dividends/interest).
        """)

# ============================================================
# 🛠️ TAB 1: THE TOOL
# ============================================================
with tab_tool:
    
    # --- SIDEBAR FILTERS ---
    with st.sidebar:
        st.header("1. Screen Assets")
        
        # 1. Region
        region_mode = st.radio("Region Focus", ["Canadian", "Global / US"], horizontal=True)
        
        # 2. Asset Class & Dataset Selection
        if region_mode == "Canadian":
            base_key = "canadian"
            asset_type = "All" 
        else:
            asset_type = st.selectbox("Asset Class", ["Equity (Sector)", "Equity (Thematic)", "Bond (Gov/Corp)", "High Yield"])
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
            st.error("Data not available for this selection.")
            st.stop()

        # 4. MASTER FILTER LOGIC
        st.markdown("### 🔍 Filters")
        active_filters = {}
        
        potential_filters = [
            "Management_Style", "ESG_Focus", "Issuer", "structure", 
            "Use_Derivative", "ETF_General_Type", "Strategic_Focus", 
            "Sector_Focus", "Theme_Focus", "Geographic_Focus",
            "Exchange_Region", "Leverage_Type"
        ]
        
        for col in potential_filters:
            col_match = next((c for c in df_info.columns if c.lower() == col.lower()), None)
            if col_match:
                unique_vals = sorted([str(x).strip() for x in df_info[col_match].dropna().unique() if str(x).strip() != ""])
                if unique_vals:
                    unique_vals = ["All"] + unique_vals
                    label = col_match.replace("_", " ").replace("exposure", "").title()
                    sel = st.selectbox(label, unique_vals, key=f"filt_{base_key}_{col_match}")
                    if sel != "All":
                        active_filters[col_match] = sel

    # --- MAIN CONTENT ---
    
    # 1. Apply Filters
    filtered_df = df_info.copy()
    for col, val in active_filters.items():
        filtered_df = filtered_df[filtered_df[col].astype(str).str.strip() == val]

    # 2. Display Screening Results
    st.subheader("2. Screening Results")
    
    if filtered_df.empty:
        st.warning("No ETFs match your filters.")
    else:
        # Dynamic Column Display
        vol_col_display = get_volume_col(filtered_df)
        disp_cols = ["ticker", "name", "Expense_Ratio", "YTD_Return"]
        if vol_col_display: disp_cols.append(vol_col_display)
        
        final_disp_cols = [c for c in disp_cols if c in filtered_df.columns]
        st.dataframe(filtered_df[final_disp_cols].head(50), use_container_width=True, hide_index=True)
        st.caption(f"Showing {len(filtered_df)} matching ETFs (Top 50 displayed)")

        # 3. Black-Litterman & Optimization
        st.markdown("---")
        st.subheader("3. Portfolio Construction (Black-Litterman)")
        
        col_risk, col_views = st.columns(2)
        
        with col_risk:
            st.markdown("**Risk Profile**")
            risk_level = st.select_slider(
                "How much risk can you handle?",
                options=["Conservative", "Moderate", "Aggressive"],
                value="Moderate"
            )
            lambdas = {"Conservative": 5.0, "Moderate": 2.5, "Aggressive": 1.0}
            
        with col_views:
            st.markdown("**Market Views (Customize your Outlook)**")
            st.caption("Combine your personal intuition with our math:")
            
            views = []
            c1, c2 = st.columns(2)
            
            if c1.checkbox("Tech Boom: I expect Tech to outperform (+25%)"): 
                views.append("Tech")
            if c1.checkbox("Energy Slump: I expect Energy to lag (-5%)"): 
                views.append("Energy")
            if c1.checkbox("NA Strength: I trust the N.American economy (+15%)"): 
                views.append("NorthAmerica")
            if c2.checkbox("EM Rally: I see high growth in Emerging Mkts (+18%)"): 
                views.append("EmergingMarkets")
            if c2.checkbox("Stability: I prefer low-volatility sectors (+10%)"): 
                views.append("Stability")
            if c2.checkbox("High Yield: I want to capture Bond income (+8%)"): 
                views.append("HighYield")

        # 4. Run Optimization Button
        if st.button("🚀 Generate Optimized Portfolio", type="primary"):
            
            with st.spinner("Crunching numbers... (Calculating Covariance, BL Posteriors, Efficient Frontier)"):
                # A. Get Data
                ticker_col = get_ticker_col(filtered_df)
                if not ticker_col:
                    st.error("Could not find ticker column.")
                    st.stop()
                
                sort_col = get_volume_col(filtered_df)
                if sort_col:
                    filtered_df[sort_col] = pd.to_numeric(filtered_df[sort_col], errors='coerce').fillna(0)
                    top_liquid = filtered_df.sort_values(by=sort_col, ascending=False).head(20)
                else:
                    st.warning("Volume column not found, optimizing first 20 ETFs.")
                    top_liquid = filtered_df.head(20)

                tickers = top_liquid[ticker_col].astype(str).tolist()
                
                # --- SMART DATA MATCHING ---
                prices, found_tickers, missing_tickers = get_prices_for(base_key, tickers)
                
                # Check logic: Why did we fall below 5 assets?
                if len(found_tickers) < 5:
                    with st.expander("⚠️ Data Diagnostics (Why < 5 Assets?)", expanded=True):
                        st.warning(f"Only {len(found_tickers)} valid price histories found out of {len(tickers)} top candidates.")
                        st.write(f"Optimization fell back to relaxed constraints (Max=100%) because fewer than 5 assets were available.")
                        
                        c1, c2 = st.columns(2)
                        with c1:
                            st.error(f"❌ Missing Price Data ({len(missing_tickers)})")
                            st.write(missing_tickers)
                        with c2:
                            st.success(f"✅ Found Price Data ({len(found_tickers)})")
                            st.write(found_tickers)
                        st.info("The system used smart matching (ignoring suffixes like .TO or US) but still couldn't find these tickers in the price sheet.")

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
                
                st.success("Optimization Complete!")
                
                r_col, m_col = st.columns([1, 2])
                
                with r_col:
                    st.markdown("#### Allocation")
                    w_df = pd.DataFrame({"Ticker": weights_display.index, "Weight": weights_display.values})
                    w_df["Weight"] = w_df["Weight"].apply(lambda x: f"{x:.1%}")
                    st.dataframe(w_df, hide_index=True, use_container_width=True)
                    
                with m_col:
                    st.markdown("#### Portfolio Stats")
                    port_ret = weights_full @ mu_bl
                    port_vol = np.sqrt(weights_full @ cov @ weights_full)
                    sharpe = port_ret / port_vol
                    
                    k1, k2, k3 = st.columns(3)
                    k1.metric("Exp. Return", f"{port_ret:.1%}")
                    k2.metric("Volatility", f"{port_vol:.1%}")
                    k3.metric("Sharpe Ratio", f"{sharpe:.2f}")
                    
                    st.bar_chart(weights_display)

# Footer
st.markdown("---")
st.caption("© 2025 Deco-Robo. Built for MFIN 706. Powered by Python & Streamlit.")
