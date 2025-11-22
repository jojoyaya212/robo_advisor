# ============================================================
# 🚀 DECO-ROBO ADVISOR (Wealthsimple-Style Edition)
# ============================================================
# Features:
# 1. Caching for speed
# 2. Clean, emoji-free UI (CSS injection)
# 3. Education Center
# 4. Black-Litterman Model Integration

import os
import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import minimize

# ============================================================
# 🎨 UI CONFIGURATION & CSS STYLING
# ============================================================

st.set_page_config(
    page_title="Deco-Robo",
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
# 🧮 MATH FUNCTIONS (Black-Litterman & Optimization)
# ============================================================

def get_ticker_col(df: pd.DataFrame) -> str | None:
    for c in ["ticker", "Ticker", "TICKER", "symbol", "Symbol", "SYMBOL"]:
        if c in df.columns: return c
    return None

def get_prices_for(base_key: str, tickers: list[str]) -> pd.DataFrame:
    price_df = price_sheets.get(base_key)
    if price_df is None or price_df.empty: return pd.DataFrame()
    
    # Normalize columns
    normalized_cols = {str(c).strip().lower(): c for c in price_df.columns}
    tickers_set = {t.strip().lower() for t in tickers}
    
    # Find intersection
    matched_cols = [normalized_cols[t] for t in tickers_set if t in normalized_cols]
    
    if not matched_cols: return pd.DataFrame()
    
    # Clean data
    out = price_df[matched_cols].copy()
    out = out.dropna(how="all").dropna(axis=0, how="any") # Strict cleaning
    return out

def calculate_metrics(prices: pd.DataFrame, freq: int = 252):
    """Returns annualized mean returns and covariance matrix."""
    # Calculate log returns
    rets = np.log(prices / prices.shift(1)).dropna()
    if rets.empty: return None, None
    
    mu = rets.mean() * freq
    cov = rets.cov() * freq
    
    # Add slight regularization to Covariance to prevent singular matrix errors
    cov = cov + np.eye(cov.shape[0]) * 1e-6
    return mu, cov

def black_litterman_adjustment(mu_prior, cov, views, ticker_info):
    """
    Adjusts expected returns based on user views.
    
    mu_prior: Historical mean returns (Equilibrium)
    cov: Covariance matrix
    views: List of selected view strings
    ticker_info: DataFrame containing metadata (Sector, Region) for mapping
    """
    # 1. Setup BL Parameters
    tau = 0.025  # Standard scaling factor
    n_assets = len(mu_prior)
    tickers = mu_prior.index.tolist()
    
    # 2. Construct View Matrices (P and Q)
    # P: Link matrix (which assets are affected)
    # Q: Expected return of the view
    
    active_views = [] # Store valid (P_row, Q_val) tuples
    
    # Helper to find indices for a condition
    def get_indices(condition_col, condition_val):
        matches = ticker_info[ticker_info[condition_col] == condition_val]
        valid_tickers = [t for t in matches['ticker'] if t in tickers]
        return [tickers.index(t) for t in valid_tickers]

    # --- VIEW LOGIC DEFINITIONS ---
    
    # View 1: "Tech Sector Boom" -> Tech stocks +5% absolute return
    if "Tech Sector Boom" in views:
        idx = get_indices('Sector_Focus', 'Technology')
        if idx:
            row = np.zeros(n_assets)
            row[idx] = 1 / len(idx) # Equal weight within the view
            active_views.append((row, 0.25)) # Expect 25% return from Tech (Aggressive)

    # View 2: "Energy Slump" -> Energy stocks -5% relative to history
    if "Energy Slump" in views:
        idx = get_indices('Sector_Focus', 'Energy')
        if idx:
            row = np.zeros(n_assets)
            row[idx] = 1 / len(idx)
            active_views.append((row, -0.05)) 

    # View 3: "North American Strength" -> NA Region +3%
    if "North American Strength" in views:
        idx = get_indices('Geographic_Focus', 'North America')
        if idx:
            row = np.zeros(n_assets)
            row[idx] = 1 / len(idx)
            active_views.append((row, 0.15)) 

    # View 4: "Emerging Markets Rally" -> EM Region +8%
    if "Emerging Markets Rally" in views:
        idx = get_indices('Geographic_Focus', 'Emerging Markets')
        if idx:
            row = np.zeros(n_assets)
            row[idx] = 1 / len(idx)
            active_views.append((row, 0.18))

    # View 5: "Stability Focus" -> Low Volatility stocks outperform
    if "Stability Focus (Low Vol)" in views:
        # Assume 'Utilities' and 'Consumer Staples' are proxies for Low Vol if explicit metric missing
        idx_u = get_indices('Sector_Focus', 'Utilities')
        idx_c = get_indices('Sector_Focus', 'Consumer Staples')
        idx = idx_u + idx_c
        if idx:
            row = np.zeros(n_assets)
            row[idx] = 1 / len(idx)
            active_views.append((row, 0.10))

    # View 6: "High Yield Opportunity" -> Bond ETFs +4%
    if "High Yield Opportunity" in views:
        # Check General Type or specific sector
        idx = get_indices('ETF_General_Type', 'Bond')
        if idx:
            row = np.zeros(n_assets)
            row[idx] = 1 / len(idx)
            active_views.append((row, 0.08))

    # 3. Calculate Posterior Estimate (The BL Formula)
    if not active_views:
        return mu_prior # No views, return historical mean

    P = np.array([v[0] for v in active_views])
    Q = np.array([v[1] for v in active_views]).reshape(-1, 1)
    
    # Uncertainty matrix (Omega) - assume standard confidence
    # Omega = diag(P * (tau * Sigma) * P.T)
    omega = np.diag(np.diag(P @ (tau * cov.values) @ P.T))
    
    # Inverse calculation
    sigma_inv = np.linalg.inv(tau * cov.values)
    omega_inv = np.linalg.inv(omega)
    
    # Posterior Expected Return: E[R] = [(τΣ)^-1 + P^T Ω^-1 P]^-1 * [(τΣ)^-1 Π + P^T Ω^-1 Q]
    term1 = np.linalg.inv(sigma_inv + P.T @ omega_inv @ P)
    term2 = (sigma_inv @ mu_prior.values.reshape(-1, 1)) + (P.T @ omega_inv @ Q)
    
    mu_bl = term1 @ term2
    return pd.Series(mu_bl.flatten(), index=tickers)

def optimize_portfolio(mu, cov, lambda_risk):
    n = len(mu)
    w0 = np.ones(n) / n
    
    def objective(w):
        # Maximize: w'mu - (lambda/2) * w'Cov'w
        ret = w @ mu
        var = w @ cov @ w
        return -(ret - (lambda_risk * var)) # Negative for minimization

    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    bounds = [(0.00, 1.0) for _ in range(n)] # Long only

    res = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=cons)
    
    if not res.success:
        # Fallback to Inverse Volatility
        iv = 1 / np.sqrt(np.diag(cov))
        return pd.Series(iv / iv.sum(), index=mu.index)
        
    return pd.Series(res.x, index=mu.index)

# ============================================================
# 📄 TABS LAYOUT
# ============================================================

st.title("Deco-Robo Advisor")
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
        
        # 2. Asset Class
        if region_mode == "Canadian":
            base_key = "canadian"
            asset_type = "All" # Canadian sheet is mixed
        else:
            asset_type = st.selectbox("Asset Class", ["Equity (Sector)", "Equity (Thematic)", "Bond (Gov/Corp)", "High Yield"])
            
            # Map selection to sheet key
            key_map = {
                "Equity (Sector)": "sector_etfs",
                "Equity (Thematic)": "thematic",
                "Bond (Gov/Corp)": "bond_etfs",
                "High Yield": "high_yield"
            }
            base_key = key_map[asset_type]

        # 3. Dynamic Filters based on dataset
        df_info = info_sheets.get(base_key, pd.DataFrame()).copy()
        
        if df_info.empty:
            st.error("Data not available for this selection.")
            st.stop()

        # Filter Logic
        active_filters = {}
        
        # Generate selectboxes for categorical columns dynamically
        filter_cols = ["Management_Style", "ESG_Focus", "Sector_Focus", "Strategic_Focus"]
        
        for col in filter_cols:
            if col in df_info.columns:
                unique_vals = ["All"] + sorted(list(df_info[col].dropna().astype(str).unique()))
                sel = st.selectbox(f"Filter by {col.replace('_',' ')}", unique_vals)
                if sel != "All":
                    active_filters[col] = sel

    # --- MAIN CONTENT ---
    
    # 1. Apply Filters
    filtered_df = df_info.copy()
    for col, val in active_filters.items():
        filtered_df = filtered_df[filtered_df[col].astype(str) == val]

    # 2. Display Screening Results
    st.subheader("2. Screening Results")
    
    if filtered_df.empty:
        st.warning("No ETFs match your filters.")
    else:
        # Create a clean display table
        disp_cols = [c for c in ["ticker", "name", "Expense_Ratio", "1d_volume", "YTD_Return"] if c in filtered_df.columns]
        st.dataframe(filtered_df[disp_cols].head(50), use_container_width=True, hide_index=True)
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
            # Risk Aversion Parameter (Lambda)
            lambdas = {"Conservative": 5.0, "Moderate": 2.5, "Aggressive": 1.0}
            
        with col_views:
            st.markdown("**Market Views (Black-Litterman)**")
            st.caption("Select your outlooks to adjust the model:")
            
            views = []
            c1, c2 = st.columns(2)
            if c1.checkbox("Tech Boom (+25%)"): views.append("Tech Sector Boom")
            if c1.checkbox("Energy Slump (-5%)"): views.append("Energy Slump")
            if c1.checkbox("NA Strength (+15%)"): views.append("North American Strength")
            if c2.checkbox("EM Rally (+18%)"): views.append("Emerging Markets Rally")
            if c2.checkbox("Stability (Low Vol)"): views.append("Stability Focus (Low Vol)")
            if c2.checkbox("High Yield (+8%)"): views.append("High Yield Opportunity")

        # 4. Run Optimization Button
        if st.button("🚀 Generate Optimized Portfolio", type="primary"):
            
            with st.spinner("Crunching numbers... (Calculating Covariance, BL Posteriors, Efficient Frontier)"):
                # A. Get Data
                ticker_col = get_ticker_col(filtered_df)
                # Take top 20 liquid ETFs to keep optimization fast & stable
                top_liquid = filtered_df.sort_values(by="1d_volume", ascending=False).head(20)
                tickers = top_liquid[ticker_col].astype(str).tolist()
                
                prices = get_prices_for(base_key, tickers)
                
                if len(prices.columns) < 2:
                    st.error("Not enough price history available for the selected assets.")
                else:
                    # B. Calculate Stats
                    mu_hist, cov = calculate_metrics(prices)
                    
                    if mu_hist is None:
                        st.error("Insufficient data points for covariance calculation.")
                    else:
                        # C. Apply Black-Litterman
                        # Pass the filtered DF to map views to tickers
                        mu_bl = black_litterman_adjustment(mu_hist, cov, views, top_liquid)
                        
                        # D. Optimize
                        weights = optimize_portfolio(mu_bl, cov, lambdas[risk_level])
                        
                        # E. Display Results
                        weights = weights[weights > 0.01].sort_values(ascending=False) # Filter small weights
                        
                        st.success("Optimization Complete!")
                        
                        r_col, m_col = st.columns([1, 2])
                        
                        with r_col:
                            st.markdown("#### Allocation")
                            # Clean table for weights
                            w_df = pd.DataFrame({"Ticker": weights.index, "Weight": weights.values})
                            w_df["Weight"] = w_df["Weight"].apply(lambda x: f"{x:.1%}")
                            st.dataframe(w_df, hide_index=True, use_container_width=True)
                            
                        with m_col:
                            st.markdown("#### Portfolio Stats")
                            # Calc portfolio metrics
                            port_ret = weights @ mu_bl
                            port_vol = np.sqrt(weights @ cov @ weights)
                            sharpe = port_ret / port_vol
                            
                            k1, k2, k3 = st.columns(3)
                            k1.metric("Exp. Return", f"{port_ret:.1%}")
                            k2.metric("Volatility", f"{port_vol:.1%}")
                            k3.metric("Sharpe Ratio", f"{sharpe:.2f}")
                            
                            st.bar_chart(weights)

# Footer
st.markdown("---")
st.caption("© 2025 Deco-Robo. Built for MFIN 706. Powered by Python & Streamlit.")
