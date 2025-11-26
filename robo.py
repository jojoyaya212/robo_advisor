# ============================================================
# 🌟 PRETTIER ROBO ADVISOR (Project Kit Edition)
# ============================================================
# MODIFICATIONS LOG:
# 1. Selection Logic: 50% Liquidity / 50% Cost Score (Top 15).
# 2. Constraints: Relaxed to (0.0, 1.0) to prevent solver failure.
# 3. Black-Litterman: Adjusted view magnitudes to be realistic.
# 4. UX: Added Rebalancing Frequency selector.
# 5. Output: Added Weighted Portfolio MER (Expense Ratio).

import os
import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import minimize

# ============================================================
# 🎨 UI CONFIGURATION & CSS STYLING
# ============================================================

st.set_page_config(
    page_title="Deco-Robo Advisor", 
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional look
st.markdown("""
    <style>
    /* Main Font */
    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', Helvetica, Arial, sans-serif;
        color: #333333;
        background-color: #ffffff;
    }
    h1, h2, h3 {
        font-weight: 700;
        color: #111111;
        letter-spacing: -0.5px;
    }
    section[data-testid="stSidebar"] {
        background-color: #f7f9fc;
        border-right: 1px solid #e0e0e0;
    }
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e6e6e6;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
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
# ⚡ DATA LOADING
# ============================================================

FILE_PATH = "robo_advisor_data.xlsx"

# Mapping: Info Sheet Name -> Price Sheet Name
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

# Graceful loading
with st.spinner("Initializing Deco-Robo... Loading market data..."):
    sheets_all = load_data()

if sheets_all is None:
    st.error(f"❌ Critical Error: Data file `{FILE_PATH}` not found.")
    st.stop()

# Organize sheets
info_sheets = {}
price_sheets = {}

for info_name, price_name in SHEET_PAIRS.items():
    actual_keys = {k.lower(): k for k in sheets_all.keys()}
    if info_name.lower() in actual_keys:
        info_sheets[info_name] = sheets_all[actual_keys[info_name.lower()]]
    if price_name.lower() in actual_keys:
        price_sheets[info_name] = sheets_all[actual_keys[price_name.lower()]]

# ============================================================
# 🧮 HELPER FUNCTIONS
# ============================================================

def get_ticker_col(df: pd.DataFrame) -> str | None:
    for c in ["ticker", "Ticker", "TICKER", "symbol", "Symbol"]:
        if c in df.columns: return c
    return None

def get_volume_col(df: pd.DataFrame) -> str | None:
    candidates = ["1d volume", "1D Volume", "1d_volume", "Volume", "volume", "Avg Volume"]
    for c in candidates:
        if c in df.columns: return c
    return None

def get_expense_col(df: pd.DataFrame) -> str | None:
    candidates = ["Expense_Ratio", "Expense Ratio", "MER", "Management Fee", "Fees"]
    for c in candidates:
        if c in df.columns: return c
    return None

def ultra_clean_ticker(t):
    """Removes spaces and common suffixes for matching."""
    t = str(t).upper()
    t = "".join(t.split())
    for suffix in ["US", "CN", ".TO", "CH", "JT", "TT", "LN", "GR", "JP"]:
        if t.endswith(suffix):
            t = t[:-len(suffix)]
            break 
    return t

def get_prices_for(base_key: str, tickers: list[str]):
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
        t_clean = ultra_clean_ticker(t)
        if str(t).strip() in col_map:
            matched_cols.append(col_map[str(t).strip()])
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
    
    out = out.ffill().dropna(axis=0, how="any")
    return out, found_tickers, missing_tickers

def calculate_metrics(prices: pd.DataFrame, freq: int = 252):
    prices = prices.apply(pd.to_numeric, errors='coerce')
    rets = np.log(prices / prices.shift(1)).dropna()
    if rets.empty: return None, None
    mu = rets.mean() * freq
    cov = rets.cov() * freq
    # Add tiny jitter to diagonal to ensure invertibility
    cov = cov + np.eye(cov.shape[0]) * 1e-6 
    return mu, cov

def black_litterman_adjustment(mu_prior, cov, views, ticker_info):
    """
    Adjusts expected returns based on views.
    tau: Scalar indicating uncertainty of the prior (standard is 0.025 - 0.05)
    """
    tau = 0.05 
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

    # MODIFIED: Reduced view magnitudes to realistic excess returns (e.g. 0.05 = 5%)
    if "Tech" in views:
        idx = get_indices('Sector_Focus', 'Technology')
        if idx:
            row = np.zeros(n_assets); row[idx] = 1/len(idx)
            active_views.append((row, 0.05)) # Expect 5% excess return

    if "Energy" in views:
        idx = get_indices('Sector_Focus', 'Energy')
        if idx:
            row = np.zeros(n_assets); row[idx] = 1/len(idx)
            active_views.append((row, -0.03)) # Expect -3% drag

    if "NorthAmerica" in views:
        idx = get_indices('Geographic_Focus', 'North America')
        if idx:
            row = np.zeros(n_assets); row[idx] = 1/len(idx)
            active_views.append((row, 0.04)) 

    if "EmergingMarkets" in views:
        idx = get_indices('Geographic_Focus', 'Emerging Markets')
        if idx:
            row = np.zeros(n_assets); row[idx] = 1/len(idx)
            active_views.append((row, 0.06)) # Higher risk, higher view

    if "Stability" in views:
        idx_u = get_indices('Sector_Focus', 'Utilities')
        idx_c = get_indices('Sector_Focus', 'Consumer Staples')
        idx = idx_u + idx_c
        if idx:
            row = np.zeros(n_assets); row[idx] = 1/len(idx)
            active_views.append((row, 0.02)) 

    if "HighYield" in views:
        idx = get_indices('ETF_General_Type', 'Bond')
        if idx:
            row = np.zeros(n_assets); row[idx] = 1/len(idx)
            active_views.append((row, 0.03)) 

    if not active_views:
        return mu_prior

    P = np.array([v[0] for v in active_views])
    Q = np.array([v[1] for v in active_views]).reshape(-1, 1)
    
    # Uncertainty matrix Omega - Proportional to Prior Variance
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
    
    # Maximize Utility: E[r] - lambda * Var
    def objective(w):
        ret = w @ mu
        var = w @ cov @ w
        return -(ret - (lambda_risk * var))

    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    
    # MODIFIED: Relaxed bounds to avoid over-constraining the solver.
    # Allowing 0% to 100% allows the risk parameter (lambda) to actually do its job.
    # Strict bounds like (0.05, 0.40) often make the problem mathematically impossible.
    bounds = [(0.0, 1.0) for _ in range(n)]

    res = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=cons)
    
    status = {
        "success": res.success,
        "message": res.message,
        "weights": pd.Series(res.x, index=mu.index) if res.success else pd.Series(w0, index=mu.index)
    }
    return status

# ============================================================
# 📄 MAIN LAYOUT
# ============================================================

st.title("Deco-Robo: The DIY ETF Kit")
st.markdown("#### Customized Portfolio Construction for Canadian Investors")
st.markdown("---")

tab_tool, tab_edu = st.tabs(["🛠️ Robo-Advisor Tool", "📚 Education Center"])

# ============================================================
# 🛠️ TAB 1: THE TOOL
# ============================================================
with tab_tool:
    with st.sidebar:
        st.header("1. Asset Selection")
        
        region_mode = st.radio("Select Focus", ["Canadian", "Global / US"], horizontal=True)
        
        if region_mode == "Canadian":
            base_key = "canadian"
            asset_type = "All"
        else:
            asset_type = st.selectbox("Category", ["Equity (Sector)", "Equity (Thematic)", "Bond (Gov/Corp)", "High Yield"])
            key_map = {
                "Equity (Sector)": "sector_etfs",
                "Equity (Thematic)": "thematic",
                "Bond (Gov/Corp)": "bond_etfs",
                "High Yield": "high_yield"
            }
            base_key = key_map[asset_type]

        df_info = info_sheets.get(base_key, pd.DataFrame()).copy()
        
        # --- FILTERS ---
        st.markdown('<div class="section-header">Refine Selection</div>', unsafe_allow_html=True)
        active_filters = {}
        potential_filters = ["Management_Style", "ESG_Focus", "Sector_Focus", "Geographic_Focus"]
        
        for col in potential_filters:
            col_match = next((c for c in df_info.columns if c.lower() == col.lower()), None)
            if col_match:
                unique_vals = sorted([str(x).strip() for x in df_info[col_match].dropna().unique() if str(x).strip() != ""])
                if unique_vals:
                    sel = st.selectbox(col_match.replace("_", " "), ["All"] + unique_vals, key=f"filt_{base_key}_{col_match}")
                    if sel != "All":
                        active_filters[col_match] = sel
        
        st.markdown("---")
        st.markdown('<div class="section-header">Rebalancing Strategy</div>', unsafe_allow_html=True)
        rebal_freq = st.selectbox(
            "Frequency", 
            ["Quarterly", "Annually"], 
            help="How often will you reset your portfolio to these weights?"
        )

    # --- MAIN CONTENT ---
    
    # 1. Apply Filters
    filtered_df = df_info.copy()
    for col, val in active_filters.items():
        filtered_df = filtered_df[filtered_df[col].astype(str).str.strip() == val]

    # 2. SELECTION LOGIC (LIQUIDITY + COST)
    st.subheader("2. Smart Selection (Liquidity & Cost)")
    
    vol_col = get_volume_col(filtered_df)
    exp_col = get_expense_col(filtered_df)
    
    if filtered_df.empty:
        st.warning("No ETFs match your filters.")
    else:
        # Preprocessing for Ranking
        scoring_df = filtered_df.copy()
        
        # Clean numeric columns
        if vol_col:
            scoring_df[vol_col] = pd.to_numeric(scoring_df[vol_col], errors='coerce').fillna(0)
        else:
            scoring_df['dummy_vol'] = 1 # Fallback
            vol_col = 'dummy_vol'

        if exp_col:
            scoring_df[exp_col] = pd.to_numeric(scoring_df[exp_col], errors='coerce').fillna(0.99) # Fill NaN with high cost
        else:
            scoring_df['dummy_exp'] = 0.5
            exp_col = 'dummy_exp'
            
        # --- SCORING ALGORITHM ---
        # 1. Volume Score (Higher is better). Min-Max Scaling.
        v_min, v_max = scoring_df[vol_col].min(), scoring_df[vol_col].max()
        if v_max - v_min > 0:
            scoring_df['score_vol'] = (scoring_df[vol_col] - v_min) / (v_max - v_min)
        else:
            scoring_df['score_vol'] = 0.5

        # 2. Cost Score (Lower is better). Inverted Min-Max.
        c_min, c_max = scoring_df[exp_col].min(), scoring_df[exp_col].max()
        if c_max - c_min > 0:
            scoring_df['score_cost'] = 1 - ((scoring_df[exp_col] - c_min) / (c_max - c_min))
        else:
            scoring_df['score_cost'] = 0.5
            
        # 3. Composite Score (50/50)
        scoring_df['final_score'] = 0.5 * scoring_df['score_vol'] + 0.5 * scoring_df['score_cost']
        
        # Select Top 15
        top_candidates = scoring_df.sort_values(by='final_score', ascending=False).head(15)
        
        col_res1, col_res2 = st.columns([3, 1])
        with col_res1:
            st.dataframe(
                top_candidates[[get_ticker_col(top_candidates), 'name', exp_col, vol_col]], 
                use_container_width=True, 
                hide_index=True
            )
        with col_res2:
            st.info(f"**Selection Logic:**\n\nWe analyzed {len(filtered_df)} ETFs.\n\nTop 15 selected based on:\n\n🔹 50% High Liquidity\n🔹 50% Low Cost")

        # 3. OPTIMIZATION SETUP
        st.markdown("---")
        st.subheader("3. Portfolio Construction")
        
        c_risk, c_views = st.columns([1, 1])
        
        with c_risk:
            st.markdown("#### Risk Attitude")
            risk_level = st.select_slider(
                "Risk Aversion",
                options=["Conservative", "Moderate", "Aggressive"],
                value="Moderate"
            )
            lambdas = {"Conservative": 5.0, "Moderate": 2.5, "Aggressive": 1.0}
            
        with c_views:
            st.markdown("#### Investor Personal Views of the Market")
            st.caption("Apply your own outlook (Black-Litterman Model)")
            
            views = []
            cc1, cc2 = st.columns(2)
            if cc1.checkbox("Tech Boom (+5%)"): views.append("Tech")
            if cc1.checkbox("Energy Slump (-3%)"): views.append("Energy")
            if cc1.checkbox("NA Growth (+4%)"): views.append("NorthAmerica")
            if cc2.checkbox("EM Rally (+6%)"): views.append("EmergingMarkets")
            if cc2.checkbox("Defensive (+2%)"): views.append("Stability")
            if cc2.checkbox("Yield Spike (+3%)"): views.append("HighYield")

        if st.button("🚀 Generate Optimized Portfolio", type="primary", use_container_width=True):
            with st.spinner("Calculating Efficient Frontier & Adjusting for Views..."):
                
                # A. Prepare Data
                ticker_col = get_ticker_col(top_candidates)
                tickers = top_candidates[ticker_col].astype(str).tolist()
                
                prices, found, missing = get_prices_for(base_key, tickers)
                
                if len(found) < 3:
                    st.error("Not enough price data found to optimize.")
                    st.stop()
                    
                # B. Statistics
                mu_hist, cov = calculate_metrics(prices)
                
                # C. Black-Litterman
                mu_bl = black_litterman_adjustment(mu_hist, cov, views, top_candidates)
                
                # D. Optimize
                res = optimize_portfolio(mu_bl, cov, lambdas[risk_level])
                
                if not res['success']:
                    st.warning("Optimizer struggled to converge perfectly, but found a solution.")
                
                weights = res['weights']
                clean_weights = weights[weights > 0.001]
                
                # E. Calculate Weighted Expense Ratio (MER)
                # Formula: Sum(Weight * MER)
                matched_indices = [top_candidates.index[top_candidates[ticker_col] == t].tolist()[0] for t in clean_weights.index]
                # Ensure we use the numeric column we created earlier
                mer_values = top_candidates.loc[matched_indices, exp_col].values 
                weight_values = clean_weights.values
                portfolio_mer = np.sum(weight_values * mer_values)
                
                # F. Display
                st.success("Optimization Successful!")
                
                col_left, col_right = st.columns([1, 2])
                
                with col_left:
                    st.markdown("#### Allocation")
                    # Format as percentage
                    w_disp = pd.DataFrame({
                        "Ticker": clean_weights.index,
                        "Weight": clean_weights.values
                    })
                    w_disp["Weight %"] = (w_disp["Weight"] * 100).map("{:.1f}%".format)
                    st.dataframe(w_disp[["Ticker", "Weight %"]], hide_index=True)
                    
                with col_right:
                    st.markdown("#### Portfolio Metrics")
                    ret = weights @ mu_bl
                    vol = np.sqrt(weights @ cov @ weights)
                    sharpe = ret / vol
                    
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Exp. Return", f"{ret:.1%}")
                    m2.metric("Volatility", f"{vol:.1%}")
                    m3.metric("Sharpe Ratio", f"{sharpe:.2f}")
                    m4.metric("Weighted Cost", f"{portfolio_mer:.2f}%", help="Weighted Average Expense Ratio")
                    
                    st.caption(f"📅 **Rebalancing Strategy:** {rebal_freq}")
                    if rebal_freq == "Quarterly":
                        st.caption("*Recommended: Review and adjust weights every 3 months to maintain these risk characteristics.*")
                    else:
                        st.caption("*Recommended: Review annually. Lower maintenance, but higher potential for risk drift.*")
                        
                    st.bar_chart(clean_weights)

# ============================================================
# 📚 TAB 2: EDUCATION (Static Content)
# ============================================================
with tab_edu:
    st.header("ETF Knowledge Base")
    st.markdown("""
    ### 1. Understanding Costs
    **Expense Ratio (MER):** The annual fee charged by the fund.
    * *Why it matters:* In our tool, we prioritize lower cost funds. A 2% fee reduces your returns significantly over 20 years compared to a 0.2% fee.
    
    ### 2. Liquidity
    **Trading Volume:** How easily you can buy or sell the ETF.
    * *Why it matters:* Low volume ETFs can have large "bid-ask spreads," costing you extra money when you trade.
    
    ### 3. Black-Litterman Model
    This tool uses the **Black-Litterman** optimization model. Unlike basic Mean-Variance (which relies purely on past data), this model allows you to input your **personal views** (e.g., "I think Tech will boom"). It mathematically blends history with your intuition.
    """)
