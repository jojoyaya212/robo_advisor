# ============================================================
# 🌟 PRETTIER ROBO ADVISOR (UX/UI Enhanced Edition)
# ============================================================
# Based on the robust logic of 'robo.py', but with a focus on:
# 1. Cleaner, modern aesthetic (Custom CSS)
# 2. Improved User Flow
# 3. Clearer Explanations

'''
# Notes: Import libraries needed for a prototype robo.
os - file existence checks (ensures the Excel data file is present).
numpy — numeric operations, linear algebra, returns/volatility calculations.
pandas — dataframes for ETF metadata and price series.
streamlit — builds the web UI (inputs, outputs, layout).
scipy.optimize — "minimize" for mean–variance optimization.
re — string handling, parsing. 
'''
import os
import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import minimize
import re

# ============================================================
# 🎨 UI CONFIGURATION & CSS STYLING
# ============================================================


# Notes: UI configuration -
# Set the page title, icon, layout, sidebar state.

st.set_page_config(
    page_title="ETF Robo Advisor", # this defines the website page title on browser
    page_icon="✨",  # add a icon infront of the website title (not the website content, but how the website looks on the browser header bar)
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Notes: Custom CSS for a cleaner, more professional look
# Set font; headings; sidebar; make metrics cars look like "cards" with subtle shadows
# style buttons, tables and expanders 
# injecting custom CSS styling into a Streamlit to override Streamlit’s default design.
"""
CSS = Cascading Style Sheets
It is the language used to control how things look on a webpage or app.
HTML = the content
CSS = the appearance
JavaScript = the behavior
"""
st.markdown("""
    <style>
    /* Main Font */
    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', Helvetica, Arial, sans-serif;
        color: #333333;
        background-color: #ffffff;
    }
    # Sets the font of the entire app to Inter/Segoe UI/Helvetica.
    # Sets the default text color to dark grey (#333).
    # Sets the background color of the app to white (#fff).
    
    /* Headings */
    h1, h2, h3 {
        font-weight: 700;
        color: #111111;
        letter-spacing: -0.5px;
    }
    # Makes headings bold (700 weight).
    # Sets heading color darker (#111).
    # Slight negative letter-spacing makes text more compact and modern.
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #f7f9fc;
        border-right: 1px solid #e0e0e0;
    }
    # Changes sidebar background to a light soft blue-grey.
    # Adds a right border to visually separate sidebar and main area.
    
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
    # White background
    # Light border
    # Soft rounded corners
    # Subtle shadow (card effect)
    # Padding inside each card
    
  /* Primary Buttons - Wealthsimple Style (Clean Dark) */
    /* Target both specific Streamlit primary buttons and general buttons to override red default */
    div.stButton > button, button[kind="primary"] {
        background-color: #1F1F1F !important; /* Force Matte Black always */
        color: #ffffff !important;
        border: 1px solid #1F1F1F !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.2rem !important;
        font-weight: 600 !important;
        transition: all 0.2s ease-in-out !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }
    
    /* Hover State */
    div.stButton > button:hover, button[kind="primary"]:hover {
        background-color: #333333 !important; /* Lighter Charcoal on Hover */
        border-color: #333333 !important;
        color: white !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15) !important;
    }
    # used a wealthsimple style botton, but overwrite by streamlit, so it alwasys show red before the click, didn't solve this in the end.
    # Makes buttons black with white text
    # Adds rounded corners
    # Removes the default border
    # Adds padding to make them larger
    # On hover: button becomes lighter black (#333)
    
    /* DataFrame Styling */
    div[data-testid="stDataFrame"] {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        overflow: hidden;
    }
    # this works on the the big white ETF table under “2. Screening Results”
    /* Expanders */
    .streamlit-expanderHeader {
        font-weight: 600;
        background-color: #ffffff;
        border-radius: 5px;
    }
    # expander titled “View ETF Table”
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
    # headings like “2. Screening Results”, “3. Portfolio Construction”
    </style>
""", unsafe_allow_html=True)

# ============================================================
# ⚡ DATA LOADING (CACHED & EXPLICITLY MAPPED)
# ============================================================

# Notes: A single Excel workbook as the data source.
FILE_PATH = "robo_advisor_data.xlsx"  # this is the file in github repository

# Notes: EXPLICIT MAPPING: Info Sheet Name -> Price Sheet Name
# Maps logical categories (e.g., sector_etfs) to the expected sheet names that contain price data (e.g., sector_price).

SHEET_PAIRS = {
    "sector_etfs": "sector_price",
    "bond_etfs": "bond_etf_price",
    "high_yield": "high yield price",
    "thematic": "thematic price",
    "canadian": "canadian_price",
    "em_bonds": "em_bonds_price"
}
# this is mapping the sheets with our website structure
# Notes: Cathes the loaded workbook to keep the UI responsive
# Repeated UI actions don’t re-read the file (speeds the demo) with caching and reliable data loading.
@st.cache_data(show_spinner=False)
def load_data():
    if not os.path.exists(FILE_PATH):
        return None
    return pd.read_excel(FILE_PATH, sheet_name=None)

# Notes: Graceful loading with a clean spinner. 
# Important: Provides a loading spinner and a clear error message if the workbook is not found.
with st.spinner("Initializing Deco-Robo... Loading market data..."):
    sheets_all = load_data()
# what to show when loading data
if sheets_all is None:
    st.error(f"❌ Critical Error: Data file `{FILE_PATH}` not found. Please ensure it is in the application folder.")
    st.stop()

# Notes: Build info_sheets and price_sheets dictionaries separately based on explicit pairs
# Populates two dictionaries: (1)info_sheets contains metadata DataFrame (2)price_sheets contains price DataFrame
# Function: keeps code modular; each ETF category uses the corresponding price matrix.
    
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
# reading data into the website

# ============================================================
# 🧮 HELPER FUNCTIONS
# ============================================================

# Notes: Scans for common column names: "ticker", "symbol", ..." and returns the first match.
def get_ticker_col(df: pd.DataFrame) -> str | None:
    """Find the ticker column robustly."""
    for c in ["ticker", "Ticker", "TICKER", "symbol", "Symbol", "SYMBOL"]:
        if c in df.columns: return c
    return None

# Notes: Checks a list of likely volume column names, then fallbacks to any column that contains "volume" in its name.
# Excluding 30 to avoid 30-day columns
def get_volume_col(df: pd.DataFrame) -> str | None:
    """Find the volume column robustly."""
    candidates = ["1d volume", "1D Volume", "1d_volume", "Volume", "volume", "1D volume"]
    for c in candidates:
        if c in df.columns: return c
    for c in df.columns:
        if "volume" in str(c).lower() and "30" not in str(c): 
            return c
    return None

# Notes: Searches for Expense Ratio labels.
# MER is used later for cost scoring and to compute the portfolio weighted MER.
def get_expense_col(df: pd.DataFrame) -> str | None:
    """Find the expense ratio column robustly."""
    candidates = ["Expense_Ratio", "Expense Ratio", "MER", "Management Fee", "Fees", "expense_ratio"]
    for c in candidates:
        if c in df.columns: return c
    return None

# Notes: Removes % and parses numeric values.
# (Clean format) Make sure data in columns like MER, Yield in the same format.
def clean_percentage_col(series):
    """Cleans a column that might be strings with % signs."""
    if series.dtype == 'object':
        return series.astype(str).str.replace('%', '', regex=False).apply(pd.to_numeric, errors='coerce')
    return pd.to_numeric(series, errors='coerce')

# Notes: Standardize the Tickers 
# (Clean format) Uppercases, removes whitespace, and strips known suffixes.
def ultra_clean_ticker(t):
    """Aggressive cleaner: removes ALL spaces and common suffixes."""
    t = str(t).upper()
    t = "".join(t.split())
    for suffix in ["US", "CN", ".TO", "CH", "JT", "TT", "LN", "GR", "JP", "AU", "SW"]:
        if t.endswith(suffix):
            t = t[:-len(suffix)]
            break 
    return t

# ============================================================
#  Robust Ticker Matching and Time-series Alignment
# ============================================================
#  Matching for correct return & covariance estimation.
def get_prices_for(base_key: str, tickers: list[str]) -> tuple[pd.DataFrame, list, list]:
    """Returns: (Price DataFrame, List of Found Tickers, List of Missing Tickers)"""

# Notes: Pulls the appropriate price sheet for the chosen ETF category.
    price_df = price_sheets.get(base_key)
    
    if price_df is None or price_df.empty: 
        return pd.DataFrame(), [], tickers
    
# Notes: Quickly check if a requested ticker exists in any of the column forms.
    col_map = {}
    for c in price_df.columns:
        col_map[str(c).strip()] = c
        col_map[ultra_clean_ticker(c)] = c
    
    found_tickers = []
    missing_tickers = []
    matched_cols = []

# Notes: Loop through requested tickers. If no matches at all, then return early.
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
    
# Notes: take matched price columns. 
    out = price_df[list(set(matched_cols))].copy()

# Notes: Convert to datetime ad set as index.    
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"])
        out.set_index("Date", inplace=True)

# Notes: Forward-fill missing daily prices handling holidays and missing ticks
# Remove rows without full coverage.
    out = out.ffill() 
    out = out.dropna(axis=0, how="any") 
    
    return out, found_tickers, missing_tickers

# Notes: Calculate metrics: log returns, annualize mean returns, annualize covariance matrix.
def calculate_metrics(prices: pd.DataFrame, freq: int = 252):
    """Returns annualized mean returns and covariance matrix."""
    prices = prices.apply(pd.to_numeric, errors='coerce')
    rets = np.log(prices / prices.shift(1)).dropna()
    if rets.empty: return None, None
    
    mu = rets.mean() * freq
    cov = rets.cov() * freq
# Notes: Add tiny jitter to diagonal to avoid singular matrices.
    cov = cov + np.eye(cov.shape[0]) * 1e-6 
    return mu, cov

# ============================================================
#    BLACK-LITTERMAN POSTERIOR
# Step 1 — Start With What History Says (“Prior returns”)
# Step 2 — Look at the Client’s Opinions (“Views”)
# Step 3 — Turn “Views” Into Math (the P and Q matrices)
# Step 4 — Ask: “How confident is the user in their view?”
# Step 5 — Combine History + Views Into One Final Return (the Posterior)
# ============================================================
def black_litterman_adjustment(mu_prior, cov, views, ticker_info, view_confidence=0.5):
    tau = 0.05  # Standard tau  # Tau controls how uncertain you believe the market equilibrium returns are, but here we just gave it a initial value, this will change
    n_assets = len(mu_prior)
    tickers = mu_prior.index.tolist()
    active_views = [] 
 
# Notes: Translate a smantic view of input into the specific ETFs it affects.
    def get_indices(condition_col, condition_val):
        if condition_col not in ticker_info.columns: return []
        matches = ticker_info[ticker_info[condition_col] == condition_val]  #filters ticker_info for matches
        ticker_col = get_ticker_col(ticker_info)  # finds the ticker column
        target_clean = set(matches[ticker_col].apply(ultra_clean_ticker).tolist())  # Clean the tickers
        indices = []
        for i, t_price in enumerate(tickers):
            if ultra_clean_ticker(t_price) in target_clean:
                indices.append(i)
        return indices

# Notes: For each pre-coded view label, call get_indices to find which ETFs in the current candidate universe match.
# -------- Correct BL view magnitudes matching UI labels --------
        # ============================
    # DEFINE USER VIEWS (updated labels)
    # ============================

    # Tech Boom (+5%)
    if "Tech" in views:
        idx = get_indices('Sector_Focus', 'Technology')
        if idx:
            row = np.zeros(n_assets)
            row[idx] = 1 / len(idx)
            active_views.append((row, 0.05))

    # Energy Slump (-3%)
    if "Energy" in views:
        idx = get_indices('Sector_Focus', 'Energy')
        if idx:
            row = np.zeros(n_assets)
            row[idx] = 1 / len(idx)
            active_views.append((row, -0.03))

    # North America Strength (+15%)
    if "NorthAmerica" in views:
        idx = get_indices('Geographic_Focus', 'North America')
        if idx:
            row = np.zeros(n_assets)
            row[idx] = 1 / len(idx)
            active_views.append((row, 0.15))

    # Emerging Markets Rally (+6%)  (renamed & expanded logic)
    if "EmergingMarketsRally" in views:
        idx_geo = get_indices('Geographic_Focus', 'Emerging Markets')

        # add EM bond ETFs (from the separate EM table)
        if "em_bonds" in info_sheets:
            ticker_col = get_ticker_col(info_sheets["em_bonds"])
            bond_list = info_sheets["em_bonds"][ticker_col].apply(ultra_clean_ticker).tolist()
            idx_bond = [i for i, t in enumerate(tickers) if ultra_clean_ticker(t) in bond_list]
        else:
            idx_bond = []

        idx = idx_geo + idx_bond

        if idx:
            row = np.zeros(n_assets)
            row[idx] = 1 / len(idx)
            active_views.append((row, 0.06))

    # Health Care (+2%)
    if "HealthCare" in views:
        idx = get_indices('Sector_Focus', 'Health Care')
        if idx:
            row = np.zeros(n_assets)
            row[idx] = 1 / len(idx)
            active_views.append((row, 0.02))

    # Communication Services (+3%)
    if "CommServices" in views:
        idx = get_indices('Sector_Focus', 'Communication Services')
        if idx:
            row = np.zeros(n_assets)
            row[idx] = 1 / len(idx)
            active_views.append((row, 0.03))

    if not active_views:
        return mu_prior  # no change if no active views.

# Notes: Build matrices: P is the view matrix while Q is the view returns.
    P = np.array([v[0] for v in active_views])
    Q = np.array([v[1] for v in active_views]).reshape(-1, 1)
    
    # Uncertainty matrix Omega
    omega = np.diag(np.diag(P @ (tau * cov.values) @ P.T))

# Notes: (Key)The view_confidence slider from the UI scales omega.
# Higher confidence → smaller omega → views get more weight  
# Give users direct control over their views’ influence.

    safe_conf = max(view_confidence, 0.01) # Avoid div by zero
    scaler = 1.0 / safe_conf
    omega = omega * scaler

# Notes: Compute BL posterior.
    try:
        sigma_inv = np.linalg.inv(tau * cov.values)
        omega_inv = np.linalg.inv(omega)
        term1 = np.linalg.inv(sigma_inv + P.T @ omega_inv @ P)
        term2 = (sigma_inv @ mu_prior.values.reshape(-1, 1)) + (P.T @ omega_inv @ Q)
        mu_bl = term1 @ term2
        return pd.Series(mu_bl.flatten(), index=tickers)
    except:
        return mu_prior  #If any matrix inversion fails, fallback mu_prior (no active views)

# ============================================================
#   CONSTRAINED MEAN-VARIANCE OPTIMIZATION IMPLEMENTATION
# ============================================================
# Notes: Define a mean–variance optimizer.
# Logic: given mu and cov (after BL adjustment) and a lambda_risk from the user, compute a diversified, implementable portfolio that sums to 100% and respects no-short and maximum weight rules.
# portfolio will always have between 5 and 15 ETFs, but mathematically it is biased to stay close to 5 or 6 due to the 20% cap and correlations.
def optimize_portfolio(mu, cov, lambda_risk):
    n = len(mu)
    w0 = np.ones(n) / n #equal-weight starting guess
    
    def objective(w):
        ret = w @ mu
        var = w @ cov @ w
        return -(ret - (lambda_risk * var)) # minimizing equivalent to maximizing ret - lambda*var

    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}) # Enforces full-investment
    
    # Default to 20% max weight per asset to force diversification (min 5 assets), constrain is here
    max_weight = 0.20
    if n < 5:
        max_weight = 1.0 / n # distribute weight evenly (no unrealistic single-ETF concentration when few ETFs).
        
    bounds = [(0.0, max_weight) for _ in range(n)] # enforces no shorting and max weight

# Notes: using minimizer to maximize. SLSQP as a standard solver for smooth constrained problems.
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

# Notes: Create two major tabs: Robot-advisor Tool & Education Center
# Header Area
st.title("ETF Robo Advisor")  # website big name/title
st.markdown("#### Empowering Canadian Investors with Institutional-Grade Tools")  # website slogan
st.markdown("---")

# Create Tabs
tab_tool, tab_edu = st.tabs(["🛠️ Robo-Advisor Tool", "📚 Education Center"])  # create the education tab

# ============================================================
# 📚 TAB 2: EDUCATION CENTER
# ============================================================
with tab_edu:
    st.header("ETF Education Center")
    st.caption("Master the basics before you invest.")
    
    col1, col2 = st.columns([2, 1])

# Notes: Collapsible sections explaining ETF related knowledge.
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

# Notes: User selects ETFs
with tab_tool:    

# Notes: SIDEBAR FILTERS
    with st.sidebar:
        st.header("1. Asset Selection")
        
        # Debug info toggle (hidden by default for cleaner UI)
        if st.checkbox("Show Data Diagnostics"):
            with st.expander("Debug Info"):
                st.write(f"Info Sheets Loaded: {len(info_sheets)}")
                st.write(f"Price Sheets Loaded: {len(price_sheets)}")
                st.write("Active Pairs:", list(info_sheets.keys()))

        # 1. Region selection
        st.markdown('<div class="section-header">Region</div>', unsafe_allow_html=True)
        region_mode = st.radio("Select Focus", ["Canadian", "Global / US"], horizontal=True, label_visibility="collapsed")
        
        # 2. Asset Class & Dataset Selection
        st.markdown('<div class="section-header">Category</div>', unsafe_allow_html=True)
        if region_mode == "Canadian":  # If Canadian → base_key = "canadian" and the full Canadian universe is loaded.
            base_key = "canadian"
            asset_type = "All" 
            st.caption("Accessing full Canadian ETF universe.")
        else:
            # Else select a category such as Sector, Thematic, Bond, High Yield.
            asset_type = st.selectbox("Select Category", ["Equity (Sector)", "Equity (Thematic)", "Bond (Gov/Corp)", "High Yield", "Bond (Emerging Mkts)"], label_visibility="collapsed")
            # Use keys from SHEET_PAIRS
            key_map = {
                "Equity (Sector)": "sector_etfs",
                "Equity (Thematic)": "thematic",
                "Bond (Gov/Corp)": "bond_etfs",
                "High Yield": "high_yield",
                "Bond (Emerging Mkts)": "em_bonds"
            }
            base_key = key_map[asset_type]
# our six tables logic was implemented above
        # 3. Load Data: load metadata for the chosen category.
        df_info = info_sheets.get(base_key, pd.DataFrame()).copy()
        
        if df_info.empty:
            st.error(f"Data not found for key: {base_key}")
            st.stop()

        # 4. Master Filter Logic
        st.markdown('<div class="section-header">Refine Selection</div>', unsafe_allow_html=True)
        active_filters = {}
        # a list of columns to be used as interactive filters
        potential_filters = [
            "Management_Style", "ESG_Focus", "Issuer", "structure", 
            "Use_Derivative", "ETF_General_Type", "Strategic_Focus", 
            "Sector_Focus", "Theme_Focus", "Geographic_Focus",
            "Exchange_Region", "Leverage_Type"
        ]
        
        # Create filters without cluttering the UI 
        # Loop over potential_filters: for each use next() to find the matching column name.
        count_filters = 0
        for col in potential_filters:
            col_match = next((c for c in df_info.columns if c.lower() == col.lower()), None) #  find the matching column name
            if col_match:
                unique_vals = sorted([str(x).strip() for x in df_info[col_match].dropna().unique() if str(x).strip() != ""])
                if unique_vals:
                    count_filters += 1
                    unique_vals = ["All"] + unique_vals
                    label = col_match.replace("_", " ").replace("exposure", "").title()
                    sel = st.selectbox(label, unique_vals, key=f"filt_{base_key}_{col_match}") # Build a st.selectbox containing "All" + unique values for that column.
                    if sel != "All": #If user selects a specific value, add it to active_filters.
                        active_filters[col_match] = sel
        
        if count_filters == 0:
            st.info("No additional filters available for this category.")

# Notes: Build Rebalancing Frequency selector
        st.markdown('<div class="section-header">Strategy Settings</div>', unsafe_allow_html=True)
        rebal_freq = st.selectbox("Rebalancing Frequency", ["Quarterly", "Annually"])

# ============================================================
#     Main Content Area
# ============================================================
    
# Notes: Apply Filters
    filtered_df = df_info.copy()
    for col, val in active_filters.items():
        filtered_df = filtered_df[filtered_df[col].astype(str).str.strip() == val]

# Notes: Display Screening Results (Metrics and table)
# Display what the custom ETF universe is like 
    st.subheader("2. Screening Results")
    
    col_count, col_msg = st.columns([1, 3])
    with col_count:
        st.metric("ETFs Found", len(filtered_df))
    
    if filtered_df.empty:
        st.warning("No ETFs match your filters. Please adjust your selection in the sidebar.")
    else:
        # Dynamic Column Display
        vol_col_display = get_volume_col(filtered_df)
        exp_col_display = get_expense_col(filtered_df)
        disp_cols = ["ticker", "name", "Expense_Ratio", "YTD_Return"]
        if vol_col_display and vol_col_display not in disp_cols: disp_cols.append(vol_col_display)
        
        final_disp_cols = [c for c in disp_cols if c in filtered_df.columns]
        
        with st.expander("View ETF Table", expanded=True):
            st.dataframe(filtered_df[final_disp_cols].head(50), use_container_width=True, hide_index=True)
            st.caption(f"Showing matches: {len(filtered_df)}")

# Notes: Black-Litterman & Optimization
        st.markdown("---")
        st.subheader("3. Portfolio Construction")
        st.caption("We use the Black-Litterman model to combine historical data with your personal market views.")
        
        # Layout for inputs
        col_risk, col_views = st.columns([1, 2])

        # Risk profile input: three levels
        with col_risk:
            st.markdown("#### Your Risk Profile")
            risk_level = st.select_slider(
                "Select your comfort level:",
                options=["Conservative", "Moderate", "Aggressive"],
                value="Moderate"
            )
            lambdas = {"Conservative": 2.0, "Moderate": 1.0, "Aggressive": 0.3}
            
            st.info(f"**Strategy:** {risk_level} optimization focuses on {'capital preservation' if risk_level=='Conservative' else 'growth' if risk_level=='Aggressive' else 'balanced returns'}.")
        # Investor views (BL)    
        with col_views:
            st.markdown("#### Investor Personal Views of the Market")
            st.caption("Select any specific outlooks you have for the market:")
            
            views = []
            c1, c2 = st.columns(2)
             
            if c1.checkbox("Tech Boom (+5%)"): views.append("Tech")
            if c1.checkbox("Energy Slump (-3%)"): views.append("Energy")
            if c1.checkbox("North America Strength (+15%)"): views.append("NorthAmerica")
            
            # renamed for investor clarity
            if c2.checkbox("Emerging Markets Rally (+6%)"): 
                views.append("EmergingMarketsRally")
            
            # substituted new categories
            if c2.checkbox("Health Care (+2%)"): views.append("HealthCare")
            if c2.checkbox("Communication Services (+3%)"): views.append("CommServices")

            
            # View Confidence Slider
            st.markdown("---")
            view_conf = st.slider(
                "How confident are you in these views?",
                min_value=0.0, max_value=1.0, value=0.5, step=0.1,
                help="Higher confidence means your views will have a stronger impact on the portfolio weights."
            )
            # here we fetch confidence, discard the default value if user choose their own confidence.

        st.write("") # Spacer
        
        # Notes: Run Optimization Button
        # Centered button for better UX
        col_spacer1, col_btn, col_spacer2 = st.columns([1, 2, 1])
        with col_btn:
            run_btn = st.button(" Generate My Own Optimized Portfolio", type="primary", use_container_width=True)

        if run_btn:
            with st.spinner("Crunching numbers... (Calculating Covariance, BL Posteriors, Efficient Frontier)"):
                # Data selection & matching
                ticker_col = get_ticker_col(filtered_df)
                if not ticker_col:
                    st.error("Could not find ticker column.")
                    st.stop()
                
                # Notes: A. Scoring logic: 50/50 Liquidity/Cost Score 
            
                scoring_df = filtered_df.copy()
                
                # Volume score
                vol_col = get_volume_col(scoring_df)
                if vol_col:
                    scoring_df[vol_col] = pd.to_numeric(scoring_df[vol_col], errors='coerce').fillna(0)
                    v_min, v_max = scoring_df[vol_col].min(), scoring_df[vol_col].max()
                    if v_max - v_min > 0:
                        scoring_df['score_vol'] = (scoring_df[vol_col] - v_min) / (v_max - v_min)
                    else:
                        scoring_df['score_vol'] = 0.5
                else:
                    st.warning("Volume column not found, using pure cost/default for volume score.")
                    scoring_df['score_vol'] = 0.5

                # Cost score
                exp_col = get_expense_col(scoring_df)
                if exp_col:
                    # Clean percentage signs
                    scoring_df[exp_col] = clean_percentage_col(scoring_df[exp_col]).fillna(0.99)
                    
                    c_min, c_max = scoring_df[exp_col].min(), scoring_df[exp_col].max()
                    # Invert because lower cost is better
                    if c_max - c_min > 0:
                        scoring_df['score_cost'] = 1 - ((scoring_df[exp_col] - c_min) / (c_max - c_min))
                    else:
                        scoring_df['score_cost'] = 0.5
                else:
                    scoring_df['score_cost'] = 0.5

                # Final Composite Score
                scoring_df['final_score'] = 0.5 * scoring_df['score_vol'] + 0.5 * scoring_df['score_cost']
                
                # Take Top 15
                top_liquid = scoring_df.sort_values(by='final_score', ascending=False).head(15)

                tickers = top_liquid[ticker_col].astype(str).tolist()
                
                # Notes: SMART DATA MATCHING. Data sanity check.
                prices, found_tickers, missing_tickers = get_prices_for(base_key, tickers)
                
                # Check logic: Why did we fall below 5 assets?
                if len(found_tickers) < 3:
                    st.error(f"Optimization Failed: Only {len(found_tickers)} valid price histories were found.")
                    with st.expander("View Missing Data Details"):
                        st.write(f"**Missing Tickers:** {missing_tickers}")
                    st.stop()

                if len(prices) < 10: 
                    st.error("Optimization Failed: Insufficient overlapping price history.")
                    st.stop()

                # B. Calculate Stats: Compute annualized mu and cov from the price matrix (log returns).
                mu_hist, cov = calculate_metrics(prices)
                
                if mu_hist is None or mu_hist.isnull().values.any():
                    st.error("Optimization Failed: Covariance matrix contains NaNs.")
                    st.stop()
                
                # C. Apply Black-Litterman: Produces a posterior return vector that blends historical estimates with the user’s views weighted by view_conf.
                # Pass view confidence
                mu_bl = black_litterman_adjustment(mu_hist, cov, views, top_liquid, view_conf)
                
                # D. Optimize: Solve the MVO with the chosen lambda and diversification bounds.
                result = optimize_portfolio(mu_bl, cov, lambdas[risk_level])
                weights_full = result["weights"]
                
                if not result["success"]:
                    st.warning(f"Optimization Warning: Solver failed to converge. (Reason: {result['message']})")
                
                # E. Display Results
                weights_display = weights_full[weights_full > 0.001].sort_values(ascending=False) # filter extremely small weights for readability.
                
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

                
                    # Weighted MER calculation. Show investors the cost drag due to underlying ETF fees.

                    portfolio_mer = 0.0
                    if exp_col:
                        matched_indices = [top_liquid.index[top_liquid[ticker_col] == t].tolist()[0] for t in weights_display.index]
                        mer_values = top_liquid.loc[matched_indices, exp_col].values
                        portfolio_mer = np.sum(weights_display.values * mer_values)
                    
                    # Notes: Rebalancing Cost Impact
                    # Quarterly = Higher Turnover (~0.20% estimated drag)
                    # Annually = Lower Turnover (~0.05% estimated drag)
                    trading_cost_est = 0.0020 if rebal_freq == "Quarterly" else 0.0005
                    net_return = port_ret - trading_cost_est
                    # Show metrics
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Net Exp. Return", f"{net_return:.1%}", delta=f"-{trading_cost_est:.2%} Trading Cost")
                    k2.metric("Volatility", f"{port_vol:.1%}", delta_color="inverse")
                    k3.metric("Sharpe Ratio", f"{sharpe:.2f}")
                    # UPDATED LABEL TO FULL NAME
                    k4.metric("Weighted Management Expense Ratio", f"{portfolio_mer:.2f}%", help="Weighted Average Expense Ratio")
                    # Visualization
                    st.bar_chart(weights_display) # visual weight distribution.
                    st.caption(f"📅 **Rebalancing Strategy:** {rebal_freq}") # show rebalancing strategy explanation.
                    if rebal_freq == "Quarterly":
                        st.caption("*Optimization Note: Quarterly rebalancing incurs higher estimated trading costs (0.20%), reducing your net return slightly, but keeps risk tighter.*")
                    else:
                        st.caption("*Optimization Note: Annual rebalancing saves trading costs (est. 0.05%), but allows portfolio risk to drift further from targets.*")

# Footer
st.markdown("---")
st.caption("© 2025 Deco-Robo. Built for MFIN 706. Powered by Python & Streamlit.")
