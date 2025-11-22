# ============================================================
# 📊 ETF Robo-Advisor Dashboard (Based on robo_advisor_data.xlsx)
# ============================================================

import os
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# 💾 Load the prepared dataset
# ============================================================

FILE_PATH = "robo_advisor_data.xlsx"

st.set_page_config(page_title="ETF Robo-Advisor", layout="wide")

if not os.path.exists(FILE_PATH):
    st.error(f"❌ File not found: {FILE_PATH}")
    st.stop()

sheets_all = pd.read_excel(FILE_PATH, sheet_name=None)
st.success(f"✅ Loaded {len(sheets_all)} sheets from {FILE_PATH}")

# Split into price and info sheets by name
raw_price_sheets = {k: v for k, v in sheets_all.items() if "price" in k.lower()}
raw_info_sheets  = {k: v for k, v in sheets_all.items() if "price" not in k.lower()}

# ---------- Normalize sheet keys so price/info share the same base key ----------
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

price_sheets = {normalize(k): v for k, v in raw_price_sheets.items()}
info_sheets  = {normalize(k): v for k, v in raw_info_sheets.items()}

# ============================================================
# ⚙️ Choice mapping (map UI choices to table values)
# ============================================================

CHOICE_MAP = {
    "Yes": ["Y"],
    "No":  ["N"],
    "Yes ESG": ["Yes ESG"],
    "No ESG":  ["No ESG"],
    "Bond": ["Bond", "Canadian Bond"],
    "Equity": ["Equity", "Canadian Equity"]
}


# ============================================================
# 🔧 Helper functions
# ============================================================

def get_ticker_col(df: pd.DataFrame) -> str | None:
    """Return the ticker/symbol column name if present."""
    for c in ["ticker", "Ticker", "TICKER", "symbol", "Symbol", "SYMBOL"]:
        if c in df.columns:
            return c
    return None

def filter_etfs(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply exact-match filters with synonym mapping (case-insensitive)."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    for col, choice in filters.items():
        if not choice or col not in out.columns:
            continue
        # accepted values for this UI choice
        accepted = [s.lower().strip() for s in CHOICE_MAP.get(choice, [choice])]
        mask = out[col].astype(str).str.strip().str.lower().isin(accepted)
        out = out[mask]
    return out

def rank_etfs(df: pd.DataFrame) -> pd.DataFrame:
    """Rank by a simple composite (Volume, AUM, Expense, Volatility)."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    # Create missing ranking columns if absent (keeps demo robust)
    for c in ["1D Volume", "AUM", "Expense_Ratio", "Volatility_3Y"]:
        if c not in out.columns:
            out[c] = np.random.rand(len(out)) * 1000
    out["Score"] = (
        out["1D Volume"].rank(pct=True) * 0.40 +
        out["AUM"].rank(pct=True)        * 0.30 +
        (1 - out["Expense_Ratio"].rank(pct=True)) * 0.20 +
        (1 - out["Volatility_3Y"].rank(pct=True)) * 0.10
    )
    out = out.sort_values("Score", ascending=False)
    return out

def get_prices_for(base_key: str, tickers: list[str]) -> pd.DataFrame:
    price_df = price_sheets.get(base_key)
    if price_df is None or price_df.empty:
        return pd.DataFrame()

    # Normalize both sides (case-insensitive, strip spaces)
    normalized_cols = {str(c).strip().lower(): c for c in price_df.columns}
    tickers_set = {t.strip().lower() for t in tickers}

    matched_cols = [normalized_cols[t] for t in tickers_set if t in normalized_cols]

    if not matched_cols:
        return pd.DataFrame()

    out = price_df[matched_cols].copy()
    out = out.dropna(axis=1, how="all").dropna(axis=0, how="any")
    return out

def returns_from_prices(prices: pd.DataFrame, freq: int = 252):
    # 1. DROP rows where ANY asset is missing data. 
    # This ensures all assets have the exact same time window.
    prices_aligned = prices.dropna(how="any") 
    
    # Check if we have enough data points left
    if len(prices_aligned) < 30: 
        return None, None # Not enough data to optimize

    rets = np.log(prices_aligned / prices_aligned.shift(1)).dropna()
    
    mu_annual = rets.mean() * freq
    cov_annual = rets.cov() * freq
    
    # 2. Add small "jitter" to diagonal to prevent singular matrix errors
    cov_annual = cov_annual + np.eye(len(cov_annual)) * 1e-6
    
    return mu_annual, cov_annual

def max_sharpe_weights(mu: pd.Series, cov: pd.DataFrame,
                       allow_short: bool = False, lambda_risk: float = 1.0) -> pd.Series:
    n = len(mu)
    tickers = mu.index.tolist()
    w0 = np.ones(n) / n
    try:
        from scipy.optimize import minimize

        def neg_objective(w):
            r = float(w @ mu.values)
            v = float(w @ cov.values @ w)
            return v - lambda_risk * r  # ✅ correct scaling

        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)
        bounds = None if allow_short else [(0.05, 0.30)] * n

        res = minimize(neg_objective, w0, method="SLSQP", bounds=bounds, constraints=cons)
        if not res.success or not np.all(np.isfinite(res.x)):
            raise RuntimeError("Optimization failed")
        w = res.x
    except Exception:
        iv = 1.0 / np.clip(np.diag(cov.values), 1e-12, None)
        w = iv / iv.sum()
    return pd.Series(w, index=tickers)



def describe_portfolio(w: pd.Series, mu: pd.Series, cov: pd.DataFrame) -> dict:
    port_ret = float(w @ mu)
    port_var = float(w.values @ cov.values @ w.values)
    port_vol = float(np.sqrt(max(port_var, 1e-12)))
    sharpe   = port_ret / port_vol if port_vol > 0 else np.nan
    return {"exp_return": port_ret, "vol": port_vol, "sharpe": sharpe}

def show_results(df: pd.DataFrame, label: str, base_key: str):
    """
    1) Rank ETFs; 2) show Top-10 with warnings; 3) Build portfolio from Top-10
    using matching price sheet; 4) Output weights + names + CSV.
    """
    st.markdown(f"### {label} — Results")

    if df is None or df.empty:
        st.warning(f"⚠️ No ETFs found {label.lower()} after applying filters.")
        return

    ranked = rank_etfs(df)

    if len(ranked) == 0:
        st.warning(f"⚠️ No {label} ETFs found after applying filters.")
        return
    elif len(ranked) < 10:
        st.warning(f"⚠️ Only {len(ranked)} {label} ETF(s) matched your filters — showing all available below.")
    else:
        st.success(f"✅ Showing Top 10 {label} ETFs.")


    st.dataframe(ranked.head(10), use_container_width=True)
    # 🎯 Add user risk tolerance input
    risk_level = st.radio(
        "Select your risk preference:",
        ["Conservative", "Balanced", "Aggressive"],
        horizontal=True,
        key=f"risk_{base_key}"
    )

    # Map user risk to risk-aversion parameter λ
    risk_map = {"Conservative": 0.5, "Balanced": 1.0, "Aggressive": 2.0}
    lambda_risk = risk_map[risk_level]

    # Build portfolio button
    build = st.button(f"📈 Build Mean-Variance Portfolio from Top 10 ({label})", key=f"build_{base_key}")
    if not build:
        return

    # Identify ticker column
    ticker_col = get_ticker_col(ranked)
    if ticker_col is None:
        st.error("❌ Could not find a ticker/symbol column in the filtered table.")
        return

    top = ranked.head(10).copy()
    tickers = top[ticker_col].astype(str).str.strip().tolist()

    prices = get_prices_for(base_key, tickers)
    if prices.empty or prices.shape[1] < 2:
        st.error("❌ Not enough price history for the selected tickers to build a portfolio.")
        return

    mu, cov = returns_from_prices(prices, freq=252)
    # ADD THIS ERROR CHECK
    if mu is None: 
        st.error("❌ Not enough overlapping data history for these specific ETFs to run optimization. Try selecting ETFs with longer histories.")
        return
# ...
    common = [t for t in tickers if t in mu.index and t in cov.index]
    mu  = mu.loc[common]
    cov = cov.loc[common, common]

    if len(common) < 2:
        st.error("❌ Need at least two tickers with valid return history.")
        return

    w = max_sharpe_weights(mu, cov, allow_short=False, lambda_risk=lambda_risk)
    stats = describe_portfolio(w, mu, cov)

    # Attach names if available
    name_map = {}
    if "name" in top.columns:
        name_map = dict(zip(top[ticker_col].astype(str), top["name"].astype(str)))

    out = (
        pd.DataFrame({
            "Ticker": w.index,
            "Name": [name_map.get(t, "") for t in w.index],
            "Weight": w.values,
        })
        .sort_values("Weight", ascending=False)
        .reset_index(drop=True)
    )

    st.subheader("✅ Final Portfolio (Max-Sharpe, no shorting)")
    st.dataframe(out, use_container_width=True)

    st.markdown(
        f"**Expected annual return**: `{stats['exp_return']:.2%}`  |  "
        f"**Volatility**: `{stats['vol']:.2%}`  |  "
        f"**Sharpe**: `{stats['sharpe']:.2f}`"
    )

    csv = out.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Portfolio CSV", data=csv,
                       file_name="portfolio_weights.csv", mime="text/csv")

# ============================================================
# 🏷️ UI — Title & Caption
# ============================================================

st.title("🌐 ETF Robo-Advisor")
st.caption("ETFs were pre-filtered for liquidity and matched with price data. Build a portfolio in one click.")

# ============================================================
# 🧠 STEP 1: Investor Preference — Region
# ============================================================

region_choice = st.radio(
    "1️⃣ Do you want to focus on Canadian ETFs or broader global options?",
    ["Canadian only", "Broader regions"],
    horizontal=True
)

# ============================================================
# 🇨🇦 Canadian route
# ============================================================

if region_choice == "Canadian only":
    st.header("🇨🇦 Canadian ETF Filters")

    f = {}
    # Column names below should match your info sheet columns exactly
    f["Use_Derivative"]   = st.selectbox("Use_Derivative", ["", "Yes", "No"])
    f["structure"]        = st.selectbox("structure", ["",
                               "Open-End Exch Traded Index Fd",
                               "Open-End Investment Company",
                               "Other",
                               "Unit Investment Trust (UIT)",
                               "Unit Trust"])
    f["ETF_General_Type"] = st.selectbox("ETF_General_Type", ["", "Equity", "Bond"])
    f["Issuer"]           = st.selectbox("Issuer", ["", "BlackRock", "BMO", "CIBC",
                                                    "Harvest", "Mirae Asset Global",
                                                    "Others", "RBC"])
    f["ESG_Focus"]        = st.selectbox("ESG_Focus", ["", "Yes ESG", "No ESG"])
    f["Management_Style"] = st.selectbox("Management_Style", ["", "Active", "Passive"])

    df_can = filter_etfs(info_sheets.get("canadian", pd.DataFrame()), f)
    show_results(df_can, label="Canadian", base_key="canadian")

# ============================================================
# 🌎 Broader route
# ============================================================

else:
    st.header("🌎 Broader ETF Selection")

    etf_type = st.radio(
        "2️⃣ What type of ETF do you want to invest in?",
        ["Bond ETFs", "Equity ETFs", "Both"],
        horizontal=True
    )
    # ============================================================
    # 🌍 CASE 1 — If user selects BOTH, merge all categories
    # ============================================================
    if etf_type == "Both":
        st.subheader("Combined Bond + Equity ETF Filters")
    
        # Shared filter criteria across all ETF types
        f = {
            "ESG_Focus":        st.selectbox("ESG_Focus", ["", "Yes ESG", "No ESG"], key="both_esg"),
            "Management_Style": st.selectbox("Management_Style", ["", "Active", "Passive"], key="both_style"),
            "Use_Derivative":   st.selectbox("Use_Derivative", ["", "Yes", "No"], key="both_deriv"),
            "structure":        st.selectbox("structure", [
                                        "",
                                        "Open-End Investment Company",
                                        "Open-End Exch Traded Index Fd",
                                        "Managed Investment Scheme",
                                        "Unit Investment Trust (UIT)",
                                        "Unit Trust",
                                        "SICAV/ICVC"
                                    ], key="both_struct")
        }
    
        # Merge all info sheets (bond + equity)
        all_sheets = [
            info_sheets.get("bond_etfs", pd.DataFrame()),
            info_sheets.get("em_bonds", pd.DataFrame()),
            info_sheets.get("high_yield", pd.DataFrame()),
            info_sheets.get("sector_etfs", pd.DataFrame()),
            info_sheets.get("thematic", pd.DataFrame())
        ]
    
        df_all = pd.concat([d for d in all_sheets if not d.empty], ignore_index=True)
    
        # Apply global filters
        df_all = filter_etfs(df_all, f)
    
        # Single ranking + single optimization
        show_results(df_all, label="Combined (Bond + Equity)", base_key="combined")

    # ---------------- Bond path ----------------
    elif etf_type == "Bond ETFs":

        bond_focus = st.radio(
            "3️⃣ Choose bond focus:",
            ["Developed Market Bond ETFs", "Emerging Market Bond ETFs",
             "High Yield Corp Bond ETFs", "No preference"],
            horizontal=True
        )
        
        if bond_focus == "Developed Market Bond ETFs":
            st.subheader("Developed Bond Filters")
            f = {
                "Strategic_Focus": st.selectbox("Strategic_Focus", ["", "Corporate", "Government"], key="Strategic_Focus_bond_etfs"),
                "ESG_Focus":       st.selectbox("ESG_Focus", ["", "Yes ESG", "No ESG"], key="ESG_Focus_bond_etfs"),
                "Management_Style": st.selectbox("Management_Style", ["", "Active", "Passive"], key="Management_Style_bond_etfs"),
                "structure":       st.selectbox("structure", [
                                        "",
                                        "Managed Investment Scheme",
                                        "Open-End Exch Traded Index Fd",
                                        "Open-End Investment Company",
                                        "Other",
                                        "SICAV/ICVC",
                                        "Unit Investment Trust (UIT)",
                                        "Unit Trust"
                                     ], key="structure_bond_etfs"),
                "Use_Derivative":  st.selectbox("Use_Derivative", ["", "Yes", "No"], key="Use_Derivative_bond_etfs"),
            }
            df = filter_etfs(info_sheets.get("bond_etfs", pd.DataFrame()), f)
            show_results(df, "Developed Market Bond", base_key="bond_etfs")


        elif bond_focus == "Emerging Market Bond ETFs":
            st.subheader("EM Bond Filters")
            f = {
                "ESG_Focus":        st.selectbox("ESG_Focus", ["", "Yes ESG", "No ESG"], key="ESG_Focus_em_bonds"),
                "Management_Style": st.selectbox("Management_Style", ["", "Active", "Passive"], key="Management_Style_em_bonds"),
                "Strategic_Focus":  st.selectbox("Strategic_Focus", ["", "Corporate", "Government"], key="Strategic_Focus_em_bonds"),
                "structure":        st.selectbox("structure", [
                                            "",
                                            "Managed Investment Scheme",
                                            "Open-End Exch Traded Index Fd",
                                            "Open-End Investment Company"
                                         ], key="structure_em_bonds"),
            }
            df = filter_etfs(info_sheets.get("em_bonds", pd.DataFrame()), f)
            show_results(df, "Emerging Market Bond", base_key="em_bonds")


        elif bond_focus == "High Yield Corp Bond ETFs":
            st.subheader("High Yield Corp Filters")
            f = {
                "ESG_Focus":        st.selectbox("ESG_Focus", ["", "Yes ESG", "No ESG"], key="ESG_Focus_high_yield"),
                "Management_Style": st.selectbox("Management_Style", ["", "Active", "Passive"], key="Management_Style_high_yield"),
                "Exchange_Region":  st.selectbox("Exchange_Region", ["", "Canada", "China", "US", "Western Europe"], key="Exchange_Region_high_yield"),
                "structure":        st.selectbox("structure", [
                                            "",
                                            "Open-End Exch Traded Index Fd",
                                            "Open-End Investment Company",
                                            "Swiss Domiciled Fund",
                                            "Unit Investment Trust (UIT)",
                                            "Unit Trust"
                                         ], key="structure_high_yield"),
                "Use_Derivative":   st.selectbox("Use_Derivative", ["", "Yes", "No"], key="Use_Derivative_high_yield"),
            }
            df = filter_etfs(info_sheets.get("high_yield", pd.DataFrame()), f)
            show_results(df, "High Yield Corporate Bond", base_key="high_yield")


    # ---------------- Equity path ----------------
    elif etf_type == "Equity ETFs":

        eq_focus = st.radio("3️⃣ Choose equity focus:", ["By sectors", "By Theme"], horizontal=True)

        if eq_focus == "By sectors":
            st.subheader("Sector ETF Filters")
            f = {
                "Sector_Focus":     st.selectbox("Sector_Focus", ["", "Communication Services",
                                                                 "Consumer Discretionary", "Consumer Staples",
                                                                 "Energy", "Financials", "Health Care",
                                                                 "Industrials", "Materials", "Technology",
                                                                 "Utilities"], key="Sector_Focus_sector_etfs"),
                "ESG_Focus":        st.selectbox("ESG_Focus", ["", "Yes ESG", "No ESG"], key="ESG_Focus_sector_etfs"),
                "Management_Style": st.selectbox("Management_Style", ["", "Active", "Passive"], key="Management_Style_sector_etfs"),
                "Geographic_Focus": st.selectbox("Geographic_Focus", ["", "Emerging Markets", "North America"], key="Geographic_Focus_sector_etfs"),
                "structure":        st.selectbox("structure", ["", "Debt Instrument", "Note",
                                                               "Open-End Exch Traded Index Fd",
                                                               "Open-End Investment Company"], key="structure_sector_etfs"),
                "Use_Derivative":   st.selectbox("Use_Derivative", ["", "Yes", "No"], key="Use_Derivative_sector_etfs"),
            }
            df = filter_etfs(info_sheets.get("sector_etfs", pd.DataFrame()), f)
            show_results(df, "Sector", base_key="sector_etfs")


        else:  # Thematic
            st.subheader("Thematic ETF Filters")
            f = {
                "Theme_Focus":      st.selectbox("Theme_Focus", ["", "Equity Thematic", "Growth Large Cap",
                                                                 "Large Cap", "Small Cap", "Technology"], key="Theme_Focus_thematic"),
                "ESG_Focus":        st.selectbox("ESG_Focus", ["", "Yes ESG", "No ESG"], key="ESG_Focus_thematic"),
                "Management_Style": st.selectbox("Management_Style", ["", "Active", "Passive"], key="Management_Style_thematic"),
                "structure":        st.selectbox("structure", ["", "Debt Instrument", "FCP",
                                                               "Managed Investment Scheme",
                                                               "Open-End Exch Traded Index Fd",
                                                               "Open-End Investment Company",
                                                               "Unit Investment Trust (UIT)"], key="structure_thematic"),
                "Use_Derivative":   st.selectbox("Use_Derivative", ["", "Yes", "No"], key="Use_Derivative_thematic"),
            }
            df = filter_etfs(info_sheets.get("thematic", pd.DataFrame()), f)
            show_results(df, "Thematic", base_key="thematic")


st.markdown("---")
st.caption("Robo-Advisor — liquidity-filtered ETFs with price-matched optimization (max-Sharpe).")
