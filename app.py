"""
Financial Capstone — Unified Pipeline
Market Screener → DCF Valuation → Portfolio Optimizer
Built with Streamlit · yfinance · PyPortfolioOpt · Plotly
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings
import datetime

warnings.filterwarnings("ignore")

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Financial Capstone — Screener + DCF + Optimizer",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* App background */
.stApp { background-color: #111827; color: #f0f4f8; }

#MainMenu, footer, header { visibility: hidden; }

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #1a2234 !important;
    border-right: 1px solid #2d3748 !important;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p { color: #a0aec0 !important; font-size: 0.83rem; }
[data-testid="stSidebar"] h2 {
    color: #f0f4f8 !important;
    font-size: 1rem;
    font-weight: 700;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
    color: #718096 !important;
    font-size: 0.7rem !important;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.25rem !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #1a2234;
    border-radius: 6px;
    padding: 3px;
    gap: 3px;
    border: 1px solid #2d3748;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #718096;
    font-weight: 600;
    border-radius: 5px;
    padding: 0.45rem 1.1rem;
    font-size: 0.84rem;
}
.stTabs [aria-selected="true"] {
    background: #2563eb !important;
    color: #ffffff !important;
}

/* Hero banner */
.hero {
    background: #1a2234;
    border: 1px solid #2d3748;
    border-left: 4px solid #2563eb;
    border-radius: 8px;
    padding: 1.4rem 1.8rem;
    margin-bottom: 1.2rem;
}
.hero h1 {
    font-size: 1.45rem;
    font-weight: 700;
    color: #f0f4f8;
    margin: 0 0 0.3rem;
}
.hero p {
    color: #a0aec0;
    font-size: 0.875rem;
    margin: 0;
    line-height: 1.6;
}

/* Metric cards */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(155px, 1fr));
    gap: 0.75rem;
    margin: 0.9rem 0 1.3rem;
}
.metric-card {
    background: #1a2234;
    border: 1px solid #2d3748;
    border-radius: 6px;
    padding: 0.9rem 1.1rem;
}
.metric-label {
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #718096;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-size: 1.55rem;
    font-weight: 700;
    color: #f0f4f8;
    line-height: 1.1;
}
.metric-value.green { color: #48bb78; }
.metric-value.blue  { color: #4299e1; }
.metric-value.amber { color: #ed8936; }
.metric-value.red   { color: #fc8181; }

/* Section headers */
.section-header {
    font-size: 0.9rem;
    font-weight: 700;
    color: #e2e8f0;
    margin: 1.4rem 0 0.65rem;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid #2d3748;
    letter-spacing: 0.01em;
}

/* Valuation badges */
.badge {
    display: inline-block;
    padding: 0.18rem 0.6rem;
    border-radius: 4px;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.badge-under { background: #1c4532; color: #68d391; border: 1px solid #2f855a; }
.badge-over  { background: #3d1515; color: #fc8181; border: 1px solid #c53030; }
.badge-na    { background: #1a202c; color: #90cdf4; border: 1px solid #2b6cb0; }

/* Info / warning boxes */
.info-box {
    background: #1a2a45;
    border: 1px solid #2b6cb0;
    border-left: 4px solid #2563eb;
    border-radius: 6px;
    padding: 0.85rem 1.1rem;
    font-size: 0.85rem;
    color: #90cdf4;
    line-height: 1.6;
    margin: 0.65rem 0;
}
.warn-box {
    background: #2d1f00;
    border: 1px solid #c05621;
    border-left: 4px solid #ed8936;
    border-radius: 6px;
    padding: 0.85rem 1.1rem;
    font-size: 0.85rem;
    color: #fbd38d;
    line-height: 1.6;
    margin: 0.65rem 0;
}

/* Pipeline step badges */
.pipeline-row {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    margin: 0.65rem 0;
    font-size: 0.81rem;
    color: #a0aec0;
}
.pipe-step {
    background: #263040;
    border: 1px solid #2d3748;
    border-radius: 4px;
    padding: 0.22rem 0.65rem;
    font-weight: 600;
    font-size: 0.75rem;
    color: #e2e8f0;
    white-space: nowrap;
}
.pipe-step.active { background: #1a3a6e; border-color: #2563eb; color: #90cdf4; }
.pipe-arrow { color: #4a5568; font-size: 0.95rem; }

/* Dataframe */
[data-testid="stDataFrame"] {
    border-radius: 6px;
    overflow: hidden;
    border: 1px solid #2d3748 !important;
}

.stAlert { border-radius: 6px !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_screener_data(tickers: tuple) -> pd.DataFrame:
    """
    Fetch basic screening metrics (Price, P/E, Debt/Equity, Market Cap, Sector)
    for each ticker. Returns a DataFrame with one row per ticker.
    """
    rows = []
    for ticker in tickers:
        try:
            t    = yf.Ticker(ticker)
            info = t.info  # primary source

            # Price: try info first, fall back to fast_info
            price = (info.get("currentPrice")
                     or info.get("regularMarketPrice")
                     or getattr(t.fast_info, "last_price", None))

            pe      = info.get("trailingPE")
            de      = info.get("debtToEquity")
            mktcap  = info.get("marketCap") or getattr(t.fast_info, "market_cap", None)
            sector  = info.get("sector", "N/A")
            name    = info.get("shortName") or info.get("longName", ticker)
            beta    = info.get("beta")
            div_yld = info.get("dividendYield")

            rows.append({
                "Ticker":          ticker,
                "Name":            name,
                "Sector":          sector,
                "Price ($)":       round(float(price), 2)        if price   else None,
                "P/E Ratio":       round(float(pe), 2)           if pe      else None,
                "Debt/Equity":     round(float(de) / 100, 2)     if de      else None,
                "Market Cap ($B)": round(float(mktcap) / 1e9, 2) if mktcap  else None,
                "Beta":            round(float(beta), 2)         if beta    else None,
                "Div Yield (%)":   round(float(div_yld) * 100, 2) if div_yld else None,
            })
        except Exception as exc:
            rows.append({"Ticker": ticker, "Name": ticker,
                         "Sector": f"Fetch error: {type(exc).__name__}",
                         "Price ($)": None, "P/E Ratio": None, "Debt/Equity": None,
                         "Market Cap ($B)": None, "Beta": None, "Div Yield (%)": None})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_dcf_inputs(tickers: tuple) -> dict:
    """
    Fetch FCF, shares outstanding, and net cash for each ticker.
    Returns a dict keyed by ticker.
    """
    results = {}
    for ticker in tickers:
        try:
            t    = yf.Ticker(ticker)
            info = t.info

            fcf    = info.get("freeCashflow")
            shares = (info.get("sharesOutstanding")
                      or info.get("impliedSharesOutstanding")
                      or getattr(t.fast_info, "shares", None))
            cash   = info.get("totalCash", 0) or 0
            debt   = info.get("totalDebt", 0) or 0
            price  = (info.get("currentPrice")
                      or info.get("regularMarketPrice")
                      or getattr(t.fast_info, "last_price", None))

            results[ticker] = {
                "fcf":      fcf,
                "shares":   shares,
                "net_cash": float(cash) - float(debt),
                "price":    float(price) if price else None,
            }
        except Exception:
            results[ticker] = {"fcf": None, "shares": None, "net_cash": 0, "price": None}
    return results


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_prices(tickers: tuple, years: int = 3) -> pd.DataFrame:
    """Download adjusted close prices for the given tickers."""
    end   = datetime.date.today()
    start = end - datetime.timedelta(days=years * 365)
    raw   = yf.download(list(tickers), start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = list(tickers)
    prices.dropna(how="all", inplace=True)
    return prices


def run_dcf(fcf: float, shares: float, net_cash: float,
            wacc: float, growth_rate: float, terminal_growth: float,
            projection_years: int = 5) -> float | None:
    """
    2-stage DCF: grow FCF for `projection_years` at `growth_rate`,
    then apply Gordon Growth Model terminal value.
    Returns intrinsic value per share (or None if inputs invalid).
    """
    if not fcf or not shares or shares == 0:
        return None
    if wacc <= terminal_growth:
        return None

    pv_fcfs = 0.0
    cf = fcf
    for yr in range(1, projection_years + 1):
        cf *= (1 + growth_rate)
        pv_fcfs += cf / (1 + wacc) ** yr

    # Terminal value (Gordon Growth)
    terminal_cf = cf * (1 + terminal_growth)
    terminal_val = terminal_cf / (wacc - terminal_growth)
    pv_terminal  = terminal_val / (1 + wacc) ** projection_years

    equity_value = pv_fcfs + pv_terminal + net_cash
    return equity_value / shares


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_TICKERS = "AAPL, MSFT, GOOGL, AMZN, NVDA, JPM, V, UNH, TSLA, XOM, SBUX, DIS, PG, META, JNJ"

with st.sidebar:
    st.markdown("## Financial Capstone")
    st.markdown("<div style='color:#64748b;font-size:0.78rem;margin-top:-0.5rem;margin-bottom:1rem'>Screener → DCF → Optimizer</div>", unsafe_allow_html=True)
    st.markdown("---")

    # ── Ticker Universe ──────────────────────────────────────────────────────
    st.markdown("### Ticker Universe")
    raw_input = st.text_area(
        "Enter 10–20 tickers (comma-separated)",
        value=DEFAULT_TICKERS,
        height=120,
        help="Enter 10–20 US stock tickers. ETFs and REITs may lack FCF data and will be excluded from the DCF step.",
        key="ticker_input",
    )

    st.markdown("---")

    # ── DCF Parameters ───────────────────────────────────────────────────────
    st.markdown("### DCF Parameters")
    st.markdown("<div style='color:#64748b;font-size:0.72rem;margin-bottom:0.5rem'>These drive the Valuation & filter the Optimizer</div>", unsafe_allow_html=True)

    wacc = st.slider(
        "WACC (%)",
        min_value=5.0, max_value=15.0, value=7.0, step=0.25,
        help="Weighted Average Cost of Capital — the discount rate applied to future free cash flows. Higher WACC → lower intrinsic value.",
        key="wacc_slider",
    ) / 100.0

    growth_rate = st.slider(
        "FCF Growth Rate (%)",
        min_value=1.0, max_value=20.0, value=8.0, step=0.5,
        help="Expected annual FCF growth rate for the 5-year projection period.",
        key="growth_slider",
    ) / 100.0

    terminal_growth = st.slider(
        "Terminal Growth Rate (%)",
        min_value=1.0, max_value=4.0, value=2.5, step=0.25,
        help="Long-run perpetual growth rate after the projection period (typically close to GDP growth).",
        key="terminal_slider",
    ) / 100.0

    margin_of_safety = st.slider(
        "Margin of Safety (%)",
        min_value=0, max_value=40, value=0, step=5,
        help="Requires Market Price < Intrinsic Value × (1 − MoS). A 20% MoS means you only buy at a 20% discount to IV. Higher = stricter filter.",
        key="mos_slider",
    ) / 100.0

    st.markdown("---")

    # ── Optimizer Parameters ─────────────────────────────────────────────────
    st.markdown("### Optimizer Parameters")
    risk_free = st.slider(
        "Risk-Free Rate (%)",
        min_value=0.0, max_value=8.0, value=4.5, step=0.25,
        help="Annualised risk-free rate (e.g., 10-yr Treasury). Used for Sharpe Ratio calculation.",
        key="rf_slider",
    ) / 100.0

    max_weight_pct = st.slider(
        "Max Weight per Stock (%)",
        min_value=10, max_value=100, value=40, step=5,
        help="Caps maximum allocation to any single asset. Lower values enforce diversification.",
        key="maxwt_slider",
    )
    max_weight = max_weight_pct / 100.0

    history_years = st.slider(
        "Price History (years)",
        min_value=1, max_value=5, value=3,
        help="How many years of daily price data to use for the optimizer.",
        key="history_slider",
    )

    st.markdown("---")
    st.markdown(f"""
    <div style='color:#475569;font-size:0.72rem;line-height:1.7'>
    <b style='color:#64748b'>Pipeline</b><br>
    &bull; WACC: {wacc*100:.2f}% &middot; MoS: {margin_of_safety*100:.0f}%<br>
    &bull; Growth: {growth_rate*100:.1f}% &middot; Terminal g: {terminal_growth*100:.2f}%<br>
    &bull; RF Rate: {risk_free*100:.2f}%<br>
    &bull; Max Wt: {max_weight_pct}% &middot; History: {history_years}yr
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    if st.button(
        "Recalculate & Sync",
        help="Clears all cached data and re-fetches live from yfinance. Use this if tickers show errors or data looks stale.",
        key="refresh_btn",
        use_container_width=True,
    ):
        st.cache_data.clear()
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PARSE & STORE TICKERS IN SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════

tickers_raw = [t.strip().upper() for t in raw_input.split(",") if t.strip()]
tickers_raw = list(dict.fromkeys(tickers_raw))  # deduplicate, preserve order

if len(tickers_raw) < 3:
    st.error("Please enter **at least 3** tickers to begin.")
    st.stop()
if len(tickers_raw) > 20:
    st.warning(f"You entered {len(tickers_raw)} tickers — truncating to the first 20.")
    tickers_raw = tickers_raw[:20]

# Store in session state so all components can reference it
st.session_state["tickers"] = tickers_raw
tickers = tuple(tickers_raw)


# ═══════════════════════════════════════════════════════════════════════════════
# PRE-COMPUTE DCF (hoisted so funnel counts are available before tabs render)
# ═══════════════════════════════════════════════════════════════════════════════

with st.spinner(f"Loading financial data for {len(tickers)} tickers — this may take up to 30 seconds on first load..."):
    dcf_inputs = fetch_dcf_inputs(tickers)   # cached after first run

dcf_rows = []
for _ticker in tickers:
    _d  = dcf_inputs[_ticker]
    _iv = run_dcf(
        fcf=_d["fcf"], shares=_d["shares"], net_cash=_d["net_cash"],
        wacc=wacc, growth_rate=growth_rate, terminal_growth=terminal_growth,
    )
    _price = _d["price"]
    if _iv is None or _price is None or _price == 0:
        _status  = "N/A"
        _upside  = None
    else:
        _threshold = _iv * (1 - margin_of_safety)
        if _price < _threshold:
            _status = "UNDERVALUED"
            _upside = (_iv - _price) / _price * 100
        else:
            _status = "OVERVALUED"
            _upside = (_iv - _price) / _price * 100
    dcf_rows.append({
        "Ticker":                  _ticker,
        "Market Price ($)":        round(_price, 2) if _price else None,
        "Intrinsic Value ($)":     round(_iv, 2)    if _iv    else None,
        "Upside / Downside (%)":   round(_upside, 1) if _upside is not None else None,
        "Status":                  _status,
        "FCF ($B)":                round(_d["fcf"] / 1e9, 2) if _d["fcf"] else None,
    })

dcf_df      = pd.DataFrame(dcf_rows)
undervalued = dcf_df[dcf_df["Status"] == "UNDERVALUED"]["Ticker"].tolist()
st.session_state["undervalued"] = undervalued
st.session_state["dcf_df"]      = dcf_df

n_under_global = len(undervalued)
n_portfolio    = st.session_state.get("n_active_positions", "—")

# ═══════════════════════════════════════════════════════════════════════════════
# HERO BANNER + FUNNEL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

# Funnel node helper
_mos_label    = f" (MoS {margin_of_safety*100:.0f}%)" if margin_of_safety > 0 else ""
_funnel_color = "#22c55e" if n_under_global > 0 else "#ef4444"

_funnel_html = """
<div class="hero">
  <h1>Financial Capstone &mdash; Unified Pipeline</h1>
  <p>
    A three-stage system: screen stocks for fundamentals, calculate a DCF intrinsic value,
    then run portfolio optimization exclusively on <strong>undervalued candidates</strong>.
    Adjust WACC or Margin of Safety sliders &mdash; the funnel updates instantly.
  </p>
  <div style="display:flex;align-items:center;gap:0;margin-top:1.1rem;flex-wrap:wrap">
    <div style="background:#1e2d45;border:1px solid #334155;border-radius:8px;
                padding:0.7rem 1.2rem;text-align:center;min-width:140px">
      <div style="font-size:0.65rem;font-weight:700;text-transform:uppercase;
                  letter-spacing:0.08em;color:#64748b;margin-bottom:0.25rem">Tickers Screened</div>
      <div style="font-size:1.9rem;font-weight:700;color:#3b82f6;
                  font-family:'Space Grotesk',sans-serif">{n_tickers}</div>
    </div>
    <div style="display:flex;flex-direction:column;align-items:center;padding:0 0.6rem">
      <div style="color:#334155;font-size:1.1rem">&rarr;</div>
      <div style="color:#475569;font-size:0.65rem;white-space:nowrap">DCF{mos}</div>
    </div>
    <div style="background:#1e2d45;border:1px solid #334155;border-radius:8px;
                padding:0.7rem 1.2rem;text-align:center;min-width:140px">
      <div style="font-size:0.65rem;font-weight:700;text-transform:uppercase;
                  letter-spacing:0.08em;color:#64748b;margin-bottom:0.25rem">Passed Valuation</div>
      <div style="font-size:1.9rem;font-weight:700;color:{fcolor};
                  font-family:'Space Grotesk',sans-serif">{n_under}</div>
    </div>
    <div style="display:flex;flex-direction:column;align-items:center;padding:0 0.6rem">
      <div style="color:#334155;font-size:1.1rem">&rarr;</div>
      <div style="color:#475569;font-size:0.65rem;white-space:nowrap">Max Sharpe</div>
    </div>
    <div style="background:#1e2d45;border:1px solid #334155;border-radius:8px;
                padding:0.7rem 1.2rem;text-align:center;min-width:140px">
      <div style="font-size:0.65rem;font-weight:700;text-transform:uppercase;
                  letter-spacing:0.08em;color:#64748b;margin-bottom:0.25rem">Final Portfolio</div>
      <div style="font-size:1.9rem;font-weight:700;color:#a78bfa;
                  font-family:'Space Grotesk',sans-serif">{n_port}</div>
    </div>
  </div>
</div>
""".format(
    n_tickers=len(tickers),
    mos=_mos_label,
    fcolor=_funnel_color,
    n_under=n_under_global,
    n_port=n_portfolio,
)

st.markdown(_funnel_html, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3 = st.tabs([
    "1 · Market Screener",
    "2 · DCF Valuation",
    "3 · Portfolio Optimizer",
])


# ═══════════════════════════════════════════════════════════════════════════════
# COMPONENT 1 — MARKET SCREENER
# ═══════════════════════════════════════════════════════════════════════════════

with tab1:
    # Update pipeline banner to show active step
    st.markdown("""
    <div class="info-box">
    <strong>Component 1 — Market Screener</strong><br>
    Fetches real-time fundamental metrics for your ticker universe.
    P/E Ratio and Debt/Equity are key signals used to contextualize the DCF results in the next tab.
    </div>
    """, unsafe_allow_html=True)

    with st.spinner(f"Fetching fundamental data for {len(tickers)} tickers..."):
        screener_df = fetch_screener_data(tickers)

    # Warn if any tickers failed to fetch
    error_tickers = screener_df[screener_df["Sector"].str.startswith("Fetch error", na=False)]["Ticker"].tolist()
    if error_tickers:
        st.warning(
            f"**{len(error_tickers)} ticker(s) failed to load:** {', '.join(error_tickers)}. "
            "This is usually a temporary yfinance rate-limit. "
            "Click **Recalculate & Sync** in the sidebar to retry."
        )

    # ── Summary Metrics ───────────────────────────────────────────────────────
    valid_prices  = screener_df["Price ($)"].dropna()
    valid_pe      = screener_df["P/E Ratio"].dropna()
    valid_de      = screener_df["Debt/Equity"].dropna()
    valid_mktcap  = screener_df["Market Cap ($B)"].dropna()

    n_loaded   = screener_df["Price ($)"].notna().sum()
    med_pe     = valid_pe.median()
    med_de     = valid_de.median()
    total_mcap = valid_mktcap.sum()

    st.markdown('<div class="section-header">Universe Summary</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-grid">
      <div class="metric-card">
        <div class="metric-label">Tickers Loaded</div>
        <div class="metric-value blue">{n_loaded} / {len(tickers)}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Median P/E Ratio</div>
        <div class="metric-value amber">{f"{med_pe:.1f}x" if not np.isnan(med_pe) else "N/A"}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Median Debt/Equity</div>
        <div class="metric-value amber">{f"{med_de:.2f}x" if not np.isnan(med_de) else "N/A"}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Total Market Cap</div>
        <div class="metric-value green">${total_mcap:.1f}B</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Sector Distribution ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">Fundamental Metrics Table</div>', unsafe_allow_html=True)

    # Format and display table
    display_df = screener_df.copy()

    # Safely coerce numeric columns so N/A strings from yfinance don't crash stylers
    for _col in ["P/E Ratio", "Debt/Equity", "Price ($)", "Market Cap ($B)", "Beta", "Div Yield (%)"]:
        if _col in display_df.columns:
            display_df[_col] = pd.to_numeric(display_df[_col], errors="coerce")

    # Color-code P/E column (pandas 2.x: use .map instead of deprecated .applymap)
    def color_pe(val):
        if pd.isna(val): return "color: #718096"
        if val < 15:     return "color: #48bb78"   # cheap — green
        if val < 30:     return "color: #ed8936"   # fair  — amber
        return "color: #fc8181"                     # expensive — red

    def color_de(val):
        if pd.isna(val): return "color: #718096"
        if val < 0.5:    return "color: #48bb78"
        if val < 1.5:    return "color: #ed8936"
        return "color: #fc8181"

    if not display_df.empty:
        styled = (
            display_df.style
            .map(color_pe, subset=["P/E Ratio"])
            .map(color_de, subset=["Debt/Equity"])
            .format({
                "Price ($)":       lambda x: f"${x:,.2f}" if pd.notna(x) else "—",
                "P/E Ratio":       lambda x: f"{x:.1f}x"  if pd.notna(x) else "—",
                "Debt/Equity":     lambda x: f"{x:.2f}x"  if pd.notna(x) else "—",
                "Market Cap ($B)": lambda x: f"${x:.1f}B"  if pd.notna(x) else "—",
                "Beta":            lambda x: f"{x:.2f}"    if pd.notna(x) else "—",
                "Div Yield (%)":   lambda x: f"{x:.2f}%"   if pd.notna(x) else "—",
            }, na_rep="—")
            .set_properties(**{"font-size": "0.84rem"})
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)
    else:
        st.info("No data loaded yet. Check your ticker list and try again.")

    # ── P/E vs Debt/Equity Scatter ────────────────────────────────────────────
    st.markdown('<div class="section-header">P/E vs Debt/Equity — Universe Map</div>', unsafe_allow_html=True)

    plot_df = screener_df.dropna(subset=["P/E Ratio", "Debt/Equity"]).copy()

    if len(plot_df) >= 2:
        palette = ["#3b82f6","#8b5cf6","#06b6d4","#10b981","#f59e0b",
                   "#ef4444","#ec4899","#a855f7","#0ea5e9","#f97316",
                   "#14b8a6","#84cc16","#fb7185","#4f46e5","#fbbf24"]

        fig_scatter = go.Figure()
        for i, row in plot_df.iterrows():
            fig_scatter.add_trace(go.Scatter(
                x=[row["Debt/Equity"]],
                y=[row["P/E Ratio"]],
                mode="markers+text",
                marker=dict(size=14, color=palette[i % len(palette)],
                            line=dict(width=1.5, color="#0a0e1a")),
                text=[row["Ticker"]],
                textposition="top center",
                textfont=dict(size=10, color="#e2e8f0", family="Inter"),
                name=row["Ticker"],
                hovertemplate=(
                    f"<b>{row['Ticker']}</b><br>"
                    f"P/E: {row['P/E Ratio']:.1f}x<br>"
                    f"D/E: {row['Debt/Equity']:.2f}x<br>"
                    f"Price: ${row['Price ($)']:,.2f}<extra></extra>"
                    if pd.notna(row["Price ($)"]) else
                    f"<b>{row['Ticker']}</b><br>P/E: {row['P/E Ratio']:.1f}x<br>D/E: {row['Debt/Equity']:.2f}x<extra></extra>"
                ),
                showlegend=False,
            ))

        # Quadrant lines at medians
        fig_scatter.add_hline(y=float(med_pe), line=dict(color="#334155", width=1, dash="dot"))
        fig_scatter.add_vline(x=float(med_de), line=dict(color="#334155", width=1, dash="dot"))

        fig_scatter.update_layout(
            paper_bgcolor="#0f1421",
            plot_bgcolor="#0a0e1a",
            font=dict(family="Inter", color="#94a3b8"),
            xaxis=dict(
                title="Debt / Equity Ratio",
                showgrid=True, gridcolor="#1e2d45",
                zeroline=False, tickfont=dict(size=11, color="#64748b"),
                title_font=dict(color="#94a3b8"),
            ),
            yaxis=dict(
                title="P/E Ratio",
                showgrid=True, gridcolor="#1e2d45",
                zeroline=False, tickfont=dict(size=11, color="#64748b"),
                title_font=dict(color="#94a3b8"),
            ),
            margin=dict(l=10, r=10, t=20, b=10),
            height=420,
        )
        st.plotly_chart(fig_scatter, use_container_width=True, config={"displayModeBar": False})
        st.caption("Dotted lines = universe medians. Lower-left quadrant = cheapest valuation + least leverage.")
    else:
        st.info("Need at least 2 tickers with P/E and Debt/Equity data to render the scatter chart.")

    # ── Save screener data to session state for downstream use ────────────────
    st.session_state["screener_df"] = screener_df

    st.markdown("""
    <div class="info-box" style="margin-top:1.5rem">
    <strong>Next Step:</strong> Head to <strong>Tab 2 · DCF Valuation</strong> to see intrinsic values
    calculated using the WACC and Growth Rate sliders. The <strong>Tab 3 · Portfolio Optimizer</strong>
    will automatically use only the stocks that are undervalued.
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# COMPONENT 2 — DCF VALUATION
# ═══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("""
    <div class="info-box">
    <strong>Component 2 — DCF Valuation</strong><br>
    A 2-stage Discounted Cash Flow model: FCF grows for 5 years at the sidebar
    <em>Growth Rate</em>, then a Gordon Growth terminal value is applied using the
    <em>Terminal Growth Rate</em>. Both stages are discounted at <em>WACC</em>.
    Adjust any slider to instantly re-rank which stocks are undervalued.
    </div>
    """, unsafe_allow_html=True)

    # dcf_inputs, dcf_df, undervalued already computed above (hoisted)

    # ── Summary KPIs ─────────────────────────────────────────────────────────
    n_under  = len(undervalued)
    n_over   = (dcf_df["Status"] == "OVERVALUED").sum()
    n_na     = (dcf_df["Status"] == "N/A").sum()
    avg_up   = dcf_df.loc[dcf_df["Status"] == "UNDERVALUED", "Upside / Downside (%)"].mean()
    avg_down = dcf_df.loc[dcf_df["Status"] == "OVERVALUED",  "Upside / Downside (%)"].mean()

    st.markdown('<div class="section-header">Valuation Summary</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-grid">
      <div class="metric-card">
        <div class="metric-label">Undervalued Stocks</div>
        <div class="metric-value green">{n_under}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Overvalued Stocks</div>
        <div class="metric-value red">{n_over}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">No FCF Data (Excluded)</div>
        <div class="metric-value amber">{n_na}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Avg Upside (Undervalued)</div>
        <div class="metric-value green">{f"+{avg_up:.1f}%" if not np.isnan(avg_up) else "—"}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Avg Downside (Overvalued)</div>
        <div class="metric-value red">{f"{avg_down:.1f}%" if not np.isnan(avg_down) else "—"}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">WACC Applied</div>
        <div class="metric-value blue">{wacc*100:.2f}%</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Optimizer eligibility callout ────────────────────────────────────────
    if n_under == 0:
        st.markdown("""
        <div class="warn-box">
        <strong>No stocks meet your valuation criteria.</strong><br>
        Try lowering the <strong>WACC</strong> or <strong>Margin of Safety</strong>
        sliders to see the ripple effect — more stocks will pass the filter
        and the Portfolio Optimizer will activate automatically.
        </div>
        """, unsafe_allow_html=True)
    elif n_under < 5:
        st.markdown(f"""
        <div class="warn-box">
        <strong>Only {n_under} stock{"s" if n_under > 1 else ""} passed the undervaluation filter
        ({", ".join(undervalued)}).</strong><br>
        The Portfolio Optimizer will run on this concentrated set. Be aware that
        a portfolio of fewer than 5 positions carries <strong>high concentration risk</strong> —
        idiosyncratic events in any single holding can dominate total portfolio returns.
        Consider lowering the WACC or raising the Growth Rate to qualify more stocks.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="info-box">
        <strong>{n_under} undervalued stocks qualify for the Portfolio Optimizer:</strong>
        {", ".join(f"<strong>{t}</strong>" for t in undervalued)}<br>
        Head to <strong>Tab 3</strong> to run the optimization.
        </div>
        """, unsafe_allow_html=True)

    # ── Valuation Table ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Market Price vs. Intrinsic Value</div>', unsafe_allow_html=True)

    # Build display DataFrame with HTML badges in Status column
    def status_badge(status):
        if status == "UNDERVALUED":
            return '<span class="badge badge-under">UNDERVALUED</span>'
        elif status == "OVERVALUED":
            return '<span class="badge badge-over">OVERVALUED</span>'
        else:
            return '<span class="badge badge-na">NO FCF DATA</span>'

    table_df = dcf_df.copy()
    table_df["Status"] = table_df["Status"].apply(status_badge)

    def fmt_price(x):
        return f"${x:,.2f}" if pd.notna(x) else "—"

    def fmt_pct(x):
        if pd.isna(x): return "—"
        sign = "+" if x > 0 else ""
        return f"{sign}{x:.1f}%"

    def fmt_fcf(x):
        return f"${x:.2f}B" if pd.notna(x) else "—"

    table_display = table_df.copy()
    table_display["Market Price ($)"]        = table_display["Market Price ($)"].apply(fmt_price)
    table_display["Intrinsic Value ($)"]      = table_display["Intrinsic Value ($)"].apply(fmt_price)
    table_display["Upside / Downside (%)"]    = table_display["Upside / Downside (%)"].apply(fmt_pct)
    table_display["FCF ($B)"]                 = table_display["FCF ($B)"].apply(fmt_fcf)

    st.markdown(
        table_display.to_html(escape=False, index=False),
        unsafe_allow_html=True,
    )

    # ── Price vs Intrinsic Value Chart ─────────────────────────────────────
    st.markdown('<div class="section-header">Price vs. Intrinsic Value</div>', unsafe_allow_html=True)
    st.markdown("<div style='color:#64748b;font-size:0.8rem;margin-bottom:0.75rem'>Green bars above the market price line are undervalued. Red bars below are overvalued.</div>", unsafe_allow_html=True)

    piv_df = dcf_df.dropna(subset=["Market Price ($)", "Intrinsic Value ($)"]).copy()
    piv_df = piv_df.sort_values("Upside / Downside (%)", ascending=False)

    if not piv_df.empty:
        fig_piv = go.Figure()

        # Market Price bars (grey baseline)
        fig_piv.add_trace(go.Bar(
            name="Market Price",
            x=piv_df["Ticker"],
            y=piv_df["Market Price ($)"],
            marker=dict(color="#334155", line=dict(color="#475569", width=1)),
            hovertemplate="<b>%{x}</b><br>Market Price: $%{y:,.2f}<extra></extra>",
        ))

        # Intrinsic Value bars (green if IV > price, red if IV < price)
        iv_colors = [
            "#22c55e" if row["Status"] == "UNDERVALUED" else "#ef4444"
            for _, row in piv_df.iterrows()
        ]
        fig_piv.add_trace(go.Bar(
            name="Intrinsic Value (DCF)",
            x=piv_df["Ticker"],
            y=piv_df["Intrinsic Value ($)"],
            marker=dict(color=iv_colors, opacity=0.85,
                        line=dict(color="rgba(0,0,0,0)", width=0)),
            hovertemplate="<b>%{x}</b><br>Intrinsic Value: $%{y:,.2f}<extra></extra>",
        ))

        fig_piv.update_layout(
            barmode="overlay",
            paper_bgcolor="#0f1421", plot_bgcolor="#0a0e1a",
            font=dict(family="Inter", color="#94a3b8"),
            xaxis=dict(showgrid=False, tickfont=dict(size=12, color="#e2e8f0")),
            yaxis=dict(
                title="Price ($)",
                showgrid=True, gridcolor="#1e2d45",
                tickprefix="$", tickfont=dict(size=11, color="#64748b"),
                title_font=dict(color="#94a3b8"),
            ),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                font=dict(size=11, color="#94a3b8"), bgcolor="rgba(0,0,0,0)",
            ),
            margin=dict(l=10, r=10, t=40, b=10),
            height=380,
            bargap=0.25,
        )
        st.plotly_chart(fig_piv, use_container_width=True, config={"displayModeBar": False})
        st.caption("Grey bar = Market Price. Green/Red bar = DCF Intrinsic Value. Green above grey = undervalued; red below grey = overvalued.")
    else:
        st.info("No tickers with both a market price and intrinsic value available.")

    # ── Upside / Downside Bar Chart ──────────────────────────────────
    st.markdown('<div class="section-header">Upside / Downside to Intrinsic Value</div>', unsafe_allow_html=True)

    chart_df = dcf_df.dropna(subset=["Upside / Downside (%)"]).copy()
    chart_df = chart_df.sort_values("Upside / Downside (%)", ascending=True)

    bar_colors = [
        "#22c55e" if v >= 0 else "#ef4444"
        for v in chart_df["Upside / Downside (%)"]
    ]

    fig_bar = go.Figure(go.Bar(
        x=chart_df["Upside / Downside (%)"],
        y=chart_df["Ticker"],
        orientation="h",
        marker=dict(color=bar_colors, line=dict(color="rgba(0,0,0,0)", width=0)),
        text=[f"{v:+.1f}%" for v in chart_df["Upside / Downside (%)"]],
        textposition="outside",
        textfont=dict(family="Inter", size=11, color="#94a3b8"),
        hovertemplate="<b>%{y}</b><br>%{x:+.1f}%<extra></extra>",
        width=0.6,
    ))
    fig_bar.add_vline(x=0, line=dict(color="#334155", width=1.5))
    fig_bar.update_layout(
        paper_bgcolor="#0f1421",
        plot_bgcolor="#0a0e1a",
        font=dict(family="Inter", color="#94a3b8"),
        xaxis=dict(
            title="Upside (+) / Downside (−) to Intrinsic Value (%)",
            showgrid=True, gridcolor="#1e2d45",
            zeroline=False, tickfont=dict(size=11, color="#64748b"),
            title_font=dict(color="#94a3b8"),
        ),
        yaxis=dict(showgrid=False, tickfont=dict(size=12, color="#e2e8f0")),
        margin=dict(l=10, r=60, t=10, b=10),
        bargap=0.25,
        height=max(300, len(chart_df) * 38 + 60),
    )
    st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

    # ── WACC Sensitivity Heatmap ─────────────────────────────────────────────
    st.markdown('<div class="section-header">WACC Sensitivity — Intrinsic Value per Share ($)</div>', unsafe_allow_html=True)
    st.markdown("<div style='color:#64748b;font-size:0.8rem;margin-bottom:0.75rem'>How intrinsic value changes across different WACC assumptions (Growth Rate held constant at sidebar value).</div>", unsafe_allow_html=True)

    wacc_range    = np.arange(0.06, 0.145, 0.01)
    sense_tickers = [r["Ticker"] for r in dcf_rows if dcf_inputs[r["Ticker"]]["fcf"] is not None][:10]

    if sense_tickers:
        heat_data = {}
        for tkr in sense_tickers:
            d = dcf_inputs[tkr]
            row_vals = []
            for w in wacc_range:
                iv_s = run_dcf(d["fcf"], d["shares"], d["net_cash"],
                               w, growth_rate, terminal_growth)
                row_vals.append(round(iv_s, 2) if iv_s else None)
            heat_data[tkr] = row_vals

        heat_df   = pd.DataFrame(heat_data, index=[f"{w*100:.0f}%" for w in wacc_range])
        x_labels  = heat_df.columns.tolist()
        y_labels  = heat_df.index.tolist()
        z_values  = heat_df.values.T.tolist()

        # Replace None with 0 for plotting
        z_clean = [[v if v is not None else 0 for v in row] for row in z_values]

        # Compute text annotations
        z_text = [
            [f"${v:.0f}" if v and v > 0 else "—" for v in row]
            for row in z_values
        ]

        fig_heat = go.Figure(go.Heatmap(
            z=z_clean,
            x=y_labels,
            y=x_labels,
            colorscale=[
                [0.0,  "#450a0a"],
                [0.35, "#991b1b"],
                [0.55, "#1c1a05"],
                [0.75, "#14532d"],
                [1.0,  "#052e16"],
            ],
            text=z_text,
            texttemplate="%{text}",
            textfont=dict(size=10, color="#e2e8f0"),
            showscale=True,
            colorbar=dict(
                title=dict(text="IV ($)", font=dict(color="#94a3b8", size=11)),
                tickfont=dict(color="#64748b", size=10),
                thickness=14, len=0.75, outlinewidth=0,
            ),
            hovertemplate="<b>%{y}</b> at WACC %{x}<br>Intrinsic Value: %{text}<extra></extra>",
        ))

        # Highlight current WACC column
        curr_wacc_label = f"{round(wacc * 100):.0f}%"
        if curr_wacc_label in y_labels:
            idx = y_labels.index(curr_wacc_label)
            fig_heat.add_shape(
                type="rect",
                x0=idx - 0.5, x1=idx + 0.5,
                y0=-0.5, y1=len(x_labels) - 0.5,
                line=dict(color="#3b82f6", width=2),
                fillcolor="rgba(59,130,246,0.07)",
            )

        fig_heat.update_layout(
            paper_bgcolor="#0f1421",
            plot_bgcolor="#0a0e1a",
            font=dict(family="Inter", color="#94a3b8"),
            xaxis=dict(
                title="WACC",
                tickfont=dict(size=11, color="#64748b"),
                title_font=dict(color="#94a3b8"),
            ),
            yaxis=dict(
                title="Ticker",
                tickfont=dict(size=11, color="#e2e8f0"),
                title_font=dict(color="#94a3b8"),
            ),
            margin=dict(l=10, r=60, t=20, b=10),
            height=max(300, len(sense_tickers) * 42 + 80),
        )
        st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False})
        st.caption(f"Blue outline = current WACC ({wacc*100:.2f}%). Warmer = higher intrinsic value. Up to 10 tickers shown.")
    else:
        st.info("No tickers with FCF data available for sensitivity analysis.")

    st.markdown("""
    <div class="info-box" style="margin-top:1.5rem">
    <strong>Next Step:</strong> Switch to <strong>Tab 3 · Portfolio Optimizer</strong> to run
    Mean-Variance optimization on the undervalued stocks identified above.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# COMPONENT 3 — PORTFOLIO OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════════════

from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier

with tab3:

    # ── Pipeline banner ───────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="hero" style="border-left-color:#22c55e">
      <h1>Portfolio Optimizer <span style="font-size:1rem;color:#4ade80;font-weight:500">— Undervalued Stocks Only</span></h1>
      <p>
        Mean-Variance optimization (Max Sharpe Ratio) applied exclusively to stocks
        the DCF model identifies as <strong>undervalued</strong> at the current WACC of
        <strong>{wacc*100:.2f}%</strong>.
        Change the WACC slider and this portfolio re-optimizes automatically.
      </p>
      <div class="pipeline-row" style="margin-top:0.75rem">
        <span class="pipe-step">1 · Market Screener</span>
        <span class="pipe-arrow">→</span>
        <span class="pipe-step">2 · DCF Valuation</span>
        <span class="pipe-arrow">→</span>
        <span class="pipe-step active">3 · Portfolio Optimizer</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Read undervalued set from session state ───────────────────────────────
    opt_tickers = st.session_state.get("undervalued", [])

    if len(opt_tickers) == 0:
        st.markdown("""
        <div class="warn-box">
        <strong>No undervalued stocks to optimize.</strong><br>
        Return to <strong>Tab 2 · DCF Valuation</strong> and lower the WACC slider
        or raise the FCF Growth Rate so that at least 2 stocks pass the filter.
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    if len(opt_tickers) == 1:
        st.markdown(f"""
        <div class="warn-box">
        <strong>Only 1 undervalued stock ({opt_tickers[0]}) — need at least 2 to optimize.</strong><br>
        A single-stock portfolio cannot be mean-variance optimized.
        Lower WACC or raise FCF Growth Rate to qualify more stocks.
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # ── Concentration warning (2-4 stocks) ───────────────────────────────────
    if len(opt_tickers) < 5:
        st.markdown(f"""
        <div class="warn-box">
        <strong>Concentration Risk Warning</strong><br>
        Only <strong>{len(opt_tickers)} stock{"s" if len(opt_tickers) > 1 else ""}</strong>
        passed the undervaluation filter: <strong>{", ".join(opt_tickers)}</strong>.<br>
        The optimizer will run, but this portfolio is <strong>highly concentrated</strong>.
        Idiosyncratic events in any single holding can dominate total returns.
        This is a direct consequence of the strict DCF valuation criteria at WACC = {wacc*100:.2f}%.
        Lower WACC or raise the Growth Rate to diversify.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="info-box">
        Optimizing across <strong>{len(opt_tickers)} undervalued stocks</strong>:
        {", ".join(f"<strong>{t}</strong>" for t in opt_tickers)}.
        </div>
        """, unsafe_allow_html=True)

    # ── Fetch price history for undervalued tickers only ─────────────────────
    opt_tuple = tuple(opt_tickers)

    with st.spinner(f"Fetching {history_years}yr price history for {len(opt_tickers)} undervalued tickers…"):
        try:
            opt_prices = fetch_prices(opt_tuple, years=history_years)
        except Exception as e:
            st.error(f"Price data fetch failed: {e}")
            st.stop()

    # Drop tickers that came back with insufficient rows
    valid_opt = [t for t in opt_tickers if t in opt_prices.columns
                 and opt_prices[t].notna().sum() > 60]
    if len(valid_opt) < 2:
        st.error("Fewer than 2 undervalued tickers returned enough price history. "
                 "Try a shorter history window in the sidebar.")
        st.stop()

    opt_prices = opt_prices[valid_opt]

    # ── Run optimization ──────────────────────────────────────────────────────
    # For very small universes (2 stocks) relax per-asset weight cap to 100%
    # so Max Sharpe is always feasible regardless of the sidebar cap setting.
    _wt_bounds = (0, 1.0) if len(valid_opt) <= 2 else (0, max_weight)
    try:
        mu    = expected_returns.mean_historical_return(opt_prices)
        sigma = risk_models.sample_cov(opt_prices)
        ef    = EfficientFrontier(mu, sigma, weight_bounds=_wt_bounds)
        ef.max_sharpe(risk_free_rate=risk_free)
        weights_raw = ef.clean_weights()
        exp_ret, ann_vol, sharpe = ef.portfolio_performance(
            verbose=False, risk_free_rate=risk_free
        )
    except Exception as e:
        try:
            ef2 = EfficientFrontier(mu, sigma, weight_bounds=(0, 1.0))
            ef2.min_volatility()
            weights_raw = ef2.clean_weights()
            exp_ret, ann_vol, sharpe = ef2.portfolio_performance(
                verbose=False, risk_free_rate=risk_free
            )
            st.info("Max Sharpe infeasible with this universe — showing Minimum Volatility portfolio instead.")
        except Exception as e2:
            st.error(f"Optimization failed: {e2}")
            st.stop()

    active_weights = {k: v for k, v in weights_raw.items() if v > 1e-4}

    # ── Performance Summary ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">Portfolio Performance Summary</div>', unsafe_allow_html=True)
    sharpe_color = "green" if sharpe >= 1.0 else ("blue" if sharpe >= 0.5 else "amber")
    n_active     = len(active_weights)
    # Write to session state so the funnel "Final Portfolio" node updates
    st.session_state["n_active_positions"] = n_active

    st.markdown(f"""
    <div class="metric-grid">
      <div class="metric-card">
        <div class="metric-label">Expected Annual Return</div>
        <div class="metric-value green">{exp_ret*100:.2f}%</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Annual Volatility (Risk)</div>
        <div class="metric-value blue">{ann_vol*100:.2f}%</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Sharpe Ratio (RF={risk_free*100:.2f}%)</div>
        <div class="metric-value {sharpe_color}">{sharpe:.3f}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Active Positions</div>
        <div class="metric-value">{n_active} / {len(valid_opt)}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Excess Return over RF</div>
        <div class="metric-value green">{(exp_ret - risk_free)*100:+.2f}%</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Max Weight Cap</div>
        <div class="metric-value amber">{max_weight_pct}%</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Weight Charts ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Optimal Allocation</div>', unsafe_allow_html=True)

    col_bar, col_pie = st.columns([1.3, 1], gap="large")

    PALETTE = ["#3b82f6","#8b5cf6","#06b6d4","#10b981","#f59e0b",
               "#ef4444","#ec4899","#a855f7","#0ea5e9","#f97316",
               "#14b8a6","#84cc16","#fb7185","#4f46e5","#fbbf24"]

    with col_bar:
        # Sort ascending so largest bar is at top of chart
        pairs  = sorted(active_weights.items(), key=lambda x: x[1])
        tkrs   = [p[0] for p in pairs]
        vals   = [p[1] * 100 for p in pairs]
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(tkrs))]

        fig_wt = go.Figure(go.Bar(
            x=vals, y=tkrs, orientation="h",
            marker=dict(color=colors, line=dict(color="rgba(0,0,0,0)", width=0)),
            text=[f"{v:.1f}%" for v in vals],
            textposition="outside",
            textfont=dict(family="Inter", size=12, color="#94a3b8"),
            hovertemplate="<b>%{y}</b>: %{x:.2f}%<extra></extra>",
            width=0.6,
        ))
        fig_wt.update_layout(
            paper_bgcolor="#0f1421", plot_bgcolor="#0a0e1a",
            font=dict(family="Inter", color="#94a3b8"),
            xaxis=dict(title="Portfolio Weight (%)", showgrid=True,
                       gridcolor="#1e2d45", zeroline=False,
                       tickfont=dict(size=11, color="#64748b"),
                       range=[0, max(vals) * 1.22] if vals else [0, 100]),
            yaxis=dict(showgrid=False, tickfont=dict(size=12, color="#e2e8f0")),
            margin=dict(l=10, r=55, t=10, b=10), bargap=0.28,
            height=max(280, len(tkrs) * 48 + 60),
        )
        st.plotly_chart(fig_wt, use_container_width=True, config={"displayModeBar": False})

    with col_pie:
        all_tkrs = list(active_weights.keys())
        all_vals = list(active_weights.values())
        pie_colors = [PALETTE[i % len(PALETTE)] for i in range(len(all_tkrs))]

        fig_pie = go.Figure(go.Pie(
            labels=all_tkrs, values=all_vals,
            hole=0.58,
            marker=dict(colors=pie_colors, line=dict(color="#0a0e1a", width=2)),
            textinfo="label+percent",
            textfont=dict(family="Inter", size=12, color="#e2e8f0"),
            hovertemplate="<b>%{label}</b><br>Weight: %{percent}<extra></extra>",
        ))
        fig_pie.update_layout(
            paper_bgcolor="#0f1421", plot_bgcolor="#0a0e1a",
            font=dict(family="Inter", color="#94a3b8"),
            annotations=[dict(
                text="<b>Portfolio</b>", x=0.5, y=0.5,
                font=dict(size=13, color="#e2e8f0", family="Space Grotesk"),
                showarrow=False,
            )],
            legend=dict(orientation="v", x=1.03, y=0.5,
                        font=dict(color="#94a3b8", size=11),
                        bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=10, r=10, t=10, b=10),
            height=max(280, len(tkrs) * 48 + 60),
        )
        st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

    # ── Weights Table ─────────────────────────────────────────────────────────
    dcf_ref = st.session_state.get("dcf_df", pd.DataFrame())
    wt_rows = []
    for tkr, wt in sorted(active_weights.items(), key=lambda x: -x[1]):
        iv_row  = dcf_ref[dcf_ref["Ticker"] == tkr] if not dcf_ref.empty else pd.DataFrame()
        iv_val  = iv_row["Intrinsic Value ($)"].values[0] if not iv_row.empty else None
        mkt_val = iv_row["Market Price ($)"].values[0]   if not iv_row.empty else None
        upside  = iv_row["Upside / Downside (%)"].values[0] if not iv_row.empty else None
        exp_ind = mu[tkr] * 100 if tkr in mu.index else None
        wt_rows.append({
            "Ticker":               tkr,
            "Weight (%)":           f"{wt*100:.2f}%",
            "Market Price ($)":     f"${mkt_val:,.2f}" if mkt_val else "—",
            "Intrinsic Value ($)":  f"${iv_val:,.2f}"  if iv_val  else "—",
            "DCF Upside (%)":       f"+{upside:.1f}%"  if upside  else "—",
            "Hist. Return (ann.)":  f"{exp_ind:.2f}%"  if exp_ind else "—",
        })
    wt_df_display = pd.DataFrame(wt_rows)
    st.markdown('<div class="section-header">Position Detail</div>', unsafe_allow_html=True)
    st.dataframe(wt_df_display, use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════════════════
    # RIPPLE EFFECT DASHBOARD
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("""
    <div class="section-header" style="margin-top:2rem">
      The Ripple Effect — WACC Sensitivity Cascade
    </div>
    <div style="color:#64748b;font-size:0.82rem;line-height:1.7;margin-bottom:1rem">
      Shows how a change in WACC propagates through the entire pipeline:
      WACC → Intrinsic Values → Undervalued Count → Portfolio Sharpe Ratio.
      Each row below is a scenario evaluated independently at that WACC.
    </div>
    """, unsafe_allow_html=True)

    ripple_waccs = np.arange(0.06, 0.145, 0.01)
    ripple_rows  = []

    dcf_in = fetch_dcf_inputs(tickers)   # already cached

    for w in ripple_waccs:
        # Count undervalued at this WACC
        n_uv = 0
        uv_list = []
        for tkr in tickers:
            d   = dcf_in[tkr]
            iv  = run_dcf(d["fcf"], d["shares"], d["net_cash"],
                          w, growth_rate, terminal_growth)
            prc = d["price"]
            if iv and prc and prc < iv:
                n_uv += 1
                uv_list.append(tkr)

        # Attempt optimization at this WACC scenario
        scen_sharpe = None
        scen_ret    = None
        scen_vol    = None
        if len(uv_list) >= 2:
            try:
                p_sub = fetch_prices(tuple(uv_list), years=history_years)
                valid_sub = [t for t in uv_list
                             if t in p_sub.columns and p_sub[t].notna().sum() > 60]
                if len(valid_sub) >= 2:
                    p_sub = p_sub[valid_sub]
                    mu_s  = expected_returns.mean_historical_return(p_sub)
                    sg_s  = risk_models.sample_cov(p_sub)
                    ef_s  = EfficientFrontier(mu_s, sg_s, weight_bounds=(0, max_weight))
                    ef_s.max_sharpe(risk_free_rate=risk_free)
                    scen_ret, scen_vol, scen_sharpe = ef_s.portfolio_performance(
                        verbose=False, risk_free_rate=risk_free
                    )
            except Exception:
                pass

        ripple_rows.append({
            "WACC (%)":             round(w * 100, 1),
            "Undervalued Count":    n_uv,
            "Tickers Qualify":      ", ".join(uv_list) if uv_list else "—",
            "Port. Return (%)":     round(scen_ret * 100, 2) if scen_ret else None,
            "Port. Volatility (%)": round(scen_vol * 100, 2) if scen_vol else None,
            "Sharpe Ratio":         round(scen_sharpe, 3) if scen_sharpe else None,
        })

    ripple_df = pd.DataFrame(ripple_rows)

    # ── Ripple chart: 3-panel linked line charts ──────────────────────────────
    from plotly.subplots import make_subplots

    fig_rip = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=(
            "Undervalued Stock Count",
            "Portfolio Expected Return (%)",
            "Portfolio Sharpe Ratio",
        ),
        vertical_spacing=0.10,
    )

    line_kw = dict(width=2.2)
    marker_kw = dict(size=7)

    # Panel 1 — Undervalued Count
    fig_rip.add_trace(go.Scatter(
        x=ripple_df["WACC (%)"], y=ripple_df["Undervalued Count"],
        mode="lines+markers",
        line=dict(color="#22c55e", **line_kw),
        marker=dict(color="#22c55e", **marker_kw),
        name="Undervalued Count",
        hovertemplate="WACC %{x:.1f}%<br>Undervalued: %{y}<extra></extra>",
    ), row=1, col=1)

    # Panel 2 — Portfolio Return
    valid_ret = ripple_df.dropna(subset=["Port. Return (%)"])
    fig_rip.add_trace(go.Scatter(
        x=valid_ret["WACC (%)"], y=valid_ret["Port. Return (%)"],
        mode="lines+markers",
        line=dict(color="#3b82f6", **line_kw),
        marker=dict(color="#3b82f6", **marker_kw),
        name="Port. Return",
        hovertemplate="WACC %{x:.1f}%<br>Return: %{y:.2f}%<extra></extra>",
    ), row=2, col=1)

    # Panel 3 — Sharpe Ratio
    valid_sr = ripple_df.dropna(subset=["Sharpe Ratio"])
    fig_rip.add_trace(go.Scatter(
        x=valid_sr["WACC (%)"], y=valid_sr["Sharpe Ratio"],
        mode="lines+markers",
        line=dict(color="#a78bfa", **line_kw),
        marker=dict(color="#a78bfa", **marker_kw),
        name="Sharpe Ratio",
        hovertemplate="WACC %{x:.1f}%<br>Sharpe: %{y:.3f}<extra></extra>",
    ), row=3, col=1)

    # Current WACC vertical line across all panels
    curr_w = round(wacc * 100, 1)
    for row_n in [1, 2, 3]:
        fig_rip.add_vline(
            x=curr_w,
            line=dict(color="#f59e0b", width=1.5, dash="dash"),
            row=row_n, col=1,
        )

    fig_rip.update_layout(
        paper_bgcolor="#0f1421", plot_bgcolor="#0a0e1a",
        font=dict(family="Inter", color="#94a3b8"),
        height=560,
        showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    for ann in fig_rip.layout.annotations:
        ann.font = dict(color="#94a3b8", size=12, family="Inter")

    for axis in ["xaxis", "xaxis2", "xaxis3"]:
        fig_rip.layout[axis].update(
            showgrid=True, gridcolor="#1e2d45", zeroline=False,
            tickfont=dict(size=11, color="#64748b"),
        )
    for axis in ["yaxis", "yaxis2", "yaxis3"]:
        fig_rip.layout[axis].update(
            showgrid=True, gridcolor="#1e2d45", zeroline=False,
            tickfont=dict(size=11, color="#64748b"),
        )
    fig_rip.layout.xaxis3.title = dict(text="WACC (%)", font=dict(color="#94a3b8", size=12))

    st.plotly_chart(fig_rip, use_container_width=True, config={"displayModeBar": False})
    st.caption(f"Amber dashed line = current WACC ({curr_w}%). Gaps in panels 2–3 = fewer than 2 undervalued stocks at that WACC.")

    # ── Ripple Table ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Ripple Effect — Scenario Table</div>', unsafe_allow_html=True)

    def hilight_row(row):
        if abs(row["WACC (%)"] - curr_w) < 0.05:
            return ["background-color:#1d3a6e; color:#93c5fd"] * len(row)
        return [""] * len(row)

    ripple_styled = (
        ripple_df.style
        .apply(hilight_row, axis=1)
        .format({
            "WACC (%)":             "{:.1f}%",
            "Port. Return (%)":     lambda x: f"{x:.2f}%" if pd.notna(x) else "—",
            "Port. Volatility (%)": lambda x: f"{x:.2f}%" if pd.notna(x) else "—",
            "Sharpe Ratio":         lambda x: f"{x:.3f}"  if pd.notna(x) else "—",
        }, na_rep="—")
        .set_properties(**{"font-size": "0.83rem"})
    )
    st.dataframe(ripple_styled, use_container_width=True, hide_index=True)

# ── Global Footer ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#475569;font-size:0.75rem;padding:0.5rem;line-height:1.8'>
  <strong style='color:#64748b'>Financial Capstone</strong> — Market Screener · DCF Valuation · Portfolio Optimizer<br>
  Data via <strong>yfinance</strong> · Optimization via <strong>PyPortfolioOpt</strong> ·
  DCF engine: 2-stage Gordon Growth Model<br>
  <em>For educational purposes only — not financial advice.</em>
</div>
""", unsafe_allow_html=True)

