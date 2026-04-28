# Financial Capstone — DRIVER Framework

> **D · R · I · V · E · R** is the structured methodology used to design, build, and reflect on this unified financial analysis pipeline.

---

## D — Discover

**What problem are we solving?**

Most retail investors treat stock screening, valuation, and portfolio construction as three separate, disconnected activities. A stock might look cheap on a P/E basis but have no free cash flow to support that price. A stock might be genuinely undervalued but still make a portfolio worse by introducing correlated risk. The core insight of this project is that **these three steps are a single pipeline, not three separate tools**.

The research question: *Can we build a system where one assumption — the discount rate (WACC) — automatically cascades through screening, valuation, and portfolio allocation so that every layer of the decision is consistent?*

The answer is the **Ripple Effect**: changing the WACC slider by 1% changes which stocks are undervalued, which changes the optimizer's input universe, which changes the final portfolio weights and Sharpe Ratio — all in real time.

---

## R — Represent

**How is the data structured and how does it flow?**

### Architecture: Three-Stage Data Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                        SIDEBAR (Global Controls)                │
│  Tickers · WACC · FCF Growth · Terminal Growth · Margin of      │
│  Safety · Risk-Free Rate · Max Weight · History Window          │
└─────────────────────┬───────────────────────────────────────────┘
                      │  (all sliders trigger a full re-render)
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│              STAGE 1 — Market Screener (Tab 1)                  │
│                                                                 │
│  Input:  10–20 ticker symbols                                   │
│  Fetch:  yfinance .info → P/E, Debt/Equity, Market Cap,         │
│          Beta, Dividend Yield, Current Price                    │
│  Output: Fundamentals table · P/E vs D/E scatter chart          │
│  Store:  st.session_state["screener_df"]                        │
│                                                                 │
│  Cache:  fetch_screener_data(tickers) — TTL 1 hour              │
└─────────────────────┬───────────────────────────────────────────┘
                      │  ticker list + current price
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│              STAGE 2 — DCF Valuation (Tab 2)                    │
│                                                                 │
│  Input:  Ticker list + WACC + FCF Growth Rate +                 │
│          Terminal Growth Rate + Margin of Safety                │
│                                                                 │
│  Fetch:  yfinance .info → Free Cash Flow, Shares Out,           │
│          Total Cash, Total Debt (cached, TTL 1 hour)            │
│                                                                 │
│  Model:  2-Stage Gordon Growth DCF                             │
│  ┌─────────────────────────────────────────────────────┐       │
│  │  Stage 1: PV = Σ [FCF × (1+g)^t / (1+WACC)^t]     │       │
│  │           for t = 1 … 5                             │       │
│  │  Stage 2: Terminal Value = FCF₆ / (WACC − g_term)  │       │
│  │           PV(TV) = TV / (1+WACC)^5                 │       │
│  │  IV/share = (PV₁ + PV₂ + Net Cash) / Shares        │       │
│  └─────────────────────────────────────────────────────┘       │
│                                                                 │
│  Filter:  UNDERVALUED if Price < IV × (1 − Margin of Safety)   │
│                                                                 │
│  Output:  Price vs IV chart · Upside/Downside bar chart         │
│           WACC sensitivity heatmap · Valuation table            │
│  Store:  st.session_state["undervalued"] (filtered list)        │
│          st.session_state["dcf_df"]      (full results)         │
│                                                                 │
│  ⚡ Runs on every render — sliders feel instantaneous           │
└─────────────────────┬───────────────────────────────────────────┘
                      │  undervalued ticker list only
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│            STAGE 3 — Portfolio Optimizer (Tab 3)                │
│                                                                 │
│  Input:  ONLY tickers where Price < IV × (1 − MoS)             │
│          (reads from st.session_state["undervalued"])           │
│                                                                 │
│  Fetch:  yfinance daily price history (1–5yr window, cached)    │
│                                                                 │
│  Model:  Mean-Variance Optimization via PyPortfolioOpt          │
│  ┌─────────────────────────────────────────────────────┐       │
│  │  μ     = Mean Historical Return (annualized)        │       │
│  │  Σ     = Sample Covariance Matrix                   │       │
│  │  Solve: max (μ·w − rf) / √(wᵀΣw)   [Max Sharpe]   │       │
│  │  s.t.  Σwᵢ = 1,  0 ≤ wᵢ ≤ max_weight              │       │
│  └─────────────────────────────────────────────────────┘       │
│                                                                 │
│  Output:  Weights bar + donut charts · Performance KPIs         │
│           Position detail table · Ripple Effect Dashboard       │
│  Store:   st.session_state["n_active_positions"] → Funnel       │
└─────────────────────────────────────────────────────────────────┘

                        ▲  always visible  ▲
┌─────────────────────────────────────────────────────────────────┐
│             FUNNEL SUMMARY (Hero Banner — every tab)            │
│                                                                 │
│  [Tickers Screened] → DCF Filter → [Passed Valuation]          │
│                    → Max Sharpe  → [Final Portfolio]            │
│                                                                 │
│  All three numbers update live on every slider change           │
└─────────────────────────────────────────────────────────────────┘
```

### Caching Strategy

| Function | Cache Key | TTL | Why |
|---|---|---|---|
| `fetch_screener_data(tickers)` | ticker tuple | 1 hour | Slow API; fundamentals don't change in minutes |
| `fetch_dcf_inputs(tickers)` | ticker tuple | 1 hour | FCF/balance sheet data is stable intraday |
| `fetch_prices(tickers, years)` | ticker tuple + years | 1 hour | Full price history; expensive to re-download |
| DCF calculation loop | *(not cached)* | instant | Pure math — must be live so sliders respond immediately |
| Ripple Effect loop | *(not cached)* | seconds | Sweeps 8 WACC scenarios; re-runs on every render |

---

## I — Implement

**What was built?**

A single-file Streamlit application (`app.py`, ~1,460 lines) with three integrated components:

### Component 1 — Market Screener (Tab 1)
- Accepts 10–20 comma-separated ticker symbols via sidebar text area
- Fetches P/E Ratio, Debt/Equity, Market Cap, Beta, Dividend Yield, and Current Price via `yfinance`
- Displays a color-coded fundamentals table (green = cheap, amber = fair, red = expensive)
- Renders a P/E vs. Debt/Equity scatter map with universe median quadrant lines
- Outputs: `st.session_state["screener_df"]`

### Component 2 — DCF Valuation (Tab 2)
- Runs the 2-stage DCF model for every ticker (live, uncached) using sidebar sliders
- Applies Margin of Safety filter: `UNDERVALUED` if `Price < IV × (1 − MoS)`
- Displays:
  - Valuation summary KPI cards (# undervalued, # overvalued, avg upside)
  - **Price vs. Intrinsic Value** grouped bar chart (grey = market price, green/red = DCF value)
  - Upside/Downside horizontal bar chart (sorted by largest upside)
  - WACC sensitivity heatmap (shows how IV changes across WACC 6–14%, highlightes current WACC)
  - Concentration risk warning when < 5 stocks pass the filter
- Outputs: `st.session_state["undervalued"]`, `st.session_state["dcf_df"]`

### Component 3 — Portfolio Optimizer (Tab 3)
- **Only receives tickers from `session_state["undervalued"]`** — the valuation filter is hard-enforced
- Fetches multi-year daily price history for the filtered subset
- Runs Max Sharpe Ratio optimization (automatic fallback to Min Volatility if infeasible)
- For universes ≤ 2 stocks, weight bounds are relaxed to (0, 100%) so the optimizer can always run
- Displays:
  - Performance summary (Return, Volatility, Sharpe, Active Positions)
  - Side-by-side weight bar chart + donut chart
  - Position detail table combining DCF context with optimizer weights
  - **Ripple Effect Dashboard**: 3-panel chart + scenario table sweeping WACC 6–14% and recording Undervalued Count, Portfolio Return, and Sharpe Ratio at each scenario
- Outputs: `st.session_state["n_active_positions"]` → feeds back into Funnel Summary

---

## V — Validate

**How do we know the model is correct?**

| Validation Check | Method |
|---|---|
| Syntax correctness | `python3 -c "import ast; ast.parse(open('app.py').read())"` |
| DCF monotonicity | Higher WACC → lower intrinsic value at every ticker (verified via sensitivity heatmap) |
| Filter integrity | `session_state["undervalued"]` only contains tickers where `Price < IV × (1 − MoS)` — cross-checked against valuation table |
| Margin of Safety | Setting MoS = 20% should reduce the undervalued count vs MoS = 0% — verified manually by toggling slider |
| Optimizer feasibility | 2-stock universes run successfully with weight bounds relaxed to (0, 1.0) |
| Ripple Effect | Increasing WACC in the scenario table monotonically reduces undervalued count — confirmed in Ripple table |
| Edge cases | 0 undervalued: friendly error with actionable advice; 1 stock: error; 2–4 stocks: runs + concentration warning; ≥5: normal |

---

## E — Evolve

**What would a production version look like?**

1. **Real WACC per Company** — compute WACC from scratch using CAPM (cost of equity) + interest expense / total debt (cost of debt), weighted by capital structure, rather than using a universal slider
2. **Forward FCF Estimates** — replace trailing FCF with analyst consensus forward estimates via a financial data API (e.g., Intrinio, Polygon.io, Alpha Vantage)
3. **3-Stage DCF** — high growth phase (y1–3), transition phase (y4–7), terminal value — more realistic for growth companies
4. **Monte Carlo Valuation** — sample WACC and growth from distributions (e.g., Normal ± 2σ) and output a probability distribution of intrinsic values, not a point estimate
5. **Backtesting Engine** — run the full pipeline historically (e.g., monthly rebalance) and report historical alpha vs. S&P 500
6. **Sector-Relative Benchmarking** — compare each stock's P/E and D/E to sector peers rather than the full universe median

---

## R — Reflect

**What does this project demonstrate?**

### Technical Lessons

- **Caching architecture matters**: `@st.cache_data` keyed on the ticker tuple makes slow yfinance calls feel instantaneous on slider changes, while the DCF loop runs uncached so WACC sensitivity feels live
- **Session state as a pipeline bus**: `st.session_state` acts as a data bus connecting the three components — the optimizer reads from `undervalued`, writes back `n_active_positions` to the funnel
- **f-string + HTML pitfall**: Python f-strings with CSS `color:{variable}` syntax inside triple-quoted strings can be misinterpreted by Streamlit's Markdown renderer as code fences — use `.format()` for HTML templates that contain curly braces
- **PyPortfolioOpt infeasibility**: Max Sharpe becomes infeasible when the weight cap is tighter than `1/n` for a 2-stock universe — always relax bounds for small universes and wrap in try/except with a min-vol fallback

### Financial Lessons

- **WACC is the most sensitive assumption in any DCF.** Moving from 7% to 10% can shift a stock from 40% undervalued to 30% overvalued. The sensitivity heatmap and Ripple Effect dashboard make this viscerally visible
- **A strict valuation filter is not free.** Requiring `Price < IV × (1 − MoS)` means fewer stocks qualify, which means higher concentration risk in the optimizer. This trade-off is the central tension the Ripple Effect Dashboard is designed to surface
- **Value investing + Mean-Variance optimization is a coherent strategy.** By pre-filtering on fundamentals (DCF) and then letting the math optimize on risk (Markowitz), the system implements a type of "quality-at-a-reasonable-price" (QARP) portfolio construction

---

*Built with Python 3.12 · Streamlit · yfinance · PyPortfolioOpt · Plotly*
*For educational purposes only — not financial advice.*
