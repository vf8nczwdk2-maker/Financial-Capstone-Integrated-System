# AI Assistance Disclosure

## Project: Financial Capstone — Market Screener · DCF Valuation · Portfolio Optimizer

---

## Statement of Use

This project was developed using **Antigravity**, an AI coding assistant created by Google DeepMind, as a collaborative tool for code generation and user interface implementation.

The use of AI assistance is disclosed here in the interest of full academic and professional transparency.

---

## What the AI Did

Antigravity was used to assist with the following **technical implementation tasks**:

- **Code generation**: Writing Python functions for data fetching (`yfinance`), the DCF calculation loop, the optimizer wrapper (`PyPortfolioOpt`), the Ripple Effect sensitivity sweep, and the Streamlit layout code
- **UI/UX layout**: Designing the dark Bloomberg-terminal CSS theme, the three-tab architecture, the metric card grid, the pipeline step banner, and the Funnel Summary hero section
- **Debugging**: Identifying and fixing technical bugs including the f-string HTML rendering issue in Streamlit, the PyPortfolioOpt infeasibility error with 2-stock universes, and the caching architecture for live slider responsiveness
- **Boilerplate**: Writing Plotly chart configuration code (axis labels, color palettes, hover templates, layout parameters)

---

## What the Student Did

The following **intellectual and financial contributions** were made entirely by the student:

- **Financial logic and model design**: Selecting the 2-stage Gordon Growth DCF as the appropriate valuation model; defining the formula structure (Stage 1 FCF projection + Stage 2 terminal value); choosing which yfinance fields (Free Cash Flow, Shares Outstanding, Net Cash) map to each model input
- **Integration strategy**: Designing the three-component pipeline architecture — the idea that the DCF output (Intrinsic Value) should act as a hard filter for the portfolio optimizer's input universe, and that changing WACC should cascade through all three stages automatically
- **The "Ripple Effect" concept**: Conceptualizing and specifying the WACC sensitivity cascade dashboard — showing how a single assumption change propagates from Intrinsic Value → Undervalued Count → Portfolio Sharpe Ratio across a sweep of scenarios
- **Mathematical verification**: Manually verifying DCF outputs against known intrinsic value estimates for reference stocks (SBUX) from prior work; confirming that higher WACC monotonically decreases intrinsic value; validating Margin of Safety filter behavior
- **Parameter selection**: Choosing default values for WACC (7%), FCF Growth Rate (8%), Terminal Growth Rate (2.5%), and Risk-Free Rate (4.5%) based on current macroeconomic conditions and financial theory
- **Prompt engineering and direction**: Writing all prompts that directed the AI's work, specifying exactly what each component should do, what data it should fetch, how components should connect, and what edge cases to handle (concentration risk, 0-stock filter, 1-stock edge case)
- **Quality control**: Reviewing all generated code for correctness, identifying bugs, requesting fixes, and approving the final implementation

---

## Summary

| Contribution | Student | AI (Antigravity) |
|---|:---:|:---:|
| Financial model design (DCF formula) | ✓ | |
| Pipeline integration strategy | ✓ | |
| Ripple Effect concept | ✓ | |
| Mathematical verification | ✓ | |
| Parameter selection & defaults | ✓ | |
| Prompt engineering & direction | ✓ | |
| Python code generation | | ✓ |
| Streamlit UI layout & CSS | | ✓ |
| Plotly chart configuration | | ✓ |
| Bug diagnosis & fixing | Collaborative | Collaborative |

---

## Tool Information

| Field | Detail |
|---|---|
| **Tool Name** | Antigravity |
| **Developer** | Google DeepMind |
| **Version / Model** | Claude Sonnet 4.6 (Thinking) via Antigravity interface |
| **Date of Use** | April 2026 |
| **Interface** | VS Code Extension (Antigravity) |

---

*This disclosure was prepared in accordance with academic integrity guidelines regarding the use of AI-assisted tools in coursework and project development.*
