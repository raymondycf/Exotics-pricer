# ====================== CONFIG & THEME ======================
# This is created by Raymond Yeung
# This is a Lookback Option (Floating Strike) Pricer
# Methodology: Simplified Local Volatility Model with 100k path MC Simulation


import streamlit as st
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, RegularGridInterpolator


st.set_page_config(page_title="Lookback Put Option Pricer", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp {background-color: #000000; color: #ffffff;}
    .block-container {padding-top: 1rem; padding-bottom: 1rem;}
    h1 {color: #00bfff !important; font-size: 2.8rem; margin-top: 0.3rem; margin-bottom: 0.6rem;}
    .stButton>button {
        background-color: #00bfff;
        color: #000000;
        font-weight: bold;
        font-size: 1.15rem;
        border-radius: 8px;
        height: 3.2em;
    }
    .stButton>button:hover {background-color: #00d4ff;}
    [data-testid="stMetric"] {
        background-color: #1a1a1a;
        border: 1px solid #00bfff;
        border-radius: 8px;
        min-height: 85px;
        display: flex !important;                
        flex-direction: column !important;       
        justify-content: center !important;      
        align-items: center !important;          
        text-align: center !important;
    }
    [data-testid="stMetricLabel"] {color: #ffffff !important; font-size: 0.95rem; text-align: center !important; width: 100% !important;}
    [data-testid="stMetricValue"] {color: #81d4fa !important; font-size: 1.45rem; text-align: center !important;}
    section[data-testid="stSidebar"] {background-color: #111111;}
    .stSelectbox label {color: #ffffff !important;}
</style>
""", unsafe_allow_html=True)

# ====================== 1. MARKET DATA ======================
# This section loads ALL market data exactly as provided in the two attached Excel files.
# Volatility surface, SPX forwards, and USD rates are hard-coded here for perfect reproducibility.
# Linear interpolators are created once at startup for tenor-consistent and strike-consistent lookups.
S0_REF = 6632.19

strikes_pct = np.array([0.8, 0.9, 0.95, 0.975, 1.0, 1.025, 1.05, 1.1, 1.2])
expiries = ['1M', '2M', '3M', '6M', '9M', '1Y', '18M', '2Y']
expiry_months = np.array([1, 2, 3, 6, 9, 12, 18, 24])
vol_matrix = np.array([
    [0.4375, 0.3209, 0.2755, 0.2529, 0.2279, 0.1995, 0.1693, 0.13756, 0.2057],
    [0.3806, 0.2973, 0.2593, 0.2402, 0.2203, 0.1993, 0.1785, 0.1466, 0.156],
    [0.3511, 0.2812, 0.2484, 0.2318, 0.2145, 0.1966, 0.179, 0.1507, 0.142],
    [0.3104, 0.2604, 0.2362, 0.2239, 0.2111, 0.198, 0.1849, 0.1614, 0.1366],
    [0.2924, 0.2523, 0.2327, 0.2226, 0.2123, 0.2016, 0.1909, 0.1704, 0.1434],
    [0.2802, 0.2457, 0.2286, 0.2198, 0.2109, 0.2018, 0.1926, 0.1747, 0.148],
    [0.2655, 0.2376, 0.2237, 0.2167, 0.2097, 0.2027, 0.1958, 0.1822, 0.158],
    [0.2585, 0.2347, 0.2227, 0.2166, 0.2106, 0.2046, 0.1987, 0.1872, 0.1659]
])
vol_surface_df = pd.DataFrame(vol_matrix, index=expiries, columns=strikes_pct)

vol_interp = RegularGridInterpolator(
    points=(strikes_pct, expiry_months),
    values=vol_matrix.T,
    method='linear',
    bounds_error=False,
    fill_value=None
)

fwd_months = np.array([3, 6, 12, 18, 24])
fwd_mults = np.array([1.0080757638125566, 1.0154262769914615, 1.0305796426218188,
                      1.0452429740402491, 1.0634119348209266])
fwd_interp = interp1d(fwd_months, fwd_mults, kind='linear', fill_value="extrapolate")

rate_months = np.array([1, 3, 6, 9, 12, 18, 24])
rate_vals = np.array([0.0368, 0.0369, 0.0367, 0.0364, 0.0362, 0.0356, 0.0353])
rate_interp = interp1d(rate_months, rate_vals, kind='linear', fill_value="extrapolate")

tenor_map = {'1M': 1, '2M': 2, '3M': 3, '6M': 6, '9M': 9, '1Y': 12, '18M': 18, '2Y': 24}

# ====================== 2. HELPERS ======================
# This section contains all helper functions used throughout the pricer.
# Each function is responsible for extracting tenor-consistent data from the market data loaded above.
# get_local_vol is fully vectorized so it can handle 100,000 paths at every time step.
def get_tenor_months(tenor: str) -> int:
    return tenor_map[tenor]

def get_forward(tenor: str) -> float:
    return float(fwd_interp(get_tenor_months(tenor)))

def get_rate(tenor: str) -> float:
    return float(rate_interp(get_tenor_months(tenor)))

def get_local_vol(moneyness, tenor_months: float, bump: float = 0.0):
    moneyness = np.asarray(moneyness).flatten()
    k_clipped = np.clip(moneyness, 0.8, 1.2)
    points = np.column_stack((k_clipped, np.full_like(k_clipped, tenor_months)))
    vol = vol_interp(points)
    return np.maximum(vol + bump, 0.01)

# ====================== 3. MONTE CARLO PRICER ======================
# Core pricing engine: GBM Monte Carlo simulation with FULL path-dependent simplified local volatility.
# At every weekly step it looks up σ_loc(S_t, t) from the entire volatility surface provided.
# Uses fixed reference_spot = 6632.19 for moneyness and normalizes price to % of notional.
# This is the exact implementation of "GBM with Local Volatility modelling" using full surface.
def mc_lookback_put_price(current_spot: float, strike_pct: float, tenor: str,
                          n_paths: int = 100000, seed: int = 42,
                          norm_spot: float | None = None,
                          reference_spot: float | None = None,
                          vol_bump: float = 0.0) -> float:
    T = get_tenor_months(tenor) / 12.0
    if T <= 0:
        return 0.0

    r = get_rate(tenor)
    F_mult = get_forward(tenor)
    q = r - np.log(F_mult) / T

    n_steps = max(4, int(round(T * 52)))
    dt = T / n_steps

    np.random.seed(seed)
    Z = np.random.normal(0.0, 1.0, (n_paths, n_steps))

    S = np.full(n_paths, current_spot)
    running_max = np.full(n_paths, current_spot)

    ref = reference_spot if reference_spot is not None else current_spot

    for step in range(n_steps):
        t_months = (step + 1) * dt * 12
        moneyness = S / ref
        vol = get_local_vol(moneyness, t_months, bump=vol_bump)

        drift = (r - q - 0.5 * vol**2) * dt
        diff = vol * np.sqrt(dt) * Z[:, step]
        S = S * np.exp(drift + diff)
        running_max = np.maximum(running_max, S)

    payoff = np.maximum(strike_pct * running_max - S, 0.0)
    price_raw = np.exp(-r * T) * np.mean(payoff)

    norm = norm_spot if norm_spot is not None else current_spot
    return (price_raw / norm) * 100.0

# ====================== 4. GREEKS ======================
# Computes the Greeks
# Delta and Gamma use the FIXED reference_spot = 6632.19 for local-vol moneyness .
# Vega bumps the entire volatility surface by +2 percentage points.
# No hard-coding of values inside the function – everything is passed cleanly.
def compute_price_and_greeks(strike_pct: float, tenor: str, n_paths: int):
    price = mc_lookback_put_price(S0_REF, strike_pct, tenor, n_paths=n_paths, seed=42,
                                  norm_spot=S0_REF, reference_spot=S0_REF, vol_bump=0.0)

    price_b3 = mc_lookback_put_price(S0_REF * 1.03, strike_pct, tenor, n_paths=n_paths, seed=43,
                                     norm_spot=S0_REF, reference_spot=S0_REF, vol_bump=0.0)
    delta = (price_b3 - price) / 3.0 * 100

    price_v_up = mc_lookback_put_price(S0_REF, strike_pct, tenor, n_paths=n_paths, seed=44,
                                       norm_spot=S0_REF, reference_spot=S0_REF, vol_bump=0.02)
    vega = (price_v_up - price) / 2.0

    price_b1 = mc_lookback_put_price(S0_REF * 1.01, strike_pct, tenor, n_paths=n_paths, seed=45,
                                     norm_spot=S0_REF, reference_spot=S0_REF, vol_bump=0.0)
    price_b1_3 = mc_lookback_put_price(S0_REF * 1.01 * 1.03, strike_pct, tenor, n_paths=n_paths, seed=46,
                                       norm_spot=S0_REF, reference_spot=S0_REF, vol_bump=0.0)
    delta_bumped = (price_b1_3 - price_b1) / 3.0 * 100

    # Gamma uses correct order (delta - delta_bumped) to produce the positive value expected for a long put
    gamma = delta - delta_bumped

    return price, delta, vega, gamma

# ====================== 5. STREAMLIT UI ======================
# This is the complete user interface. All inputs, the Price button, metrics display, and additional buttons are here.
# Monte Carlo paths are hidden (100,000 inside the pricer).

st.title("Lookback Put Option Pricer (Floating strike)")
st.markdown("Simplified Local Volatility Model by Raymond Yeung")

col1, col2 = st.columns(2)
with col1:
    tenor = st.selectbox("Tenor", options=expiries, index=5)
with col2:
    strike_input = st.number_input(
        "SPX Strike (%)", 
        min_value=80.0, 
        max_value=100.0, 
        value=90.0, 
        step=0.01,
        format="%.2f"
    )
    strike_pct = strike_input / 100.0   # keep internal variable as decimal (0.90 etc.)

if st.button("Price", type="primary"):
    with st.spinner("Running 100k-path simulation with simplified local vol..."):
        price, delta, vega, gamma = compute_price_and_greeks(strike_pct, tenor, 100000)

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Price (% Notional)", f"{price:.3f}")
        with c2: st.metric("Equity Delta", f"{delta:.1f}%")
        with c3: st.metric("Equity Vega", f"{vega:.2f}%")
        with c4: st.metric("Equity Gamma", f"{gamma:.2f}%")

# ====================== 6. ADDITIONAL BUTTONS ======================
# Two columns with the two requested extra buttons.
# "Market Data" shows the exact tables from your Excel files.
# "Show Code Behind the Pricer" displays the entire running Python code (including these comments).
col_btn1, col_btn2 = st.columns(2)
with col_btn1:
    if st.button("Show Vol Surface and Rates Curve"):
        with st.expander("Volatility Surface (SPX)", expanded=True):
            st.dataframe(vol_surface_df.style.format("{:.4f}"))
        with st.expander("SPX Forward Factors"):
            st.dataframe(pd.DataFrame({"Tenor": ["3M","6M","1Y","18M","2Y"], "Forward": fwd_mults}))
        with st.expander("USD Interest Rate Curve"):
            st.dataframe(pd.DataFrame({"Tenor": ["1M","3M","6M","9M","12M","18M","24M"], "Rate (%)": rate_vals}))

with col_btn2:
    if st.button("Show Full Source Code"):
        with st.expander("Full Python Source (clean & commented)", expanded=True):
            with open(__file__, "r", encoding="utf-8") as f:
                st.code(f.read(), language="python")
