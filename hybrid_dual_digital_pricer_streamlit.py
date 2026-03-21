import streamlit as st
import numpy as np
import pandas as pd

# ====================== EXACT DATA FROM YOUR EXCEL FILES ======================
SPX_SPOT = 6632.19
FX_SPOT = 1.1487
N_PATHS = 100000
tenors_months = {'1M': 1, '2M': 2, '3M': 3, '6M': 6, '9M': 9, '1Y': 12, '18M': 18, '2Y': 24}
spx_fwd_dict = {
    '3M': 1.0080757638125566, '6M': 1.0154262769914615,
    '1Y': 1.0305796426218188, '18M': 1.0452429740402491, '2Y': 1.0634119348209266
}
usd_dict = {'1M': 0.0368, '3M': 0.0369, '6M': 0.0367, '9M': 0.0364,
            '12M': 0.0362, '18M': 0.0356, '24M': 0.0353}
eur_dict = {'1M': 0.01944, '3M': 0.0198, '6M': 0.0201, '9M': 0.0218,
            '12M': 0.0226, '18M': 0.0233, '24M': 0.0237}
spx_strikes = np.array([0.8, 0.9, 0.95, 0.975, 1.0, 1.025, 1.05, 1.1, 1.2])
fx_strikes = np.array([0.92, 0.95, 1.0, 1.05, 1.08])
spx_vol_dict = {
    '1M': [0.4375, 0.3209, 0.2755, 0.2529, 0.2279, 0.1995, 0.1693, 0.13756, 0.2057],
    '2M': [0.3806, 0.2973, 0.2593, 0.2402, 0.2203, 0.1993, 0.1785, 0.1466, 0.156],
    '3M': [0.3511, 0.2812, 0.2484, 0.2318, 0.2145, 0.1966, 0.179, 0.1507, 0.142],
    '6M': [0.3104, 0.2604, 0.2362, 0.2239, 0.2111, 0.198, 0.1849, 0.1614, 0.1366],
    '9M': [0.2924, 0.2523, 0.2327, 0.2226, 0.2123, 0.2016, 0.1909, 0.1704, 0.1434],
    '1Y': [0.2802, 0.2457, 0.2286, 0.2198, 0.2109, 0.2018, 0.1926, 0.1747, 0.148],
    '18M': [0.2655, 0.2376, 0.2237, 0.2167, 0.2097, 0.2027, 0.1958, 0.1822, 0.158],
    '2Y': [0.2585, 0.2347, 0.2227, 0.2166, 0.2106, 0.2046, 0.1987, 0.1872, 0.1659]
}
fx_vol_dict = {
    '1M': [0.1119, 0.0962, 0.0855, 0.0817, 0.0839],
    '2M': [0.1057, 0.0913, 0.0819, 0.0794, 0.0835],
    '3M': [0.1009, 0.0874, 0.079, 0.0776, 0.0826],
    '6M': [0.0943, 0.0823, 0.0763, 0.0777, 0.0857],
    '9M': [0.0917, 0.0804, 0.0759, 0.07789, 0.0888],
    '1Y': [0.0896, 0.0788, 0.075, 0.0791, 0.0902],
    '18M': [0.0923, 0.0812, 0.0773, 0.0813, 0.0927],
    '2Y': [0.0935, 0.0823, 0.078, 0.0824, 0.094]
}


# ====================== HELPERS ======================
def get_months(k: str) -> int:
    if k.endswith('M'): return int(k[:-1])
    if k.endswith('Y'): return int(k[:-1]) * 12
    return int(k)


def get_spx_fwd_factor(tenor: str) -> float:
    T_m = tenors_months[tenor]
    keys = list(spx_fwd_dict.keys())
    xs = np.array([get_months(k) for k in keys])
    ys = np.array(list(spx_fwd_dict.values()))
    return np.interp(T_m, xs, ys)


def get_rate(tenor: str, rate_dict: dict) -> float:
    T_m = tenors_months[tenor]
    keys = list(rate_dict.keys())
    xs = np.array([get_months(k) for k in keys])
    ys = np.array(list(rate_dict.values()))
    return np.interp(T_m, xs, ys)


def get_vol(tenor: str, strike_decimal: float, is_spx: bool = True) -> float:
    strikes = spx_strikes if is_spx else fx_strikes
    vols = spx_vol_dict[tenor] if is_spx else fx_vol_dict[tenor]
    return np.interp(strike_decimal, strikes, vols)


# ====================== MC PRICER ======================
def mc_price(spx_s0, fx_s0, spx_k, fx_k, spx_dir, fx_dir, rho, T, usd_r,
             spx_fwd, fx_fwd, spx_vol, fx_vol, z1, z_indep):
    spx_sigT = spx_vol * np.sqrt(T)
    fx_sigT = fx_vol * np.sqrt(T)
    spx_drift = np.log(spx_fwd / spx_s0) - 0.5 * spx_vol ** 2 * T
    fx_drift = np.log(fx_fwd / fx_s0) - 0.5 * fx_vol ** 2 * T
    z2 = rho * z1 + np.sqrt(1 - rho ** 2) * z_indep
    spx_log = np.log(spx_s0) + spx_drift + spx_sigT * z1
    fx_log = np.log(fx_s0) + fx_drift + fx_sigT * z2
    spx_t = np.exp(spx_log)
    fx_t = np.exp(fx_log)
    spx_hit = (spx_t < spx_k) if spx_dir == '<' else (spx_t > spx_k)
    fx_hit = (fx_t < fx_k) if fx_dir == '<' else (fx_t > fx_k)
    prob = np.mean(spx_hit & fx_hit)
    df = np.exp(-usd_r * T)
    return float(prob * df * 100)


# ====================== STREAMLIT UI ======================
st.set_page_config(page_title="Hybrid Dual Digital Pricer", layout="wide")
# Dark theme + centered metrics + no top blank space + MORE COMPACT METRIC BOXES
st.markdown("""
<style>
    .stApp {
        background-color: #000000;
        color: #ffffff;
        padding-top: 0rem !important;
    }
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }
    h1 {
        color: #00bfff !important;
        font-size: 2.8rem;
        margin-top: 0.2rem !important;
        margin-bottom: 0.5rem !important;
    }
    h2, h3 {
        color: #00bfff;
        text-align: center;
    }
    .stButton>button {
        background-color: #00bfff;
        color: #000000;
        font-weight: bold;
        font-size: 1.15rem;
        border-radius: 10px;
        height: 3.2em;
    }
    .stButton>button:hover {
        background-color: #00d4ff;
    }
    [data-testid="stMetric"] {
        background-color: #1a1a1a;
        border: 1px solid #00bfff;
        border-radius: 10px;
        text-align: center !important;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        min-height: 90px;          /* <<< MADE MORE COMPACT */
        padding: 8px 0 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #ffffff !important;
        font-size: 0.95rem;        /* <<< SMALLER FOR COMPACTNESS */
        text-align: center;
    }
    [data-testid="stMetricValue"] {
        color: #81d4fa !important;
        font-size: 1.45rem;        /* <<< SMALLER FOR COMPACTNESS */
        text-align: center;
    }
    section[data-testid="stSidebar"] {
        background-color: #111111;
    }
    .stSelectbox label, .stNumberInput label {
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)
st.title("Hybrid Dual Digital Pricer")
st.markdown("**Multivariate GBM with constant correlation Copula** \n*by Raymond Yeung*")
with st.sidebar:
    st.header("Inputs")
    tenor = st.selectbox("Tenor", options=list(tenors_months.keys()), index=3)
    spx_strike_pct = st.number_input("SPX Digital Strike (%)", 80.0, 120.0, 90.0, 0.1)
    spx_dir = st.selectbox("SPX Direction", ["<", ">"])
    fx_strike_pct = st.number_input("EUR/USD Digital Strike (%)", 92.0, 108.0, 95.0, 0.1)
    fx_dir = st.selectbox("EUR/USD Direction", ["<", ">"])
    corr = st.number_input("EQ/FX Correlation (%)", 0.0, 100.0, 13.0, 0.1)
# Centered main button
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    if st.button("Price Option", type="primary", use_container_width=True):
        with st.spinner("Running 100,000 paths..."):
            T = tenors_months[tenor] / 12.0
            usd_r = get_rate(tenor, usd_dict)
            eur_r = get_rate(tenor, eur_dict)
            spx_fwd_factor = get_spx_fwd_factor(tenor)
            spx_fwd = SPX_SPOT * spx_fwd_factor
            fx_fwd = FX_SPOT * np.exp((usd_r - eur_r) * T)
            spx_k = SPX_SPOT * (spx_strike_pct / 100)
            fx_k = FX_SPOT * (fx_strike_pct / 100)
            spx_vol = get_vol(tenor, spx_strike_pct / 100, True)
            fx_vol = get_vol(tenor, fx_strike_pct / 100, False)
            np.random.seed(42)
            z1 = np.random.randn(N_PATHS)
            z_indep = np.random.randn(N_PATHS)
            rho = corr / 100.0
            base = mc_price(SPX_SPOT, FX_SPOT, spx_k, fx_k, spx_dir, fx_dir, rho, T, usd_r,
                            spx_fwd, fx_fwd, spx_vol, fx_vol, z1, z_indep)
            # Greeks
            p_eq = mc_price(SPX_SPOT * 1.05, FX_SPOT, spx_k, fx_k, spx_dir, fx_dir, rho, T, usd_r,
                            spx_fwd * 1.05, fx_fwd, spx_vol, fx_vol, z1, z_indep)
            eq_delta = ((p_eq - base) / 0.05) if base > 0 else 0
            p_fx = mc_price(SPX_SPOT, FX_SPOT * 1.02, spx_k, fx_k, spx_dir, fx_dir, rho, T, usd_r,
                            spx_fwd, fx_fwd * 1.02, spx_vol, fx_vol, z1, z_indep)
            fx_delta = ((p_fx - base) / 0.02) if base > 0 else 0
            p_eqv = mc_price(SPX_SPOT, FX_SPOT, spx_k, fx_k, spx_dir, fx_dir, rho, T, usd_r,
                             spx_fwd, fx_fwd, spx_vol + 0.02, fx_vol, z1, z_indep)
            eq_vega = ((p_eqv - base) / 2) if base > 0 else 0
            p_fxv = mc_price(SPX_SPOT, FX_SPOT, spx_k, fx_k, spx_dir, fx_dir, rho, T, usd_r,
                             spx_fwd, fx_fwd, spx_vol, fx_vol + 0.01, z1, z_indep)
            fx_vega = (p_fxv - base) if base > 0 else 0
            p_corr = mc_price(SPX_SPOT, FX_SPOT, spx_k, fx_k, spx_dir, fx_dir, (corr + 5) / 100, T, usd_r,
                              spx_fwd, fx_fwd, spx_vol, fx_vol, z1, z_indep)
            corr_delta = ((p_corr - base) / 5) if base > 0 else 0
            st.session_state.results = {
                'price': base,
                'eq_delta': eq_delta,
                'fx_delta': fx_delta,
                'eq_vega': eq_vega,
                'fx_vega': fx_vega,
                'corr_delta': corr_delta,
                'spx_spot': SPX_SPOT,
                'fx_spot': FX_SPOT,
                'spx_vol': spx_vol * 100,
                'fx_vol': fx_vol * 100
            }
# Display results + market data used (PRICING RESULTS NOW ABOVE MARKET DATA USED)
if 'results' in st.session_state:
    r = st.session_state.results

    # === PRICING RESULTS (moved to top) ===
    st.markdown("### Pricing Results")
    # 2 rows × 3 columns
    res_row1 = st.columns(3)
    res_row2 = st.columns(3)
    res_row1[0].metric("Price (% Notional)", f"{r['price']:.2f}%")
    res_row1[1].metric("EQ Delta", f"{r['eq_delta']:.2f}%")
    res_row1[2].metric("FX Delta", f"{r['fx_delta']:.2f}%")
    res_row2[0].metric("EQ Vega", f"{r['eq_vega']:.2f}%")
    res_row2[1].metric("FX Vega", f"{r['fx_vega']:.2f}%")
    res_row2[2].metric("Corr Delta", f"{r['corr_delta']:.2f}%")

    # === MARKET DATA USED (now below) ===
    st.markdown("### Market Data Used")
    # 2 rows × 2 columns
    row1 = st.columns(2)
    row2 = st.columns(2)
    row1[0].metric("SPX Spot", f"{r['spx_spot']:,.2f}")
    row1[1].metric("EUR/USD Spot", f"{r['fx_spot']:.4f}")
    row2[0].metric("SPX Implied Vol", f"{r['spx_vol']:.2f}%")
    row2[1].metric("EUR/USD Implied Vol", f"{r['fx_vol']:.2f}%")

# Updated button name + shows Vol Surface & Rates
if st.button("Show Vol Surface, Rates curve and Forward", use_container_width=True):
    st.subheader("SPX Forward Factors")
    st.dataframe(pd.DataFrame(list(spx_fwd_dict.items()), columns=["Tenor", "Forward Factor"]))
    st.subheader("Interest Rates")
    st.dataframe(
        pd.DataFrame({"Tenor": list(usd_dict.keys()), "USD": list(usd_dict.values()), "EUR": list(eur_dict.values())}))
    st.subheader("SPX Vol Surface (%)")
    df_spx = pd.DataFrame(spx_vol_dict, index=[f"{s * 100:.1f}%" for s in spx_strikes]).T * 100
    st.dataframe(df_spx.style.format("{:.2f}"))
    st.subheader("EUR/USD Vol Surface (%)")
    df_fx = pd.DataFrame(fx_vol_dict, index=[f"{s * 100:.1f}%" for s in fx_strikes]).T * 100
    st.dataframe(df_fx.style.format("{:.2f}"))

# Source code
if st.button("Show Full Source Code", use_container_width=True):
    try:
        with open(__file__, 'r', encoding='utf-8') as f:
            source_code = f.read()
        with st.expander("Complete Source Code"):
            st.code(source_code, language="python")
    except Exception as e:
        st.error(f"Error displaying source code: {str(e)}")
