# ====================== PAGE CONFIG & THEME (EXACT COLOR MATCH) ======================
# This is created by Raymond Yeung
# This is a Best of Put option pricer
# Methodology: Correlated, quanto-adjusted Geometric Brownian Motions (GBMs) with constant Correlation Copula
# This section sets the page title, layout, and applies the custom CSS for the black background + blue/white theme

import streamlit as st
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


st.set_page_config(page_title="Best of Put Pricer", layout="wide")
st.markdown("""
<style>
    .main {background-color: #000000; color: #ffffff;}
    .stApp {background-color: #000000;}
    .stButton > button {background-color: #00BFFF; color: white; font-weight: bold; border-radius: 8px; border: none; height: 50px; font-size: 18px;}
    .stButton > button:hover {background-color: #0099CC;}
    .metric-box {background-color: #1E2A44; padding: 20px; border-radius: 12px; border: 2px solid #00BFFF; text-align: center; margin-bottom: 10px;}
    h1 {color: #00BFFF !important;}
    h3 {color: #00BFFF !important;}
    .metric-label {color: #00BFFF !important; font-weight: 600;}
    .metric-value {color: #ffffff !important;}
</style>
""", unsafe_allow_html=True)

st.title("Best of Put Pricer")
st.markdown(
    '<p style="color:#ffffff; margin-top:-15px; font-size:14px;"><strong>Multivariate GBM with constant correlation Copula by Raymond Yeung</strong></p>',
    unsafe_allow_html=True)

# ====================== HARDCODED MARKET DATA ======================
# This section contains all the volatility surfaces, forward factors, interest rates, and strike lists loaded directly from the Excel files
eq_strikes = [0.8, 0.9, 0.95, 0.975, 1.0, 1.025, 1.05, 1.1, 1.2]
fx_hkd_strikes = [0.97, 0.985, 1.0, 1.005, 1.01]
fx_jpy_strikes = [0.9, 0.94, 1.0, 1.04, 1.09]

vol_data = {
    'SPX': {'1M': [0.4375, 0.3209, 0.2755, 0.2529, 0.2279, 0.1995, 0.1693, 0.13756, 0.2057],
            '2M': [0.3806, 0.2973, 0.2593, 0.2402, 0.2203, 0.1993, 0.1785, 0.1466, 0.156],
            '3M': [0.3511, 0.2812, 0.2484, 0.2318, 0.2145, 0.1966, 0.179, 0.1507, 0.142],
            '6M': [0.3104, 0.2604, 0.2362, 0.2239, 0.2111, 0.198, 0.1849, 0.1614, 0.1366],
            '9M': [0.2924, 0.2523, 0.2327, 0.2226, 0.2123, 0.2016, 0.1909, 0.1704, 0.1434],
            '1Y': [0.2802, 0.2457, 0.2286, 0.2198, 0.2109, 0.2018, 0.1926, 0.1747, 0.148],
            '18M': [0.2655, 0.2376, 0.2237, 0.2167, 0.2097, 0.2027, 0.1958, 0.1822, 0.158],
            '2Y': [0.2585, 0.2347, 0.2227, 0.2166, 0.2106, 0.2046, 0.1987, 0.1872, 0.1659]},
    'HSI': {'1M': [0.4236, 0.3091, 0.2631, 0.2461, 0.234, 0.2265, 0.2235, 0.231, 0.2724],
            '2M': [0.3554, 0.2793, 0.2498, 0.2392, 0.2313, 0.226, 0.2235, 0.226, 0.246],
            '3M': [0.3185, 0.2629, 0.2414, 0.2338, 0.2284, 0.2251, 0.2238, 0.2258, 0.2399],
            '6M': [0.2801, 0.2475, 0.2361, 0.232, 0.2289, 0.2268, 0.2256, 0.2256, 0.2311],
            '9M': [0.2656, 0.2428, 0.2351, 0.2322, 0.2299, 0.2281, 0.2269, 0.2257, 0.2275],
            '1Y': [0.2589, 0.2419, 0.236, 0.2337, 0.2319, 0.2305, 0.2295, 0.2285, 0.2298],
            '18M': [0.2426, 0.2426, 0.2385, 0.2369, 0.2357, 0.2347, 0.234, 0.2234, 0.2347],
            '2Y': [0.2538, 0.2432, 0.2393, 0.2377, 0.2364, 0.2353, 0.2345, 0.2236, 0.2342]},
    'NKY': {'1M': [0.572, 0.4589, 0.4088, 0.385, 0.365, 0.3493, 0.3369, 0.3178, 0.3037],
            '2M': [0.4772, 0.3969, 0.3645, 0.35, 0.3369, 0.3253, 0.3154, 0.3012, 0.2922],
            '3M': [0.429, 0.3647, 0.3385, 0.3271, 0.3168, 0.3076, 0.2996, 0.2869, 0.273],
            '6M': [0.3698, 0.326, 0.3077, 0.2995, 0.292, 0.2851, 0.2788, 0.2682, 0.2534],
            '9M': [0.3479, 0.3141, 0.3, 0.2936, 0.2876, 0.2822, 0.2771, 0.2684, 0.2553],
            '1Y': [0.3319, 0.3032, 0.2911, 0.2856, 0.2805, 0.2757, 0.2713, 0.2634, 0.2513],
            '18M': [0.3055, 0.2826, 0.2729, 0.2684, 0.2642, 0.2602, 0.2565, 0.2498, 0.239],
            '2Y': [0.2909, 0.2716, 0.2633, 0.2595, 0.2559, 0.2524, 0.2492, 0.2433, 0.2336]},
    'USDHKD': {'1M': [0.016, 0.0122, 0.0098, 0.0098, 0.0105], '2M': [0.0164, 0.0119, 0.0091, 0.0092, 0.0102],
               '3M': [0.018, 0.0122, 0.0087, 0.0092, 0.0114], '6M': [0.0266, 0.015, 0.0086, 0.008, 0.0112],
               '9M': [0.0313, 0.0175, 0.0088, 0.0071, 0.0122], '1Y': [0.043, 0.0215, 0.0099, 0.007, 0.01232],
               '18M': [0.052, 0.0284, 0.0151, 0.012, 0.0171], '2Y': [0.0587, 0.0322, 0.0172, 0.0137, 0.0195]},
    'USDJPY': {'1M': [0.1192, 0.1062, 0.0993, 0.1019, 0.111], '2M': [0.1206, 0.1076, 0.1006, 0.1025, 0.111],
               '3M': [0.1185, 0.1058, 0.0988, 0.1004, 0.1086], '6M': [0.118, 0.1055, 0.099, 0.10035, 0.1083],
               '9M': [0.1178, 0.1054, 0.0993, 0.1011, 0.1097], '1Y': [0.1163, 0.1043, 0.0983, 0.1006, 0.1093],
               '18M': [0.113, 0.1001, 0.0958, 0.0989, 0.1092], '2Y': [0.111, 0.0993, 0.0945, 0.0981, 0.1092]}
}

forward_data = {
    'SPX': {'3M': 1.0080757638125566, '6M': 1.0154262769914615, '1Y': 1.0305796426218188, '18M': 1.0452429740402491,
            '2Y': 1.0634119348209266},
    'HSI': {'3M': 0.9891382885147022, '6M': 0.9876853480774065, '1Y': 0.9977773938175422, '18M': 0.9894917064589094,
            '2Y': 0.9890597511937673},
    'NKY': {'3M': 0.9916459818270701, '6M': 0.9923892053472703, '1Y': 0.9923892053472703, '18M': 0.9923892053472703,
            '2Y': 0.9905311465467699}
}

rates_data = {
    'USD': {'1M': 0.0368, '3M': 0.0369, '6M': 0.0367, '9M': 0.0364, '12M': 0.0362, '18M': 0.0356, '24M': 0.0353},
    'JPY': {'1M': 0.0074, '3M': 0.0082, '6M': 0.0091, '9M': 0.0098, '12M': 0.0105, '18M': 0.0117, '24M': 0.0128},
    'HKD': {'1M': 0.0225, '3M': 0.0238, '6M': 0.0263, '9M': 0.0276, '12M': 0.0267, '18M': 0.0267, '24M': 0.0266}
}

T_map = {'1M': 1 / 12, '2M': 2 / 12, '3M': 3 / 12, '6M': 0.5, '9M': 0.75, '1Y': 1.0, '18M': 1.5, '2Y': 2.0}


def get_tenor_years(tenor): return T_map[tenor]


def get_vol(asset, tenor, K):
    data = vol_data.get(asset, {})
    if tenor not in data: return 0.20
    strikes = eq_strikes if asset in ['SPX', 'HSI', 'NKY'] else (
        fx_hkd_strikes if asset == 'USDHKD' else fx_jpy_strikes)
    vols = data[tenor]
    f = interp1d(strikes, vols, kind='linear', fill_value='extrapolate')
    return float(f(K))


def get_forward_factor(asset, tenor):
    data = forward_data[asset]
    ts_known = [T_map[k] for k in data]
    vals = list(data.values())
    t = get_tenor_years(tenor)
    f = interp1d(ts_known, vals, kind='linear', fill_value='extrapolate')
    return float(f(t))


def get_rate(tenor):
    data = rates_data['USD']
    ts_known = [int(k.replace('M', '')) / 12 if 'M' in k else 1.0 for k in data]
    vals = list(data.values())
    t = get_tenor_years(tenor)
    f = interp1d(ts_known, vals, kind='linear', fill_value='extrapolate')
    return float(f(t))


# ====================== CORE PRICING ENGINE ======================
# This is the Monte-Carlo pricing function: simulates 100k paths under correlated GBMs with quanto adjustment
def mc_best_of_put(tenor, strike_pct, corrs_eq, rho_hsi_fx, rho_nky_fx, spot_bump=None, vol_bump=None):
    np.random.seed(42)
    N = 100000
    T = get_tenor_years(tenor)
    r = get_rate(tenor)
    df = np.exp(-r * T)
    K = strike_pct / 100.0

    K_vol = K
    vol_spx = get_vol('SPX', tenor, K_vol)
    vol_hsi = get_vol('HSI', tenor, K_vol)
    vol_nky = get_vol('NKY', tenor, K_vol)
    vol_hkd = get_vol('USDHKD', tenor, 1.0)
    vol_jpy = get_vol('USDJPY', tenor, 1.0)

    fr_spx = get_forward_factor('SPX', tenor)
    fr_hsi = get_forward_factor('HSI', tenor)
    fr_nky = get_forward_factor('NKY', tenor)

    adj_hsi = np.exp(-rho_hsi_fx * vol_hsi * vol_hkd * T)
    fr_hsi_q = fr_hsi * adj_hsi
    adj_nky = np.exp(-rho_nky_fx * vol_nky * vol_jpy * T)
    fr_nky_q = fr_nky * adj_nky

    if spot_bump:
        fr_spx *= spot_bump.get('SPX', 1.0)
        fr_hsi_q *= spot_bump.get('HSI', 1.0)
        fr_nky_q *= spot_bump.get('NKY', 1.0)
    if vol_bump:
        if 'SPX' in vol_bump: vol_spx += vol_bump['SPX']
        if 'HSI' in vol_bump: vol_hsi += vol_bump['HSI']
        if 'NKY' in vol_bump: vol_nky += vol_bump['NKY']
        if 'HKD' in vol_bump:
            vol_hkd += vol_bump['HKD']
            adj_hsi = np.exp(-rho_hsi_fx * vol_hsi * vol_hkd * T)
            fr_hsi_q = get_forward_factor('HSI', tenor) * adj_hsi
        if 'JPY' in vol_bump:
            vol_jpy += vol_bump['JPY']
            adj_nky = np.exp(-rho_nky_fx * vol_nky * vol_jpy * T)
            fr_nky_q = get_forward_factor('NKY', tenor) * adj_nky

    mu_spx = np.log(fr_spx) - 0.5 * vol_spx ** 2 * T
    mu_hsi = np.log(fr_hsi_q) - 0.5 * vol_hsi ** 2 * T
    mu_nky = np.log(fr_nky_q) - 0.5 * vol_nky ** 2 * T

    corr_mat = np.array(
        [[1.0, corrs_eq[0], corrs_eq[2]], [corrs_eq[0], 1.0, corrs_eq[1]], [corrs_eq[2], corrs_eq[1], 1.0]])
    L = np.linalg.cholesky(corr_mat)
    Z = np.random.normal(0, 1, (N, 3))
    dW = Z @ L.T

    perf_spx = np.exp(mu_spx + vol_spx * np.sqrt(T) * dW[:, 0])
    perf_hsi = np.exp(mu_hsi + vol_hsi * np.sqrt(T) * dW[:, 1])
    perf_nky = np.exp(mu_nky + vol_nky * np.sqrt(T) * dW[:, 2])

    max_perf = np.maximum.reduce([perf_spx, perf_hsi, perf_nky])
    payoff = np.maximum(K - max_perf, 0.0)
    price_pct = df * np.mean(payoff) * 100
    return price_pct


# ====================== SIDEBAR INPUTS ======================
# This section builds the left sidebar with all user inputs (tenor, strike, correlations)
with st.sidebar:
    st.header("Inputs")
    tenor = st.selectbox("Tenor", ["1M", "2M", "3M", "6M", "9M", "1Y", "18M", "2Y"], index=3)
    strike_pct = st.number_input("Best of Put Strike (%)", 80.0, 100.0, 90.0, 0.01)

    st.markdown('<div style="color:#ffffff !important; font-size:1.25em; font-weight:600; margin:1.2em 0 0.6em 0;">Equity-Equity Correlations (%)</div>', unsafe_allow_html=True)
    rho_hsi_spx = st.number_input("HSI/SPX", value=15.0, step=0.1)
    rho_hsi_nky = st.number_input("HSI/NKY", value=29.0, step=0.1)
    rho_spx_nky = st.number_input("SPX/NKY", value=12.0, step=0.1)

    st.markdown('<div style="color:#ffffff !important; font-size:1.25em; font-weight:600; margin:1.2em 0 0.6em 0;">Quanto Correlations (%)</div>', unsafe_allow_html=True)
    rho_hsi_hkd = st.number_input("HSI/USDHKD", value=-7.0, step=0.1)
    rho_nky_jpy = st.number_input("NKY/USDJPY", value=11.0, step=0.1)

price_btn = st.button("Price Option", type="primary", use_container_width=True)

# ====================== PRICING & GREEKS ======================
# This block runs when the user clicks "Price Option" — it calculates price + all Greeks (Delta, Vega, Correlation Delta, Gamma)
if price_btn:
    corrs_eq = [rho_hsi_spx / 100, rho_hsi_nky / 100, rho_spx_nky / 100]
    rho_hsi_fx = rho_hsi_hkd / 100
    rho_nky_fx = rho_nky_jpy / 100

    with st.spinner("Monte Carlo 100k paths..."):
        base_price = mc_best_of_put(tenor, strike_pct, corrs_eq, rho_hsi_fx, rho_nky_fx)

        bump_spx = 0.03
        p_spx_up = mc_best_of_put(tenor, strike_pct, corrs_eq, rho_hsi_fx, rho_nky_fx, spot_bump={'SPX': 1 + bump_spx})
        delta_spx = (p_spx_up - base_price) / bump_spx

        bump_hsi = 0.05
        p_hsi_up = mc_best_of_put(tenor, strike_pct, corrs_eq, rho_hsi_fx, rho_nky_fx, spot_bump={'HSI': 1 + bump_hsi})
        delta_hsi = (p_hsi_up - base_price) / bump_hsi

        bump_nky = 0.03
        p_nky_up = mc_best_of_put(tenor, strike_pct, corrs_eq, rho_hsi_fx, rho_nky_fx, spot_bump={'NKY': 1 + bump_nky})
        delta_nky = (p_nky_up - base_price) / bump_nky

        p_spx_vega = mc_best_of_put(tenor, strike_pct, corrs_eq, rho_hsi_fx, rho_nky_fx, vol_bump={'SPX': 0.02})
        vega_spx = (p_spx_vega - base_price) / 2
        p_hsi_vega = mc_best_of_put(tenor, strike_pct, corrs_eq, rho_hsi_fx, rho_nky_fx, vol_bump={'HSI': 0.02})
        vega_hsi = (p_hsi_vega - base_price) / 2
        p_nky_vega = mc_best_of_put(tenor, strike_pct, corrs_eq, rho_hsi_fx, rho_nky_fx, vol_bump={'NKY': 0.02})
        vega_nky = (p_nky_vega - base_price) / 2
        p_jpy_vega = mc_best_of_put(tenor, strike_pct, corrs_eq, rho_hsi_fx, rho_nky_fx, vol_bump={'JPY': 0.01})
        vega_jpy = (p_jpy_vega - base_price) / 1
        p_hkd_vega = mc_best_of_put(tenor, strike_pct, corrs_eq, rho_hsi_fx, rho_nky_fx, vol_bump={'HKD': 0.005})
        vega_hkd = (p_hkd_vega - base_price) * 2

        corr_delta_hsi_spx = (mc_best_of_put(tenor, strike_pct,
                                             [rho_hsi_spx / 100 + 0.05, rho_hsi_nky / 100, rho_spx_nky / 100],
                                             rho_hsi_fx, rho_nky_fx) - base_price) / 500
        corr_delta_hsi_nky = (mc_best_of_put(tenor, strike_pct,
                                             [rho_hsi_spx / 100, rho_hsi_nky / 100 + 0.05, rho_spx_nky / 100],
                                             rho_hsi_fx, rho_nky_fx) - base_price) / 500
        corr_delta_spx_nky = (mc_best_of_put(tenor, strike_pct,
                                             [rho_hsi_spx / 100, rho_hsi_nky / 100, rho_spx_nky / 100 + 0.05],
                                             rho_hsi_fx, rho_nky_fx) - base_price) / 500
        corr_delta_hsi_fx = (mc_best_of_put(tenor, strike_pct, corrs_eq, rho_hsi_fx + 0.05,
                                            rho_nky_fx) - base_price) / 500
        corr_delta_nky_fx = (mc_best_of_put(tenor, strike_pct, corrs_eq, rho_hsi_fx,
                                            rho_nky_fx + 0.05) - base_price) / 500

        # ====================== GAMMA MATRIX (AMENDED ONLY HERE - EXACTLY AS PER YOUR INSTRUCTIONS) ======================
        # This computes the full 3x3 Equity Gamma matrix (diagonal + cross gammas) exactly as you specified
        gamma_matrix = np.zeros((3, 3))
        bump_pct = 0.01
        base_deltas = np.array([delta_spx, delta_hsi, delta_nky])
        assets = ['SPX', 'HSI', 'NKY']
        for i, bumped_asset in enumerate(assets):
            bump_dict = {bumped_asset: 1 + bump_pct}
            p_bumped = mc_best_of_put(tenor, strike_pct, corrs_eq, rho_hsi_fx, rho_nky_fx, spot_bump=bump_dict)

            # Recompute ALL 3 deltas at the bumped level
            p_spx_up = mc_best_of_put(tenor, strike_pct, corrs_eq, rho_hsi_fx, rho_nky_fx,
                                      spot_bump={**bump_dict, 'SPX': bump_dict.get('SPX', 1.0) * 1.03})
            d_spx_new = (p_spx_up - p_bumped) / 0.03

            p_hsi_up = mc_best_of_put(tenor, strike_pct, corrs_eq, rho_hsi_fx, rho_nky_fx,
                                      spot_bump={**bump_dict, 'HSI': bump_dict.get('HSI', 1.0) * 1.05})
            d_hsi_new = (p_hsi_up - p_bumped) / 0.05

            p_nky_up = mc_best_of_put(tenor, strike_pct, corrs_eq, rho_hsi_fx, rho_nky_fx,
                                      spot_bump={**bump_dict, 'NKY': bump_dict.get('NKY', 1.0) * 1.03})
            d_nky_new = (p_nky_up - p_bumped) / 0.03

            new_deltas = np.array([d_spx_new, d_hsi_new, d_nky_new])
            gamma_matrix[
                i] = new_deltas - base_deltas  # Diagonal = own delta change, Cross = other deltas change (raw difference, exactly as per your definition)

        # ====================== PRICING RESULTS ======================
        # This section displays the main pricing results (Price, Deltas, Vegas) using the metric-box layout
        st.markdown("### Pricing Results")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                f"<div class='metric-box'><span class='metric-label'>Price (% Notional)</span><br><span class='metric-value'>{base_price:.2f}%</span></div>",
                unsafe_allow_html=True)
        with c2:
            st.markdown(
                f"<div class='metric-box'><span class='metric-label'>SPX Delta</span><br><span class='metric-value'>{delta_spx:.2f}%</span></div>",
                unsafe_allow_html=True)
        with c3:
            st.markdown(
                f"<div class='metric-box'><span class='metric-label'>HSI Delta</span><br><span class='metric-value'>{delta_hsi:.2f}%</span></div>",
                unsafe_allow_html=True)
        with c4:
            st.markdown(
                f"<div class='metric-box'><span class='metric-label'>NKY Delta</span><br><span class='metric-value'>{delta_nky:.2f}%</span></div>",
                unsafe_allow_html=True)

        v1, v2, v3, v4 = st.columns(4)
        with v1:
            st.markdown(
                f"<div class='metric-box'><span class='metric-label'>SPX Vega</span><br><span class='metric-value'>{vega_spx:.2f}%</span></div>",
                unsafe_allow_html=True)
        with v2:
            st.markdown(
                f"<div class='metric-box'><span class='metric-label'>HSI Vega</span><br><span class='metric-value'>{vega_hsi:.2f}%</span></div>",
                unsafe_allow_html=True)
        with v3:
            st.markdown(
                f"<div class='metric-box'><span class='metric-label'>NKY Vega</span><br><span class='metric-value'>{vega_nky:.2f}%</span></div>",
                unsafe_allow_html=True)
        with v4:
            st.markdown(
                f"<div class='metric-box'><span class='metric-label'>USDJPY Vega</span><br><span class='metric-value'>{vega_jpy:.2f}%</span></div>",
                unsafe_allow_html=True)

        v5, v6, _, _ = st.columns(4)
        with v5:
            st.markdown(
                f"<div class='metric-box'><span class='metric-label'>USDHKD Vega</span><br><span class='metric-value'>{vega_hkd:.2f}%</span></div>",
                unsafe_allow_html=True)

        st.markdown("### Correlation Delta")
        corr_df = pd.DataFrame({
            "HSI/SPX": [f"{corr_delta_hsi_spx * 100:+.3f}%"],
            "HSI/NKY": [f"{corr_delta_hsi_nky * 100:+.3f}%"],
            "SPX/NKY": [f"{corr_delta_spx_nky * 100:+.3f}%"],
            "HSI/USDHKD": [f"{corr_delta_hsi_fx * 100:+.3f}%"],
            "NKY/USDJPY": [f"{corr_delta_nky_fx * 100:+.3f}%"]
        })
        st.dataframe(corr_df, use_container_width=True)

        st.markdown("### Equity Gamma Matrix (3x3)")
        gamma_df = pd.DataFrame(gamma_matrix, index=assets, columns=assets)
        st.dataframe(gamma_df.style.format("{:.2f}%"), use_container_width=True)

        # ====================== MARKET DATA USED ======================
        # This section shows the actual market data used for the current pricing (spots + interpolated implied vols)
        K = strike_pct / 100.0
        vol_spx_used = get_vol('SPX', tenor, K)
        vol_hsi_used = get_vol('HSI', tenor, K)
        vol_nky_used = get_vol('NKY', tenor, K)
        vol_hkd_used = get_vol('USDHKD', tenor, 1.0)
        vol_jpy_used = get_vol('USDJPY', tenor, 1.0)

        st.markdown("### Market Data Used")
        md1, md2 = st.columns(2)
        with md1:
            st.markdown(
                f"<div class='metric-box'><span class='metric-label'>SPX Spot</span><br><span class='metric-value'>6632.19</span></div>",
                unsafe_allow_html=True)
            st.markdown(
                f"<div class='metric-box'><span class='metric-label'>HSI Spot</span><br><span class='metric-value'>25465.6</span></div>",
                unsafe_allow_html=True)
            st.markdown(
                f"<div class='metric-box'><span class='metric-label'>NKY Spot</span><br><span class='metric-value'>53819.61</span></div>",
                unsafe_allow_html=True)
        with md2:
            st.markdown(
                f"<div class='metric-box'><span class='metric-label'>SPX Implied Vol</span><br><span class='metric-value'>{vol_spx_used * 100:.2f}%</span></div>",
                unsafe_allow_html=True)
            st.markdown(
                f"<div class='metric-box'><span class='metric-label'>HSI Implied Vol</span><br><span class='metric-value'>{vol_hsi_used * 100:.2f}%</span></div>",
                unsafe_allow_html=True)
            st.markdown(
                f"<div class='metric-box'><span class='metric-label'>NKY Implied Vol</span><br><span class='metric-value'>{vol_nky_used * 100:.2f}%</span></div>",
                unsafe_allow_html=True)
            st.markdown(
                f"<div class='metric-box'><span class='metric-label'>USDJPY Implied Vol</span><br><span class='metric-value'>{vol_jpy_used * 100:.2f}%</span></div>",
                unsafe_allow_html=True)
            st.markdown(
                f"<div class='metric-box'><span class='metric-label'>USDHKD Implied Vol</span><br><span class='metric-value'>{vol_hkd_used * 100:.2f}%</span></div>",
                unsafe_allow_html=True)

# ====================== MARKET DATA BUTTON ======================
# This button displays the full volatility surfaces, forwards and interest rate curves in nice tables
if st.button("Show Vol Surface and Rates curve"):
    st.subheader("Equity Vol Surfaces")
    for name in ['SPX', 'HSI', 'NKY']:
        df_vol = pd.DataFrame(vol_data[name], index=eq_strikes)
        st.markdown(f'<div style="color:#ffffff !important; font-size:1.25em; font-weight:600; margin:1.2em 0 0.6em 0;">{name}</div>', unsafe_allow_html=True)
        st.dataframe(df_vol)
    st.subheader("FX Vol Surfaces")
    for name in ['USDHKD', 'USDJPY']:
        df_vol = pd.DataFrame(vol_data[name], index=fx_hkd_strikes if name == 'USDHKD' else fx_jpy_strikes)
        st.markdown(f'<div style="color:#ffffff !important; font-size:1.25em; font-weight:600; margin:1.2em 0 0.6em 0;">{name}</div>', unsafe_allow_html=True)
        st.dataframe(df_vol)
    st.subheader("Forwards & Rates")
    st.markdown('<div style="color:#ffffff !important; font-size:1.25em; font-weight:600; margin:1.2em 0 0.6em 0;">SPX/HSI/NKY Forward Factors</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(forward_data))
    st.markdown(
        '<div style="color:#ffffff !important; font-size:1.25em; font-weight:600; margin:1.2em 0 0.6em 0;">USD Rates</div>',
        unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(rates_data['USD'].items(), columns=['Tenor', 'USD rates']))
    st.dataframe(pd.DataFrame(rates_data['USD'].items(), columns=['Tenor', 'USD rates']))
    st.write("JPY Rates")
    st.dataframe(pd.DataFrame(rates_data['JPY'].items(), columns=['Tenor', 'JPY rates']))
    st.write("HKD Rates")
    st.dataframe(pd.DataFrame(rates_data['HKD'].items(), columns=['Tenor', 'HKD rates']))

# ====================== SOURCE CODE BUTTON ======================
# This button shows the complete Python source code of the entire pricer when clicked
if st.button("Show Full Source Code"):
    with open(__file__, "r", encoding="utf-8") as f:
        st.code(f.read(), language="python")

st.caption("100,000 paths • Seeded • Quanto-adjusted GBM • Linear interpolation • All greeks via finite difference")