import streamlit as st
import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares, minimize
from scipy.stats import norm
import time
from scipy.interpolate import interp1d
import streamlit.components.v1 as components
import inspect
import hashlib

# ====================== PICKLEABLE CLASSES ======================
class LocalVolFunc:
    def __init__(self, spline):
        self.spline = spline

    def __call__(self, t, S):
        S = np.asarray(S).clip(1e-6)
        return self.spline.ev(np.full_like(S, t), S).flatten()


class ConditionalLeverage:
    def __init__(self, cond_ev, local_vol_func):
        self.cond_ev = cond_ev
        self.local_vol_func = local_vol_func

    def __call__(self, S, t):
        S = np.asarray(S).clip(1e-6)
        t = np.asarray(t)
        out = np.ones_like(S, dtype=float)
        for ti in np.unique(t):
            mask = np.isclose(t, ti)
            if np.any(mask):
                t_key = min(self.cond_ev.keys(), key=lambda x: abs(x - ti))
                ev = self.cond_ev[t_key](S[mask])
                sigma_loc = self.local_vol_func(ti, S[mask])
                out[mask] = sigma_loc / np.sqrt(np.maximum(ev, 1e-8))
        return np.clip(out, 0.25, 3.0)


st.set_page_config(page_title="LSV Pricer - Barriers & Vanillas", layout="wide")

# ====================== UI COLOR ======================

st.markdown(
    """
   <style>
        [data-testid="stAppViewContainer"] {background-color: #000000 !important;}
        .stApp, .main .block-container {background-color: #000000 !important;}
        h1, h2, h3, h4, h5, h6, p, span, label, div.stMarkdown, .stCaption {color: #ffffff !important;}
        h1 {color: #00ccff !important;}
        [data-testid="stSidebar"] {background-color: #1a1a1a !important;}
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {color: #00ccff !important;}
        .stButton button {background-color: #00ccff !important; color: #000000 !important;}
        .stButton button:hover {background-color: #0099cc !important;}
        .stSuccess, .stInfo, .stWarning, .stError {color: #ffffff !important; background-color: #222222 !important;}

        /* === ONLY CHANGE FOR FILE UPLOADER TEXT === */
        [data-testid="stFileUploaderDropzone"] p,
        [data-testid="stFileUploaderDropzone"] span,
        [data-testid="stFileUploaderDropzone"] div {
            color: #000000 !important;
        }
        .stFileUploader label {
            color: #000000 !important;
        }

        /* === BLACK FONT FOR SOURCE CODE (on default white background) - Windows fix === */
        [data-testid="stCodeBlock"],
        .stCodeBlock,
        div[data-testid="stCodeBlock"] > div {
            background-color: #ffffff !important;
        }
        [data-testid="stCodeBlock"] pre,
        [data-testid="stCodeBlock"] code,
        .stCodeBlock pre,
        .stCodeBlock code,
        pre.highlight,
        code.highlight,
        div[data-testid="stCodeBlock"] > div pre,
        div[data-testid="stCodeBlock"] > div code {
            color: #000000 !important;
            background-color: #ffffff !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Local Stochastic Volatility (Heston) Pricer")
st.caption("by Raymond Yeung")

# ====================== TENOR PARSER ======================
def parse_tenor(t):
    t = str(t).strip().upper().replace(" ", "")
    if t.endswith("M"):
        return float(t[:-1]) / 12.0
    elif t.endswith("Y"):
        return float(t[:-1])
    else:
        try:
            return float(t)
        except:
            return np.nan


# ====================== FILE UPLOADER ======================
st.markdown("### Upload Market Data Excel File")
uploaded_file = st.file_uploader("Upload your market data Excel file",
                                 type=["xlsx"])

if uploaded_file is None:
    st.warning("Please upload the Excel file.")
    st.stop()

# ====================== ROBUST EXCEL READING ======================
try:
    fwd_df = pd.read_excel(uploaded_file, sheet_name="SPX Forward", header=0)
    fwd_df.columns = fwd_df.columns.str.strip()
    fwd_tenors = np.array([parse_tenor(x) for x in fwd_df["Tenor"]])
    fwd_factors = fwd_df["Forward"].values.astype(float)

    rate_df = pd.read_excel(uploaded_file, sheet_name="USD interest rate curve", header=0)
    rate_df.columns = rate_df.columns.str.strip()
    rate_tenors = np.array([parse_tenor(x) for x in rate_df["Tenor"]])
    usd_rates = rate_df["USD rates"].values.astype(float)

    vol_df = pd.read_excel(uploaded_file, sheet_name="Listed Volatility Screen", header=None)
    tenor_series = vol_df.iloc[2:, 1].dropna().astype(str).str.strip()
    T_vol = np.array([parse_tenor(x) for x in tenor_series])
    K_vol = vol_df.iloc[1, 2:].dropna().values.astype(float)
    vol_matrix_raw = vol_df.iloc[2:2 + len(T_vol), 2:2 + len(K_vol)].values.astype(float)

    price_df = pd.read_excel(uploaded_file, sheet_name="Last price history", header=0)
    price_df.columns = price_df.columns.str.strip()
    spx_prices = price_df["Price"].values.astype(float)
except Exception as e:
    st.error(f"Error reading Excel: {e}")
    st.stop()

log_returns = np.log(spx_prices[1:] / spx_prices[:-1])
realized_vol = np.sqrt(np.mean(log_returns ** 2) * 252)

ref_spot = float(spx_prices[-1])

st.success("Market data loaded successfully!")
# ====================== FORCE RESET ON NEW FILE UPLOAD ======================
# Create a unique hash of the uploaded file so we detect any change
file_bytes = uploaded_file.getvalue()  # safe because it's a BytesIO object
current_file_hash = hashlib.md5(file_bytes).hexdigest()

if ("last_file_hash" not in st.session_state or 
    st.session_state.last_file_hash != current_file_hash):
    
    st.info("🔄 New market data file detected — clearing old calibration & caches...")
    
    # Clear all relevant session state
    keys_to_reset = [
        'heston_params', 'L_func', 'vol_matrix_clean',
        'original_heston_params', 'original_L_func'
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]
    
    # Clear ALL cached functions (local vol, call surface, calibration, etc.)
    st.cache_data.clear()
    st.cache_resource.clear()
    
    # Remember this file so we don't reset again on the next rerun
    st.session_state.last_file_hash = current_file_hash
    
    st.rerun()  # Important: forces the whole script to re-execute with the new file




st.info(f"Loaded {len(spx_prices)} prices | Current SPX: {ref_spot:.2f} | Vol surface {len(T_vol)}×{len(K_vol)}")


# ====================== HELPERS ======================
def get_fwd(T):
    return np.interp(T, fwd_tenors, fwd_factors)


def get_r(T):
    return np.interp(T, rate_tenors, usd_rates)


def bs_call(F, K, T, sigma, r):
    if T <= 0:
        return max(F - K, 0)
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))


# ====================== ULTRA-LIGHT EXPLICIT DE-ARB + PROFESSIONAL SMOOTHING ======================
def check_negative_variance(vol):
    return np.any(vol < 0.0)


def check_calendar_arbitrage(vol, T_vol):
    total_var = (vol ** 2) * T_vol[:, np.newaxis]
    diffs = np.diff(total_var, axis=0)
    return np.any(diffs < -1e-5)


def check_butterfly_arbitrage(vol, K_vol, T_vol, ref_spot):
    for i, T in enumerate(T_vol):
        F = ref_spot * get_fwd(T)
        r = get_r(T)
        K_abs = ref_spot * K_vol
        prices = np.array([bs_call(F, k, T, v, r) for k, v in zip(K_abs, vol[i])])
        dk = np.diff(K_abs)
        dC = np.diff(prices)
        dC_dK = dC / dk
        d2C_dK2 = np.diff(dC_dK) / ((dk[:-1] + dk[1:]) / 2)
        if np.any(d2C_dK2 < -1e-7):
            return True
    return False


def clean_vol_surface(vol_mat_raw, T_vol, K_vol, ref_spot, max_trials=3):
    vol_raw = vol_mat_raw.copy().astype(float)
    for trial in range(1, max_trials + 1):
        st.info(f"🔄 Smoothing & de-arb trial {trial}/{max_trials} (ultra-light)")

        gauss_sigma = 0.15 + (trial - 1) * 0.08
        spline_s = 0.4 + (trial - 1) * 0.6

        vol = np.maximum(vol_raw, 0.01)
        vol = gaussian_filter(vol, sigma=gauss_sigma)

        spline = RectBivariateSpline(T_vol, K_vol, vol, kx=3, ky=3, s=spline_s)
        TT, KK = np.meshgrid(T_vol, K_vol, indexing='ij')
        vol = spline.ev(TT.ravel(), KK.ravel()).reshape(vol.shape)
        vol = np.maximum(vol, 0.01)

        if check_negative_variance(vol):
            st.warning(f"⚠️ Negative variance in trial {trial} → retrying")
            continue
        if check_calendar_arbitrage(vol, T_vol):
            st.warning(f"⚠️ Calendar arbitrage in trial {trial} → retrying")
            continue
        if check_butterfly_arbitrage(vol, K_vol, T_vol, ref_spot):
            st.warning(f"⚠️ Butterfly arbitrage in trial {trial} → retrying")
            continue

        signed_diff = vol - vol_mat_raw
        abs_diff = np.abs(signed_diff)
        max_idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
        i_t, i_k = max_idx

        tenor_str = f"{int(T_vol[i_t] * 12)}M" if T_vol[i_t] < 1 else f"{int(T_vol[i_t])}Y" if T_vol[
            i_t].is_integer() else f"{T_vol[i_t]:.1f}Y"
        strike_pct = int(K_vol[i_k] * 100)
        max_change_volpts = abs(signed_diff[i_t, i_k]) * 100
        direction = "lower" if signed_diff[i_t, i_k] < 0 else "higher"

        st.success(f"✅ Trial {trial} PASSED | "
                   f"Max change = {max_change_volpts:.4f} vol points at {tenor_str} {strike_pct}% strike "
                   f"(smoothed vol is {max_change_volpts:.3f} vol points {direction}) | "
                   f"Mean change = {np.mean(abs_diff) * 100:.4f} vol points")
        return vol


# ====================== APPLY CLEANING ONCE ======================
vol_matrix = clean_vol_surface(vol_matrix_raw, T_vol, K_vol, ref_spot)
st.session_state.vol_matrix_clean = vol_matrix


# ====================== NUMERICALLY ROBUST LOCAL-VOL EXTRACTION ======================
@st.cache_resource
def build_smoothed_call_surface(ref_spot, vol_mat=None):
    if vol_mat is None:
        vol_mat = st.session_state.vol_matrix_clean
    spline = RectBivariateSpline(T_vol, K_vol, vol_mat, kx=3, ky=3, s=0.8)
    T_grid = np.linspace(0.005, max(T_vol) * 1.05, 120)
    K_grid = np.linspace(0.55 * ref_spot, 1.60 * ref_spot, 200)
    TT, KK = np.meshgrid(T_grid, K_grid, indexing='ij')
    sigma_imp = spline.ev(TT.ravel(), (KK / ref_spot).ravel()).reshape(TT.shape)
    sigma_imp = gaussian_filter(sigma_imp, sigma=0.75)
    F_grid = ref_spot * get_fwd(TT)
    r_grid = get_r(TT)
    C = np.vectorize(bs_call)(F_grid, KK, TT, sigma_imp, r_grid)
    return T_grid, K_grid, C


@st.cache_resource
def compute_dupire_local_vol(ref_spot, vol_mat=None):
    T_grid, K_grid, C = build_smoothed_call_surface(ref_spot, vol_mat)
    dC_dT = np.gradient(C, T_grid, axis=0)
    dC_dK = np.gradient(C, K_grid, axis=1)
    d2C_dK2 = np.gradient(dC_dK, K_grid, axis=1)

    fwd_T = get_fwd(T_grid)
    r_grid = get_r(T_grid)
    r_minus_q = np.log(fwd_T) / T_grid
    q_grid = r_grid - r_minus_q

    K_2d = K_grid[None, :]
    r_mq_2d = r_minus_q[:, None]
    q_2d = q_grid[:, None]

    num = dC_dT + r_mq_2d * K_2d * dC_dK + q_2d * C
    den = 0.5 * K_2d ** 2 * d2C_dK2
    den = np.maximum(den, 1e-9)
    local_vol2 = np.maximum(num / den, 1e-8)
    local_vol = np.sqrt(local_vol2)
    local_vol = gaussian_filter(local_vol, sigma=0.9)
    local_vol = np.clip(local_vol, 0.04, 0.75)

    spline = RectBivariateSpline(T_grid, K_grid, local_vol, kx=3, ky=3)
    return LocalVolFunc(spline)


# ====================== HESTON CALIBRATION + LEVERAGE ======================
@st.cache_resource
def calibrate_heston_and_leverage(ref_spot, n_calib_paths=35000, n_calib_steps=110):
    atm_idx = 4
    atm_vols = st.session_state.vol_matrix_clean[:, atm_idx]
    atm_vars = atm_vols ** 2

    def atm_obj(x):
        v0, kappa, theta = x
        E_vt = theta + (v0 - theta) * np.exp(-kappa * T_vol)
        err = np.sum((E_vt - atm_vars) ** 2)
        penalty = 8 * (theta - np.mean(atm_vars[-3:])) ** 2 + 12 * max(0, v0 - 0.35) ** 2
        return err + penalty

    res1 = minimize(atm_obj, [realized_vol ** 2, 2.5, np.mean(atm_vars[-3:])],
                    bounds=[(0.005, 0.40), (0.3, 8.0), (0.01, 0.40)], method='L-BFGS-B')
    v0, kappa, theta = res1.x

    def skew_obj(x):
        xi, rho = x
        params_tmp = np.array([v0, kappa, theta, xi, rho])
        err = 0.0
        for T_test in [0.25, 0.5, 1.0, 1.5, 2.0]:
            for k_pct in [0.8, 0.90, 1.10, 1.20]:
                K_test = ref_spot * k_pct
                tenor_idx = np.argmin(np.abs(T_vol - T_test))
                market_vol = np.interp(k_pct, K_vol, st.session_state.vol_matrix_clean[tenor_idx])
                F = ref_spot * get_fwd(T_test)
                market_price = bs_call(F, K_test, T_test, market_vol, get_r(T_test))
                mc_price = price_option_mc(ref_spot, T_test, K_test, 0, "", True, False,
                                           "pure_heston", params_tmp, n_paths=n_calib_paths,
                                           n_steps=n_calib_steps, seed=42)
                err += (mc_price - market_price) ** 2
        return err

    res2 = least_squares(skew_obj, [0.75, -0.68], bounds=([0.3, -0.95], [1.6, -0.2]))
    xi, rho = res2.x
    heston_params = np.array([v0, kappa, theta, xi, rho])

    L_func = build_conditional_leverage(heston_params, ref_spot)
    return heston_params, L_func


def build_conditional_leverage(heston_params, ref_spot, n_paths=120000, n_steps=200,
                               progress_bar=None, status_text=None):
    local_vol_func = compute_dupire_local_vol(ref_spot)
    v0, kappa, theta, xi, rho = heston_params
    T_max = 2.0
    dt = T_max / n_steps
    r = get_r(T_max)
    q = np.log(get_fwd(T_max)) / T_max - r
    drift = (r - q) * dt
    sqrt_dt = np.sqrt(dt)

    S = np.full(n_paths, ref_spot, dtype=np.float64)
    V = np.full(n_paths, v0, dtype=np.float64)

    # More time samples + denser early times (most important for 1Y)
    sample_times = np.concatenate(([0.05, 0.10], np.linspace(0.15, T_max, 18)))
    step_indices = (sample_times / dt).astype(int).clip(0, n_steps - 1)

    cond_ev = {}
    np.random.seed(123)

    for step in range(n_steps):
        Z1 = np.random.normal(0, 1, n_paths)
        Z2 = rho * Z1 + np.sqrt(1 - rho ** 2) * np.random.normal(0, 1, n_paths)

        sqrtV = np.sqrt(np.maximum(V, 1e-8))
        dV = kappa * (theta - V) * dt + xi * sqrtV * sqrt_dt * Z2
        V += dV + 0.25 * xi ** 2 * dt * (Z2 ** 2 - 1)
        V = np.maximum(V, 0.0)

        sigma = sqrtV
        dlogS = (drift - 0.5 * sigma ** 2) * dt + sigma * sqrt_dt * Z1
        S = S * np.exp(dlogS)
        S = np.maximum(S, 1e-6)

        if step in step_indices:
            t = (step + 1) * dt
            # More bins + minimum 12 points per bin + light smoothing
            bins = np.percentile(S, np.linspace(0, 100, 121))  # bins
            centers = (bins[:-1] + bins[1:]) / 2
            mean_V = []
            for i in range(len(bins) - 1):
                mask = (S >= bins[i]) & (S < bins[i + 1])
                if np.sum(mask) > 12:
                    mean_V.append(np.mean(V[mask]))
                else:
                    mean_V.append(theta)
            mean_V = np.array(mean_V)

            # Light smoothing on E[V|S] curve (very cheap, reduces noise a lot)
            mean_V = gaussian_filter(mean_V, sigma=1.2)

            # Better interpolation: quadratic
            f_ev = interp1d(centers, mean_V, kind='quadratic',
                            fill_value="extrapolate", bounds_error=False)
            cond_ev[t] = f_ev

        if progress_bar is not None and step % max(1, n_steps // 15) == 0:
            percent = int((step + 1) / n_steps * 100)
            progress_bar.progress(percent)
            if status_text is not None:
                status_text.text(f"Re-calibrating leverage function... {percent}% complete")

    if progress_bar is not None:
        progress_bar.progress(100)
    if status_text is not None:
        status_text.text("✅ Leverage function re-calibration finished!")

    return ConditionalLeverage(cond_ev, local_vol_func)


# ====================== MC PRICER (Milstein/Euler scheme) ======================
def price_option_mc(spot, T, K, barrier, barrier_type, is_call, is_barrier, mode, heston_params,
                    local_vol_func=None, L_func=None, n_paths=150000, n_steps=500, seed=42, vol_mat=None):
    np.random.seed(seed)
    dt = T / n_steps
    r = get_r(T)
    q = np.log(get_fwd(T)) / T - r
    drift = (r - q) * dt
    v0, kappa, theta, xi, rho = heston_params

    if vol_mat is not None:
        local_vol_func = compute_dupire_local_vol(spot, vol_mat)

    if local_vol_func is None:
        local_vol_func = compute_dupire_local_vol(spot)

    S = np.full(n_paths, spot, dtype=np.float64)
    V = np.full(n_paths, v0, dtype=np.float64)
    hit = np.zeros(n_paths, dtype=bool)
    sqrt_dt = np.sqrt(dt)

    for step_idx in range(n_steps):
        t = (step_idx + 1) * dt
        Z1 = np.random.normal(0, 1, n_paths)

        if mode == "LV" or vol_mat is not None:
            sigma = local_vol_func(t, S)
        elif mode == "pure_heston":
            sqrtV = np.sqrt(np.maximum(V, 1e-8))
            sigma = sqrtV
            Z2 = rho * Z1 + np.sqrt(1 - rho ** 2) * np.random.normal(0, 1, n_paths)
            dV = kappa * (theta - V) * dt + xi * sqrtV * sqrt_dt * Z2
            V += dV + 0.25 * xi ** 2 * dt * (Z2 ** 2 - 1)   # Milstein correction
            V = np.maximum(V, 0.0)
        else:  # LSV
            L = L_func(S, t)
            sqrtV = np.sqrt(np.maximum(V, 1e-8))
            sigma = L * sqrtV
            Z2 = rho * Z1 + np.sqrt(1 - rho ** 2) * np.random.normal(0, 1, n_paths)
            dV = kappa * (theta - V) * dt + xi * sqrtV * sqrt_dt * Z2
            V += dV + 0.25 * xi ** 2 * dt * (Z2 ** 2 - 1)   # Milstein correction
            V = np.maximum(V, 0.0)

        sigma = np.clip(sigma, 1e-4, 2.0)
        dlogS = (drift - 0.5 * sigma ** 2) * dt + sigma * sqrt_dt * Z1
        S = S * np.exp(dlogS)
        S = np.maximum(S, 1e-8)

        if is_barrier:
            if "Down" in barrier_type:
                hit |= (S <= barrier)
            else:
                hit |= (S >= barrier)

    payoff = np.maximum(S - K, 0) if is_call else np.maximum(K - S, 0)
    if is_barrier:
        if "Out" in barrier_type:
            payoff = payoff * (~hit)
        else:
            payoff = payoff * hit

    raw_price = np.exp(-r * T) * np.mean(payoff)
    return raw_price


# ====================== HIGHCHARTS VOL SURFACE CHART ======================
def show_smoothed_vol_chart():
    raw_vol = vol_matrix_raw.tolist()
    clean_vol = st.session_state.vol_matrix_clean.tolist()
    tenors_str = [f"{int(t*12)}M" if t < 1 else f"{int(t)}Y" if t.is_integer() else f"{t:.1f}Y" for t in T_vol]
    default_idx = np.argmin(np.abs(T_vol - 1.0))

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Smoothed & De-arb Vol Surface</title>
        <script src="https://cdn.jsdelivr.net/npm/highcharts@11.4.3/highcharts.js"></script>
        <style>
            body {{ margin:0; padding:20px; background:#111; color:#fff; font-family:Arial,sans-serif; }}
            select {{ padding:10px; font-size:18px; background:#222; color:#fff; border:none; }}
        </style>
    </head>
    <body>
        <h2 style="color:#00ccff">Smoothed & De-arb Vol Surface vs Raw (Your File)</h2>
        <p><strong>Select Tenor:</strong> 
            <select id="tenorSelect" onchange="updateChart()">
                {"".join([f'<option value="{i}" {"selected" if i==default_idx else ""}>{tenors_str[i]}</option>' for i in range(len(T_vol))])}
            </select>
        </p>
        <div id="container" style="width:100%; height:650px;"></div>

        <script>
            let rawData = {raw_vol};
            let cleanData = {clean_vol};
            let K = {K_vol.tolist()};

            function updateChart() {{
                let idx = parseInt(document.getElementById("tenorSelect").value);
                Highcharts.chart('container', {{
                    chart: {{ type: 'line', backgroundColor: '#111' }},
                    title: {{ text: 'Vol Surface – Raw vs Smoothed & De-arb', style: {{ color: '#fff' }} }},
                    xAxis: {{ 
                        title: {{ text: 'Moneyness', style: {{ color: '#fff' }} }},
                        categories: K,
                        labels: {{ style: {{ color: '#ccc' }} }}
                    }},
                    yAxis: {{ 
                        title: {{ text: 'Implied Volatility', style: {{ color: '#fff' }} }},
                        labels: {{ style: {{ color: '#ccc' }} }}
                    }},
                    tooltip: {{ valueDecimals: 4 }},
                    legend: {{ itemStyle: {{ color: '#fff' }} }},
                    series: [
                        {{ name: 'Raw Listed Vol', data: rawData[idx], color: '#ff4444', dashStyle: 'Dash', lineWidth: 2 }},
                        {{ name: 'Smoothed & De-arb Vol', data: cleanData[idx], color: '#00ccff', lineWidth: 4 }}
                    ]
                }});
            }}
            window.onload = updateChart;
        </script>
    </body>
    </html>
    """
    components.html(html, height=750, width=1150, scrolling=True)


# ====================== UI ======================
if 'heston_params' not in st.session_state or 'L_func' not in st.session_state:
    st.markdown("<p style='color:#00ccff; font-weight:bold;'>Calibrating Heston Params and Leverage Function....</p>",
                unsafe_allow_html=True)
    status_text_init = st.empty()

    st.session_state.heston_params, st.session_state.L_func = calibrate_heston_and_leverage(ref_spot)
    st.session_state.original_heston_params = st.session_state.heston_params.copy()
    st.session_state.original_L_func = st.session_state.L_func

    for p in range(0, 101, 10):
        status_text_init.text(f"Calibrating Heston Params and Leverage Function.... {p}/100%")
        time.sleep(0.08)
    status_text_init.text("✅ Initial calibration completed!")

st.sidebar.header("Heston Parameters")
v0 = st.sidebar.number_input("v₀ - Initial instantaneous variance", value=float(st.session_state.heston_params[0]), format="%.4f")
kappa = st.sidebar.number_input("κ - Mean reversion speed of instantaneous variance", value=float(st.session_state.heston_params[1]), format="%.3f")
theta = st.sidebar.number_input("θ - Long-term mean of instantaneous variance", value=float(st.session_state.heston_params[2]), format="%.4f")
xi = st.sidebar.number_input("ξ - Volatility of Instantaneous variance (Vol of Vol)", value=float(st.session_state.heston_params[3]), format="%.3f")
rho = st.sidebar.number_input("ρ - Correlation between spot and variance", value=float(st.session_state.heston_params[4]), format="%.3f")

col_btn1, _ = st.sidebar.columns(2)
with col_btn1:
    if st.button("Re-calibrate Leverage Function", use_container_width=True):
        with st.spinner("Re-calibrating leverage function..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            current_params = np.array([v0, kappa, theta, xi, rho])
            new_L_func = build_conditional_leverage(current_params, ref_spot, progress_bar=progress_bar, status_text=status_text)
            st.session_state.L_func = new_L_func
            st.success("✅ Leverage function successfully re-calibrated!")
            st.rerun()

st.sidebar.header("Payoff Specification")
ref_spot_ui = st.sidebar.number_input("SPX Reference Spot", value=ref_spot, step=1.0)

tenor_options = {"3M": 0.25, "6M": 0.5, "1Y": 1.0, "18M": 1.5, "2Y": 2.0}
tenor_str = st.sidebar.selectbox("Tenor", list(tenor_options.keys()))
T = tenor_options[tenor_str]

option_type = st.sidebar.selectbox("Option Type", ["Barrier Call", "Barrier Put", "Vanilla Call", "Vanilla Put"])
is_call = "Call" in option_type
is_barrier = "Barrier" in option_type

strike_pct = st.sidebar.number_input("Strike (%)", value=100.0, step=0.1)
K = ref_spot_ui * strike_pct / 100.0

if is_barrier:
    barrier_pct = st.sidebar.number_input("Barrier (%)", value=80.0, step=0.1)
    B = ref_spot_ui * barrier_pct / 100.0
    barrier_types = ["Down & Out", "Down & In", "Up & Out", "Up & In"]
    barrier_type = st.sidebar.selectbox("Barrier Type", barrier_types)
else:
    barrier_type = ""
    B = 0.0

mode_label = st.sidebar.radio("Model Mode", ["LV (Local Vol only)", "LSV (Local + Stochastic Vol)"])
mode = "LV" if "LV" in mode_label else "LSV"
st.markdown("### Click to Price the Option")
if st.button("PRICE NOW", type="primary", use_container_width=True):
    with st.spinner("Running Monte Carlo paths..."):
        start = time.time()
        
        pricing_paths = 120000
        greeks_paths  = 400000      # higher = rock-solid
        mc_steps = 320
        seed = 42
        
        h = 0.001                   # 0.1% bump (critical)
        vol_bump = 0.01

        heston_params = np.array([v0, kappa, theta, xi, rho])
        L_func = st.session_state.L_func if mode == "LSV" else None

        # ==================== BASE PRICE ====================
        base_raw = price_option_mc(ref_spot_ui, T, K, B, barrier_type, is_call, is_barrier, mode,
                                   heston_params, local_vol_func=None, L_func=L_func,
                                   n_paths=pricing_paths, n_steps=mc_steps, seed=seed)
        base_pct = (base_raw / ref_spot_ui) * 100

        # ==================== HELPER: price at bumped spot ====================
        def price_at_spot(S):
            if mode == "LV":
                local_vol_func_bumped = compute_dupire_local_vol(S)   # rebuild for LV
            else:
                local_vol_func_bumped = compute_dupire_local_vol(ref_spot_ui)
            return price_option_mc(S, T, K, B, barrier_type, is_call, is_barrier, mode,
                                   heston_params, local_vol_func=local_vol_func_bumped, L_func=L_func,
                                   n_paths=greeks_paths, n_steps=mc_steps, seed=seed)

        raw_up   = price_at_spot(ref_spot_ui * (1 + h))
        raw_down = price_at_spot(ref_spot_ui * (1 - h))

        pct_up   = (raw_up   / ref_spot_ui) * 100
        pct_down = (raw_down / ref_spot_ui) * 100

        
        # ====================== GREEKS (FIXED FOR PUTS - POINT 1) ======================
        # Use absolute raw prices + more paths + smaller bump + common random seed logic
        seed = 42
        n_greeks = 150000
        h = 0.005

        # =========
        # Use THE SAME seed for base + all bumps.
        # This makes the Monte Carlo paths identical except for the intentional bump.
        # Finite-difference Greeks become extremely stable

        raw_base = price_option_mc(ref_spot_ui, T, K, B, barrier_type, is_call, is_barrier,
                                   mode, heston_params, local_vol_func=local_vol_func,
                                   L_func=L_func, n_paths=n_greeks, seed=seed)
        base_pct = (raw_base / ref_spot_ui) * 100

        raw_up = price_option_mc(ref_spot_ui * (1 + h), T, K, B, barrier_type, is_call, is_barrier,
                                 mode, heston_params, local_vol_func=local_vol_func,
                                 L_func=L_func, n_paths=n_greeks, seed=seed)  # ← same seed

        raw_down = price_option_mc(ref_spot_ui * (1 - h), T, K, B, barrier_type, is_call, is_barrier,
                                   mode, heston_params, local_vol_func=local_vol_func,
                                   L_func=L_func, n_paths=n_greeks, seed=seed)  # ← same seed

        pct_up = (raw_up / ref_spot_ui) * 100
        pct_down = (raw_down / ref_spot_ui) * 100

        # Your original formulas are already correct for "% notional" convention
        delta = (pct_up - pct_down) / (2 * h)
        gamma = (pct_up - 2 * base_pct + pct_down) / (h ** 2) / 100

        # Vega (same seed again)
        bumped_vol = vol_matrix + 0.02
        raw_vega = price_option_mc(ref_spot_ui, T, K, B, barrier_type, is_call, is_barrier,
                                   mode, heston_params,
                                   local_vol_func=None, L_func=None, vol_mat=bumped_vol,
                                   n_paths=n_greeks, seed=seed)  # ← same seed

        pct_vega = (raw_vega / ref_spot_ui) * 100
        vega = (pct_vega - base_pct) / 2.0

        col1, col2, col3 = st.columns(3)
        col1.metric("Delta (% notional)", f"{delta:.2f}")
        col2.metric("Gamma (% notional)", f"{gamma:.4f}")
        col3.metric("Vega (% notional)", f"{vega:.2f}")


# ====================== HIGHCHARTS BUTTON ======================
st.markdown("---")
col_chart, col_code = st.columns(2)
with col_chart:
    if st.button("Smoothed & De-arb Vol Surface", use_container_width=True):
        show_smoothed_vol_chart()

# ======================  SOURCE CODE BUTTON ======================
with col_code:
    if st.button("Show full source code", use_container_width=True):
        try:
            source_code = inspect.getsource(__import__("__main__"))

            # Custom dark HTML code block (bypasses all st.code() styling issues)
            html_code = f"""
            <div style="background-color: #1a1a1a; padding: 20px; border-radius: 8px; overflow-x: auto; max-height: 700px;">
                <pre style="margin:0; padding:0; background:#1a1a1a; color:#e6e6e6; font-family: 'Courier New', monospace; font-size: 14px; line-height: 1.4; white-space: pre;">
                    <code style="background:#1a1a1a; color:#e6e6e6;">{source_code}</code>
                </pre>
            </div>
            """
            components.html(html_code, height=750, width=1150, scrolling=True)

        except Exception as e:
            st.error(f"Could not retrieve source code: {e}")
            st.info("The full source code is the script you are currently running.")

# ====================== METHODOLOGY BUTTON ======================
if st.button("LSV Model Methodology", use_container_width=True):
    st.markdown("### LSV Model Methodology")
    st.markdown(r"""
1. **Raw data input**  
   Load SPX forwards, USD rate curve, listed implied-vol surface (tenor × moneyness), and historical prices.  
   Extract reference spot \(S_0\) and realized volatility from log-returns.

2. **Smoothing and de-arb**  
   Apply Gaussian filter + RectBivariateSpline (cubic) smoothing on the raw implied-vol matrix.  
   Explicit arbitrage checks (negative variance, calendar spreads, butterfly via finite-difference \( \partial_K^2 C < 0 \)) with up to 3 iterative trials until fully arbitrage-free.

3. **Dupire Local Vol model implementation**  
   Convert smoothed implied vols to dense call prices \(C(T,K)\) via Black-Scholes formula.  
   Extract local volatility using the exact Dupire formula with central finite differences:  
   $$\sigma_{\rm loc}^2(K,T)=\frac{\partial_T C+(r-q)K\,\partial_K C+qC}{\frac12 K^2\,\partial_K^2 C}$$  
   Followed by Gaussian smoothing and clipping.

4. **Heston Stochastic Vol model parameters calibration (2 phases)**  
   - **ATM phase**: Least-squares fit of \(v_0,\kappa,\theta\) to ATM implied variance curve using  
     $$\mathbb{E}[V_T]=\theta+(v_0-\theta)e^{-\kappa T}$$ plus penalties.  
   - **Skew phase**: Least-squares on \(\xi\) (vol-of-vol) and \(\rho\) (spot-vol correlation) by minimising pure-Heston MC price errors versus market prices on all OTM strikes.

5. **Leverage function implementation**  
   Run long pure-Heston simulation (Milstein scheme) and compute conditional expectation \(\mathbb{E}[V_t\mid S_t]\) at 10 discrete times via percentile binning + linear interpolation.  
   Construct leverage function:  
   $$L(S,t)=\frac{\sigma_{\rm loc}(S,t)}{\sqrt{\mathbb{E}[V_t\mid S_t]}}$$  
   (clipped to \([0.25,3.0]\)).

6. **Monte-Carlo pricing engine**  
   Simulate under full LSV dynamics with Euler scheme on \(\ln S\) and Milstein scheme on variance:  
   $$dS_t=(r-q)S_t\,dt+L(S_t,t)\sqrt{V_t}\,S_t\,dW_1$$  
   $$dV_t=\kappa(\theta-V_t)\,dt+\xi\sqrt{V_t}\,dW_2,\quad\langle dW_1,dW_2\rangle=\rho\,dt$$  
   Supports pure LV, pure Heston, or LSV modes with continuous barrier monitoring.

7. **Price and Greeks output**  
   Option price = discounted Monte-Carlo average of payoff.  
   Greeks computed via central finite differences **with Common Random Numbers** (identical seed across base/bumped paths) for noise-free, stable Delta, Gamma and Vega.
    """, unsafe_allow_html=True)
    st.caption(" ")
