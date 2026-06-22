import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import math
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Quantitative Finance Pricing Engine", layout="wide")

# ==========================================
# 共用數學與定價函數
# ==========================================
def black_scholes_call_price(S0, K, T, r, q, sigma):
    if T <= 0: return max(S0 - K, 0)
    d1 = (np.log(S0/K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def black_scholes_put_price(S0, K, T, r, q, sigma):
    if T <= 0: return max(K - S0, 0)
    d1 = (np.log(S0/K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * np.exp(-q * T) * norm.cdf(-d1)

def black_scholes_price(S0, K, T, r, q, sigma, option_type):
    if option_type == 'CALL':
        return black_scholes_call_price(S0, K, T, r, q, sigma)
    else:
        return black_scholes_put_price(S0, K, T, r, q, sigma)

def payoff_func(S, K, option_type):
    if option_type == 'CALL':
        return np.maximum(S - K, 0)
    else:
        return np.maximum(K - S, 0)

def floor_decimal(number, decimals=2):
    factor = 10 ** decimals
    return math.floor(number * factor) / factor

# ==========================================
# 模組 1：FDM 
# ==========================================
def implicit_FDM(S0, K, r, q, sigma, T, Smin, Smax, m, n, is_American=False, Callput='Call'):
    dt = T / n
    dS = (Smax - Smin) / m
    grid = np.zeros((m+1, n+1))

    if Callput == 'Call':
        for k in range(n+1):
            grid[0, k] = (Smax - K)
            grid[m, k] = 0
        for _ in range(m+1):
            grid[_,-1] = max(0, Smin + (m - _) * dS - K)
    else:
        for k in range(n+1):
            grid[0, k] = 0
            grid[m, k] = K
        for _ in range(m+1):
            grid[_,-1] = max(0, K - (Smin + (m - _) * dS))

    def coefficients(j):
        aj = -0.5 * sigma**2 * j**2 * dt - 0.5 * (r - q) * j * dt
        bj = 1 + sigma**2 * j**2 * dt + r * dt
        cj = -0.5 * sigma**2 * j**2 * dt + 0.5 * (r - q) * j * dt
        return cj, bj, aj

    A = np.zeros((m-1, m-1))
    if m > 1:
        A[0, 0] = coefficients(m-1)[1]
        if m > 2: A[0, 1] = coefficients(m-1)[0]
        for j in range(1, m-2):
            node = m - j - 1
            c, b, a = coefficients(node)
            A[j, j-1] = a
            A[j, j] = b
            A[j, j+1] = c
        if m > 2:
            A[-1, -1] = coefficients(1)[1]
            A[-1, -2] = coefficients(1)[2]

        A_inv = np.linalg.inv(A)
        for i in range(n-1, -1, -1):
            B = np.zeros(m-1)
            for j in range(1, m):
                B[j-1] = grid[j, i+1]
            B[0]   -= coefficients(m-1)[2] * grid[0, i]
            B[m-2] -= coefficients(1)[0]   * grid[m, i]
            x = A_inv @ B
            for j in range(1, m):
                grid[j, i] = x[j-1]
                if is_American:
                    current_S = Smin + (m - j) * dS
                    grid[j, i] = max(grid[j, i], current_S - K if Callput == 'Call' else K - current_S)

    j0 = (S0 - Smin) / dS
    row0 = m - j0
    row_low  = int(np.floor(row0))
    row_high = min(row_low + 1, m)
    w = row0 - row_low
    return (1-w) * grid[row_low, 0] + w * grid[row_high, 0]

def explicit_FDM(S0, K, r, q, sigma, T, Smin, Smax, m, n, is_American=False, Callput='Call'):
    dt = T / n
    dS = (Smax - Smin) / m
    grid = np.zeros((m+1, n+1))

    if Callput == 'Call':
        for k in range(n+1):
            grid[0, k] = (Smax - K)
            grid[m, k] = 0
        for _ in range(m+1):
            grid[_,-1] = max(0, Smin + (m - _) * dS - K)
    else:
        for k in range(n+1):
            grid[0, k] = 0
            grid[m, k] = K
        for _ in range(m+1):
            grid[_,-1] = max(0, K - (Smin + (m - _) * dS))

    def coefficients(j):
        aj = 0.5 * sigma**2 * j**2 * dt + 0.5 * (r - q) * j * dt
        bj = 1 - sigma**2 * j**2 * dt - r * dt
        cj = 0.5 * sigma**2 * j**2 * dt - 0.5 * (r - q) * j * dt
        return cj, bj, aj

    for i in range(n-1, -1, -1):
        for j in range(1, m):
            node = m - j
            c, b, a = coefficients(node)
            grid[j, i] = a * grid[j-1, i+1] + b * grid[j, i+1] + c * grid[j+1, i+1]
            if is_American:
                current_S = Smin + (m - j) * dS
                grid[j, i] = max(grid[j, i], current_S - K if Callput == 'Call' else K - current_S)

    j0 = (S0 - Smin) / dS
    row0 = m - j0
    row_low  = int(np.floor(row0))
    row_high = min(row_low + 1, m)
    w = row0 - row_low
    return (1-w) * grid[row_low, 0] + w * grid[row_high, 0]

# ==========================================
# 介面渲染
# ==========================================
st.title("📈 Quantitative Finance Pricing Engine")
st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    " Finte differencee model ",
    " standard options pricing ",
    " combination of n(assets numbers) vanilla options ",
    " Lookback option pricing ",
    " Rainbow option pricing"
])

# ==========================================
# Tab 1：Finite Difference Method
# ==========================================
with tab1:
    st.header("Finite Difference Method (Implicit & Explicit)")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Asset & Strike")
        s0  = st.slider("S₀ (Spot Price)",     min_value=10.0,  max_value=200.0, value=50.0,  step=1.0)
        k   = st.slider("K (Strike Price)",     min_value=10.0,  max_value=200.0, value=50.0,  step=1.0)
        T   = st.slider("T (Years to Expiry)",  min_value=0.1,   max_value=5.0,   value=0.5,   step=0.1)
    with col2:
        st.subheader("Rates & Volatility")
        r     = st.slider("r – Risk-free Rate (%)",  min_value=0.0,  max_value=20.0, value=10.0, step=0.5) / 100
        q     = st.slider("q – Dividend Yield (%)",  min_value=0.0,  max_value=10.0, value=5.0,  step=0.5) / 100
        sigma = st.slider("σ – Volatility (%)",      min_value=1.0,  max_value=100.0,value=40.0, step=1.0) / 100
    with col3:
        st.subheader("Grid Parameters")
        smin = st.slider("Smin",              min_value=0.0,   max_value=50.0,  value=0.0,   step=5.0)
        smax = st.slider("Smax",              min_value=50.0,  max_value=500.0, value=100.0, step=10.0)
        m    = st.slider("m – Price Steps",   min_value=20,    max_value=300,   value=100,   step=10)
        n    = st.slider("n – Time Steps",    min_value=20,    max_value=300,   value=100,   step=10)

    if st.button("Calculate FDM Prices", key="fdm_calc"):
        with st.spinner("Calculating..."):
            df = pd.DataFrame(columns=["Implicit", "Explicit"],
                              index=["European Call", "European Put", "American Call", "American Put"])

            df.loc["European Call", "Implicit"] = round(implicit_FDM(s0,k,r,q,sigma,T,smin,smax,m,n,False,"Call"), 4)
            df.loc["European Put",  "Implicit"] = round(implicit_FDM(s0,k,r,q,sigma,T,smin,smax,m,n,False,"Put"),  4)
            df.loc["American Call", "Implicit"] = round(implicit_FDM(s0,k,r,q,sigma,T,smin,smax,m,n,True, "Call"), 4)
            df.loc["American Put",  "Implicit"] = round(implicit_FDM(s0,k,r,q,sigma,T,smin,smax,m,n,True, "Put"),  4)

            df.loc["European Call", "Explicit"] = floor_decimal(explicit_FDM(s0,k,r,q,sigma,T,smin,smax,m,n,False,"Call"), 4)
            df.loc["European Put",  "Explicit"] = floor_decimal(explicit_FDM(s0,k,r,q,sigma,T,smin,smax,m,n,False,"Put"),  4)
            df.loc["American Call", "Explicit"] = floor_decimal(explicit_FDM(s0,k,r,q,sigma,T,smin,smax,m,n,True, "Call"), 4)
            df.loc["American Put",  "Explicit"] = floor_decimal(explicit_FDM(s0,k,r,q,sigma,T,smin,smax,m,n,True, "Put"),  4)

        st.dataframe(df.style.format("{:.4f}"), use_container_width=True)

# ==========================================
# Tab 2：Standard Options (BSM, CRR, BBS, MC)
# ==========================================
with tab2:
    st.header("Standard Option Pricing")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Asset & Strike")
        s0_std = st.slider("S₀ (Spot Price)",    min_value=10.0,  max_value=500.0, value=100.0, step=1.0,  key="std_s0")
        K_std  = st.slider("K (Strike Price)",    min_value=10.0,  max_value=500.0, value=100.0, step=1.0,  key="std_k")
        T_std  = st.slider("T (Years)",           min_value=0.1,  max_value=10.0,  value=1.0,  step=0.1, key="std_T")
    with col2:
        st.subheader("Rates & Volatility")
        r_std  = st.slider("r – Risk-free (%)",   min_value=0.0,   max_value=20.0,  value=5.0,   step=0.25, key="std_r") / 100
        q_std  = st.slider("q – Dividend (%)",    min_value=0.0,   max_value=10.0,  value=0.0,   step=0.25, key="std_q") / 100
        sigma_std = st.slider("σ – Volatility (%)", min_value=1.0,  max_value=100.0, value=20.0, step=1.0, key="std_sigma") / 100
    with col3:
        st.subheader("Model Parameters")
        n_std     = st.slider("Tree Steps (n)",      min_value=10,   max_value=500,   value=100,  step=10,  key="std_n")
        sim_times = st.slider("MCS Simulations",     min_value=1000, max_value=50000, value=10000, step=1000, key="std_sim")
        reps      = st.slider("MCS Repetitions",     min_value=1,    max_value=50,    value=20,   step=1, key="std_rep")

    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        option_type_std = st.radio("Option Type", ["CALL", "PUT"], horizontal=True, key="std_type")
    with col_opt2:
        option_style = st.radio("Option Style", ["European", "American"], horizontal=True, key="std_style")
    
    is_euro = (option_style == "European")

    if st.button("Calculate Prices", key="std_calc"):
        with st.spinner("Calculating BSM, CRR, BBS, and Monte Carlo..."):
            ans_dict = {}

            # 1. BSM (Euro Only)
            if is_euro:
                ans_dict["BSM (Analytic)"] = f"{black_scholes_price(s0_std, K_std, T_std, r_std, q_std, sigma_std, option_type_std):.4f}"
            else:
                ans_dict["BSM (Analytic)"] = "N/A (European Only)"

            # 2. Monte Carlo (Euro Only)
            if is_euro:
                mcs_results = []
                for _ in range(reps):
                    drift = (r_std - q_std - 0.5 * (sigma_std**2)) * T_std
                    diffusion = sigma_std * np.sqrt(T_std) * np.random.normal(size=sim_times)
                    S_T_arr = s0_std * np.exp(drift + diffusion)
                    payoffs = payoff_func(S_T_arr, K_std, option_type_std)
                    mcs_results.append(np.mean(np.exp(-r_std * T_std) * payoffs))
                
                mcs_mean = np.mean(mcs_results)
                ci_gap = 2 * np.std(mcs_results, ddof=1)
                
                ans_dict["Monte Carlo (Mean)"] = f"{mcs_mean:.4f}"
                ans_dict["Monte Carlo (95% CI)"] = f"[{mcs_mean - ci_gap:.4f}, {mcs_mean + ci_gap:.4f}]"
            else:
                ans_dict["Monte Carlo (Mean)"] = "N/A"
                ans_dict["Monte Carlo (95% CI)"] = "N/A"

            # 3. CRR & BBS Preparation
            dt = T_std / n_std 
            u = np.exp(sigma_std * np.sqrt(dt))
            d = np.exp(-1 * sigma_std * np.sqrt(dt)) 
            p = (np.exp((r_std - q_std) * dt) - d ) / (u - d) 

            # CRR 1D
            bonus1 = np.zeros(n_std+1) 
            for j in range(n_std+1): 
                bonus1[j] = payoff_func(s0_std * (u ** (n_std-j)) * (d ** j), K_std, option_type_std) 
            
            for k_step in range(n_std , 0, -1):
                for i in range(k_step):
                    hold = np.exp(-r_std * dt) * (p * bonus1[i] + (1-p) * bonus1[i+1])
                    if not is_euro:
                        exe = payoff_func(s0_std * (u ** (k_step-1-i)) * (d ** i), K_std, option_type_std)
                        bonus1[i] = max(hold, exe)
                    else:
                        bonus1[i] = hold
            ans_dict["CRR (1D Tree)"] = f"{bonus1[0]:.4f}"

            # BBS
            bbs_stock = np.zeros((n_std + 1, n_std + 1))
            bbs_opt = np.zeros((n_std + 1, n_std + 1))
            for k_step in range(n_std + 1):
                for j in range(k_step + 1):
                    bbs_stock[j, k_step] = s0_std * (u ** (k_step - j)) * (d ** j)

            for j in range(n_std):
                s_node = bbs_stock[j, n_std-1]
                bs_val = black_scholes_price(s_node, K_std, dt, r_std, q_std, sigma_std, option_type_std)
                if not is_euro: 
                    bbs_opt[j, n_std-1] = max(bs_val, payoff_func(s_node, K_std, option_type_std))
                else: 
                    bbs_opt[j, n_std-1] = bs_val

            for k_step in range(n_std - 2, -1, -1):
                for j in range(k_step + 1):
                    disc_exp = np.exp(-r_std * dt) * (p * bbs_opt[j, k_step+1] + (1 - p) * bbs_opt[j+1, k_step+1])
                    if not is_euro:
                        bbs_opt[j, k_step] = max(disc_exp, payoff_func(bbs_stock[j, k_step], K_std, option_type_std))
                    else:
                        bbs_opt[j, k_step] = disc_exp
            ans_dict["BBS (Tree)"] = f"{bbs_opt[0, 0]:.4f}"

            st.write("### 定價結果比較表")
            df_res = pd.DataFrame.from_dict(ans_dict, orient='index', columns=['Estimated Value'])
            st.table(df_res)

# ==========================================
# Tab 3：Structured Payoff Option
# ==========================================
with tab3:
    st.header("Structured Payoff Option")
    st.markdown("Payoff structured across four strikes — statically replicated via long/short vanilla calls.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Asset & Time")
        s0_sp = st.slider("S₀ (Spot Price)", min_value=10.0,  max_value=300.0, value=100.0, step=1.0,  key="sp_s0")
        T_sp  = st.slider("T (Years)",        min_value=0.1,   max_value=5.0,   value=1.0,   step=0.1,  key="sp_T")
    with col2:
        st.subheader("Rates & Vol")
        r_sp     = st.slider("r – Risk-free (%)",  min_value=0.0,  max_value=20.0,  value=5.0,  step=0.25, key="sp_r")     / 100
        q_sp     = st.slider("q – Dividend (%)",   min_value=0.0,  max_value=10.0,  value=0.0,  step=0.25, key="sp_q")     / 100
        sigma_sp = st.slider("σ – Volatility (%)", min_value=1.0,  max_value=100.0, value=20.0, step=1.0,  key="sp_sigma") / 100
    with col3:
        st.subheader("Four Strikes")
        K1 = st.slider("K1", min_value=10.0,  max_value=200.0, value=80.0,  step=1.0, key="sp_K1")
        K2 = st.slider("K2", min_value=10.0,  max_value=200.0, value=90.0,  step=1.0, key="sp_K2")
        K3 = st.slider("K3", min_value=10.0,  max_value=200.0, value=110.0, step=1.0, key="sp_K3")
        K4 = st.slider("K4", min_value=10.0,  max_value=200.0, value=120.0, step=1.0, key="sp_K4")

    if st.button("Calculate Structured Price", key="sp_calc"):
        if not (K1 < K2 < K3 < K4):
            st.error("Please ensure K1 < K2 < K3 < K4")
        else:
            p1 = black_scholes_price(s0_sp, K1, T_sp, r_sp, q_sp, sigma_sp, "CALL")
            p2 = black_scholes_price(s0_sp, K2, T_sp, r_sp, q_sp, sigma_sp, "CALL")
            p3 = black_scholes_price(s0_sp, K3, T_sp, r_sp, q_sp, sigma_sp, "CALL")
            p4 = black_scholes_price(s0_sp, K4, T_sp, r_sp, q_sp, sigma_sp, "CALL")
            weight = (K2 - K1) / (K4 - K3)
            result = p1 - p2 - weight * p3 + weight * p4
            st.success(f"**Analytical Price (Static Replication): {result:.6f}**")

# ==========================================
# Tab 4：Lookback 
# ==========================================
with tab4:
    st.header("Lookback Options")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st_val    = st.slider("St (Current Price)", min_value=10.0, max_value=200.0, value=50.0, step=1.0, key="lb_st")
        smax_t    = st.slider("S_max_t (Hist Max)", min_value=10.0, max_value=200.0, value=50.0, step=1.0, key="lb_smax")
        T_minus_t = st.slider("T - t (Remaining)",  min_value=0.1,  max_value=5.0,   value=0.25, step=0.05, key="lb_t")
    with col2:
        r_lb      = st.slider("r (%)", min_value=0.0, max_value=20.0, value=10.0, step=0.5, key="lb_r") / 100
        q_lb      = st.slider("q (%)", min_value=0.0, max_value=10.0, value=0.0,  step=0.5, key="lb_q") / 100
        sigma_lb  = st.slider("σ (%)", min_value=1.0, max_value=100.0,value=40.0, step=1.0, key="lb_sig") / 100
    with col3:
        n_lb      = st.slider("Tree Steps (n)",  min_value=5,    max_value=100,   value=10,    step=5,    key="lb_n")
        sims_lb   = st.slider("MCS Simulations", min_value=1000, max_value=20000, value=10000, step=1000, key="lb_sim")
        reps_lb   = st.slider("MCS Repetitions", min_value=1,    max_value=50,    value=20,    step=1,    key="lb_rep")

    is_american_lb = st.checkbox("American Style?", value=False, key="lb_am")

    if st.button("Calculate Lookback Price", key="lb_calc"):
        with st.spinner('Calculating (Fast 1D Array + MCS)...'):
            
            # Tree Method
            dt = T_minus_t / n_lb
            u = np.exp(sigma_lb * np.sqrt(dt))
            d = 1.0 / u
            mu = np.exp((r_lb - q_lb) * dt)  
            r_disc = np.exp(r_lb * dt)      
            q_prob = (mu * u - 1.0) / (mu * (u - d))
            
            k0 = int(round(np.log(max(smax_t / st_val, 1.0)) / np.log(u)))
            max_k = n_lb + k0
            V = np.array([u**k - 1.0 for k in range(max_k + 1)])
            
            for j in range(n_lb - 1, -1, -1):
                V_new = np.zeros(j + k0 + 1)
                for k in range(j + k0 + 1):
                    k_up = max(k - 1, 0)
                    k_down = k + 1
                    hold = (q_prob * V[k_up] + (1 - q_prob) * V[k_down]) * mu / r_disc
                    if is_american_lb:
                        V_new[k] = max(hold, u**k - 1.0)
                    else:
                        V_new[k] = hold
                V = V_new
                
            tree_price = st_val * V[k0]

            # MCS Method
            mcs_prices = []
            drift = (r_lb - q_lb - 0.5 * sigma_lb**2) * dt
            vol = sigma_lb * np.sqrt(dt)

            for _ in range(reps_lb):
                S_path = np.zeros((sims_lb, n_lb + 1))
                S_path[:, 0] = st_val
                
                Z = np.random.normal(size=(sims_lb, n_lb))
                for i in range(1, n_lb + 1):
                    S_path[:, i] = S_path[:, i-1] * np.exp(drift + vol * Z[:, i-1])
                
                max_S = np.maximum(smax_t, np.max(S_path, axis=1))
                payoffs = np.maximum(max_S - S_path[:, -1], 0)
                mcs_prices.append(np.mean(np.exp(-r_lb * T_minus_t) * payoffs))

            st.write(f"**Tree Method Price:** {tree_price:.6f}")
            st.write(f"**MCS Mean ({reps_lb} reps):** {np.mean(mcs_prices):.6f}")

# ==========================================
# Tab 5：Rainbow Options 
# ==========================================
with tab5:
    st.header("Rainbow Option Pricing (MCS, AV, MM)")
    
    col_top1, col_top2, col_top3 = st.columns(3)
    with col_top1:
        # 改為純數字輸入框，提供更大的輸入彈性
        n_assets = st.number_input("Number of Assets (n)", min_value=2, max_value=10, value=2, step=1, key="rb_n_assets")
    with col_top2:
        K_rb = st.number_input("Strike Price (K)", min_value=1.0, value=100.0, step=1.0, key="rb_k")
        T_rb = st.number_input("T (Years)",        min_value=0.01, value=1.0, step=0.1, key="rb_T")
    with col_top3:
        r_rb     = st.number_input("r – Risk-free (%)", min_value=0.0, value=5.0, step=0.1, key="rb_r") / 100
        sims_rb  = st.number_input("MCS Simulations",   min_value=100, value=10000, step=1000, key="rb_sims")
        reps_rb  = st.number_input("MCS Repetitions",   min_value=1,   value=10,  step=1, key="rb_reps")

    st.subheader("個別資產參數 (S₀, q, σ)")
    
    # 建立彈性的 Columns 數量，避免過多資產時版面擠壓
    cols_assets = st.columns(min(n_assets, 5)) 
    s0_list, q_list, sigma_list = [], [], []
    
    for i in range(n_assets):
        with cols_assets[i % 5]:
            st.markdown(f"**Asset {i+1}**")
            s0_list.append(st.number_input(f"S0_{i+1}", min_value=1.0, value=100.0, step=1.0, key=f"rb_s0_{i}"))
            q_list.append(st.number_input(f"q_{i+1} (%)", min_value=0.0, value=0.0, step=0.1, key=f"rb_q_{i}") / 100)
            sigma_list.append(st.number_input(f"σ_{i+1} (%)", min_value=0.1, value=20.0, step=1.0, key=f"rb_sig_{i}") / 100)

    st.subheader("相關係數矩陣 (Correlation)")
    corr_matrix = np.eye(n_assets)
    idx = 0
    corr_cols = st.columns(min(3, n_assets * (n_assets - 1) // 2))
    
    # 動態生成相關係數 "輸入框" (number_input)
    for i in range(n_assets):
        for j in range(i+1, n_assets):
            with corr_cols[idx % 3]:
                val = st.number_input(f"Corr (Asset {i+1} & {j+1})", min_value=-1.0, max_value=1.0, value=0.5, step=0.05, key=f"corr_{i}_{j}")
                corr_matrix[i, j] = val
                corr_matrix[j, i] = val
                idx += 1

    if st.button("Calculate Rainbow Options", key="rb_calc"):
        with st.spinner("Simulating Cholesky Decomposition & MCS..."):
            
            cov_matrix = np.zeros((n_assets, n_assets))
            for i in range(n_assets):
                for j in range(n_assets):
                    cov_matrix[i, j] = sigma_list[i] * sigma_list[j] * corr_matrix[i, j] * T_rb

            try:
                A = np.linalg.cholesky(cov_matrix)
            except np.linalg.LinAlgError:
                st.error("Correlation matrix is not positive-definite! Please adjust the correlation values.")
                st.stop()
            
            s0_arr = np.array(s0_list)
            q_arr = np.array(q_list)
            sig_arr = np.array(sigma_list)
            drift = (r_rb - q_arr - 0.5 * sig_arr**2) * T_rb

            res_basic, res_bonus, res_new = [], [], []

            def calc_rainbow_price(G_paths):
                St = s0_arr * np.exp(drift + G_paths)
                max_prices = St.max(axis=1)
                payoffs = np.maximum(max_prices - K_rb, 0)
                return np.mean(np.exp(-r_rb * T_rb) * payoffs)

            for _ in range(int(reps_rb)):
                # 1. Basic MCS
                Z = np.random.standard_normal((int(sims_rb), n_assets))
                G = Z.dot(A.T)
                
                # 2. AV + MM
                Z_bonus = Z.copy()
                Z_bonus[int(sims_rb)//2:] = -Z_bonus[:int(sims_rb)//2]
                Z_bonus = (Z_bonus - Z_bonus.mean(axis=0)) / Z_bonus.std(axis=0)
                G_bonus = Z_bonus.dot(A.T)

                # 3. Inverse Cholesky
                cov_star = np.cov(Z_bonus, rowvar=False)
                B = np.linalg.cholesky(cov_star)
                B_inv = np.linalg.inv(B)
                G_new = Z_bonus.dot(B_inv.T).dot(A.T)

                res_basic.append(calc_rainbow_price(G))
                res_bonus.append(calc_rainbow_price(G_bonus))
                res_new.append(calc_rainbow_price(G_new))

            df_res = pd.DataFrame({
                "Method": ["Basic MCS", "AV + MM", "Inverse Chol"],
                "Mean Price": [np.mean(res_basic), np.mean(res_bonus), np.mean(res_new)],
                "Std Error": [np.std(res_basic, ddof=1), np.std(res_bonus, ddof=1), np.std(res_new, ddof=1)]
            })
            
            # 加入 95% CI 輸出供 Rainbow 參考
            df_res["95% CI Lower"] = df_res["Mean Price"] - 2 * df_res["Std Error"]
            df_res["95% CI Upper"] = df_res["Mean Price"] + 2 * df_res["Std Error"]

            st.write("### Rainbow Option Results")
            st.dataframe(df_res.style.format({
                "Mean Price": "{:.4f}", 
                "Std Error": "{:.6f}",
                "95% CI Lower": "{:.4f}",
                "95% CI Upper": "{:.4f}"
            }), use_container_width=True)
