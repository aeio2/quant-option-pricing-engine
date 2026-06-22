import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import math

st.set_page_config(page_title="Quantitative Finance Pricing Engine", layout="wide")

# ==========================================
# 模組 1：FDM (Finite Difference Method)
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
                    if Callput == 'Call':
                        grid[j, i] = max(grid[j, i], current_S - K)
                    else:
                        grid[j, i] = max(grid[j, i], K - current_S)

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
                if Callput == 'Call':
                    grid[j, i] = max(grid[j, i], current_S - K)
                else:
                    grid[j, i] = max(grid[j, i], K - current_S)

    j0 = (S0 - Smin) / dS
    row0 = m - j0
    row_low  = int(np.floor(row0))
    row_high = min(row_low + 1, m)
    w = row0 - row_low
    return (1-w) * grid[row_low, 0] + w * grid[row_high, 0]


def floor_decimal(number, decimals=2):
    factor = 10 ** decimals
    return math.floor(number * factor) / factor


# ==========================================
# 模組 2：BSM
# ==========================================
def black_scholes_price(S0, K, T, r, q, sigma, option_type):
    d1 = (np.log(S0/K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'CALL':
        return S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * np.exp(-q * T) * norm.cdf(-d1)


# ==========================================
# Header
# ==========================================
st.title("📈 Quantitative Finance Pricing Engine")
st.markdown("---")

# ==========================================
# Tab Navigation (replaces sidebar selectbox)
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔢 FDM",
    "📊 Standard Options",
    "🏗️ Structured Payoff",
    "🔭 Lookback",
    "🌈 Rainbow"
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
# Tab 2：Standard Options
# ==========================================
with tab2:
    st.header("Standard Option Pricing (BSM)")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Asset & Strike")
        s0_std = st.slider("S₀ (Spot Price)",    min_value=10.0,  max_value=500.0, value=100.0, step=1.0,  key="std_s0")
        K_std  = st.slider("K (Strike Price)",    min_value=10.0,  max_value=500.0, value=100.0, step=1.0,  key="std_k")
        r_std  = st.slider("r – Risk-free (%)",   min_value=0.0,   max_value=20.0,  value=5.0,   step=0.25, key="std_r") / 100
        q_std  = st.slider("q – Dividend (%)",    min_value=0.0,   max_value=10.0,  value=0.0,   step=0.25, key="std_q") / 100
    with col2:
        st.subheader("Vol & Time")
        sigma_std = st.slider("σ – Volatility (%)", min_value=1.0,  max_value=100.0, value=20.0, step=1.0, key="std_sigma") / 100
        T_std     = st.slider("T (Years)",           min_value=0.1,  max_value=10.0,  value=1.0,  step=0.1, key="std_T")
        n_std     = st.slider("Tree Steps (n)",      min_value=10,   max_value=500,   value=100,  step=10,  key="std_n")

    option_type_std = st.radio("Option Type", ["CALL", "PUT"], horizontal=True, key="std_type")

    if st.button("Calculate BSM Price", key="bsm_calc"):
        bsm_price = black_scholes_price(s0_std, K_std, T_std, r_std, q_std, sigma_std, option_type_std)
        st.success(f"**BSM Price: {bsm_price:.4f}**")

        st.subheader("Convergence Plot (CRR vs BBS)")
        with st.spinner("Generating plot..."):
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.axhline(y=bsm_price, color='blue', linestyle='-', linewidth=1.5, label=f'BSM = {bsm_price:.4f}')
            ax.set_xlabel("Number of Steps (n)")
            ax.set_ylabel("Option Price")
            ax.set_title("Option Price Convergence (add CRR / BBS results here)")
            ax.legend()
            st.pyplot(fig)


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

            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("Call(K1)", f"{p1:.4f}")
            col_b.metric("Call(K2)", f"{p2:.4f}")
            col_c.metric("Call(K3)", f"{p3:.4f}")
            col_d.metric("Call(K4)", f"{p4:.4f}")


# ==========================================
# Tab 4：Lookback (placeholder)
# ==========================================
with tab4:
    st.header("Lookback Options")
    st.info("🚧 Add your Lookback pricing logic here (MCS, closed-form, etc.)")


# ==========================================
# Tab 5：Rainbow Options
# ==========================================
with tab5:
    st.header("Rainbow Option Pricing (MCS, AV, MM)")
    st.info("Input values are comma-separated, e.g. S0 = `100, 100, 100`")

    s0_input   = st.text_input("S₀ for n assets",            value="100, 100",  key="rb_s0")
    sigma_input = st.text_input("σ (%) for n assets",         value="20, 20",    key="rb_sigma")
    corr_input  = st.text_input("Correlation matrix values",  value="0.5",       key="rb_corr")

    col1, col2 = st.columns(2)
    with col1:
        r_rb  = st.slider("r – Risk-free (%)", min_value=0.0,  max_value=20.0, value=5.0,  step=0.25, key="rb_r")  / 100
        T_rb  = st.slider("T (Years)",          min_value=0.1,  max_value=5.0,  value=1.0,  step=0.1,  key="rb_T")
    with col2:
        n_sim = st.slider("Monte Carlo Simulations", min_value=1000, max_value=100000, value=10000, step=1000, key="rb_n")

    if st.button("Run Monte Carlo", key="rb_calc"):
        st.write("⚙️ Executing Cholesky Decomposition and Simulation...")
        st.info("Add your Rainbow MCS logic here.")
