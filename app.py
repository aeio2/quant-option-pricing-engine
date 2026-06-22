import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import math

# 設定網頁基本資訊
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
            grid[_,-1] = max(0 , Smin + (m - _ ) * dS - K)
    else: 
        for k in range(n+1):
            grid[0, k] = 0
            grid[m , k] = K 
        for _ in range(m+1):
            grid[_,-1] = max(0 , K - (Smin + (m - _ ) * dS))

    def coefficients(j):
        aj = - 0.5 * sigma**2 * j**2 * dt - 0.5 * (r - q) * j * dt
        bj = 1 + sigma**2 * j**2 * dt + r * dt
        cj = - 0.5 * sigma**2 * j**2 * dt + 0.5 * (r - q) * j * dt
        return cj , bj , aj 
    
    A = np.zeros((m-1,m-1))
    if m > 1:
        A[0,0]  = coefficients(m-1)[1]
        if m > 2: A[0,1]  = coefficients(m-1)[0]
        for j in range(1,m-2): 
            node = m - j - 1
            c , b , a = coefficients(node)
            A[j, j-1] = a
            A[j, j] = b
            A[j, j+1] = c
        if m > 2:
            A[-1, -1] = coefficients(1)[1]
            A[-1, -2] = coefficients(1)[2]
        
        A_inv = np.linalg.inv(A)
        for i in range (n-1 ,-1, -1):
            B = np.zeros(m-1)
            for j in range(1 , m):
                B[j-1] = grid[ j , i+1]
            B[0]    -= coefficients(m-1)[2] * grid[0, i]   
            B[m-2]  -= coefficients(1)[0]   * grid[m, i]   
            x = A_inv @ B    
            for j in range(1 , m):
                grid[j,i] = x[j-1] 
                if is_American:
                    current_S = Smin + (m - j) * dS
                    if Callput == 'Call':
                        grid[j,i] = max(grid[j,i], current_S - K)
                    else:
                        grid[j,i] = max(grid[j,i], K - current_S)

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
            grid[_,-1] = max(0 , Smin + (m - _ ) * dS - K)
    else:
        for k in range(n+1):
            grid[0, k] = 0
            grid[m , k] = K 
        for _ in range(m+1):
            grid[_,-1] = max(0 , K - (Smin + (m - _ ) * dS))

    def coefficients(j):
        aj = 0.5 * sigma**2 * j**2 * dt + 0.5 * (r - q) * j * dt
        bj = 1 - sigma**2 * j**2 * dt - r * dt
        cj = 0.5 * sigma**2 * j**2 * dt - 0.5 * (r - q) * j * dt
        return cj , bj , aj 
    
    for i in range (n-1 ,-1, -1):
        for j in range(1 , m):
            node = m - j
            c , b , a = coefficients(node)
            grid[j, i] = a * grid[j-1, i+1] + b * grid[j, i+1] + c * grid[j+1, i+1]
            
            if is_American:
                current_S = Smin + (m - j) * dS
                if Callput == 'Call':
                    grid[j,i] = max(grid[j,i], current_S - K)
                else:
                    grid[j,i] = max(grid[j,i], K - current_S)

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
# 模組 2：標準選擇權 (CRR, BBS, BSM)
# ==========================================
def black_scholes_price(S0, K, T, r, q, sigma, option_type):
    d1 = (np.log(S0/K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'CALL':
        return S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * np.exp(-q * T) * norm.cdf(-d1)

# 為了保持程式碼簡潔，這裡省略了 Lookback 和 Rainbow 的完整函數，
# 概念上你只需將腳本中的函數貼到這裡即可。

# ==========================================
# 側邊欄導航 (Sidebar Navigation)
# ==========================================
st.sidebar.title("Option Pricing Engine")
page = st.sidebar.selectbox("Choose a Model:", [
    "1. Finite Difference Method (FDM)",
    "2. Standard Options (CRR, BBS, MCS)",
    "3. Structured Payoff Option",
    "4. Lookback Options",
    "5. Rainbow Options"
])

# ==========================================
# 頁面 1：Finite Difference Method
# ==========================================
if page == "1. Finite Difference Method (FDM)":
    st.header("Finite Difference Method (Implicit & Explicit)")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        s0 = st.number_input("S0", value=50)
        k = st.number_input("K", value=50)
        T = st.number_input("T (years)", value=0.5)
    with col2:
        r = st.number_input("r (%)", value=10.0) / 100
        q = st.number_input("q (%)", value=5.0) / 100
        sigma = st.number_input("sigma (%)", value=40.0) / 100
    with col3:
        smin = st.number_input("Smin", value=0.0)
        smax = st.number_input("Smax", value=100.0)
        m = st.number_input("m (Price steps)", value=100, step=10)
        n = st.number_input("n (Time steps)", value=100, step=10)

    if st.button("Calculate FDM Prices"):
        with st.spinner('Calculating...'):
            df = pd.DataFrame(columns=["implicit" , "explicit"] , index= ["EC" , "EP" , "AC" , "AP"])
            
            # Implicit
            df.loc["EC", "implicit"] = round(implicit_FDM(s0,k,r,q,sigma,T,smin,smax,m,n,False, "Call"), 4)
            df.loc["EP", "implicit"] = round(implicit_FDM(s0,k,r,q,sigma,T,smin,smax,m,n,False, "Put"), 4)
            df.loc["AC", "implicit"] = round(implicit_FDM(s0,k,r,q,sigma,T,smin,smax,m,n,True, "Call"), 4)
            df.loc["AP", "implicit"] = round(implicit_FDM(s0,k,r,q,sigma,T,smin,smax,m,n,True, "Put"), 4)

            # Explicit
            df.loc["EC", "explicit"] = floor_decimal(explicit_FDM(s0,k,r,q,sigma,T,smin,smax,m,n,False, "Call"), 4)
            df.loc["EP", "explicit"] = floor_decimal(explicit_FDM(s0,k,r,q,sigma,T,smin,smax,m,n,False, "Put"), 4)
            df.loc["AC", "explicit"] = floor_decimal(explicit_FDM(s0,k,r,q,sigma,T,smin,smax,m,n,True, "Call"), 4)
            df.loc["AP", "explicit"] = floor_decimal(explicit_FDM(s0,k,r,q,sigma,T,smin,smax,m,n,True, "Put"), 4)

            st.dataframe(df, use_container_width=True)

# ==========================================
# 頁面 2：Standard Options
# ==========================================
elif page == "2. Standard Options (CRR, BBS, MCS)":
    st.header("Standard Option Pricing")
    
    col1, col2 = st.columns(2)
    with col1:
        s0 = st.number_input("S0", value=100.0)
        K = st.number_input("Strike Price (K)", value=100.0)
        r = st.number_input("r (%)", value=5.0) / 100
        q = st.number_input("q (%)", value=0.0) / 100
    with col2:
        sigma = st.number_input("sigma (%)", value=20.0) / 100
        T = st.number_input("T (years)", value=1.0)
        n = st.number_input("Tree Steps (n)", value=100, step=10)
        
    option_type = st.radio("Option Type", ["CALL", "PUT"])
    
    if st.button("Calculate BSM & Plot Convergence"):
        bsm_price = black_scholes_price(s0, K, T, r, q, sigma, option_type)
        st.success(f"BSM Price: {bsm_price:.4f}")
        
        # Plotting CRR vs BBS convergence logic
        st.subheader("Convergence Plot (CRR vs BBS)")
        with st.spinner("Generating plot..."):
            # 這裡簡化呼叫邏輯，請將你的 crr_2_dimensional 和 bbs 函式補上
            fig, ax = plt.subplots(figsize=(10, 5))
            # ax.plot(steps, crr_results, 'go--', label='CRR', linewidth=1, markersize=1.5)
            # ax.plot(steps, bbs_results, 'ro--', label='BBS', linewidth=1, markersize=1.5)
            ax.set_xlabel('Number of steps (n)')
            ax.set_ylabel('Option Price')
            ax.set_title('Option Price by CRR and BBS')
            ax.legend()
            st.pyplot(fig)

# ==========================================
# 頁面 3：Structured Payoff Option
# ==========================================
elif page == "3. Structured Payoff Option":
    st.header("Structured Payoff Option (K1, K2, K3, K4)")
    st.markdown("""
    The option payoff is structured across four strikes. 
    Can be decomposed into standard long/short vanilla positions.
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        s0 = st.number_input("S0", value=100.0)
        T = st.number_input("T (years)", value=1.0)
    with col2:
        r = st.number_input("r (%)", value=5.0) / 100
        q = st.number_input("q (%)", value=0.0) / 100
        sigma = st.number_input("sigma (%)", value=20.0) / 100
    with col3:
        K1 = st.number_input("K1", value=80.0)
        K2 = st.number_input("K2", value=90.0)
        K3 = st.number_input("K3", value=110.0)
        K4 = st.number_input("K4", value=120.0)
        
    if st.button("Calculate Structured Price"):
        # Validate strikes
        if not (K1 < K2 < K3 < K4):
            st.error("Please ensure K1 < K2 < K3 < K4")
        else:
            price_1 = black_scholes_price(s0, K1, T, r, q, sigma, "CALL")
            price_2 = black_scholes_price(s0, K2, T, r, q, sigma, "CALL")
            price_3 = black_scholes_price(s0, K3, T, r, q, sigma, "CALL")
            price_4 = black_scholes_price(s0, K4, T, r, q, sigma, "CALL")
            
            weight = (K2 - K1) / (K4 - K3)
            sum_vanilla = price_1 - price_2 - (weight * price_3) + (weight * price_4)
            
            st.success(f"Analytical Price (Static Replication): {sum_vanilla:.6f}")

# ==========================================
# 頁面 4 & 5 (概念保留區)
# ==========================================
elif page == "5. Rainbow Options":
    st.header("Rainbow Option Pricing (MCS, AV, MM)")
    st.info("輸入參數請使用逗號分隔，例如 S0 輸入 `100, 100, 100`")
    
    s0_input = st.text_input("S0 for n assets", value="100, 100")
    sigma_input = st.text_input("Sigma (%) for n assets", value="20, 20")
    corr_input = st.text_input("Correlation matrix values", value="0.5")
    
    if st.button("Run Monte Carlo"):
        st.write("Executing Cholesky Decomposition and Simulation...")
        # 你的 Rainbow 邏輯放置處
        