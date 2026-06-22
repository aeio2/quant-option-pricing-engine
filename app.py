import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import math
import warnings

warnings.filterwarnings("ignore")

# 設定網頁基本資訊
st.set_page_config(page_title="Quantitative Finance Pricing Engine", layout="wide")

# ==========================================
# 共用函數區 (BSM, Payoff 等)
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

def payoff(S, K, option_type):
    if option_type == 'CALL':
        return np.maximum(S - K, 0)
    else:
        return np.maximum(K - S, 0)

def floor_decimal(number, decimals=2):
    factor = 10 ** decimals
    return math.floor(number * factor) / factor

# ==========================================
# 模組 1：FDM (Finite Difference Method) 的函數
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
            
            df.loc["EC", "implicit"] = round(implicit_FDM(s0,k,r,q,sigma,T,smin,smax,m,n,False, "Call"), 4)
            df.loc["EP", "implicit"] = round(implicit_FDM(s0,k,r,q,sigma,T,smin,smax,m,n,False, "Put"), 4)
            df.loc["AC", "implicit"] = round(implicit_FDM(s0,k,r,q,sigma,T,smin,smax,m,n,True, "Call"), 4)
            df.loc["AP", "implicit"] = round(implicit_FDM(s0,k,r,q,sigma,T,smin,smax,m,n,True, "Put"), 4)

            df.loc["EC", "explicit"] = floor_decimal(explicit_FDM(s0,k,r,q,sigma,T,smin,smax,m,n,False, "Call"), 4)
            df.loc["EP", "explicit"] = floor_decimal(explicit_FDM(s0,k,r,q,sigma,T,smin,smax,m,n,False, "Put"), 4)
            df.loc["AC", "explicit"] = floor_decimal(explicit_FDM(s0,k,r,q,sigma,T,smin,smax,m,n,True, "Call"), 4)
            df.loc["AP", "explicit"] = floor_decimal(explicit_FDM(s0,k,r,q,sigma,T,smin,smax,m,n,True, "Put"), 4)

            st.dataframe(df, use_container_width=True)

# ==========================================
# 頁面 2：Standard Options (CRR, BBS, MCS)
# ==========================================
elif page == "2. Standard Options (CRR, BBS, MCS)":
    st.header("Standard Option Pricing")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        s0 = st.number_input("S0", value=100.0)
        K = st.number_input("Strike Price (K)", value=100.0)
        T = st.number_input("T (years)", value=1.0)
    with col2:
        r = st.number_input("r (%)", value=5.0) / 100
        q = st.number_input("q (%)", value=0.0) / 100
        sigma = st.number_input("sigma (%)", value=20.0) / 100
    with col3:
        n = st.slider("Tree Steps (n)", 10, 500, 100)
        sim_times = st.slider("MCS 模擬次數", 1000, 50000, 10000, step=1000)
        reps = st.slider("MCS 重複次數", 1, 50, 20)

    col4, col5 = st.columns(2)
    with col4:
        option_type = st.radio("Option Type", ["CALL", "PUT"])
    with col5:
        option_style = st.radio("Option Style", ["European", "American"])

    style_code = 'E' if option_style == "European" else 'A'

    if st.button("Calculate Prices"):
        with st.spinner('Calculating...'):
            ans_dict = {}

            # 1. BSM (European Only)
            if style_code == 'E':
                ans_dict["BSM"] = black_scholes_price(s0, K, T, r, q, sigma, option_type)
            else:
                ans_dict["BSM"] = "N/A (European Only)"

            # 2. Monte Carlo (European Only for standard MCS)
            if style_code == 'E':
                mcs_results = []
                for _ in range(reps):
                    drift = (r - q - 0.5 * (sigma**2)) * T
                    diffusion = sigma * np.sqrt(T) * np.random.normal(size=sim_times)
                    S_T_arr = s0 * np.exp(drift + diffusion)
                    payoffs = payoff(S_T_arr, K, option_type)
                    mcs_results.append(np.mean(np.exp(-r * T) * payoffs))
                
                ans_dict["MCS (Mean)"] = np.mean(mcs_results)
                ci_gap = 2 * np.std(mcs_results, ddof=1)
                ans_dict["MCS 95% CI"] = f"[{np.mean(mcs_results)-ci_gap:.4f}, {np.mean(mcs_results)+ci_gap:.4f}]"
            else:
                ans_dict["MCS (Mean)"] = "N/A"
                ans_dict["MCS 95% CI"] = "N/A"

            # Tree parameters
            delta_t = T / n 
            u = np.exp(sigma * np.sqrt(delta_t))
            d = np.exp(-1 * sigma * np.sqrt(delta_t)) 
            p = (np.exp((r - q ) * delta_t) - d ) / (u - d) 

            # 3. CRR (1D Array - Bonus 1)
            bonus1 = np.zeros(n+1) 
            for j in range(n+1): 
                bonus1[j] = payoff(s0 * (u ** (n-j)) * (d ** j), K, option_type) 
            
            for k_step in range(n , 0, -1):
                for i in range(k_step):
                    hold = np.exp(-r * delta_t) * (p * bonus1[i] + (1-p) * bonus1[i+1])
                    if style_code == 'A':
                        exe = payoff(s0 * (u ** (k_step-1-i)) * (d ** i), K, option_type)
                        bonus1[i] = max(hold, exe)
                    else:
                        bonus1[i] = hold
            ans_dict["CRR (1D)"] = bonus1[0]

            # 4. BBS (Bonus 2)
            bbs_stock = np.zeros((n + 1, n + 1))
            bbs_opt = np.zeros((n + 1, n + 1))
            for k_step in range(n + 1):
                for j in range(k_step + 1):
                    bbs_stock[j, k_step] = s0 * (u ** (k_step - j)) * (d ** j)

            for j in range(n):
                s_node = bbs_stock[j, n-1]
                bs_val = black_scholes_price(s_node, K, delta_t, r, q, sigma, option_type)
                if style_code == 'A': 
                    bbs_opt[j, n-1] = max(bs_val, payoff(s_node, K, option_type))
                else: 
                    bbs_opt[j, n-1] = bs_val

            for k_step in range(n - 2, -1, -1):
                for j in range(k_step + 1):
                    discounted_exp = np.exp(-r * delta_t) * (p * bbs_opt[j, k_step+1] + (1 - p) * bbs_opt[j+1, k_step+1])
                    if style_code == 'A':
                        bbs_opt[j, k_step] = max(discounted_exp, payoff(bbs_stock[j, k_step], K, option_type))
                    else:
                        bbs_opt[j, k_step] = discounted_exp
            ans_dict["BBS"] = bbs_opt[0, 0]

            st.write("### Pricing Results")
            st.table(pd.DataFrame.from_dict(ans_dict, orient='index', columns=['Price']))


# ==========================================
# 頁面 3：Structured Payoff Option
# ==========================================
elif page == "3. Structured Payoff Option":
    st.header("Structured Payoff Option (K1, K2, K3, K4)")
    st.markdown("The option payoff is structured across four strikes. Can be decomposed into standard long/short vanilla positions.")
    
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
# 頁面 4：Lookback Options
# ==========================================
elif page == "4. Lookback Options":
    st.header("Lookback Option Pricing")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st_val = st.number_input("St (Current Price)", value=50.0)
        smax_t = st.number_input("S_max_t (Historical Max)", value=50.0)
        T_minus_t = st.number_input("T - t (Remaining Time)", value=0.25)
    with col2:
        r = st.number_input("r (%)", value=10.0) / 100
        q = st.number_input("q (%)", value=0.0) / 100
        sigma = st.number_input("sigma (%)", value=40.0) / 100
    with col3:
        n = st.slider("Tree Steps (n)", 5, 100, 10)
        sims = st.slider("MCS Simulations", 1000, 20000, 10000)
        reps = st.slider("MCS Repetitions", 1, 50, 20)

    is_american = st.checkbox("American Style?", value=False)

    if st.button("Calculate Lookback Price"):
        with st.spinner('Calculating...'):
            
            # 1. Tree Method (Bonus 2 logic - Fast 1D Array Backward Induction)
            dt = T_minus_t / n
            u = np.exp(sigma * np.sqrt(dt))
            d = 1.0 / u
            mu = np.exp((r - q) * dt)  
            r_disc = np.exp(r * dt)      
            q_prob = (mu * u - 1.0) / (mu * (u - d))
            
            # Using max to avoid log domain errors if st_val > smax_t somehow
            k0 = int(round(np.log(max(smax_t / st_val, 1.0)) / np.log(u)))
            max_k = n + k0
            V = np.array([u**k - 1.0 for k in range(max_k + 1)])
            
            for j in range(n - 1, -1, -1):
                V_new = np.zeros(j + k0 + 1)
                for k in range(j + k0 + 1):
                    k_up = max(k - 1, 0)
                    k_down = k + 1
                    hold = (q_prob * V[k_up] + (1 - q_prob) * V[k_down]) * mu / r_disc
                    
                    if is_american:
                        V_new[k] = max(hold, u**k - 1.0)
                    else:
                        V_new[k] = hold
                V = V_new
                
            tree_price = st_val * V[k0]

            # 2. MCS Method
            mcs_prices = []
            drift = (r - q - 0.5 * sigma**2) * dt
            vol = sigma * np.sqrt(dt)

            for _ in range(reps):
                S_path = np.zeros((sims, n + 1))
                S_path[:, 0] = st_val
                
                Z = np.random.normal(size=(sims, n))
                for i in range(1, n + 1):
                    S_path[:, i] = S_path[:, i-1] * np.exp(drift + vol * Z[:, i-1])
                
                max_S = np.maximum(smax_t, np.max(S_path, axis=1))
                payoffs = np.maximum(max_S - S_path[:, -1], 0)
                mcs_prices.append(np.mean(np.exp(-r * T_minus_t) * payoffs))

            st.write("### Results")
            st.write(f"**Tree Method (Bonus 2 API):** {tree_price:.6f}")
            
            ci_gap = 2 * np.std(mcs_prices, ddof=1)
            st.write(f"**MCS Mean ({reps} reps):** {np.mean(mcs_prices):.6f}")
            st.write(f"**MCS 95% CI:** [{np.mean(mcs_prices) - ci_gap:.6f}, {np.mean(mcs_prices) + ci_gap:.6f}]")


# ==========================================
# 頁面 5：Rainbow Options
# ==========================================
elif page == "5. Rainbow Options":
    st.header("Rainbow Option Pricing (MCS, AV, MM)")
    
    col_top1, col_top2, col_top3 = st.columns(3)
    with col_top1:
        n = st.slider("Number of Assets (n)", 2, 5, 2)
    with col_top2:
        K = st.number_input("Strike Price (K)", value=100.0)
        T = st.number_input("T (years)", value=1.0)
    with col_top3:
        r = st.number_input("r (%)", value=5.0) / 100
        sims = st.slider("MCS Simulations", 1000, 50000, 10000, step=1000)
        reps = st.slider("MCS Repetitions", 1, 50, 10)

    st.subheader("Asset Parameters")
    cols = st.columns(n)
    s0_list, q_list, sigma_list = [], [], []
    
    for i in range(n):
        with cols[i]:
            st.markdown(f"**Asset {i+1}**")
            s0_list.append(st.number_input(f"S0_{i+1}", value=100.0, key=f"s{i}"))
            q_list.append(st.number_input(f"q_{i+1} (%)", value=0.0, key=f"q{i}") / 100)
            sigma_list.append(st.number_input(f"sigma_{i+1} (%)", value=20.0, key=f"sig{i}") / 100)

    st.subheader("Correlation Matrix")
    corr_matrix = np.eye(n)
    idx = 0
    corr_cols = st.columns(3)
    
    # Dynamically generate sliders for every pair of assets
    for i in range(n):
        for j in range(i+1, n):
            with corr_cols[idx % 3]:
                val = st.slider(f"Corr: Asset {i+1} & {j+1}", -1.0, 1.0, 0.5, key=f"corr_{i}_{j}")
                corr_matrix[i, j] = val
                corr_matrix[j, i] = val
                idx += 1

    if st.button("Calculate Rainbow Options"):
        with st.spinner("Simulating..."):
            
            # Construct Covariance Matrix
            cov_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    cov_matrix[i, j] = sigma_list[i] * sigma_list[j] * corr_matrix[i, j] * T

            # Cholesky Decomposition
            try:
                A = np.linalg.cholesky(cov_matrix)
            except np.linalg.LinAlgError:
                st.error("Correlation matrix is not positive-definite! Please adjust the correlation values.")
                st.stop()
            
            s0_arr = np.array(s0_list)
            q_arr = np.array(q_list)
            sig_arr = np.array(sigma_list)
            drift = (r - q_arr - 0.5 * sig_arr**2) * T

            res_basic, res_bonus, res_new = [], [], []

            # Shared Payoff evaluation logic
            def calc_rainbow_price(G_paths):
                St = s0_arr * np.exp(drift + G_paths)
                max_prices = St.max(axis=1)
                payoffs = np.maximum(max_prices - K, 0)
                return np.mean(np.exp(-r * T) * payoffs)

            for _ in range(reps):
                # 1. Basic MCS
                Z = np.random.standard_normal((sims, n))
                G = Z.dot(A.T)
                
                # 2. AV + MM (Bonus 1)
                Z_bonus = Z.copy()
                Z_bonus[sims//2:] = -Z_bonus[:sims//2] # Antithetic Variates
                Z_bonus = (Z_bonus - Z_bonus.mean(axis=0)) / Z_bonus.std(axis=0) # Moment Matching
                G_bonus = Z_bonus.dot(A.T)

                # 3. New Method (Inverse Cholesky - Bonus 2)
                cov_star = np.cov(Z_bonus, rowvar=False)
                B = np.linalg.cholesky(cov_star)
                B_inv = np.linalg.inv(B)
                G_new = Z_bonus.dot(B_inv.T).dot(A.T)

                res_basic.append(calc_rainbow_price(G))
                res_bonus.append(calc_rainbow_price(G_bonus))
                res_new.append(calc_rainbow_price(G_new))

            st.success("Simulation Complete!")
            
            df_res = pd.DataFrame({
                "Method": ["Basic MCS", "Bonus 1 (AV+MM)", "Bonus 2 (Inv Chol)"],
                "Mean Price": [np.mean(res_basic), np.mean(res_bonus), np.mean(res_new)],
                "Std Error": [np.std(res_basic, ddof=1), np.std(res_bonus, ddof=1), np.std(res_new, ddof=1)]
            })
            df_res["95% CI Lower"] = df_res["Mean Price"] - 2 * df_res["Std Error"]
            df_res["95% CI Upper"] = df_res["Mean Price"] + 2 * df_res["Std Error"]
            df_res["CI Range"] = df_res["95% CI Upper"] - df_res["95% CI Lower"]
            
            st.dataframe(df_res.style.format({
                "Mean Price": "{:.4f}", 
                "Std Error": "{:.6f}", 
                "95% CI Lower": "{:.4f}", 
                "95% CI Upper": "{:.4f}", 
                "CI Range": "{:.4f}"
            }), use_container_width=True)
