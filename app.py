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
# 共用函數 (BSM, Payoff 等)
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

def payoff(S, K, option_type):
    if option_type == 'CALL':
        return np.maximum(S - K, 0)
    else:
        return np.maximum(K - S, 0)

def floor_decimal(number, decimals=2):
    factor = 10 ** decimals
    return math.floor(number * factor) / factor

# ==========================================
# 側邊欄導航 (Sidebar Navigation)
# ==========================================
st.sidebar.title("量化金融選擇權定價引擎")
page = st.sidebar.radio("選擇定價模組:", [
    "1. Finite Difference Method (FDM)",
    "2. Standard Options (BSM, CRR, BBS, MCS)",
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
        s0 = st.number_input("S0 (現價)", value=50.0)
        k = st.number_input("K (履約價)", value=50.0)
        T = st.number_input("T (年期)", value=0.5)
    with col2:
        r = st.number_input("r (%)", value=10.0) / 100
        q = st.number_input("q (%)", value=5.0) / 100
        sigma = st.number_input("sigma (%)", value=40.0) / 100
    with col3:
        smin = st.number_input("Smin", value=0.0)
        smax = st.number_input("Smax", value=100.0)
        m = st.number_input("m (Price steps)", value=100, step=10)
        n = st.number_input("n (Time steps)", value=100, step=10)

    if st.button("計算 FDM 價格"):
        with st.spinner('計算中...'):
            # 由於字數限制，請將您原本的 implicit_FDM 和 explicit_FDM 函數內容放在這個 if block 上方或獨立檔案 import
            st.warning("請確保 implicit_FDM 與 explicit_FDM 函數已宣告於腳本中。")
            # df.loc["EC", "implicit"] = implicit_FDM(...) 

# ==========================================
# 頁面 2：Standard Options (整合版)
# ==========================================
elif page == "2. Standard Options (BSM, CRR, BBS, MCS)":
    st.header("標準選擇權定價 (Standard Options)")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        s0 = st.number_input("S0 (現價)", value=100.0)
        K = st.number_input("K (履約價)", value=100.0)
        T = st.number_input("T (年期)", value=1.0)
    with col2:
        r = st.number_input("r (%)", value=5.0) / 100
        q = st.number_input("q (%)", value=0.0) / 100
        sigma = st.number_input("sigma (%)", value=20.0) / 100
    with col3:
        n = st.slider("Tree Steps (n)", 10, 500, 100)
        sim_times = st.slider("MCS 模擬次數", 1000, 50000, 10000, step=1000)
        reps = st.slider("MCS 重複次數", 1, 50, 20)

    option_type = st.radio("選擇權類型 (Option Type)", ["CALL", "PUT"])
    option_style = st.radio("選擇權風格 (Option Style)", ["European", "American"])
    style_code = 'E' if option_style == "European" else 'A'

    if st.button("執行定價分析"):
        with st.spinner('正在計算所有模型...'):
            ans_dict = {}

            # 1. BSM (僅適用 European)
            if style_code == 'E':
                if option_type == 'CALL':
                    ans_dict["BSM"] = black_scholes_call_price(s0, K, T, r, q, sigma)
                else:
                    ans_dict["BSM"] = black_scholes_put_price(s0, K, T, r, q, sigma)
            else:
                ans_dict["BSM"] = "N/A (僅適用 European)"

            # 2. MCS (僅適用 European)
            if style_code == 'E':
                mcs_results = []
                for _ in range(reps):
                    drift = (r - q - 0.5 * (sigma**2)) * T
                    diffusion = sigma * np.sqrt(T) * np.random.normal(size=sim_times)
                    S_T_arr = s0 * np.exp(drift + diffusion)
                    payoffs = payoff(S_T_arr, K, option_type)
                    mcs_results.append(np.mean(np.exp(-r * T) * payoffs))
