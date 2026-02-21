import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
import pandas as pd

def black_scholes_with_greeks(S, K, T, r, sigma, right="C"):
    S = np.asarray(S)
    if T <= 0:
        price = np.maximum(S - K, 0) if right == "C" else np.maximum(K - S, 0)
        return price, np.zeros_like(S), np.zeros_like(S), np.zeros_like(S), np.zeros_like(S), np.zeros_like(S)
        
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    pdf_d1 = norm.pdf(d1)
    cdf_d1 = norm.cdf(d1)
    cdf_d2 = norm.cdf(d2)
    gamma = pdf_d1 / (S * sigma * np.sqrt(T))
    vega = (S * pdf_d1 * np.sqrt(T)) / 100
    
    if right == "C":
        price = S * cdf_d1 - K * np.exp(-r * T) * cdf_d2
        delta = cdf_d1
        theta = (- (S * pdf_d1 * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * cdf_d2) / 365
        rho = (K * T * np.exp(-r * T) * cdf_d2) / 100
    else:
        cdf_neg_d1 = norm.cdf(-d1)
        cdf_neg_d2 = norm.cdf(-d2)
        price = K * np.exp(-r * T) * cdf_neg_d2 - S * cdf_neg_d1
        delta = cdf_d1 - 1
        theta = (- (S * pdf_d1 * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * cdf_neg_d2) / 365
        rho = (-K * T * np.exp(-r * T) * cdf_neg_d2) / 100
        
    return price, delta, gamma, theta, vega, rho

def monte_carlo(S, K, T, r, sigma, option_type="call", num_simulations=10000):
    S = np.asarray(S)
    if T <= 0:
        price = np.maximum(S - K, 0) if option_type == "call" else np.maximum(K - S, 0)
        return price, np.zeros_like(S), np.zeros_like(S), np.zeros_like(S), np.zeros_like(S), np.zeros_like(S)

    np.random.seed(0)
    dt = max(T / 252, 1e-5)
    steps = max(int(T / dt), 1)
    random_normals = np.random.normal(size=(num_simulations, steps))
    multipliers = np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * random_normals)
    final_prices = S[:, np.newaxis] * np.prod(multipliers, axis=1)
    payoffs = np.maximum(final_prices - K, 0) if option_type == "call" else np.maximum(K - final_prices, 0)
    price = np.exp(-r * T) * np.mean(payoffs, axis=1)
    return price, np.zeros_like(S), np.zeros_like(S), np.zeros_like(S), np.zeros_like(S), np.zeros_like(S)

def binomial_tree(S, K, T, r, sigma, option_type="call", num_steps=100):
    S = np.asarray(S)
    if T <= 0:
        price = np.maximum(S - K, 0) if option_type == "call" else np.maximum(K - S, 0)
        return price, np.zeros_like(S), np.zeros_like(S), np.zeros_like(S), np.zeros_like(S), np.zeros_like(S)

    dt = T / num_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    discount = np.exp(-r * dt)
    j_indices = np.arange(num_steps + 1)
    multipliers = (u ** (num_steps - j_indices)) * (d ** j_indices)
    stock_prices_T = S[:, np.newaxis] * multipliers
    option_values = np.maximum(stock_prices_T - K, 0) if option_type == "call" else np.maximum(K - stock_prices_T, 0)
    for i in range(num_steps - 1, -1, -1):
        option_values = discount * (p * option_values[:, :-1] + (1 - p) * option_values[:, 1:])
    price = option_values[:, 0]
    return price, np.zeros_like(S), np.zeros_like(S), np.zeros_like(S), np.zeros_like(S), np.zeros_like(S)

def price_option_with_greeks(model, S, K, T, r, sigma, right="C", **kwargs):
    if model == "black_scholes":
        return black_scholes_with_greeks(S, K, T, r, sigma, right)
    elif model == "binomial":
        return binomial_tree(S, K, T, r, sigma, "call" if right=="C" else "put", num_steps=kwargs.get("num_steps", 100))
    elif model == "monte_carlo":
        return monte_carlo(S, K, T, r, sigma, "call" if right=="C" else "put", num_simulations=kwargs.get("num_sims", 10000))
    return black_scholes_with_greeks(S, K, T, r, sigma, right)

def find_breakevens(price_range, strategy_pnl, max_count=2):
    abs_pnl = np.abs(strategy_pnl)
    closest_indices = np.argsort(abs_pnl)[:max_count]
    return np.sort(np.round(price_range[closest_indices], 1))

class OptionAnalytics:
    pricing_model = "black_scholes"
    
    def __init__(self, S, K, T, r, sigma, right, premium, num_contracts=1):
        self.S, self.K, self.T = S, K, T
        self.r, self.sigma = r, sigma
        self.right = right.upper()
        self.premium = premium
        self.num_contracts = num_contracts

    def calculate_metrics(self, S_eval, eval_time, model_params=None):
        remaining_T = max(self.T - eval_time, 0)
        model = model_params["model"] if model_params else self.pricing_model
        kwargs = model_params.get("kwargs", {})
        
        price, delta, gamma, theta, vega, rho = price_option_with_greeks(
            model, S_eval, self.K, remaining_T, self.r, self.sigma, self.right, **kwargs
        )
        
        mult = 100 * self.num_contracts
        pnl = mult * (price - self.premium)
        return pnl, delta * mult, gamma * mult, theta * mult, vega * mult, rho * mult

def main():
    st.set_page_config(layout="wide")
    st.title("Options Strategy PnL & Greeks Visualizer")

    if 'legs_data' not in st.session_state:
        st.session_state.legs_data = [
            {"id": 0, "right": "C", "K": 690.0, "sigma_pct": 10.0, "T_days": 3, "premium": 2.0, "contracts": -1},
            {"id": 1, "right": "C", "K": 690.0, "sigma_pct": 10.0, "T_days": 7, "premium": 3.5, "contracts": 1}
        ]
        st.session_state.next_id = 2

    def add_leg():
        st.session_state.legs_data.append({
            "id": st.session_state.next_id, "right": "C", "K": 690.0, 
            "sigma_pct": 10.0, "T_days": 7, "premium": 1.0, "contracts": 1
        })
        st.session_state.next_id += 1

    def remove_leg(leg_id):
        st.session_state.legs_data = [leg for leg in st.session_state.legs_data if leg["id"] != leg_id]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        underlying = st.number_input("Spot Price", value=689.32)
        RISK_FREE_RATE = st.number_input("Risk-Free Rate", value=0.0361, format="%.4f")
    with col2:
        global_model = st.selectbox("Pricing Model", ["black_scholes", "binomial", "monte_carlo"])
        model_params = {"model": global_model}
        if global_model == "binomial":
            model_params["kwargs"] = {"num_steps": st.slider("Binomial Steps", 50, 500, 100)}
        elif global_model == "monte_carlo":
            model_params["kwargs"] = {"num_sims": st.slider("MC Sims", 1000, 20000, 5000)}
    with col3:
        eval_days = st.slider("Days from Now", 0, 90, 0)
        evaluation_time = eval_days / 365.0
    with col4:
        show_greeks = st.multiselect("Overlay Greeks", ["Delta", "Gamma", "Theta", "Vega", "Rho"])

    price_min = st.number_input("Price Min", value=650.0)
    price_max = st.number_input("Price Max", value=730.0)
    price_range = np.linspace(price_min, price_max, 500)

    st.subheader("Strategy Legs")
    st.button("✚ Add Leg", on_click=add_leg)

    legs = []
    for i, leg_data in enumerate(st.session_state.legs_data):
        leg_id = leg_data["id"]
        with st.expander(f"Leg {i+1} ({leg_data['right']}{leg_data['K']})", expanded=True):
            c1, c2, c3, c4, c5, c6, c7 = st.columns([1,1,1,1,1,1,0.5])
            
            with c1: leg_data["right"] = st.selectbox("Type", ["C", "P"], index=0 if leg_data["right"]=="C" else 1, key=f"r_{leg_id}")
            with c2: leg_data["K"] = st.number_input("Strike", value=float(leg_data["K"]), key=f"k_{leg_id}")
            with c3: leg_data["sigma_pct"] = st.number_input("IV %", value=float(leg_data["sigma_pct"]), format="%.1f", key=f"iv_{leg_id}")
            with c4: leg_data["T_days"] = st.number_input("DTE", value=int(leg_data["T_days"]), min_value=1, key=f"dte_{leg_id}")
            with c5: leg_data["premium"] = st.number_input("Premium", value=float(leg_data["premium"]), key=f"p_{leg_id}")
            with c6: leg_data["contracts"] = st.number_input("Contracts", value=int(leg_data["contracts"]), key=f"c_{leg_id}")
            with c7: 
                st.write("")
                st.write("")
                st.button("✕", key=f"del_{leg_id}", on_click=remove_leg, args=(leg_id,))
            
            T = leg_data["T_days"] / 365.0
            sigma = leg_data["sigma_pct"] / 100.0
            legs.append(OptionAnalytics(underlying, leg_data["K"], T, RISK_FREE_RATE, sigma, leg_data["right"], leg_data["premium"], leg_data["contracts"]))

    if legs:
        st.subheader("Visualization")
        
        with st.spinner("Computing..."):
            total_pnl = np.zeros_like(price_range)
            total_greeks = {g: np.zeros_like(price_range) for g in ["Delta", "Gamma", "Theta", "Vega", "Rho"]}
            leg_pnls = []
            
            for leg in legs:
                pnl, d, g, t, v, r = leg.calculate_metrics(price_range, evaluation_time, model_params)
                leg_pnls.append(pnl)
                total_pnl += pnl
                total_greeks["Delta"] += d
                total_greeks["Gamma"] += g
                total_greeks["Theta"] += t
                total_greeks["Vega"] += v
                total_greeks["Rho"] += r
        
        max_loss = round(np.min(total_pnl), 2)
        max_profit = round(np.max(total_pnl), 2)
        breakevens = find_breakevens(price_range, total_pnl)
        breakeven_text = " | ".join(map(str, breakevens)) if len(breakevens) else "None"
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Max Loss", f"${max_loss}")
        c2.metric("Max Profit", f"${max_profit}")
        c3.metric("Breakevens", breakeven_text)
        net_premium = sum(leg.num_contracts * leg.premium * 100 for leg in legs)
        c4.metric("Net Premium", f"${round(net_premium,2)}", f"{'Credit' if net_premium>0 else 'Debit'}")
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(go.Scatter(x=price_range, y=total_pnl, name='Total PnL', line=dict(color='blue', width=4)), secondary_y=False)
        
        colors = ['red','green','orange','purple','brown','pink']
        for i, (leg, pnl, color) in enumerate(zip(legs, leg_pnls, colors)):
            fig.add_trace(go.Scatter(x=price_range, y=pnl, name=f'L{i+1}: {leg.right}{leg.K}', 
                                line=dict(color=color, dash='dash'), visible='legendonly'), secondary_y=False)
        
        greek_colors = {"Delta": "cyan", "Gamma": "magenta", "Theta": "yellow", "Vega": "lightgreen", "Rho": "lightcoral"}
        if global_model != "black_scholes" and show_greeks:
            st.warning("⚠️ Greeks are only calculated analytically via Black-Scholes.")
        else:
            for greek in show_greeks:
                fig.add_trace(go.Scatter(x=price_range, y=total_greeks[greek], name=f'Total {greek}', 
                                    line=dict(color=greek_colors[greek], width=2)), secondary_y=True)

        fig.add_hline(y=0, line_dash="dash", line_color="black")
        fig.add_vline(x=underlying, line_dash="dot", line_color="gray", annotation_text=f"Spot ${underlying}")
        
        fig.update_layout(title=f"PnL & Greeks @ {eval_days}d", hovermode='x unified', template="plotly_dark", height=600)
        fig.update_yaxes(title_text="PnL ($)", secondary_y=False)
        fig.update_yaxes(title_text="Greeks Value", secondary_y=True, showgrid=False)
        
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Add at least one leg!")

if __name__ == "__main__":
    main()
