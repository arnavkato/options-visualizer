# Options Strategy PnL & Greeks Visualizer

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://options-visualizer.streamlit.app/)

An advanced, interactive web application built with **Streamlit** and **Plotly** that allows traders and quantitative analysts to visualize complex multi-leg options strategies. 

**Live Demo:** [options-visualizer.streamlit.app](https://options-visualizer.streamlit.app/)

## What It Does

This tool evaluates the Profit & Loss (PnL) and risk metrics of any combination of options legs (Calls and Puts) across a user-defined range of underlying asset prices. 

Key capabilities include:
- **Dynamic Leg Management:** Add or remove an unlimited number of option legs on the fly. Configure Type (Call/Put), Strike, Time to Expiration (DTE), Premium paid/received, Implied Volatility (IV%), and position size.
- **Time-Dependent PnL:** Shift the "Evaluation Date" forward in time to visualize how theta decay and changing time horizons affect your strategy before expiration (ideal for calendar and diagonal spreads).
- **Greeks Overlay:** Plotly dual-axis integration allows you to overlay position-scaled Greeks ($\Delta, \Gamma, \Theta, \nu, \rho$) directly on top of your PnL curve.
- **Robust Metrics:** Automatically calculates Net Premium (Credit/Debit), Maximum Profit, Maximum Loss, and exact Breakeven points using mathematical sign-change detection.

---

## The Three Pricing Methods

This visualizer allows users to toggle between three distinct quantitative finance models to price the options. Each model calculates the theoretical value of the options across the underlying price range, which is then used to construct the continuous PnL curve.

### 1. Black-Scholes Model (Analytical)
The standard closed-form solution for pricing European options. 
* **How it works here:** The app uses continuous-time mathematical formulas based on the normal distribution (`scipy.stats.norm`) to instantly calculate the option's theoretical price. 
* **Greeks Engine:** Because Black-Scholes is analytical, it is the exclusive method used in this app to calculate the continuous Greeks (the partial derivatives of the option price with respect to price, time, volatility, and interest rates).

### 2. Binomial Tree Model (Numerical / Lattice)
A discrete-time model that traces the evolution of the option's key underlying variables over time.
* **How it works here:** The user defines the number of discrete steps (`num_steps`). The app builds a forward-looking binomial lattice of possible stock prices (multiplying by up-factors $u$ and down-factors $d$). It evaluates the intrinsic payoff at expiration, and then works *backwards* through the tree step-by-step, applying risk-neutral probabilities and discounting by the risk-free rate to find the present theoretical value. 
* **Implementation:** The code is heavily vectorized using NumPy broadcasting to simultaneously calculate the tree for all 500+ data points on the X-axis instantly.

### 3. Monte Carlo Simulation (Stochastic)
A computational algorithm that relies on repeated random sampling to obtain numerical results.
* **How it works here:** The app simulates thousands of possible future price paths for the underlying asset (`num_sims`) using Geometric Brownian Motion (GBM). It calculates the terminal payoff of the option at the end of every single random path, and then takes the discounted expected value (the mathematical mean) of all those simulated payoffs to arrive at today's price.
* **Implementation:** Using NumPy's random normal generator, it creates a massive matrix of price multipliers. By vectorizing the cumulative product across the time steps, it processes millions of simulated steps in fractions of a second without Python `for` loops.

---

## ⚙️ How to Run Locally

If you want to run or modify this project on your local machine:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/arnavkato/options-visualizer.git
   cd options-visualizer
2. **Create Virtual Environment**
   
3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
4. **Run the Streamlit server:**
   ```bash
   streamlit run visualizer.py
