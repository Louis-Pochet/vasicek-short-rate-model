# Short-Rate Modeling and Zero-Coupon Bond Pricing under the Vasicek Framework

This project implements the Vasicek short-rate model within an affine term structure framework.  
It combines exact stochastic simulation, analytical bond pricing, Monte Carlo validation, and parameter calibration in a structured quantitative finance setting.

The objective is to provide a clean numerical implementation of a classical interest rate model and to verify the consistency between theoretical formulas and simulation-based methods.

---

## Model Specification

Under the risk-neutral measure, the short rate follows:

dr_t = kappa (theta - r_t) dt + sigma dW_t

where:

- kappa : mean-reversion speed  
- theta : long-run mean level  
- sigma : volatility  
- r_t   : short rate  

The affine structure implies a closed-form expression for zero-coupon bond prices.

---

## Methodology

### Simulation

The Vasicek process is simulated using its exact discrete transition.  
This preserves the Gaussian conditional distribution and avoids Euler discretization bias.

### Bond Pricing

Zero-coupon bond prices are computed in two ways:

- Analytical closed-form solution  
- Monte Carlo estimation based on  

P(0,T) = E[ exp( - ∫ r_t dt ) ]

Monte Carlo results converge toward the analytical solution, validating the numerical implementation.

### Yield Curve

The implied term structure is obtained from model bond prices:

y(0,T) = - (1/T) log P(0,T)

Parameter sensitivity, notably volatility, is analyzed through shifts in the yield curve.

### Calibration

Using the discrete representation

r_{t+Δ} = a + b r_t + epsilon_t

parameters are estimated via OLS.  
Calibration on simulated data illustrates the econometric properties of mean-reverting processes.

---

## Results

- Monte Carlo bond pricing converges to the analytical solution as the number of paths increases.  
- The long-run mean (theta) and volatility (sigma) are robustly estimated.  
- The mean-reversion speed (kappa) is more sensitive at high sampling frequency, as the AR(1) coefficient b is close to 1.  

The project focuses on internal consistency and numerical validation rather than empirical market performance.

---

## Code Organization

The implementation is modular:

- `src/models` contains the stochastic rate dynamics.  
- `src/pricing` implements analytical and Monte Carlo bond pricing.  
- `src/calibration` contains OLS parameter estimation.  
- `notebooks` includes numerical experiments and validation.  

---

## Requirements

Install dependencies with:

pip install -r requirements.txt

Main libraries used:

- numpy  
- scipy  
- matplotlib  
- statsmodels  

---

## How to Run

1. Create and activate a virtual environment  
2. Install dependencies  
3. Open `notebooks/01_vasicek_pricing.ipynb`  
4. Run all cells  

---

Louis Pochet  
Quantitative Finance Projects