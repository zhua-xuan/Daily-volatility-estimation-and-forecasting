
# Financial Volatility Forecasting with GARCH and GAS Models

This repository contains the code and analysis for our financial econometrics case study at Vrije Universiteit Amsterdam. We apply advanced time series models to high-frequency stock data of **Cisco Systems, Inc. (CSCO)** to estimate and forecast **volatility** between 2018 and 2025.  

---

## Overview

The goal of this project is to model and forecast stock price volatility using high-frequency data. We utilize various econometric models including GARCH-type and GAS-type models under multiple distributional assumptions. Additionally, we incorporate **realized volatility measures** such as the **Realized Kernel** to account for market microstructure noise.

---

## Dataset

- **Source:** Wharton Research Data Services (WRDS)
- **Stock:** Cisco Systems, Inc. (CSCO)
- **Period:**  
  - In-sample: *2018-01-05 to 2023-01-03* (30M+ observations)  
  - Out-of-sample: *2023-01-04 to 2025-01-03* (12M+ observations)
- **Data Type:** High-frequency trade data (filtered and cleaned)

---

## Methodology

### Volatility Estimation

- **Realized Kernel (RK):** Handles microstructure noise using the Parzen kernel.
- **Realized Variance (RV):** Used for comparative purposes.

### Models

We estimate and compare the following models:

#### GARCH Family
- GARCH(1,1)
- GJR-GARCH
- EGARCH
- Realized GARCH

#### GAS Models (Generalized Autoregressive Score)
- GAS
- Realized GAS

### Distributional Assumptions

- Normal
- Student-t
- Skewed Student-t

### Evaluation Criteria

- **In-sample estimation:** via log-likelihood, AIC, BIC
- **Out-of-sample forecasting:** evaluated by RMSE, MSE, MAE
- **Diebold-Mariano test**: for statistical significance between models

---

## Key Findings

- Realized Kernel better captures true volatility than standard Realized Variance.
- GAS models, especially with Student-t or Skewed Student-t distributions, consistently outperform traditional GARCH models in forecasting accuracy.
- Out-of-sample RMSE for GAS-STD and GAS-SSTD: **0.012184**

---

## References

- Barndorff-Nielsen et al. (2009) – Realized kernels in practice  
- Bollerslev (1986) – GARCH  
- Glosten et al. (1993) – GJR-GARCH  
- Nelson (1991) – EGARCH  
- Creal et al. (2013) – GAS  
- Diebold & Mariano (1995) – Forecast accuracy test

---

## Contact

For questions or collaboration inquiries, please contact jx.zhu@yahoo.com


