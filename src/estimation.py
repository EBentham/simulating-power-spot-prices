import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from .parameters import MeanReversionParams, JumpParams


def estimate_mean_reversion_params(prices: pd.Series, dt: float) -> MeanReversionParams:
    """
    Estimates Ornstein-Uhlenbeck (Vasicek) mean-reversion parameters
    (alpha, mu, sigma) for log-prices using OLS.

    Based on the discretized equation:
    dx_t = x_{t} - x_{t-1} = alpha * (mu - x_{t-1}) * dt + sigma * sqrt(dt) * Z_t
    Rearranging for OLS: dx_t = (alpha*mu*dt) + (-alpha*dt)*x_{t-1} + error
                             = intercept + slope * x_{t-1} + error

    Args:
        prices: pandas Series of historical spot prices.
        dt: Time step between observations in years (e.g., 1/252 for daily).

    Returns:
        Estimated MeanReversionParams.

    Raises:
        ValueError: If estimation fails or results are nonsensical.
    """
    if prices.empty or len(prices) < 2:
        raise ValueError("Price series must contain at least two data points.")
    if dt <= 0:
        raise ValueError("Time step dt must be positive.")

    log_prices = np.log(prices)
    dx = log_prices.diff().dropna()
    x_lagged = log_prices.shift(1).dropna()

    # Ensure dx and x_lagged are aligned and have the same length
    aligned_len = min(len(dx), len(x_lagged))
    dx = dx[-aligned_len:]
    x_lagged = x_lagged[-aligned_len:]

    if len(dx) < 2:  # Need at least 2 points for regression
        raise ValueError("Not enough data points remaining after diff and alignment for regression.")

    # Prepare data for OLS (add constant for intercept)
    X_ols = sm.add_constant(x_lagged)
    y_ols = dx

    # Perform OLS regression
    model = sm.OLS(y_ols, X_ols)
    results = model.fit()

    intercept = results.params['const']
    slope = results.params['Price']  # Name matches the input Series name

    # --- Derive parameters ---
    # slope = -alpha * dt  => alpha = -slope / dt
    # intercept = alpha * mu * dt => mu = intercept / (alpha * dt) = intercept / (-slope)

    if slope >= 0:
        print(f"Warning: OLS slope is non-negative ({slope:.4f}), indicating no mean reversion.")
        # Handle this case: perhaps set alpha low or raise error depending on context
        alpha = 0.0  # Or a small positive value
        mu = np.mean(log_prices)  # Fallback to sample mean
    else:
        alpha = -slope / dt
        mu = intercept / (-slope)  # Long term mean of log-price

    # Estimate sigma from residual standard deviation
    residuals_std = np.std(results.resid)
    # Variance of residual = sigma^2 * dt => sigma = residuals_std / sqrt(dt)
    sigma = residuals_std / np.sqrt(dt)

    if alpha < 0 or sigma <= 0:
        raise ValueError(f"Estimated parameters are invalid: alpha={alpha:.4f}, sigma={sigma:.4f}")

    print("\n--- Mean Reversion Estimation Results ---")
    print(results.summary())
    print(f"\nDerived Parameters:")
    print(f"  alpha (Mean Reversion Speed): {alpha:.4f}")
    print(f"  mu (Long-Term Mean Log-Price): {mu:.4f} (exp(mu)={np.exp(mu):.2f})")
    print(f"  sigma (Base Volatility): {sigma:.4f}")

    return MeanReversionParams(alpha=alpha, mu=mu, sigma=sigma)


def estimate_jump_params(prices: pd.Series, dt: float, k: float = 3.0, max_iter: int = 10, tol: float = 1e-5) -> tuple[
    JumpParams, float]:
    """
    Estimates jump diffusion parameters (lambda_j, mu_j, sigma_j) and
    diffusion volatility (sigma) using an iterative filtering approach.

    Args:
        prices: pandas Series of historical spot prices.
        dt: Time step between observations in years.
        k: Threshold multiplier (e.g., 3 standard deviations) to identify jumps.
        max_iter: Maximum iterations for the filter.
        tol: Tolerance for volatility convergence.

    Returns:
        Tuple containing:
            - Estimated JumpParams.
            - Estimated diffusion sigma (volatility excluding jumps).

    Raises:
        ValueError: If estimation fails or data is insufficient.
    """
    if prices.empty or len(prices) < 2:
        raise ValueError("Price series must contain at least two data points.")
    if dt <= 0:
        raise ValueError("Time step dt must be positive.")

    log_returns = np.log(prices / prices.shift(1)).dropna()

    if len(log_returns) < 2:
        raise ValueError("Not enough data points remaining after calculating returns.")

    n = len(log_returns)
    total_time_years = n * dt

    # --- Iterative Filter ---
    sigma_diffusion_prev = np.std(log_returns) / np.sqrt(dt)  # Initial guess
    sigma_diffusion = 0.0

    print("\n--- Jump Diffusion Estimation (Iterative Filter) ---")
    for i in range(max_iter):
        print(f"Iteration {i + 1}: sigma_diffusion_prev = {sigma_diffusion_prev:.5f}")
        threshold = k * sigma_diffusion_prev * np.sqrt(dt)

        # Identify non-jump returns
        non_jump_returns = log_returns[np.abs(log_returns) < threshold]

        if len(non_jump_returns) < 2:
            print("Warning: Very few non-jump returns identified. Check threshold or data.")
            # Keep previous sigma or handle as error
            sigma_diffusion = sigma_diffusion_prev
            break

            # Re-estimate diffusion volatility from non-jump returns
        sigma_diffusion = np.std(non_jump_returns) / np.sqrt(dt)

        # Check for convergence
        if abs(sigma_diffusion - sigma_diffusion_prev) < tol:
            print(f"Converged after {i + 1} iterations.")
            break
        sigma_diffusion_prev = sigma_diffusion
    else:  # Loop finished without break (no convergence)
        print(f"Warning: Jump filter did not converge after {max_iter} iterations.")

    if sigma_diffusion <= 0:
        raise ValueError(f"Estimated diffusion sigma is non-positive: {sigma_diffusion:.4f}")

    # --- Estimate Jump Parameters ---
    final_threshold = k * sigma_diffusion * np.sqrt(dt)
    jump_returns = log_returns[np.abs(log_returns) >= final_threshold]
    num_jumps = len(jump_returns)

    if num_jumps == 0:
        print("No jumps detected with the current threshold.")
        lambda_j = 0.0
        mu_j = 0.0
        sigma_j = 0.0
    else:
        lambda_j = num_jumps / total_time_years  # Avg jumps per year
        mu_j = np.mean(jump_returns)  # Mean jump size (log %)
        sigma_j = np.std(jump_returns)  # Std dev of jump size (log %)
        # Ensure sigma_j is non-negative (can be 0 if only one jump)
        sigma_j = max(sigma_j, 0.0)

    print(f"\nFinal Diffusion Sigma: {sigma_diffusion:.4f}")
    print(f"Number of Jumps Detected: {num_jumps} (out of {n} returns)")
    print(f"Final Threshold (log-return): {final_threshold:.5f}")
    print(f"Estimated lambda_j (Jump Intensity / year): {lambda_j:.4f}")
    print(f"Estimated mu_j (Mean Jump Size Log%): {mu_j:.4f}")
    print(f"Estimated sigma_j (Jump Volatility Log%): {sigma_j:.4f}")

    jump_params = JumpParams(lambda_j=lambda_j, mu_j=mu_j, sigma_j=sigma_j)
    return jump_params, sigma_diffusion