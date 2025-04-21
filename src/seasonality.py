import math


class SeasonalityModel:
    """
    Models deterministic seasonality in model parameters.
    Simple example: Sinusoidal yearly seasonality for mean and volatility.
    """
    def __init__(self, base_mu: float, mu_amplitude: float, mu_freq: float,
                 base_sigma: float, sigma_amplitude: float, sigma_freq: float):
        self._base_mu = base_mu
        self._mu_amplitude = mu_amplitude
        self._mu_freq = mu_freq # Typically 1 for yearly cycle
        self._base_sigma = base_sigma
        self._sigma_amplitude = sigma_amplitude
        self._sigma_freq = sigma_freq # Typically 1 for yearly cycle

    def get_long_term_mean(self, t: float) -> float:
        """Gets the seasonally adjusted long-term mean (log-price)."""
        # Example: sinusoidal yearly pattern
        seasonal_adjustment = self._mu_amplitude * math.sin(2 * math.pi * self._mu_freq * t)
        return self._base_mu + seasonal_adjustment

    def get_volatility(self, t: float) -> float:
        """Gets the seasonally adjusted volatility."""
        # Example: sinusoidal yearly pattern, ensuring volatility > 0
        seasonal_adjustment = self._sigma_amplitude * math.sin(2 * math.pi * self._sigma_freq * t)
        # Ensure volatility doesn't go below a small positive number or zero
        vol = self._base_sigma + seasonal_adjustment
        return max(vol, 0.01) # Add a floor to prevent negative/zero vol