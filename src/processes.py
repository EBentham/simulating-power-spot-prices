from abc import ABC, abstractmethod

from src.parameters import MeanReversionParams, JumpParams
from src.seasonality import SeasonalityModel


class SpotPriceProcess(ABC):
    """
    Abstract Base Class for different spot price evolution models (Strategy Interface).
    """
    @abstractmethod
    def evolve(self, x_t: float, t: float, dt: float, dw: float, dj_indicator: float, jump_size: float) -> float:
        """
        Evolves the state variable (e.g., log-price) by one time step dt.

        Args:
            x_t: Current value of the state variable (log-price).
            t: Current time.
            dt: Time step size.
            dw: Realization of the Wiener process increment (sqrt(dt)*Z).
            dj_indicator: 1 if a jump occurs in dt, 0 otherwise.
            jump_size: The size of the jump (in log-price terms) if one occurs.

        Returns:
            The value of the state variable at time t + dt.
        """
        pass

# --- 4. Concrete Strategy: Mean-Reverting Jump Process ---

class MeanRevertingJumpProcess(SpotPriceProcess):
    """
    Implements a mean-reverting process with jumps for the log-price (Concrete Strategy).
    Based on combining elements from Chapter 2 (e.g., Eq 2.2 for MR part, Eq 2.14/2.17 for jump part).
    Uses log-price x = ln(S) as the state variable.
    """
    def __init__(self, mr_params: MeanReversionParams, jump_params: JumpParams, seasonality: SeasonalityModel = None):
        self.mr = mr_params
        self.jump = jump_params
        self.seasonality = seasonality

    def evolve(self, x_t: float, t: float, dt: float, dw: float, dj_indicator: float, jump_size: float) -> float:
        """Evolves the log-price using Euler-Maruyama discretization."""

        # Get time-dependent parameters if seasonality model is provided
        if self.seasonality:
            mu_t = self.seasonality.get_long_term_mean(t)
            sigma_t = self.seasonality.get_volatility(t)
        else:
            mu_t = self.mr.mu
            sigma_t = self.mr.sigma

        # Ensure parameters are valid
        if sigma_t <= 0:
            raise ValueError(f"Volatility sigma_t must be positive, but got {sigma_t} at time {t}")
        if self.mr.alpha < 0:
            raise ValueError(f"Mean reversion speed alpha cannot be negative, but got {self.mr.alpha}")


        # Discretization based on dx = [alpha(mu - x) - 0.5*sigma^2] dt + sigma*dW + dJ
        # Note: dw already contains sqrt(dt)
        drift_term = (self.mr.alpha * (mu_t - x_t) - 0.5 * sigma_t**2) * dt
        diffusion_term = sigma_t * dw
        jump_term = dj_indicator * jump_size

        x_t_plus_dt = x_t + drift_term + diffusion_term + jump_term
        return x_t_plus_dt