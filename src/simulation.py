import numpy as np

from src.parameters import SimulationParams
from src.processes import SpotPriceProcess


class SimulationEngine:
    """
    Runs Monte Carlo simulations using a specified price process strategy (Context).
    """

    def __init__(self, process_strategy: SpotPriceProcess, sim_params: SimulationParams):
        if not isinstance(process_strategy, SpotPriceProcess):
            raise TypeError("process_strategy must be an instance of SpotPriceProcess")
        if not isinstance(sim_params, SimulationParams):
            raise TypeError("sim_params must be an instance of SimulationParams")

        self.process = process_strategy
        self.params = sim_params

    def run_simulation(self) -> np.ndarray:
        """
        Performs the Monte Carlo simulation.

        Returns:
            A NumPy array containing the simulated spot price paths.
            Shape: (num_time_steps + 1, num_paths)
        """
        num_steps = int((self.params.T - self.params.t0) / self.params.dt)
        times = np.linspace(self.params.t0, self.params.T, num_steps + 1)

        # Initialize log-price paths array
        log_price_paths = np.zeros((num_steps + 1, self.params.num_paths))
        log_price_paths[0, :] = np.log(self.params.s0)  # Start from log price

        # Pre-generate random numbers for efficiency
        std_norm_random = np.random.standard_normal(size=(num_steps, self.params.num_paths))

        # Jump simulation components
        jump_intensity_dt = 0.0
        if hasattr(self.process, 'jump') and self.process.jump.lambda_j > 0:
            jump_intensity_dt = self.process.jump.lambda_j * self.params.dt
            poisson_random = np.random.poisson(lam=jump_intensity_dt, size=(num_steps, self.params.num_paths))
            jump_sizes_random = np.random.normal(
                loc=self.process.jump.mu_j,
                scale=self.process.jump.sigma_j,
                size=(num_steps, self.params.num_paths)
            )
        else:  # No jumps configured for the process
            poisson_random = np.zeros((num_steps, self.params.num_paths))
            jump_sizes_random = np.zeros((num_steps, self.params.num_paths))

        # --- Simulation Loop ---
        sqrt_dt = np.sqrt(self.params.dt)
        for i in range(num_steps):
            t = times[i]
            dw = std_norm_random[i, :] * sqrt_dt

            # Determine if jump occurs (dj_indicator) and its size
            # Here poisson_random > 0 indicates at least one jump event in dt
            dj_indicator = (poisson_random[i, :] > 0).astype(float)
            # For simplicity, assume at most one jump per small dt, use pre-generated size
            jump_size = jump_sizes_random[i, :]

            # Evolve log-price using the strategy
            log_price_paths[i + 1, :] = self.process.evolve(
                log_price_paths[i, :], t, self.params.dt, dw, dj_indicator, jump_size
            )

        # Convert log-prices back to spot prices
        spot_price_paths = np.exp(log_price_paths)
        return spot_price_paths, times