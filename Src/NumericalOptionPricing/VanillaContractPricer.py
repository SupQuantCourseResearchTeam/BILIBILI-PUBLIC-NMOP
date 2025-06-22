from typing_extensions import Self
import numpy as np
from numpy import exp, log, sqrt
from scipy.stats import norm
from Src.NumericalOptionPricing.Contracts import VanillaContract
from Src.NumericalOptionPricing.Utils import SequentialMeanCalculator, generate_tridiagonal_matrix


class VanillaContractPricer:
    def __init__(self: Self, contract: VanillaContract) -> None:
        self.__contract = contract
        self.__contract_type = contract.contract_type
        self.__s0 = contract.s0
        self.__T = contract.T
        self.__K = contract.K
        self.__r = contract.r
        self.__sigma = contract.sigma
        self.__algorithm_type = contract.algorithm_type
        self.__control_params = contract.control_params

    @property
    def contract(self: Self) -> VanillaContract:
        return self.__contract

    def set_contract(self: Self, contract: VanillaContract) -> None:
        # Use a new contract to replace the current contract, so that the new contract can be priced
        self.__init__(contract=contract)

    def _get_analytic_price(self: Self) -> float:
        # Use the Black-Scholes formula to compute the call price
        d1 = (log(self.__s0 / self.__K) + (self.__r + 0.5 * self.__sigma**2) * self.__T) / (
            self.__sigma * sqrt(self.__T)
        )
        d2 = d1 - self.__sigma * sqrt(self.__T)
        call_price = float(
            self.__s0 * norm.cdf(d1) - self.__K * exp(-self.__r * self.__T) * norm.cdf(d2)
        )

        if self.__contract_type == "VanillaCall":
            return call_price
        elif self.__contract_type == "VanillaPut":
            # Use put-call parity to caompute the put price
            return call_price + self.__K * exp(-self.__r * self.__T) - self.__s0
        else:
            raise ValueError(f"Invalid contract type detected: {self.__contract_type}.")

    def _get_monte_carlo_price(self: Self) -> float:
        # Prepare the payoff function
        if self.__contract_type == "VanillaCall":
            payoff_func = lambda s: np.maximum(s - self.__K, 0)
        elif self.__contract_type == "VanillaPut":
            payoff_func = lambda s: np.maximum(self.__K - s, 0)
        else:
            raise ValueError(f"Invalid contract type detected: {self.__contract_type}.")

        mean_calculator = SequentialMeanCalculator()

        # Estimate the optimal value of lambda and the mean of control variate
        #   if use_control_variate is True
        if self.__control_params.use_control_variate:
            sT = self.__s0 * exp(
                (self.__r - 0.5 * self.__sigma**2) * self.__T
                + self.__sigma * sqrt(self.__T) * np.random.randn(200)
            )
            payoffs = payoff_func(sT)
            lbd = np.corrcoef(sT, payoffs)[0, 1] * np.std(payoffs) / np.std(sT)
            mean_sT = self.__s0 * np.exp(self.__r * self.__T)

        for _ in range(self.__control_params.n_sample_paths):
            # Generate the terminal stock price
            innovation = sqrt(self.__T) * np.random.randn()
            sT = self.__s0 * exp(
                (self.__r - 0.5 * self.__sigma**2) * self.__T + self.__sigma * innovation
            )

            # Generate the payoff and update the mean
            if self.__control_params.use_control_variate:
                mean_calculator.step(payoff_func(sT) - lbd * (sT - mean_sT))
            else:
                mean_calculator.step(payoff_func(sT))

        # The mean should be discounted to get the price at time 0
        return exp(-self.__r * self.__T) * mean_calculator.mean

    def _get_pde_price(self: Self) -> float:
        S_max = 4 * self.__s0
        M, N = self.__control_params.n_steps_S, self.__control_params.n_steps_T
        dS, dt = S_max / M, self.__T / N
        table = np.zeros((M + 1, N + 1))

        if self.__contract_type == "VanillaCall":
            # Handle the terminal condition
            table[:, -1] = np.maximum(np.arange(M + 1) * dS - self.__K, 0)
            # Handle the boundary conditions
            table[0, :] = 0
            table[-1, :] = S_max - self.__K * exp(-self.__r * dt * np.arange(N, -1, -1))
        elif self.__contract_type == "VanillaPut":
            # Handle the terminal condition
            table[:, -1] = np.maximum(self.__K - np.arange(M + 1) * dS, 0)
            # Handle the boundary conditions
            table[0, :] = self.__K * exp(-self.__r * dt * np.arange(N, -1, -1))
            table[-1, :] = 0
        else:
            raise ValueError(f"Invalid contract type detected: {self.__contract_type}.")

        # Compute the table in a backward order
        for j in range(N, 0, -1):
            grid = np.arange(1, M)
            neg_coefs = 0.5 * dt * (self.__sigma**2 * grid**2 - self.__r * grid)
            main_coefs = 1 - (self.__r + self.__sigma**2 * grid**2) * dt
            pos_coefs = 0.5 * dt * (self.__sigma**2 * grid**2 + self.__r * grid)
            coef_matrix = generate_tridiagonal_matrix(
                neg=neg_coefs,
                main=main_coefs,
                pos=pos_coefs,
            )
            table[1:M, j - 1] = np.dot(coef_matrix, table[:, j])
        index_low = int(self.__s0 / dS)
        return table[index_low, 0] + (table[index_low + 1, 0] - table[index_low, 0]) / dS * (
            self.__s0 - index_low * dS
        )

    def get_price(self: Self) -> float:
        if self.__algorithm_type == "Analytical":
            return self._get_analytic_price()
        elif self.__algorithm_type == "MonteCarlo":
            return self._get_monte_carlo_price()
        elif self.__algorithm_type == "PDE":
            return self._get_pde_price()
        else:
            raise ValueError(f"Invalid algorithm type detected: {self.__algorithm_type}.")
