import sympy as sp


class BlackScholesSolver:
    def __init__(self, S, K, sigma, T, r=None, C=None) -> None:
        """Initialize Black-Scholes parameters."""
        self.S = S          # Stock price
        self.K = K          # Strike price
        self.sigma = sigma  # Volatility
        self.T = T          # Time to expiration
        self.C = C          # Call option price
        self.r = r          # Interest rate to be solved
        
        # Define symbolic variables
        self._define_equation()


    def _define_equation(self) -> None:
        """Define the Black-Scholes equation symbolically"""
        S, K, sigma, T, r, C = sp.symbols('S K sigma T r C')

        # Define d1 and d2
        d1 = (sp.ln(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sp.sqrt(T))
        d2 = d1 - sigma * sp.sqrt(T)

        # Normal CDF approximation using erf
        N_d1 = 0.5 * (1 + sp.erf(d1 / sp.sqrt(2)))
        N_d2 = 0.5 * (1 + sp.erf(d2 / sp.sqrt(2)))

        # Black-Scholes equation for a call option
        self.black_scholes_equation = sp.Eq(C, S * N_d1 - K * sp.exp(-r * T) * N_d2)


    def set_C(self, C) -> None:
        self.C = C


    def set_r(self, r) -> None:
        self.r = r


    def solve_for_C(self, initial_guess=3) -> float:
        if self.r is None:
            raise Exception("Error: Value of r is not set")

        params = {
            sp.Symbol('S'): self.S,
            sp.Symbol('K'): self.K,
            sp.Symbol('sigma'): self.sigma,
            sp.Symbol('T'): self.T,
            sp.Symbol('r'): self.r,
        }

        c_solution = sp.nsolve(self.black_scholes_equation.subs(params), sp.Symbol('C'), initial_guess)
        return float(c_solution)


    def solve_for_r(self, initial_guess=0.05) -> float:
        """Solve for r using numerical methods."""
        if self.C is None:
            raise Exception("Error: Value of call is not set")

        params = {
            sp.Symbol('S'): self.S,
            sp.Symbol('K'): self.K,
            sp.Symbol('sigma'): self.sigma,
            sp.Symbol('T'): self.T,
            sp.Symbol('C'): self.C,
        }

        r_solution = sp.nsolve(self.black_scholes_equation.subs(params), sp.Symbol('r'), initial_guess)
        return float(r_solution)
