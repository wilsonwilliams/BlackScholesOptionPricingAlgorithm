import sympy as sp


class BlackScholesSolver:
    def __init__(self) -> None:
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


    def solve_for_C(self, S, K, sigma, T, r, initial_guess=3) -> float:
        params = {
            sp.Symbol('S'): S,
            sp.Symbol('K'): K,
            sp.Symbol('sigma'): sigma,
            sp.Symbol('T'): T,
            sp.Symbol('r'): r,
        }

        c_solution = sp.nsolve(self.black_scholes_equation.subs(params), sp.Symbol('C'), initial_guess)
        return float(c_solution)


    def solve_for_r(self, S, K, sigma, T, C, initial_guess=0.05) -> float:
        params = {
            sp.Symbol('S'): S,
            sp.Symbol('K'): K,
            sp.Symbol('sigma'): sigma,
            sp.Symbol('T'): T,
            sp.Symbol('C'): C,
        }

        r_solution = sp.nsolve(self.black_scholes_equation.subs(params), sp.Symbol('r'), initial_guess)
        return float(r_solution)
