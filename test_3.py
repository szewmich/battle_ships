# import numpy as np
# import random

# random.seed(10)
# #x = np.array([0.224, 0.176, 0.873, 0.242, 0.173, 0.843, 0.993, 0.705, 0.713, 0.912, 0.301, 0.719, 0.340, 0.924, 0.746, 0.667, 0.170, 0.268, 0.145, 0.901])   # your sample array
# x = np.random.rand(10_000)
# mu_target = 0.76

# mu_hat = x.mean()
# nu_hat = (x**2).mean()
# xmin, xmax = x.min(), x.max()

# D = (mu_target*mu_hat - nu_hat) / (mu_hat - mu_target)

# if D >= -xmin:
#     beta = 1.0 / (xmax + D)   # max-efficiency choice
#     def a(xi): return beta*(xi + D)
#     print(f"Using linear acceptance a(x) = (x + {D})/({xmax} + {D})")
# else:
#     # positive-part fallback: find c such that mean of x>x_c equals mu_target
#     xs = np.sort(x)
#     found_c = None
#     for k in range(len(xs)-1):
#         c = 0.5*(xs[k] + xs[k+1])
#         sel = x[x > c]
#         if sel.size == 0:
#             continue
#         c_temp = abs(sel.mean() - mu_target) 

#         if found_c is not None:
#             if c_temp < found_c:   
#                 found_c = c_temp
#         else:
#             found_c = c_temp   

#     if found_c is not None:
#         c = found_c
#         gamma = 1.0 / np.max(np.maximum(0, x - c))
#         def a(xi): return gamma * max(0.0, xi - c)
#         print("Using positive-part acceptance a(x) = gamma*max(0,x-c)")
#         print(f"Found c = {c}")
#     else:
#         print("No exact threshold found; consider numerical solve with logistic or accept approximate c.")


# #apply a(xi) on each element of x array
# axi_values = np.array([a(xi) for xi in x])
# y = np.array([xi*a(xi) for xi in x])

# # print(axi_values)
# # print(y)
# print(y.sum()/axi_values.sum())



import numpy as np
from scipy.optimize import minimize

def logistic_acceptance(x, k, t):
    """Logistic acceptance probabilities."""
    return 1.0 / (1.0 + np.exp(-k*(x - t)))

def accepted_mean(x, k, t):
    p = logistic_acceptance(x, k, t)
    num = np.sum(x * p)
    den = np.sum(p)
    return num / den if den > 0 else np.nan

def fit_logistic_acceptance(x, mu_target, k0=10.0, t0=None):
    """
    Fit logistic acceptance parameters (k,t) so accepted mean ≈ mu_target.
    Uses simple least squares minimization.
    """
    x = np.asarray(x, dtype=float)
    if t0 is None:
        t0 = np.median(x)  # sensible initial threshold

    def objective(params):
        k, t = params
        m = accepted_mean(x, k, t)
        if np.isnan(m):
            return 1e6
        return (m - mu_target)**2

    res = minimize(objective, x0=[k0, t0], method='Nelder-Mead')
    k_opt, t_opt = res.x
    return k_opt, t_opt, res.fun

# Example usage
if __name__ == "__main__":
    np.random.seed(1)
    #x = np.random.rand(10000)  # some sample
    x = np.array([0.224, 0.176, 0.873, 0.242, 0.173, 0.843, 0.993, 0.705, 0.713, 0.912, 0.301, 0.719, 0.340, 0.924, 0.746, 0.667, 0.170, 0.268, 0.145, 0.901])   # your sample array
    mu_target = 0.993

    k, t, loss = fit_logistic_acceptance(x, mu_target)
    print(f"Optimal parameters: k={k:.4f}, t={t:.4f}, objective={loss:.3e}")
    m = accepted_mean(x, k, t)
    print(f"Achieved mean = {m:.6f}, target = {mu_target}")

    #apply a(xi) on each element of x array
    axi_values = np.array([logistic_acceptance(xi, k, t) for xi in x])
    y = np.array([xi*logistic_acceptance(xi, k, t) for xi in x])

    # print(axi_values)
    # print(y)
    print(y.sum()/axi_values.sum())