"""chaos_tools: research-grade analysis for chaotic time series.

Given only a scalar signal -- no equations, no known state dimension --
these tools reconstruct a phase space, estimate the largest Lyapunov
exponent, estimate the correlation (fractal) dimension, and build
recurrence plots. The same questions this repo's playable toys ask
visually (does this diverge? is there structure hiding in the noise?),
answered numerically, for data you bring yourself.

    from chaos_tools import embedding, lyapunov, dimension, recurrence

    tau = embedding.estimate_delay(x)
    dim, _ = embedding.estimate_dimension(x, tau)
    lle, times, log_div = lyapunov.largest_lyapunov_exponent(x, dim, tau, dt=dt)
    d2, radii, corr_sum = dimension.correlation_dimension(x, dim, tau)
    rp = recurrence.recurrence_matrix(x, dim, tau)
"""

from . import dimension, embedding, lyapunov, recurrence

__all__ = ["embedding", "lyapunov", "dimension", "recurrence"]
__version__ = "0.1.0"
