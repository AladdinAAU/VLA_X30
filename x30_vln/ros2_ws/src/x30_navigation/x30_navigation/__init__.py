"""x30_navigation: odometry, 2D mapping, Nav2 glue and frontier exploration for the X30."""
import os

# numpy/scipy start one BLAS thread per core and the idle threads spin: a 3x3 SVD per scan made cloud_leveler burn ~10 cores and starve Nav2.
# The per-scan problems here are tiny, so single-threaded BLAS is faster. Must be set before numpy is imported.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
