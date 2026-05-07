include("HeatEqnSolvers.jl")

problem1 = ProblemSpec(a=0.0, b=1.0, nBit=3, t=0.5, fCoeff=[1.0])
problem2 = ProblemSpec(a=0.0, b=1.0, nBit=7, t=1.0, fCoeff=[1.0])

# time node values, time nodes count, x node values, x nodes count
(ts2, Nt2, xs2, Nx2) = getNodes(problem2)

# true solution to problem2
u2_true = trueSoln(problem2)

# approximate solution using classical finite differences
u2_est_cl = approxSoln(problem2, MatrixSolver(), ForwardEuler())

u2_err_cl = [norm(u2_true[:, t_i] - u2_est_cl[:, t_i]) / norm(u2_true[:, t_i]) for t_i in 1:Nt2]

# approximate solution using quantics MPS / tensor train 
# NOTE: This will take a long time since we are doing global forward time steps
# and SVD truncations
u2_est_qtt = approxSoln(problem2, QTTSolver(2, 2), ForwardEuler())

u2_err_qtt = [norm(u2_true[:, t_i] - u2_est_qtt[:, t_i]) / norm(u2_true[:, t_i]) for t_i in 1:Nt2]

# approximate solution using Fourier spectral basis
u2_est_fft = approxSoln(problem2, FourierSolver(), ForwardEuler())

u2_err_fft = [norm(u2_true[:, t_i] - u2_est_fft[:, t_i]) / norm(u2_true[:, t_i]) for t_i in 1:Nt2]
