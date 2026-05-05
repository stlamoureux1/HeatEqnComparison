import ITensors, ITensorMPS, QTTFinDiff, LinearAlgebra, FFTW

using ITensors, ITensorMPS, QTTFinDiff, FFTW

using LinearAlgebra: SymTridiagonal, Diagonal, norm

abstract type SolverType end

struct MatrixSolver <: SolverType end

struct QTTSolver <: SolverType
    sourceRank::Int
    solnRank::Int # for a QTT solver, specify the maximal rank of the solution approximation
end

struct FourierSolver <: SolverType end

abstract type TimeStepping end
struct ForwardEuler <: TimeStepping end

# Solvers for the IBVP 
# u(t, x) : [0, ∞) × [a, b] → \mathbb{R}
# u_t + u_xx = f(x), a ≤ x ≤ b
# u(t, a) = 0, u(t, b) = 0
# u(0, x) = 0
# where f is given by a sum-of-sines in $x$ with constant Fourier coefficients

# Note that eigensolutions on $t$, $B_n(t) = bₙ / λₙ - (bₙ / λₙ)exp(-λₙt)
# where $bₙ$ is the $n$-th Fourier coefficient of the source term $f$.
#
# Specify $f$ as an array mapping the $i$-th position to the $i$-th Fourier coefficient

struct ProblemSpec
    a::Float64 # lower bound of spatial domain
    b::Float64 # upper bound of spatial domain
    nBit::Int # "bits of resolution" -- spatial grid has 2^nBit + 2 nodes
    t::Float64 # upper bound of time marching
    fCoeff::Vector{Float64} # array of Fourier sine-series coefficients for source term
end

function dtCFL(dx)
    # calculate dt based on CFL condition
    return dx^2 * 0.4
end

function getNodes(a::Float64, b::Float64, nBit::Int, t::Float64)
    Nx = 2^nBit
    xs = range(a, b, length=Nx + 2)
    dx = step(xs)

    dt = dtCFL(dx)
    # calculate number of time steps based on upper bound and time-step size
    Nt = Int(floor(t / dt))

    ts = range(0, t, Nt)

    return ts, Nt, xs, Nx
end

function getNodes(problem::ProblemSpec)
    return getNodes(problem.a, problem.b, problem.nBit, problem.t)
end

function trueSoln(a::Float64, b::Float64, nBit::Int, t::Float64, fCoeff::Vector{Float64})
    # n-th term in sine series solution
    term(n, t_i, x_j) = (fCoeff[n] / (n * pi)^2) * (1 - exp(-(n * pi)^2 * t_i)) * sin((n * pi * x_j) / (b - a))
    u(t_i, x_j) = sum([term(n, t_i, x_j) for n in fCoeff])
    ts, Nt, xs, Nx = getNodes(a, b, nBit, t)
    xs_int = @view xs[2:end-1]

    u_true = zeros(Nx + 2, Nt)
    for t_i in 2:Nt
        u_curr = @view u_true[2:end-1, t_i]
        u_curr .= [sum([term(n, ts[t_i], x_j) for n in 1:length(fCoeff)]) for x_j in xs_int]
    end
    return u_true
end

function trueSoln(problem::ProblemSpec)
    trueSoln(problem.a, problem.b, problem.nBit, problem.t, problem.fCoeff)
end

function approxSoln(a::Float64, b::Float64, nBit::Int, t::Float64, fCoeff::Vector{Float64}, ::MatrixSolver, ::ForwardEuler)::Matrix{Float64}
    # Calculate approximation solution using Linear system
    # with sparse matrix for 1D Laplacian
    ts, Nt, xs, Nx = getNodes(a, b, nBit, t)
    xs_int = @view xs[2:end-1]
    dt = step(ts)
    dx = step(xs)

    # source term
    fs = [sin((i * pi * x) / (b - a)) for x in xs_int, i in 1:length(fCoeff)] * fCoeff

    # return value
    u_est = zeros(Nx + 2, Nt)

    # Discrete Laplacian
    L = (1 / dx)^2 * SymTridiagonal(-2 * ones(Nx), ones(Nx - 1))

    # time steps
    for t_i in 2:Nt
        u_prev = @view u_est[2:end-1, t_i-1]
        u_curr = @view u_est[2:end-1, t_i]
        u_curr .= u_prev .+ (L * u_prev .+ fs) .* dt
    end
    return u_est
end

function approxSoln(a, b, nBit, t, fCoeff, solver::QTTSolver, ::ForwardEuler)
    ts, Nt, xs, Nx = getNodes(a, b, nBit, t)
    dt = step(ts)
    dx = step(xs)
    xs_int = @view xs[2:end-1]

    sites = siteinds("QTT", nBit)

    # source term
    fs = [sin((i * pi * x) / (b - a)) for x in xs_int, i in 1:length(fCoeff)] * fCoeff
    fs_mps = MPS(fs, sites; maxdim=solver.sourceRank)

    # Laplacian MPO
    # Note that the module QTTFinDiff uses the negative convention
    L = (1 / dx)^2 * -laplacian1d(sites)

    u_est = zeros(Nx + 2, Nt)

    for t_i in 2:Nt
        u_prev = @view u_est[2:end-1, t_i-1]
        u_curr = @view u_est[2:end-1, t_i]
        u_prev_mps = MPS(u_prev, sites; maxdim=solver.solnRank)
        u_curr_mps = u_prev_mps + (noprime(L * u_prev_mps) + fs_mps) * dt
        u_curr .= reshape(array(contract(u_curr_mps)), Nx)
    end
    return u_est
end

function approxSoln(a, b, nBit, t, fCoeff, ::FourierSolver, ::ForwardEuler)
    ts, Nt, xs, Nx = getNodes(a, b, nBit, t)
    dt = step(ts)

    u_est = zeros(Nx + 2, Nt)
    k = 1:Nx
    λ = -(k .* pi) .^ 2

    @assert length(fCoeff) <= Nx
    f_hat = zeros(Nx)
    f_hat[1:length(fCoeff)] = fCoeff

    for t_i in 2:Nt
        u_prev = @view u_est[2:end-1, t_i-1]
        u_curr = @view u_est[2:end-1, t_i]
        u_curr .= u_prev + (λ .* u_prev + f_hat) * dt
    end

    for t_i in 1:Nt
        u_curr = @view u_est[2:end-1, t_i]
        FFTW.r2r!(u_curr, FFTW.RODFT01)
        u_curr ./= 2
    end

    return u_est
end

function approxSoln(problem::ProblemSpec, solver::SolverType, stepper::TimeStepping)
    return approxSoln(problem.a, problem.b, problem.nBit, problem.t, problem.fCoeff, solver, stepper)
end

