# Comparison of Solution Techniques for the Heat Equation

## Goal
Get an initial comparison for accuracy across 3 numerical methods for the heat equation:
- Classical Finite Differences (linear solver)
- Finite Differences in the quantics tensor train / matrix-product state format
- Spectral method

## How-to
The file `src/HeatEqnSolvers.jl` provides methods for applying these solution techniques to heat equation problems. In its current state, it only handles 0 Dirichlet boundary conditions (implicitly periodic for the spectral method). To set up a solver create a `ProblemSpec`. For example,

```julia
problem = ProblemSpec(a=0.0, b=1.0, nBit=8, t=1.0, fCoeff=ones(10))
```
specifies a problem with spatial domain $x \in [0,1]$, 8 "bits" of resolution, i.e. 256 grid points in the spatial domain. It will run until time step `1.0`. The field `fCoeff` specifies the Fourier coefficients for the time-invariant source term, interpreted a *sine series*. So, the above gives a source term corresponding to

$$
\sum_{i=1}^{10} 1.0 \cdot \sin{\frac{i \pi x}{1}}.
$$

Currently, the initial condition is uniformly zero, i.e. $u(0, x) = 0$.

To apply the different methods for the spatial solution, pass a `SolverType` instance to the method `approxSoln`. The available types are `MatrixSolver`, for classical FD, `FourierSolver` for the spectral method, and `QTTSolver(solnRank, sourceRank)` for the quantics tensor train solver. In this last case, the fields of the `QTTSolver` instance specify the maximal rank (bond dimension) of the solution and source term, respectively. 

The only time-stepping method currently applied is forward (explicit) Euler. The time step `dt` is calculated automatically from `dx` using relevant the CFL condition.

Taken together, a solution to the above problem is available as

```julia
# Classical
soln1 = approxSoln(problem, MatrixSolver(), ForwardEuler())

# Spectral
soln2 = approxSoln(problem, FourierSolver(), ForwardEuler())

# Quantics Tensor Train with max. bond dimension 2 for the solution, and 3 for the source
soln3 = approxSoln(problem, QTTSolver(2,3), ForwardEuler())
```

## Results (tentative)
The file `src/Problems.jl` specifies `problem2` as

$$
\begin{aligned}

u(t, x) &: [0, 1] \times [0, 1] \to \mathbb{R}\\[1em]

\partial_t u - \partial_{x}^{2}u &= \sin(\pi x) \\[1em]

u(0, x) &= 0 \\[1em]

u(t, 0) &= u(t, 1) = 0

\end{aligned}
$$

with $128$ grid points. The results are available in the folder `plots`.

(Note that `problem1` is a small "warm-up" problem for compilation in the `julia` REPL.)
