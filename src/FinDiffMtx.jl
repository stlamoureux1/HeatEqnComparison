# Solve the IBVP
# 
# u_t = u_xx + sin(πx), 0 ≤ x ≤ 1, t ≥ 0
# u(t,0) = 0, u(t,1) = 0
# u(0,x) = 0
#
# Using a finite difference matrix-vector system 
# and explicit Euler time-stepping
#
# Analytic solution
# u = (1/π²)(1 - exp(-π²t))sin(πx)
# u -> (1/π²)sin(πx) as t -> ∞

using LinearAlgebra

get_u_true(t, xs) = (1 / pi^2) * (1 - exp(-pi^2 * t)) * sin.(pi .* xs)

get_dt(dx) = dx^2 * 0.4

function get_u_true(a, b, Nx, t)
    xs = range(a, b, Nx + 2)
    dx = step(xs)
    xs_int = @view xs[2:end-1]
    dt = get_dt(dx)
    Nt = Int(floor(t / dt))

    u_true = zeros(Nx + 2, Nt)
    for t_i in 2:Nt
        u_curr = @view u_true[2:end-1, t_i]
        u_curr .= get_u_true((t_i - 1) * dt, xs_int)
    end
    return u_true
end

function get_u_est(a, b, Nx, t)
    # nodes and step sizes
    xs = range(a, b, Nx + 2)
    dx = step(xs)
    xs_int = @view xs[2:end-1]

    dt = get_dt(dx)
    Nt = Int(floor(t / dt))

    # source term
    f = sin.(pi * xs_int)

    # Laplacian for interior nodes only
    L = 1 / dx^2 * SymTridiagonal(-2 * ones(Nx), ones(Nx - 1))

    u_est = zeros(Nx + 2, Nt)
    for t_i in 2:Nt
        u_prev = @view u_est[2:end-1, t_i-1]
        u_curr = @view u_est[2:end-1, t_i]
        u_curr .= u_prev .+ (L * u_prev .+ f) .* dt
    end
    return u_est
end


function get_nodes(a, b, Nx, t)
    xs = range(a, b, Nx + 2)
    dx = step(xs)
    dt = get_dt(dx)
    Nt = Int(round(t / dt))
    ts = range(0, t, Nt)
    return ts, xs
end

function get_err(u_true, u_est)
    Nt = size(u_true)[2]
    err = zeros(Nt)
    for t_i in 2:Nt
        err[t_i] = norm(u_true[:, t_i] - u_est[:, t_i]) / norm(u_true[:, t_i])
    end
    return err
end

function get_err(a, b, Nx, t)
    u_est = get_u_est(a, b, Nx, t)
    u_true = get_u_true(a, b, Nx, t)
    Nt = size(u_est)[2]
    err = zeros(Nt)
    for t_i in 2:Nt
        err[t_i] = norm(u_true[:, t_i] - u_est[:, t_i]) / norm(u_true[:, t_i])
    end
    return err
end

