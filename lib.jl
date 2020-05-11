g(x,y) = x.*y
g(x) = g(x[1],x[2])
function grad_g!(G, x,y)
    G[1] = y
    G[2] = x
    return G
end
grad_g!(G, x) = grad_g!(G, x[1], x[2])


r(x,y) = g(x,y) .- 3.0
r(x) = r(x[1], x[2])
grad_r!(G, x) = grad_g!(G, x)

f(x, y) = 1.0/2.0*r(x,y).^2
f(x) = f(x[1], x[2])

function grad_f!(F, x, y)
    grad_g!(F, x, y)
    F[:] .*= r(x,y)
    return F
end
grad_f!(F, x) = grad_f!(F, x[1], x[2])

function hess_f!(H, x, y)
    Dg = zeros(eltype(H), 2)
    grad_g!(Dg, x,y)
    H[:,:] = Dg .* Dg'
    return H
end
hess_f!(H, x) = hess_f!(H, x[1], x[2])

f_min(x) = 3 ./x

td = TwiceDifferentiable(
    f,
    (G, x) -> grad_f!(G, x[1], x[2]),
    (H, x) -> hess_f!(H, x[1], x[2]),
    randn(2)
)
function projected_levenberg_marquardt_local(df, project, x0, k_max, μ)
    x_trace = zeros(eltype(x0), length(x0), 2, k_max + 1)
    G_trace = zeros(eltype(x0), length(x0), k_max)
    H_trace = zeros(eltype(x0), length(x0), length(x0), k_max)
    x = copy(x0)
    d_k = copy(x0)

    x_trace[:, 1, 1] = x
    x_trace[:, 2, 1] = project(x)
    for k = 1:k_max
        J_k = jacobian!(df, x)
        F_k = value!(df, x)
        μ_k = μ * norm(F_k)

        H_reg = J_k' .* J_k + μ_k * I
        G_k = J_k * F_k

        d_k[:] = H_reg\(-G_k)

        x_non_projected = x + d_k
        x[:] = project(x_non_projected)

        if norm(F_k) < norm(value!(df, x))
            @warn "Cost is not decreasing"
        end

        # Save
        x_trace[:, 1, k + 1] = x_non_projected
        x_trace[:, 2, k + 1] = x
        G_trace[:, k] = G_k
        H_trace[:,:,k] = H_reg

        # println(k)
        # display(G_k)
        # display(H_reg)
    end
    Dict(
    "x_trace" => x_trace,
    "G_trace" => G_trace,
    "H_trace" => H_trace
    )
end


x_min = begin
    x = randn()
    [x, f_min(x)]
end

H_min = zeros(2,2)
hess_f!(H_min, x_min)
@assert ForwardDiff.hessian(f, x_min) == H_min "Hessian is wrong"

G_rand = zeros(2)
x_rand = randn(2)
grad_f!(G_rand, x_rand)
@assert ForwardDiff.gradient(f, x_rand) == G_rand "Gradient is wrong"
