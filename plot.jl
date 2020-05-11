function plot_function(fig, ax)
    x_range = LinRange(0.0, 4.5, 100)
    y_range = LinRange(0.0, 4.5, 200)

    X = [ x for x in x_range, y in y_range]
    Y = [ y for x in x_range, y in y_range]

    Z = f(X, Y)

    (z_min, z_max) = extrema(Z)
    z_levels = map(x -> 10^x, LinRange(-5, 3, 25))
    CS = ax.contourf(X,Y, Z, levels = z_levels,
                     locator = matplotlib.ticker.LogLocator())
    cbar = fig.colorbar(CS)


    for mask in [x_range .< 0, x_range .> 0]
        plot(x_range[mask], f_min(x_range[mask]), "r-")
    end

    # The projection
    # ||x|| = 2.0
    f_project(y) = 2.0*ones(length(y))
    plot(f_project(y_range), y_range, "b-")

    ax.set_xlim(minimum(x_range), maximum(x_range))
    ax.set_ylim(minimum(y_range), maximum(y_range))

    nothing
end
function plot_res(ax, res, c)
    x_trace = res["x_trace"]
    ax.plot(x_trace[1,1,:],x_trace[2,1,:], "$(c)-x")
    ax.plot(x_trace[1,2,:],x_trace[2,2,:], "$(c)s")
    # ax.plot(x_trace[1,2,:],x_trace[2,2,:], "b--")
end
function plot_cost(ax, res, c, label)
    x_trace = reshape(res["x_trace"], 2, :)
    trace = [f(x_trace[:,k]) for k = 1:size(x_trace, 2)]

    ax.plot(1:length(trace), trace, "$(c)-", label = label)
end
function plot_grad_norm(ax, res, c, label)
    G_trace = res["G_trace"]
    trace = [norm(G_trace[:,k]) for k = 1:size(G_trace, 2)]

    ax.plot(1:length(trace), trace, c, label = label)
end

fig, ax = plt.subplots(1,1)
plot_function(fig, ax)
plot_res(ax, res, "k")
plot_res(ax, res_2, "g")

fig, ax = plt.subplots(1,1)
plot_cost(ax, res, "k", "Cost")
plot_cost(ax, res_2, "g", "Cost")

# plot_grad_norm(ax, res, "k-", "Grad")
# plot_grad_norm(ax, res_2, "g-", "Grad")

ax.grid(true, which = "both")
ax.set_yscale("log")

plt.show()
