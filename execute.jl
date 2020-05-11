x0 = [4.0, 4.0]
dr = OnceDifferentiable(r, grad_r!, x0)
dr_auto = OnceDifferentiable(r, x0; autodiff = :forward)

project(x) = begin
    [2.0, x[2]]
end

# J = jacobian!(dr, x0)
# show(J' .*J)
res = projected_levenberg_marquardt_local(
    dr, project, x0, 10, 1.0
)

println("New case")
res_2 = projected_levenberg_marquardt_local(
    dr, project, [1.0, 1.0], 10, 10.0
)
