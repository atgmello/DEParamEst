function floudas_one(dz_dt, z, phi, t)
    r_1 = phi[1]*z[1]
    r_2 = phi[2]*z[2]

    dz_dt[1] = -r_1
    dz_dt[2] = r_1 - r_2
end

ode_fun = floudas_one
#desired_precision = BigFloat
desired_precision = Float64

data = desired_precision[1.0 0.57353 0.328937 0.188654 0.108198 0.0620545 0.0355906 0.0204132 0.011708 0.00671499;
        0.0 0.401566 0.589647 0.659731 0.666112 0.639512 0.597179 0.54867 0.499168 0.451377]
t = desired_precision[0.0, 0.111111, 0.222222, 0.333333, 0.444444, 0.555556, 0.666667, 0.777778, 0.888889, 1.0]
ini_cond = data[:,1]

function lsq_ss_estimator(time_array, phi)
    tspan = (t[1], t[end])
    ini_cond = data[:,1]
    oprob = ODEProblem(ode_fun, ini_cond, tspan, phi)
    osol  = solve(oprob, Tsit5(), saveat=t)
    estimated = reduce(hcat, osol.u)
    return vec(estimated)
end

p0 = desired_precision[5., 5.]
fit = lsf.curve_fit(lsq_ss_estimator, t, vec(data), p0)

lsq_ss(p) = (lsq_ss_estimator(t, p) - vec(data))
ss_resid = lso.optimize(lsq_ss, p0, lso.Dogleg())

lsq_ss_sum(p) = sum(lsq_ss(p).^2)
ss_sum = opt.optimize(lsq_ss_sum, p0)


tspan = (t[1], t[end])
phi = desired_precision[5.0035, 1]
prob_one = ODEProblem(ode_fun, ini_cond, tspan, phi)

lsopt_obj = build_lsoptim_objective(prob_one, tspan, t, data)
ss_resid = lso.optimize(lsopt_obj, p0, lso.Dogleg())

two_obj = two_stage_method(prob_one, t, data)
two_obj_res = opt.optimize(two_obj, p0)

los_obj = build_loss_objective(prob_one, Tsit5(), L2Loss(t,data))
los_obj_res = opt.optimize(los_obj, p0)

am_optim(p) = sum(abs2.(loss.(adams_moulton_estimator(p, data, t, ode_fun))))
res_am = optimize(am_optim, p0)

fit.param
ss_resid.minimizer
ss_sum.minimizer
two_obj_res.minimizer
los_obj_res.minimizer
res_am.minimizer


two_obj_res









res_am
