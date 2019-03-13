selected_precision = BigFloat

function floudas_one(dz_dt, z, phi, t)
    r_1 = phi[1]*z[1]
    r_2 = phi[2]*z[2]

    dz_dt[1] = -r_1
    dz_dt[2] = r_1 - r_2
end

data = selected_precision[1.0 0.57353 0.328937 0.188654 0.108198 0.0620545 0.0355906 0.0204132 0.011708 0.00671499;
        0.0 0.401566 0.589647 0.659731 0.666112 0.639512 0.597179 0.54867 0.499168 0.451377]
t = selected_precision[0.0, 0.111111, 0.222222, 0.333333, 0.444444, 0.555556, 0.666667, 0.777778, 0.888889, 1.0]

#data = floudas_samples_array[1]
#t = floudas_samples_times_array[1]

lb = selected_precision[0., 0.]
ub = selected_precision[10., 10.]

ode_fun = floudas_one

function lsq_ss_curvefit(t, phi)
    tspan = (t[1], t[end])
    ini_cond = data[:,1]
    oprob = ODEProblem(ode_fun, ini_cond, tspan, phi)
    osol  = solve(oprob, Tsit5(), saveat=reduce(vcat, t))
    estimated = reduce(hcat, osol.u)
    return vec(estimated)
end

p0 = selected_precision[5., 5.]

lsq_ss_sum(p) = sum(abs2.(lsq_ss_curvefit(t,p)-vec(data)))
res_opt_ss = opt.optimize(lsq_ss_sum, lb, ub, p0, opt.Fminbox(opt.LBFGS()))

res_lsf_ss = lsf.curve_fit(lsq_ss_curvefit, t, vec(data), p0, lower=lb, upper=ub)

lsq_ss(p) = lsq_ss_curvefit(t,p)-vec(data)
res_lso_ss = lso.optimize(lsq_ss, p0, lso.Dogleg(), lower=lb, upper=ub)

euclidean(res_lso_ss.minimizer, phi_array[1])

#res_lso_ss = lso.optimize(lsq_ss_sum, p0, lso.Dogleg(), lower=lb, upper=ub)

zeros(selected_precision,2)
#=>
lsopt_obj = build_lsoptim_objective(prob_one, tspan, t, data)}
res_lso_build = lso.optimize(lsopt_obj, p0, lso.Dogleg())

two_obj = two_stage_method(ode_prob, t, data)
res_opt_two_stage = opt.optimize(two_obj, p0)

los_obj = build_loss_objective(prob_one, Tsit5(), L2Loss(t,data))
res_opt_build = opt.optimize(los_obj, p0)
<=#

function rosenbrock(x)
	[1 - x[1], 100 * (x[2]-x[1]^2)]
end
x0 = zeros(desired_precision,2)
lso.optimize(rosenbrock, x0)
lso.optimize(rosenbrock, x0, lso.LevenbergMarquardt())
