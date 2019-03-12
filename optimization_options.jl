#=>
# ----- Build -----

#lsopt_obj = build_lsoptim_objective(prob_one, tspan, t, data)
#res_lso_build = lso.optimize(lsopt_obj, p0, lso.Dogleg())

two_obj = two_stage_method(ode_prob, t, data)
res_opt_two_stage = opt.optimize(two_obj, p0)

los_obj = build_loss_objective(prob_one, Tsit5(), L2Loss(t,data))
res_opt_build = opt.optimize(los_obj, p0)
<=#

#=>
function floudas_one(dz_dt, z, phi, t)
    r_1 = phi[1]*z[1]
    r_2 = phi[2]*z[2]

    dz_dt[1] = -r_1
    dz_dt[2] = r_1 - r_2
end

ode_fun = floudas_one


data = [1.0 0.57353 0.328937 0.188654 0.108198 0.0620545 0.0355906 0.0204132 0.011708 0.00671499;
        0.0 0.401566 0.589647 0.659731 0.666112 0.639512 0.597179 0.54867 0.499168 0.451377]
t = [0.0, 0.111111, 0.222222, 0.333333, 0.444444, 0.555556, 0.666667, 0.777778, 0.888889, 1.0]

<=#

for i in 1:6
    res_lsf_ss_dist_array = []
    res_lso_ss_dist_array = []
    res_opt_ss_dist_array = []

    res_lsf_am_dist_array = []
    res_lso_am_dist_array = []
    res_opt_am_dist_array = []

    print("\n----- Getting results for Flouda's Problem number $i -----\n")
    ode_fun = ode_fun_array[i]
    phi = phi_array[i]
    rand_range = rand_range_array[i]
    bounds = bounds_array[i]

    lb = desired_precision[]
    ub = desired_precision[]

    for i in 1:length(phi)
        push!(lb, bounds[1])
        push!(ub, bounds[2])
    end

    # Floudas Data
    data = floudas_samples_array[i]
    t = floudas_samples_times_array[i]
    ini_cond = data[:,1]

    # Artificial Data
    #t = t_array[i]
    #tspan = (t[1], t[end])
    #ode_prob = ODEProblem(ode_fun, ini_cond, tspan, phi)
    #ode_sol  = solve(ode_prob, lsoda(), saveat=reduce(vcat, t))
    #data = reduce(hcat, ode_sol.u)

    linear(x) = x
    #loss = soft_l1
    loss = linear
    # --- Defining functions ---

    function lsq_ss_curvefit(time_array, phi)
        tspan = (t[1], t[end])
        ini_cond = data[:,1]
        oprob = ODEProblem(ode_fun, ini_cond, tspan, phi)
        osol  = solve(oprob, Tsit5(), saveat=reduce(vcat, t))
        estimated = reduce(hcat, osol.u)
        return vec(estimated)
    end

    function lsq_am_curvefit(time_array, phi)
        num_state_variables, num_samples = size(data)

        estimated = zeros(num_samples*num_state_variables)
        estimated = reshape(estimated, (num_state_variables, num_samples))
        estimated[:, 1] = data[:,1] #Initial conditions are stored at x_dot_num's first column

        for i in range(1, stop=num_samples-1)
            delta_t = time_array[i+1] - time_array[i]
            x_k_0 = data[:, i]
            x_k_1 = data[:, i+1]

            f_eval_0 = zeros(num_state_variables)
            ode_fun(f_eval_0, x_k_0, phi, 0)
            f_eval_1 = zeros(num_state_variables)
            ode_fun(f_eval_1, x_k_1, phi, 0)

            x_k_1_est = x_k_0 + (1/2)*delta_t*(f_eval_0+f_eval_1)
            estimated[:, i+1] = x_k_1_est
        end

        return vec(estimated)
    end

    for j in 1:15
        print("\n--- Iteration $j---\n")
        p0 = desired_precision[rand(Uniform(rand_range[1], rand_range[end])) for i in 1:length(phi)]

        # ----- Adams-Moulton -----

        try
            res_lsf_am = lsf.curve_fit(lsq_am_curvefit, t, vec(data), p0, lower=lb, upper=ub)
            res_lsf_am_dist = euclidean(phi, res_lsf_am.param)
            push!(res_lsf_am_dist_array, res_lsf_am_dist)
        catch
            push!(res_lsf_am_dist_array, -.5)
        end

        try
            lsq_am(p) = adams_moulton_estimator(p, data, t, ode_fun)
            res_lso_am = lso.optimize(lsq_am, p0, lso.Dogleg(), lower=lb, upper=ub)
            res_lso_am_dist = euclidean(phi, res_lso_am.minimizer)
            push!(res_lso_am_dist_array, res_lso_am_dist)
        catch
            push!(res_lso_am_dist_array, -.5)
        end

        try
            lsq_am_sum(p) = sum(abs2.(loss.(adams_moulton_estimator(p, data, t, ode_fun))))
            res_opt_am = opt.optimize(lsq_am_sum, lb, ub, p0)
            res_opt_am_dist = euclidean(phi, res_opt_am.minimizer)
            push!(res_opt_am_dist_array, res_opt_am_dist)
        catch
            push!(res_opt_am_dist_array, -.5)
        end

        # ----- Single Shooting -----

        try
            res_lsf_ss = lsf.curve_fit(lsq_ss_curvefit, t, vec(data), p0, lower=lb, upper=ub)
            res_lsf_ss_dist = euclidean(phi, res_lsf_ss.param)
            push!(res_lsf_ss_dist_array, res_lsf_ss_dist)
        catch
            push!(res_lsf_ss_dist_array, -.5)
        end

        try
            lsq_ss(p) = single_shooting_estimator(p, data, t, ode_fun)
            res_lso_ss = lso.optimize(lsq_ss, p0, lso.Dogleg(), lower=lb, upper=ub)
            res_lso_ss_dist = euclidean(phi, res_lso_ss.minimizer)
            push!(res_lso_ss_dist_array, res_lso_ss_dist)
        catch
            push!(res_lso_ss_dist_array, -.5)
        end

        try
            lsq_ss_sum(p) = sum(abs2.(loss.(lsq_ss(p))))
            res_opt_ss = opt.optimize(lsq_ss_sum, lb, ub, p0)
            res_opt_ss_dist = euclidean(phi, res_opt_ss.minimizer)
            push!(res_opt_ss_dist_array, res_opt_ss_dist)
        catch
            push!(res_opt_ss_dist_array, -.5)
        end

    end
    p_am = plot(res_lsf_ss_dist_array, title="Single Shooting Errors for Problem $i", label="LsqFit")
    plot!(p_am, res_lso_ss_dist_array, label="LsqOptim")
    #plot!(p_am, res_opt_ss_dist_array, label="Optim")
    display(p_am)

    p_ss = plot(res_lsf_am_dist_array, title="Adams-Moulton Errors for Problem $i", label="LsqFit")
    plot!(p_ss, res_lso_am_dist_array, label="LsqOptim")
    #plot!(p_ss, res_opt_am_dist_array, label="Optim")
    display(p_ss)

    p_am_ss = plot(res_lsf_ss_dist_array, title="SS vs AM Errors for Problem $i", label="Single Shooting LsqFit")
    plot!(p_am_ss, res_lsf_am_dist_array, label="Adams-Moulton LsqFit")
    plot!(p_am_ss, res_lso_ss_dist_array, label="Single Shooting LSOptim")
    plot!(p_am_ss, res_lso_am_dist_array, label="Adams-Moulton LSOptim")
    display(p_am_ss)
end
