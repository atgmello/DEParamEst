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

fun = floudas_one


data = [1.0 0.57353 0.328937 0.188654 0.108198 0.0620545 0.0355906 0.0204132 0.011708 0.00671499;
        0.0 0.401566 0.589647 0.659731 0.666112 0.639512 0.597179 0.54867 0.499168 0.451377]
t = [0.0, 0.111111, 0.222222, 0.333333, 0.444444, 0.555556, 0.666667, 0.777778, 0.888889, 1.0]

<=#

for i in 1:6
    res_lsf_ss_dist_array = []
    res_lso_ss_dist_array = []
    res_opt_ss_dist_array = []
    res_opt_ss_dist_rand_array = []

    res_lsf_am_dist_array = []
    res_lso_am_dist_array = []
    res_opt_am_dist_array = []

    print("\n----- Getting results for Flouda's Problem number $i -----\n")
    p_solve = problem_set[i]

    # Floudas Data

    # Artificial Data
    #t = t_array[i]
    #tspan = (t[1], t[end])
    #ode_prob = ODEProblem(fun, ini_cond, tspan, phi)
    #ode_sol  = solve(ode_prob, lsoda(), saveat=reduce(vcat, t))
    #data = reduce(hcat, ode_sol.u)

    linear(x) = x
    #loss = soft_l1
    loss = linear
    # --- Defining functions ---

    function lsq_ss_curvefit(time_array, phi)
        tspan = (p_solve.t[1], p_solve.t[end])
        ini_cond = p_solve.data[:,1]
        oprob = ODEProblem(p_solve.fun, ini_cond, tspan, p_solve.phi)
        osol  = solve(oprob, Tsit5(), saveat=reduce(vcat, t))
        estimated = reduce(hcat, osol.u)
        return vec(estimated)
    end

    function lsq_am_curvefit(time_array, phi)
        num_state_variables, num_samples = size(p_solve.data)
        set_precision = typeof(phi[1])

        estimated = zeros(num_samples*num_state_variables)
        estimated = reshape(estimated, (num_state_variables, num_samples))
        estimated[:, 1] = p_solve.data[:,1] #Initial conditions are stored at x_dot_num's first column
        estimated = convert(Array{set_precision}, estimated)

        for i in range(1, stop=num_samples-1)
            delta_t = p_solve.t[i+1] - p_solve.t[i]
            x_k_0 = p_solve.data[:, i]
            x_k_1 = p_solve.data[:, i+1]

            f_eval_0 = zeros(set_precision, num_state_variables)
            p_solve.fun(f_eval_0, x_k_0, phi, 0)
            f_eval_1 = zeros(set_precision, num_state_variables)
            p_solve.fun(f_eval_1, x_k_1, phi, 0)

            x_k_1_est = x_k_0 + (1/2)*delta_t*(f_eval_0+f_eval_1)
            estimated[:, i+1] = x_k_1_est
        end

        return vec(estimated)
    end

    for j in 1:5
        print("\n--- Iteration $j---\n")
        p0 = [rand(Uniform(p_solve.bounds[1][i], p_solve.bounds[end][i]))
                        for i in 1:length(p_solve.phi)]
        # ----- Adams-Moulton -----

        lsq_am(p) = adams_moulton_estimator(p, p_solve.data, p_solve.t, p_solve.fun)
        res_lso_am = lso.optimize(lsq_am, p0, lso.Dogleg(), lower=p_solve.bounds[1], upper=p_solve.bounds[end])
        res_lso_am_dist = euclidean(p_solve.phi, res_lso_am.minimizer)
        push!(res_lso_am_dist_array, res_lso_am_dist)
        #p0 = res_lso_am.minimizer

        res_lsf_am = lsf.curve_fit(lsq_am_curvefit, p_solve.t, vec(p_solve.data), p0, lower=p_solve.bounds[1], upper=p_solve.bounds[end])
        res_lsf_am_dist = euclidean(p_solve.phi, res_lsf_am.param)
        push!(res_lsf_am_dist_array, res_lsf_am_dist)
        #p0 = res_lsf_am.param

        lsq_am_sum(p) = sum(abs2.(loss.(adams_moulton_estimator(p, p_solve.data, p_solve.t, p_solve.fun))))
        res_opt_am = opt.optimize(lsq_am_sum, p_solve.bounds[1], p_solve.bounds[end], p0)
        res_opt_am_dist = euclidean(p_solve.phi, res_opt_am.minimizer)
        push!(res_opt_am_dist_array, res_opt_am_dist)

        # ----- Single Shooting -----


        #res_lsf_ss = lsf.curve_fit(lsq_ss_curvefit, t, vec(data), p0, lower=lb, upper=ub)
        #res_lsf_ss_dist = euclidean(phi, res_lsf_ss.param)
        #push!(res_lsf_ss_dist_array, res_lsf_ss_dist)

        lsq_ss(p) = vec(single_shooting_estimator(p, p_solve.data, p_solve.t, p_solve.fun))
        #res_lso_ss = lso.optimize(lsq_ss, p0, lso.Dogleg(), lower=lb, upper=ub)
        #res_lso_ss_dist = euclidean(phi, res_lso_ss.minimizer)
        #push!(res_lso_ss_dist_array, res_lso_ss_dist)

        lsq_ss_sum(p) = sum(abs2.(loss.(lsq_ss(p))))
        res_opt_ss_rand = opt.optimize(lsq_ss_sum, convert(Array{BigFloat}, p_solve.bounds[1]), convert(Array{BigFloat}, p_solve.bounds[end]), convert(Array{BigFloat}, p0), opt.Fminbox(opt.NelderMead()))
        res_opt_ss_dist_rand = euclidean(p_solve.phi, res_opt_ss_rand.minimizer)
        push!(res_opt_ss_dist_rand_array, res_opt_ss_dist_rand)

        p0 = res_opt_am.minimizer
        res_opt_ss = opt.optimize(lsq_ss_sum, convert(Array{BigFloat}, p_solve.bounds[1]), convert(Array{BigFloat}, p_solve.bounds[end]), convert(Array{BigFloat}, p0), opt.Fminbox(opt.NelderMead()))
        res_opt_ss_dist = euclidean(p_solve.phi, res_opt_ss.minimizer)
        push!(res_opt_ss_dist_array, res_opt_ss_dist)

    end
    p_am = plot(res_opt_am_dist_array, title="Adams-Moulton Errors for Problem $i", label="Optim")
    plot!(p_am, res_lso_am_dist_array, label="LsqOptim")
    plot!(p_am, res_lsf_am_dist_array, label="LsqFit")
    display(p_am)

    p_ss = plot(res_opt_ss_dist_rand_array, title="Single Shooting Errors for Problem $i", label="Optim Rand")
    plot!(p_ss, res_opt_ss_dist_array, label="Optim Am")
    display(p_ss)

    p_am_ss = plot(res_opt_ss_dist_array, title="SS vs AM Errors for Problem $i", label="Single Shooting Optim")
    plot!(p_am_ss, res_opt_am_dist_array, label="Adams-Moulton Optim")
    #plot!(p_am_ss, res_lso_ss_dist_array, label="Single Shooting LSOptim")
    #plot!(p_am_ss, res_lso_am_dist_array, label="Adams-Moulton LSOptim")
    display(p_am_ss)
end
