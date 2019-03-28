old_precision = precision(BigFloat)
new_precision = 2048
setprecision(new_precision)
setprecision(old_precision)

for i in 1:6
    res_lsf_ss_dist_array = []
    res_lso_ss_dist_array = []
    res_opt_ss_dist_array = []
    res_opt_ss_dist_rand_array = []

    res_lsf_am_dist_array = []
    res_lso_am_dist_array = []
    res_opt_am_dist_array = []

    #print("\n----- Getting results for Flouda's Problem number $i -----\n")
    p_solve = problem_set[i]

    # Floudas Data
    #data = p_solve.data

    # Artificial Data
    t = p_solve.t
    t = range(t[1], stop=t[end], length=50)
    tspan = (t[1], t[end])
    ini_cond = p_solve.data[:,1]
    ode_prob = ODEProblem(p_solve.fun, ini_cond, tspan, p_solve.phi)
    ode_sol  = solve(ode_prob, lsoda(), saveat=reduce(vcat, t))
    data_original = reduce(hcat, ode_sol.u)
    data = copy(data_original)
    m,n = size(data)
    for i in 1:m
        for j in 1:n
            if data[i,j] != 0.
                data[i,j] += rand(Uniform(-0.2*data[i,j], 0.2*data[i,j]))
            end
        end
    end
    plot_data = plot(transpose(data))
    display(plot_data)

    linear(x) = x
    #loss = soft_l1
    loss = linear
    # --- Defining functions ---

    function lsq_ss_curvefit(time_array, phi)
        tspan = (t[1], t[end])
        ini_cond = data[:,1]
        oprob = ODEProblem(p_solve.fun, ini_cond, tspan, p_solve.phi)
        osol  = solve(oprob, Tsit5(), saveat=reduce(vcat, t))
        estimated = reduce(hcat, osol.u)
        return vec(estimated)
    end

    function lsq_am_curvefit(time_array, phi)
        num_state_variables, num_samples = size(data)
        set_precision = typeof(phi[1])

        estimated = zeros(num_samples*num_state_variables)
        estimated = reshape(estimated, (num_state_variables, num_samples))
        estimated[:, 1] = data[:,1] #Initial conditions are stored at x_dot_num's first column
        estimated = convert(Array{set_precision}, estimated)

        for i in range(1, stop=num_samples-1)
            delta_t = t[i+1] - t[i]
            x_k_0 = data[:, i]
            x_k_1 = data[:, i+1]

            f_eval_0 = zeros(set_precision, num_state_variables)
            p_solve.fun(f_eval_0, x_k_0, phi, 0)
            f_eval_1 = zeros(set_precision, num_state_variables)
            p_solve.fun(f_eval_1, x_k_1, phi, 0)

            x_k_1_est = x_k_0 + (1/2)*delta_t*(f_eval_0+f_eval_1)
            estimated[:, i+1] = x_k_1_est
        end

        return vec(estimated)
    end

    for j in 1:50
        if j%10 == 0
            data = copy(data_original)
            m,n = size(data)
            for i in 1:m
                for j in 1:n
                    if data[i,j] != 0.
                        data[i,j] += rand(Uniform(-0.2*data[i,j], 0.2*data[i,j]))
                    end
                end
            end
        end
        #print("\n--- Iteration $j---\n")
        p0 = [rand(Uniform(p_solve.bounds[1][i], p_solve.bounds[end][i]))
                        for i in 1:length(p_solve.phi)]
        # ----- Adams-Moulton -----

        #lsq_am(p) = adams_moulton_fourth_estimator(p, data, t, p_solve.fun; plot_estimated=false)
        #res_lso_am = lso.optimize(lsq_am, p0, lso.Dogleg(), lower=p_solve.bounds[1], upper=p_solve.bounds[end])
        #res_lso_am_dist = euclidean(p_solve.phi, res_lso_am.minimizer)
        #push!(res_lso_am_dist_array, res_lso_am_dist)
        #p0 = res_lso_am.minimizer

        # two_stage_method: Errors on problems 4 and 6
        #lso_am = two_stage_method(ode_prob,t,data; mpg_autodiff=false)
        #res_lso_am = opt.optimize(lso_am, p_solve.bounds[1], p_solve.bounds[end], p0)
        #res_lso_am_dist = euclidean(p_solve.phi, res_lso_am.minimizer)
        #push!(res_lso_am_dist_array, res_lso_am_dist)

        #res_lsf_am = lsf.curve_fit(lsq_am_curvefit, t, vec(data), p0, lower=p_solve.bounds[1], upper=p_solve.bounds[end])
        #res_lsf_am_dist = euclidean(p_solve.phi, res_lsf_am.param)
        #push!(res_lsf_am_dist_array, res_lsf_am_dist)
        #p0 = res_lsf_am.param

        lsq_am_sum(p) = sum(abs2.(loss.(single_multiple_adams_shooting(p, data, t, p_solve.fun; plot_estimated=false))))
        res_opt_am = opt.optimize(lsq_am_sum, p_solve.bounds[1], p_solve.bounds[end], p0; autodiff=:forward)
        res_opt_am_dist = euclidean(p_solve.phi, res_opt_am.minimizer)
        push!(res_lsf_am_dist_array, res_opt_am_dist)

        lsq_am_sum(p) = sum(abs2.(loss.(sm_mean_shooting(p, data, t, p_solve.fun; plot_estimated=false))))
        res_opt_am = opt.optimize(lsq_am_sum, p_solve.bounds[1], p_solve.bounds[end], p0; autodiff=:forward)
        res_opt_am_dist = euclidean(p_solve.phi, res_opt_am.minimizer)
        push!(res_lso_am_dist_array, res_opt_am_dist)

        lsq_am_sum(p) = sum(abs2.(loss.(adams_moulton_estimator(p, data, t, p_solve.fun; plot_estimated=false))))
        res_opt_am = opt.optimize(lsq_am_sum, p_solve.bounds[1], p_solve.bounds[end], p0; autodiff=:forward)
        res_opt_am_dist = euclidean(p_solve.phi, res_opt_am.minimizer)
        push!(res_opt_am_dist_array, res_opt_am_dist)

        # ----- Single Shooting -----


        #res_lsf_ss = lsf.curve_fit(lsq_ss_curvefit, t, vec(data), p0, lower=lb, upper=ub)
        #res_lsf_ss_dist = euclidean(phi, res_lsf_ss.param)
        #push!(res_lsf_ss_dist_array, res_lsf_ss_dist)

        lsq_ss(p) = vec(single_shooting_estimator(convert(Array{BigFloat}, p), convert(Array{BigFloat}, data), convert(Array{BigFloat}, t), p_solve.fun))
        #res_lso_ss = lso.optimize(lsq_ss, p0, lso.Dogleg(), lower=lb, upper=ub)
        #res_lso_ss_dist = euclidean(phi, res_lso_ss.minimizer)
        #push!(res_lso_ss_dist_array, res_lso_ss_dist)

        lsq_ss_sum(p) = sum(abs2.(loss.(lsq_ss(p))))
        res_opt_ss_rand = opt.optimize(lsq_ss_sum, convert(Array{BigFloat}, p_solve.bounds[1]), convert(Array{BigFloat}, p_solve.bounds[end]), convert(Array{BigFloat}, p0))
        res_opt_ss_dist_rand = euclidean(p_solve.phi, res_opt_ss_rand.minimizer)
        push!(res_opt_ss_dist_rand_array, res_opt_ss_dist_rand)

        p0 = res_opt_am.minimizer
        res_opt_ss = opt.optimize(lsq_ss_sum, convert(Array{BigFloat}, p_solve.bounds[1]), convert(Array{BigFloat}, p_solve.bounds[end]), convert(Array{BigFloat}, p0))
        res_opt_ss_dist = euclidean(p_solve.phi, res_opt_ss.minimizer)
        push!(res_opt_ss_dist_array, res_opt_ss_dist)

    end
    p_am = plot(res_opt_am_dist_array, title="Adams-Moulton Errors for Problem $i", label="Optim")
    plot!(p_am, res_lso_am_dist_array, label="SMSM")
    plot!(p_am, res_lsf_am_dist_array, label="SMS")
    display(p_am)

    p_ss = plot(res_opt_ss_dist_rand_array, title="Single Shooting Errors for Problem $i", label="Optim Rand")
    plot!(p_ss, res_opt_ss_dist_array, label="Optim Am")
    display(p_ss)

    p_am_ss = plot(res_opt_ss_dist_array, title="SS Optim AM for Problem $i")
    #plot!(p_am_ss, res_opt_am_dist_array, label="Adams-Moulton Optim")
    #plot!(p_am_ss, res_lso_ss_dist_array, label="Single Shooting LSOptim")
    #plot!(p_am_ss, res_lso_am_dist_array, label="Adams-Moulton LSOptim")
    display(p_am_ss)
end
