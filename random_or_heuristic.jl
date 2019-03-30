old_precision = precision(BigFloat)
new_precision = 2048
setprecision(new_precision)
setprecision(old_precision)

plotlyjs()
gr()

for i in 1:7
    res_am = []
    push!(res_am, [])
    push!(res_am, [])

    res_ds_1 = []
    push!(res_ds_1, [])
    push!(res_ds_1, [])

    res_ds_2 = []
    push!(res_ds_2, [])
    push!(res_ds_2, [])

    res_ds_50 = []
    push!(res_ds_50, [])
    push!(res_ds_50, [])

    res_ss = []
    push!(res_ss, [])
    push!(res_ss, [])

    res_ss_am = []
    push!(res_ss_am, [])
    push!(res_ss_am, [])

    res = []

    #print("\n----- Getting results for Flouda's Problem number $i -----\n")
    p_solve = problem_set[i]
    phi = p_solve.phi
    bounds = p_solve.bounds
    lb = bounds[1]
    ub = bounds[end]

    # Floudas Data
    #data = p_solve.data

    # Artificial Data
    t = p_solve.t
    t = range(t[1], stop=t[end], length=8)
    tspan = (t[1], t[end])
    ini_cond = p_solve.data[:,1]
    ode_prob = ODEProblem(p_solve.fun, ini_cond, tspan, phi)
    ode_sol  = solve(ode_prob, AutoVern9(Rodas5()), saveat=reduce(vcat, t))
    data_original = reduce(hcat, ode_sol.u)
    data = copy(data_original)
    var = 0.05
    add_noise!(data, var)
    plot_data = plot(transpose(data))
    display(plot_data)

    linear(x) = x
    #loss = soft_l1
    loss = linear
    SAMIN_options = opt.Options(x_tol=10^-10, f_tol=10^-20, iterations=100000, time_limit=30)
    Grad_options = opt.Options(x_tol=10^-10, f_tol=10^-20, iterations=100000, time_limit=30)
    inner_optimizer = opt.LBFGS()

    for j in 1:50
        #=>
        if j%10 == 0
            data = copy(data_original)
            add_noise!(data, var)
        end
        <=#
        p0 = rand_guess(bounds)

        lsq_am_sum(p) = sum(abs2.(loss.(data_shooting_estimator(p, data, t, p_solve.fun; step=1, plot_estimated=false))))
        #timed = @elapsed res_obj = opt.optimize(lsq_am_sum, lb, ub, p0, opt.SAMIN(), opt_options)
        od = opt.OnceDifferentiable(lsq_am_sum, p0; autodiff = :forward)
        timed = @elapsed res_obj = opt.optimize(od, lb, ub, p0, opt.Fminbox(inner_optimizer), Grad_options)
        dist = mape(phi, res_obj.minimizer[1:length(phi)])
        push!(res_ds_1[1], dist)
        push!(res_ds_1[2], timed)

        #=>
        opt_gn = nlo.Opt(:LN_NELDERMEAD, length(phi))
        nlo.lower_bounds!(opt_gn, lb)
        nlo.upper_bounds!(opt_gn, ub)
        nlo.min_objective!(opt_gn, lsq_am_sum)
        nlo.xtol_rel!(opt_gn,1e-12)
        nlo.maxeval!(opt_gn, 20000)
        nlo.maxtime!(opt_gn, 30)
        (minf,minx,ret) = NLopt.optimize(opt_gn,p0)
        dist = mape(phi, minx)
        push!(res_am, dist)
        <=#

        lsq_am_sum(p) = sum(abs2.(loss.(data_shooting_estimator(p, data, t, p_solve.fun; step=2, plot_estimated=false))))
        #timed = @elapsed res_obj = opt.optimize(lsq_am_sum, lb, ub, p0, opt.SAMIN(), opt_options)
        od = opt.OnceDifferentiable(lsq_am_sum, p0; autodiff = :forward)
        timed = @elapsed res_obj = opt.optimize(od, lb, ub, p0, opt.Fminbox(inner_optimizer), Grad_options)
        dist = mape(phi, res_obj.minimizer[1:length(phi)])
        push!(res_ds_2[1], dist)
        push!(res_ds_2[2], timed)

        #=>
        opt_gn = nlo.Opt(:LN_NELDERMEAD, length(phi))
        nlo.lower_bounds!(opt_gn, lb)
        nlo.upper_bounds!(opt_gn, ub)
        nlo.min_objective!(opt_gn, lsq_am_sum)
        nlo.xtol_rel!(opt_gn,1e-12)
        nlo.maxeval!(opt_gn, 20000)
        nlo.maxtime!(opt_gn, 30)
        (minf,minx,ret) = NLopt.optimize(opt_gn,p0)
        dist = mape(phi, minx)
        push!(res_ds_1, dist)
        <=#


        lsq_am_sum(p) = sum(abs2.(loss.(data_shooting_estimator(p, data, t, p_solve.fun; step=50, plot_estimated=false))))
        #timed = @elapsed res_obj = opt.optimize(lsq_am_sum, lb, ub, p0, opt.SAMIN(), opt_options)
        od = opt.OnceDifferentiable(lsq_am_sum, p0; autodiff = :forward)
        timed = @elapsed res_obj = opt.optimize(od, lb, ub, p0, opt.Fminbox(inner_optimizer), Grad_options)
        dist = mape(phi, res_obj.minimizer[1:length(phi)])
        push!(res_ds_50[1], dist)
        push!(res_ds_50[2], timed)

        #=>
        opt_gn = nlo.Opt(:LN_NELDERMEAD, length(phi))
        nlo.lower_bounds!(opt_gn, lb)
        nlo.upper_bounds!(opt_gn, ub)
        nlo.min_objective!(opt_gn, lsq_am_sum)
        nlo.xtol_rel!(opt_gn,1e-12)
        nlo.maxeval!(opt_gn, 20000)
        nlo.maxtime!(opt_gn, 30)
        (minf,minx,ret) = NLopt.optimize(opt_gn,p0)
        dist = mape(phi, minx)
        push!(res_ds_2, dist)
        <=#

        lsq_am_sum(p) = sum(abs2.(loss.(adams_moulton_estimator(p, data, t, p_solve.fun; plot_estimated=false))))
        #timed = @elapsed res_obj = opt.optimize(lsq_am_sum, lb, ub, p0, opt.SAMIN(), opt_options)
        od = opt.OnceDifferentiable(lsq_am_sum, p0; autodiff = :forward)
        timed = @elapsed res_obj = opt.optimize(od, lb, ub, p0, opt.Fminbox(inner_optimizer), Grad_options)
        dist = mape(phi, res_obj.minimizer[1:length(phi)])
        push!(res_am[1], dist)
        push!(res_am[2], timed)
        p0_am = res_obj.minimizer[1:length(phi)]

        #=>
        opt_gn = nlo.Opt(:LN_NELDERMEAD, length(phi))
        nlo.lower_bounds!(opt_gn, lb)
        nlo.upper_bounds!(opt_gn, ub)
        nlo.min_objective!(opt_gn, lsq_am_sum)
        nlo.xtol_rel!(opt_gn,1e-12)
        nlo.maxeval!(opt_gn, 20000)
        nlo.maxtime!(opt_gn, 30)
        (minf,minx,ret) = NLopt.optimize(opt_gn,p0)
        dist = mape(phi, minx)
        push!(res_am, dist)
        p0_am = minx
        <=#

        # ----- Single Shooting -----

        #lsq_ss(p) = vec(single_shooting_estimator(convert(Array{BigFloat}, p), convert(Array{BigFloat}, data), convert(Array{BigFloat}, t), p_solve.fun))
        lsq_ss(p) = vec(single_shooting_estimator(p, data, t, p_solve.fun))
        lsq_ss_sum(p) = sum(abs2.(loss.(lsq_ss(p))))

        try
            od = opt.OnceDifferentiable(lsq_ss_sum, p0; autodiff = :forward)
            timed = @elapsed res_obj = opt.optimize(od, lb, ub, p0, opt.Fminbox(inner_optimizer), Grad_options)
            #timed = @elapsed res_obj = opt.optimize(od, convert(Array{BigFloat}, p0), convert(Array{BigFloat}, p_solve.bounds[1]), convert(Array{BigFloat}, p_solve.bounds[end]), opt.LBFGS())
            #timed = @elapsed res_obj = opt.optimize(lsq_ss_sum, lb, ub, p0, opt.SAMIN(), opt_options)
            dist = mape(phi, res_obj.minimizer[1:length(phi)])
            push!(res_ss[1], dist)
            push!(res_ss[2], timed)
        catch e
            @show e
        end

        #=>
        opt_gn = nlo.Opt(:LN_NELDERMEAD, length(phi))
        nlo.lower_bounds!(opt_gn, lb)
        nlo.upper_bounds!(opt_gn, ub)
        nlo.min_objective!(opt_gn, lsq_ss_sum)
        nlo.xtol_rel!(opt_gn,1e-12)
        nlo.maxeval!(opt_gn, 20000)
        nlo.maxtime!(opt_gn, 30)
        (minf,minx,ret) = NLopt.optimize(opt_gn,p0)
        dist = mape(phi, minx)
        push!(res_ss, dist)
        <=#

        try
            #timed = @elapsed res_obj = opt.optimize(od, convert(Array{BigFloat}, p0_am), opt.LBFGS())
            timed = @elapsed res_obj = opt.optimize(od, lb, ub, p0_am, opt.Fminbox(inner_optimizer), Grad_options)
            #timed = @elapsed res_obj = opt.optimize(lsq_ss_sum, lb, ub, p0_am, opt.SAMIN(), opt_options)
            dist = mape(phi, res_obj.minimizer[1:length(phi)])
            push!(res_ss_am[1], dist)
            push!(res_ss_am[2], timed)
        catch e
            @show e
        end

        #=>
        opt_gn = nlo.Opt(:LN_NELDERMEAD, length(phi))
        nlo.lower_bounds!(opt_gn, lb)
        nlo.upper_bounds!(opt_gn, ub)
        nlo.min_objective!(opt_gn, lsq_ss_sum)
        nlo.xtol_rel!(opt_gn,1e-12)
        nlo.maxeval!(opt_gn, 20000)
        nlo.maxtime!(opt_gn, 30)
        (minf,minx,ret) = NLopt.optimize(opt_gn,p0_am)
        dist = mape(phi, minx)
        push!(res_ss_am, dist)
        <=#

    end

    #res = (1/4)*(res_am[1] + res_ds_1[1] + res_ds_2[1] + res_ds_50[1])
    #med = sta.quantile(res, 0.85)
    canvas_new_res = plot(title="DS: Error vs Run")
    #ylims!(canvas_new_res, (-Inf, med+0.2*med))
    plot!(canvas_new_res, res_am[1], label="Adams-Moulton")
    plot!(canvas_new_res, res_ds_1[1], label="Data Shooting 1")
    plot!(canvas_new_res, res_ds_2[1], label="Data Shooting 2")
    plot!(canvas_new_res, res_ds_50[1], label="Data Shooting 50")
    display(canvas_new_res)

    #res = (1/4)*(res_am[2] + res_ds_1[2] + res_ds_2[2] + res_ds_50[2])
    #med = sta.quantile(res, 0.85)
    canvas_new_time = plot(title="DS: Time vs Run")
    #ylims!(canvas_new_time, (-Inf, med+0.2*med))
    plot!(canvas_new_time, res_am[2], label="Adams-Moulton")
    plot!(canvas_new_time, res_ds_1[2], label="Data Shooting 1")
    plot!(canvas_new_time, res_ds_2[2], label="Data Shooting 2")
    plot!(canvas_new_time, res_ds_50[2], label="Data Shooting 50")
    display(canvas_new_time)

    #res = (1/2)*(res_ss[1] + res_ss_am[1])
    #med = sta.quantile(res, 0.85)
    canvas_cla_res = plot(title="SS: Error vs Run")
    #ylims!(canvas_cla_res, (-Inf, med+0.2*med))
    plot!(canvas_cla_res, res_ss[1], label="MS Single Shooting")
    plot!(canvas_cla_res, res_ss_am[1], label="AM Single Shooting")
    display(canvas_cla_res)

    #res = (1/2)*(res_ss[2] + res_ss_am[2])
    #med = sta.quantile(res, 0.85)
    canvas_cla_time = plot(title="SS: Time vs Run")
    #ylims!(canvas_cla_time, (-Inf, med+0.2*med))
    plot!(canvas_cla_time, res_ss[2], label="MS Single Shooting")
    plot!(canvas_cla_time, res_ss_am[2], label="AM Single Shooting")
    display(canvas_cla_time)
end
plotly()

print("End")
import Statistics
const sta = Statistics

#lsq_am(p) = adams_moulton_fourth_estimator(p, data, t, p_solve.fun; plot_estimated=false)
#res_lso_am = lso.optimize(lsq_am, p0, lso.Dogleg(), lower=p_solve.bounds[1], upper=p_solve.bounds[end])
#res_lso_am_dist = mape(phi, res_lso_am.minimizer)
#push!(res_lso_am_dist_array, res_lso_am_dist)
#p0 = res_lso_am.minimizer

# two_stage_method: Errors on problems 4 and 6
#lso_am = two_stage_method(ode_prob,t,data; mpg_autodiff=false)
#res_lso_am = opt.optimize(lso_am, p_solve.bounds[1], p_solve.bounds[end], p0)
#res_lso_am_dist = mape(phi, res_lso_am.minimizer)
#push!(res_lso_am_dist_array, res_lso_am_dist)

#res_lsf_am = lsf.curve_fit(lsq_am_curvefit, t, vec(data), p0, lower=p_solve.bounds[1], upper=p_solve.bounds[end])
#res_lsf_am_dist = mape(phi, res_lsf_am.param)
#push!(res_lsf_am_dist_array, res_lsf_am_dist)
#p0 = res_lsf_am.param

#res_lsf_ss = lsf.curve_fit(lsq_ss_curvefit, t, vec(data), p0, lower=lb, upper=ub)
#res_lsf_ss_dist = mape(phi, res_lsf_ss.param)
#push!(res_lsf_ss_dist_array, res_lsf_ss_dist)

#res_lso_ss = lso.optimize(lsq_ss, p0, lso.Dogleg(), lower=lb, upper=ub)
#res_lso_ss_dist = mape(phi, res_lso_ss.minimizer)
#push!(res_lso_ss_dist_array, res_lso_ss_dist)

f(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
function g!(storage, x)
    storage[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
    storage[2] = 200.0 * (x[2] - x[1]^2)
end

lower = [1.25, -2.1]
upper = [Inf, Inf]
initial_x = [2.0, 2.0]
od = opt.OnceDifferentiable(f, g!, initial_x)
results = opt.optimize(od, lower, upper, initial_x,  opt.SimulatedAnnealing())

results.minimizer
