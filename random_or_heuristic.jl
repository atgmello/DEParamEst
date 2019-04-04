#=>
old_precision = precision(BigFloat)
new_precision = 2048
setprecision(new_precision)
setprecision(old_precision)
<=#
using Suppressor

plotlyjs()
gr()

struct ErrorTimeData
    error::AbstractArray
    time::AbstractArray
end

for var in [0.1]
	for sam in 50:40:50
	    for i_prob in [1,6,8]
	        res_am = ErrorTimeData([], [])

	        res_ds = ErrorTimeData([], [])

	        res_am_g = ErrorTimeData([], [])

	        res_ds_g = ErrorTimeData([], [])

	        res_ss = ErrorTimeData([], [])

	        res_ss_am = ErrorTimeData([], [])

	        res_ss_am_g = ErrorTimeData([], [])

	        res_ms = ErrorTimeData([], [])

	        #print("\n----- Getting results for Flouda's Problem number $i -----\n")
	        p_solve = problem_set[i_prob]
	        fun = p_solve.fun
	        phi = p_solve.phi
	        bounds = p_solve.bounds
	        ini_cond = p_solve.data[:,1]
	        lb = bounds[1]
	        ub = bounds[end]

	        # Floudas Data
	        #=>
	        t = p_solve.t
	        data = p_solve.data
	        plot_data = plot(t,data')
	        display(plot_data)
	        <=#

	        # Artificial Data
	        t = p_solve.t
	        if sam == 1
	            t = range(t[1], stop=t[end], length=length(t))
	        else
	            t = range(t[1], stop=t[end], length=sam)
	        end
	        tspan = (t[1], t[end])
	        ode_prob = ODEProblem(p_solve.fun, ini_cond, tspan, phi)
	        ode_sol  = solve(ode_prob, AutoVern9(Rodas5()), saveat=reduce(vcat, t))
	        data_original = reduce(hcat, ode_sol.u)
	        data = copy(data_original)
	        #var = 0.05
	        add_noise!(data, var)
	        plot_data = plot(t,data')
	        display(plot_data)

	        linear(x) = x
	        #loss = soft_l1
	        loss = linear
	        SAMIN_options = opt.Options(x_tol=10^-6, f_tol=10^-12, iterations=10^6)
	        Grad_options = opt.Options(x_tol=10^-6, f_tol=10^-12, iterations=10^6)
	        inner_optimizer = opt.LBFGS()

	        for rep in 1:5
	            data = copy(data_original)
	            add_noise!(data, var)
	            p0 = rand_guess(bounds)

	            p0_est = p0
				data_shooting_error(p) = sum(abs2.(loss.(data_shooting_estimator(p, data, t, p_solve.fun))))
				try
                    od = opt.OnceDifferentiable(data_shooting_error, p0; autodiff = :forward)
                    timed = @elapsed res_obj = opt.optimize(od, lb, ub, p0, opt.Fminbox(inner_optimizer), Grad_options)
	                #dist = mape(phi, res_obj.minimizer[1:length(phi)])
	                dist = diff_states(fun, ini_cond, (t[1],t[end]), phi, res_obj.minimizer[1:length(phi)])
	                push!(res_ds.error, dist)
	                push!(res_ds.time, timed)
	            catch e
	                @show e
	            end
				try
                    timed = @elapsed res_obj = opt.optimize(data_shooting_error, lb, ub, p0, opt.SAMIN(verbosity=0), SAMIN_options)
					dist = diff_states(fun, ini_cond, (t[1],t[end]), phi, res_obj.minimizer[1:length(phi)])
	                push!(res_ds_g.error, dist)
	                push!(res_ds_g.time, timed)
				catch e
					@show e
				end

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

	            adams_moulton_error(p) = sum(abs2.(loss.(adams_moulton_estimator(p, data, t, p_solve.fun; plot_estimated=false))))
				try
                    timed = @elapsed res_obj = opt.optimize(adams_moulton_error, lb, ub, p0, opt.SAMIN(verbosity=0), SAMIN_options)
	                dist = diff_states(fun, ini_cond, (t[1],t[end]), phi, res_obj.minimizer[1:length(phi)])
	                push!(res_am_g.error, dist)
	                push!(res_am_g.time, timed)
	                p0_est = res_obj.minimizer[1:length(phi)]
				catch e
					@show e
				end
	            try
                    od = opt.OnceDifferentiable(adams_moulton_error, p0; autodiff = :forward)
                    timed = @elapsed res_obj = opt.optimize(od, lb, ub, p0, opt.Fminbox(inner_optimizer), Grad_options)
	                #dist = mape(phi, res_obj.minimizer[1:length(phi)])
	                dist = diff_states(fun, ini_cond, (t[1],t[end]), phi, res_obj.minimizer[1:length(phi)])
	                push!(res_am.error, dist)
	                push!(res_am.time, timed)
	                p0_est = res_obj.minimizer[1:length(phi)]
	            catch e
	                @show e
	            end

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
                    timed = @elapsed res_obj = opt.optimize(lsq_ss_sum, lb, ub, p0, opt.SAMIN(verbosity=0), SAMIN_options)
                    #od = opt.OnceDifferentiable(lsq_ss_sum, p0; autodiff = :forward)
                    #timed = @elapsed res_obj = opt.optimize(od, convert(Array{BigFloat}, p0), convert(Array{BigFloat}, p_solve.bounds[1]), convert(Array{BigFloat}, p_solve.bounds[end]), opt.LBFGS())
                    #timed = @elapsed res_obj = opt.optimize(od, lb, ub, p0, opt.Fminbox(inner_optimizer), Grad_options)
	                #dist = mape(phi, res_obj.minimizer[1:length(phi)])
	                dist = diff_states(fun, ini_cond, (t[1],t[end]), phi, res_obj.minimizer[1:length(phi)])
	                push!(res_ss.error, dist)
	                push!(res_ss.time, timed)
	            catch e
	                @show e
	            end
	            """
	            Multiple shooting: one shooting node for each data point.
	            Similar to our proposed approach.
	            """
	            #=>
	            #ms_lb = zeros(Float64, (length(t)-1)*(length(data[:,1]))+length(phi))
	            mesh_div = ceil(Int, length(t)/8)
	            ms_obj = multiple_shooting_objective(ode_prob,Tsit5(),L2Loss(t,data))
	            ms_lb = zeros(Float64, floor(Int, (length(t)-1)/mesh_div)*(length(data[:,1]))+length(phi))
	            ms_ub = zeros(Float64, floor(Int, (length(t)-1)/mesh_div)*(length(data[:,1]))+length(phi))
	            for curve in 1:length(data[:,1])
	                for tlen in curve:length(data[:,1]):length(ms_lb)
	                    ms_lb[tlen] = minimum(data[curve,:])
	                    ms_ub[tlen] = maximum(data[curve,:])
	                end
	            end
	            ms_lb[end-length(phi)+1:end] = lb
	            ms_ub[end-length(phi)+1:end] = ub
	            ms_p0 = rand_guess([ms_lb, ms_ub])
	            ms_p0[end-length(phi)+1:end] = p0

	            #timed = @elapsed res_obj = opt.optimize(ms_obj, ms_lb, ms_ub, ms_p0, opt.SAMIN(), SAMIN_options)

	            od = opt.OnceDifferentiable(ms_obj, ms_p0; autodiff = :forward)
	            timed = @elapsed res_obj = opt.optimize(od, ms_lb, ms_ub, ms_p0, opt.Fminbox(inner_optimizer), Grad_options)

	            dist = mape(phi, res_obj.minimizer[end-length(phi)+1:end])
	            push!(res_ms[1], dist)
	            push!(res_ms[2], timed)
	            <=#

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
                    timed = @elapsed res_obj = opt.optimize(lsq_ss_sum, lb, ub, p0_est, opt.SAMIN(verbosity=0), SAMIN_options)
	                dist = diff_states(fun, ini_cond, (t[1],t[end]), phi, res_obj.minimizer[1:length(phi)])
	                push!(res_ss_am_g.error, dist)
	                push!(res_ss_am_g.time, timed)
					p0_est = res_obj.minimizer[1:length(phi)]
				catch e
					@show e
				end
	            try
                    od = opt.OnceDifferentiable(lsq_ss_sum, p0; autodiff = :forward)
                    #timed = @elapsed res_obj = opt.optimize(od, convert(Array{BigFloat}, p0_am), opt.LBFGS())
                    timed = @elapsed res_obj = opt.optimize(od, lb, ub, p0_est, opt.Fminbox(inner_optimizer), Grad_options)
	                #dist = mape(phi, res_obj.minimizer[1:length(phi)])
	                dist = diff_states(fun, ini_cond, (t[1],t[end]), phi, res_obj.minimizer[1:length(phi)])
	                push!(res_ss_am.error, dist)
	                push!(res_ss_am.time, timed)
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
		        println("res_am:\n$res_am")
		        println("res_am_g:\n$res_am")
		        println("res_ds:\n$res_ds")
		        println("res_ds_g:\n$res_ds_g")
		        println("res_ss:\n$res_ss")
		        println("res_ss_am:\n$res_ss_am")
		        println("res_ss_am_g:\n$res_ss_am_g")
	        end

	        #res = (1/4)*(res_am[1] + res_ds_1[1] + res_ds_2[1] + res_ds_50[1])
	        #med = sta.quantile(res, 0.85)
	        canvas_new_res = plot(title="DS: Error vs Run: $sam samples, $var noise")
	        #ylims!(canvas_new_res, (-Inf, med+0.2*med))

			filter!(e -> e < 1e5, res_am.error)
			filter!(e -> e < 1e5, res_ds.error)
			filter!(e -> e < 1e5, res_am_g.error)
			filter!(e -> e < 1e5, res_ds_g.error)

        	plot!(canvas_new_res, res_am.error, label="Adams-Moulton")
        	plot!(canvas_new_res, res_ds.error, label="Data Shooting")
        	plot!(canvas_new_res, res_am_g.error, label="G Adams-Moulton")
	        plot!(canvas_new_res, res_ds_g.error, label="G Data Shooting")
	        display(canvas_new_res)

	        #res = (1/4)*(res_am[2] + res_ds_1[2] + res_ds_2[2] + res_ds_50[2])
	        #med = sta.quantile(res, 0.85)
	        canvas_new_time = plot(title="DS: Time vs Run: $sam samples, $var noise")
	        #ylims!(canvas_new_time, (-Inf, med+0.2*med))
        	plot!(canvas_new_time, res_am.time[2:end], label="Adams-Moulton")
	        plot!(canvas_new_time, res_ds.time[2:end], label="Data Shooting")
	        plot!(canvas_new_time, res_am_g.time[2:end], label="G Adams-Moulton")
	        plot!(canvas_new_time, res_ds_g.time[2:end], label="G Data Shooting")
	        display(canvas_new_time)

	        #res = (1/2)*(res_ss[1] + res_ss_am[1])
	        #med = sta.quantile(res, 0.85)
	        canvas_cla_res = plot(title="SS: Error vs Run: $sam samples, $var noise")
	        #ylims!(canvas_cla_res, (-Inf, med+0.2*med))
			filter!(e -> e < 1e5, res_ss.error)
			filter!(e -> e < 1e5, res_ss_am.error)
			filter!(e -> e < 1e5, res_ss_am_g.error)

	        plot!(canvas_cla_res, res_ss.error, label="G Single Shooting")
	        plot!(canvas_cla_res, res_ss_am_g.error, label="G DS + Single Shooting")
	        plot!(canvas_cla_res, res_ss_am.error, label="DS + Single Shooting")
	        display(canvas_cla_res)

	        canvas_cla_res = plot(title="SS: Error vs Run: $sam samples, $var noise")
	        plot!(canvas_cla_res, res_ss_am_g.error, label="G DS + Single Shooting")
	        plot!(canvas_cla_res, res_ss_am.error, label="DS + Single Shooting")
	        display(canvas_cla_res)

	        #res = (1/2)*(res_ss[2] + res_ss_am[2])
	        #med = sta.quantile(res, 0.85)
	        canvas_cla_time = plot(title="SS: Time vs Run: $sam samples, $var noise")
	        #ylims!(canvas_cla_time, (-Inf, med+0.2*med))
	        plot!(canvas_cla_time, res_ss.time[2:end], label=" G Single Shooting")
	        plot!(canvas_cla_time, res_ss_am_g.time[2:end], label="G DS + Single Shooting")
	        plot!(canvas_cla_time, res_ss_am.time[2:end], label="DS + Single Shooting")
	        display(canvas_cla_time)

	        canvas_cla_time = plot(title="SS: Time vs Run: $sam samples, $var noise")
	        plot!(canvas_cla_time, res_ss_am_g.time[2:end], label="G DS + Single Shooting")
	        plot!(canvas_cla_time, res_ss_am.time[2:end], label="DS + Single Shooting")
	        display(canvas_cla_time)

	        #=>
	        canvas_cla_res = plot(title="MS: Error vs Run: $sam samples, $var noise")
	        plot!(canvas_cla_res, res_ms[1], label="Multiple Shooting")
	        display(canvas_cla_res)
	        canvas_cla_time = plot(title="MS: Time vs Run")
	        plot!(canvas_cla_time, res_ms[2][2:end], label="Multiple Shooting")
	        display(canvas_cla_time)
	        <=#
	    end
	end
end
