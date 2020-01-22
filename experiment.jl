using Suppressor

using DiffEqParamEstim
using DifferentialEquations
using LSODA
using NLsolve
using Statistics
using Distances
using Plots
gr()

import Distributions: Uniform

import Optim
const opt = Optim

using Revise
includet("./utils.jl")
includet("./problem_set.jl")
includet("./objective_function.jl")
import .ODEProblems: get_problem, get_problem_key
import .ObjectiveFunctions: data_shooting, single_shooting, soft_l1
import .Utils: Trace, rand_guess, make_trace, get_best_worst_traces,
				add_noise!, filter_outlier, fill_trace, scale_eval

old_precision = precision(BigFloat)
new_precision = 1024
setprecision(new_precision)
#setprecision(old_precision)
dp = Float64

#model = ["ds","ds_g","ss","ss_g"]
method = ["DS","SS"]

vars = Float16[0.00, 0.025, 0.05, 0.10, 0.20]

results = Dict()

#@suppress begin
    for p in 1:1

		prob_key = get_problem_key(p)
		problem = get_problem(prob_key)
		results[prob_key] = Dict()

		cd("/home/andrew/git/ChemParamEst/plots/experiments/")
		dir = prob_key
		mkdir(dir)
		cd(dir)

        #print("\n----- Getting results for Flouda's Problem number $i -----\n")
        fun = problem.fun
        phi = problem.phi
        bounds = problem.bounds
        ini_cond = problem.data[:,1]
        lb = bounds[1]
        ub = bounds[end]

        # Floudas Data
        #=>
        t = problem.t
        data = problem.data
        plot_data = plot(t,data')
        display(plot_data)
        <=#

		for sam in 5:50:45
			for var in vars
				res = Dict()
				for m in method
		        	res[m] = Dict("error" => dp[],
									"time" => dp[],
									"trace" => Trace([],[]),
									)
				end

		        # Artificial Data
		        t = problem.t
		        if sam == 1
		            t = range(t[1], stop=t[end], length=length(t))
		        else
		            t = range(t[1], stop=t[end], length=sam)
		        end
		        tspan = (t[1], t[end])
		        ode_prob = ODEProblem(fun, ini_cond, tspan, phi)
		        ode_sol  = solve(ode_prob, lsoda(), saveat=reduce(vcat, t))
		        data_original = reduce(hcat, ode_sol.u)
		        data = copy(data_original)
		        plot_data = plot(t,data')
		        display(plot_data)

		        linear(x) = x
		        #loss = soft_l1
		        loss = linear

		        #SAMIN_options = opt.Options(x_tol=10^-12, f_tol=10^-24, iterations=10^6)
		        #Grad_options = opt.Options(x_tol=10^-12, f_tol=10^-24, iterations=10^6)
				SAMIN_options = opt.Options(x_tol=10^-6, f_tol=10^-12,
											iterations=10^4, time_limit=120)
		        Grad_options = opt.Options(x_tol=10^-6, f_tol=10^-12,
											iterations=10^4, time_limit=120, store_trace=true)
		        inner_optimizer = opt.LBFGS()

		        for rep in 1:5
		            data = copy(data_original)
					if var > 0.0
			            add_noise!(data, var)
					end
		            p0 = rand_guess(bounds)
			    	p0_est = p0

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
					"""
					Function for retrieving the time and results (error) given estimator
					"""
					function time_res!(f, obj, opt_type)
						if opt_type == "G"
							timed = @elapsed res_obj = opt.optimize(f, lb, ub, p0, opt.SAMIN(verbosity=0), SAMIN_options)
						else
	                    	#od = opt.OnceDifferentiable(adams_moulton_error, p0; autodiff = :forward)
	                    	#timed = @elapsed res_obj = opt.optimize(od, lb, ub, p0, opt.Fminbox(inner_optimizer), Grad_options)
	                    	#timed = @elapsed res_obj = opt.optimize(f, lb, ub, p0, opt.Fminbox(inner_optimizer), Grad_options; autodiff=:forward)
	                    	#timed = @elapsed res_obj = opt.optimize(f, lb, ub, p0, opt.Fminbox(inner_optimizer), Grad_options)
	                    	timed = @elapsed res_obj = opt.optimize(f, p0, opt.NelderMead(), Grad_options)
						end

		                #dist = diff_states(fun, ini_cond, (t[1],t[end]), phi, res_obj.minimizer[1:length(phi)])
						dist = euclidean(phi,res_obj.minimizer[1:length(phi)])
						trace = make_trace(opt.trace(res_obj))

		                push!(obj["error"], dist)
		                push!(obj["time"], timed)
		                push!(obj["trace"].time, trace.time)
		                push!(obj["trace"].eval, trace.eval)
					end

		            ds_err(p) = sum(abs2.(loss.(data_shooting(p, data, t, problem.fun; plot_estimated=false))))
					"""Local"""
					time_res!(ds_err, res["DS"], "L")

					"""Global"""
					#=>
					try
						time_res!(ds_err, res["ds_g"], "G")
					catch e
						println("G DS Error:")
						@show e
					end
					<=#

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

		            #lsq_ss(p) = vec(single_shooting_estimator(convert(Array{BigFloat}, p), convert(Array{BigFloat}, data), convert(Array{BigFloat}, t), problem.fun))
		            lsq_ss(p) = vec(single_shooting(p, data, t, problem.fun))
		            lsq_ss_sum(p) = sum(abs2.(loss.(lsq_ss(p))))

					"""Local"""
		            try
						time_res!(lsq_ss_sum, res["SS"], "L")
		            catch e
						println("SS Error:")
		                @show e
		            end
					"""Global"""
					#=>
		            try
						time_res!(lsq_ss_sum, res["ss_g"], "G")
		            catch e
						println("SS Error:")
		                @show e
		            end
					<=#

					"""
					Testes: passar resultado de DS como chute inicial para SS
					"""
					#=>
					try
	                    timed = @elapsed res_obj = opt.optimize(lsq_ss_sum, lb, ub, p0_est, opt.SAMIN(verbosity=0), SAMIN_options)
		                dist = diff_states(fun, ini_cond, (t[1],t[end]), phi, res_obj.minimizer[1:length(phi)])
		                push!(res_ss_am_g.error, dist)
		                push!(res_ss_am_g.time, timed)
					catch e
						println("ss+DS/AM")
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
						println("G SS+DS/AM")
		                @show e
		            end
					<=#

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

	                #=>
			        println("res_am:\n$res_am")
			        println("res_am_g:\n$res_am")
			        println("res_ds:\n$res_ds")
			        println("res_ds_g:\n$res_ds_g")
			        println("res_ss:\n$res_ss")
			        println("res_ss_am:\n$res_ss_am")
			        println("res_ss_am_g:\n$res_ss_am_g")
					<=#
		        end

				#=>
				for k in keys(res)
					res[k]["error"] = filter_outlier(res[k]["error"], p=3.5)
				end
				<=#

				"""
				Plots
				"""

				#=>
				canvas_local_error = plot(title="$sam samples, $var noise",
											xlabel="Run",
											ylabel="Error",)
	        	plot!(canvas_local_error, res["ds"]["error"], label="Data Shooting")
		        plot!(canvas_local_error, res["ss"]["error"], label="Single Shooting")
				display(canvas_local_error)

		        canvas_local_time = plot(title="$sam samples, $var noise",
											xlabel="Run",
											ylabel="Time (s)",)
	        	plot!(canvas_local_time, res["ds"]["time"][2:end], label="Data Shooting")
		        plot!(canvas_local_time, res["ss"]["time"][2:end], label="Single Shooting")
		        display(canvas_local_time)
				<=#
				"""
				Plots End
				"""

				#=>
		        canvas_new_res = plot(title="DS: Error vs Run: $sam samples, $var noise")

	        	plot!(canvas_new_res, res_am.error, label="Adams-Moulton")
	        	#plot!(canvas_new_res, res_ds.error, label="Data Shooting")
	        	plot!(canvas_new_res, res_am_g.error, label="G Adams-Moulton")
		        #plot!(canvas_new_res, res_ds_g.error, label="G Data Shooting")
		        display(canvas_new_res)

		        canvas_new_time = plot(title="DS: Time vs Run: $sam samples, $var noise")
	        	plot!(canvas_new_time, res_am.time[2:end], label="Adams-Moulton")
		        #plot!(canvas_new_time, res_ds.time[2:end], label="Data Shooting")
		        plot!(canvas_new_time, res_am_g.time[2:end], label="G Adams-Moulton")
		        #plot!(canvas_new_time, res_ds_g.time[2:end], label="G Data Shooting")
		        display(canvas_new_time)
				<=#

				"""Comparando estratégia DS/AM -> SS"""
				#=>
		        canvas_cla_res = plot(title="SS: Error vs Run: $sam samples, $var noise")

		        plot!(canvas_cla_res, res_ss.error, label="G Single Shooting")
		        plot!(canvas_cla_res, res_ss_am_g.error, label="G DS/AM + Single Shooting")
		        plot!(canvas_cla_res, res_ss_am.error, label="DS/AM + Single Shooting")
		        display(canvas_cla_res)

		        canvas_cla_res = plot(title="SS: Error vs Run: $sam samples, $var noise")
		        plot!(canvas_cla_res, res_ss_am_g.error, label="G DS/AM + Single Shooting")
		        plot!(canvas_cla_res, res_ss_am.error, label="DS/AM + Single Shooting")
		        display(canvas_cla_res)

		        canvas_cla_time = plot(title="SS: Time vs Run: $sam samples, $var noise")
		        plot!(canvas_cla_time, res_ss.time[2:end], label=" G Single Shooting")
		        plot!(canvas_cla_time, res_ss_am_g.time[2:end], label="G DS/AM + Single Shooting")
		        plot!(canvas_cla_time, res_ss_am.time[2:end], label="DS/AM + Single Shooting")
		        display(canvas_cla_time)

		        canvas_cla_time = plot(title="SS: Time vs Run: $sam samples, $var noise")
		        plot!(canvas_cla_time, res_ss_am_g.time[2:end], label="G DS/AM + Single Shooting")
		        plot!(canvas_cla_time, res_ss_am.time[2:end], label="DS/AM + Single Shooting")
		        display(canvas_cla_time)
				<=#

		        #=>
		        canvas_cla_res = plot(title="MS: Error vs Run: $sam samples, $var noise")
		        plot!(canvas_cla_res, res_ms[1], label="Multiple Shooting")
		        display(canvas_cla_res)
		        canvas_cla_time = plot(title="MS: Time vs Run")
		        plot!(canvas_cla_time, res_ms[2][2:end], label="Multiple Shooting")
		        display(canvas_cla_time)
		        <=#
				results[prob_key][var] = res

				"""
				Plots
				Trace
				"""

				p = plot(xlabel="Time", ylabel="Function Evaluation")
				for m in method
					trace = res[m]["trace"]
					if length(trace.eval) > 0
						trace = scale_eval(fill_trace(trace))
						plot!(p, mean(trace.time), log10.(mean(trace.eval)), label="Mean "*m)
						display(p)

        				savefig(p,"./trace_$(m)_$(sam)_$(replace(string(var),"."=>"")).svg")
					end
				end
		    end
		end

		"""
		Plots
		xaxis: Noise Percentage
		yaxis: Mean Error
		"""
		plot_error = Dict()
		for m in method
			plot_error[m] = Dict()
			plot_error[m]["error_mean"] = []
			plot_error[m]["error_std"] = []
		end

		for var in vars
			for m in keys(results[prob_key][var])
				#error = filter_outlier(results[prob_key][var][m]["error"], p=5)
				error = results[prob_key][var][m]["error"]
				append!(plot_error[m]["error_mean"], mean(error))
				append!(plot_error[m]["error_std"], std(error))
			end
		end

		p = plot(x=vars, xlabel="Noise Percentage", ylabel="Mean Error")
		ylim_arr = []
		for m in keys(plot_error)
			if m == "DS"
				label = "Data Shooting"
			elseif m == "SS"
				label = "Single Shooting"
			else
				lable = "Error"
			end

			plot!(p, x=vars, plot_error[m]["error_mean"],
						grid=false, ribbon=plot_error[m]["error_std"],
						fillalpha=.5,label=label)
			display(p)
			push!(ylim_arr, ylims(p))

			savefig(p,"./error_$(m)_$(sam)_$(replace(string(var),"."=>"")).svg")
		end
		if ylim_arr[2][1] > -10
			ylims!(p, (ylim_arr[2][1],ylim_arr[1][2]))
		else
			ylims!(p, ylim_arr[1])
		end
		display(p)

		savefig(p,"./error_all_$(sam)_$(replace(string(var),"."=>"")).svg")

		"""
		Plots - End
		"""
	end
#end
