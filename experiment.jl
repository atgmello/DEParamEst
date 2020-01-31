using Suppressor

using Distributed
using DiffEqParamEstim
using DifferentialEquations
using LSODA
using NLsolve
using Statistics
using Random
using Distances
using Printf
using Plots
using StatsPlots
using Plots.PlotMeasures
using PlotThemes
gr()

import Distributions: Uniform

import Optim
const opt = Optim

using Revise
includet("./problem_set.jl")
includet("./objective_function.jl")
includet("./utils.jl")
import .ProblemSet: get_problem, get_problem_key
import .ObjectiveFunction: data_shooting, single_shooting, soft_l1, huber
import .Utils: Trace, rand_guess, make_trace, get_range_traces, add_noise, filter_outlier, fill_trace, scale_eval,
				diff_states, log_diff, success_rate, box_data, box_scatter_plot

old_precision = precision(BigFloat)
new_precision = 1024
setprecision(new_precision)
#setprecision(old_precision)
dp = Float64

methods = ["DS","SS","DSS"]

theme(:default)
cur_colors = get_color_palette(:auto, plot_color(:white), 17)

method_color = Dict()
for (m,c) in zip(methods,cur_colors)
	method_color[m] = c
end

method_label = Dict()
m_labels = ["Data Shooting", "Single Shooting", "Data to Single Shooting"]
for (m,l) in zip(methods,m_labels)
	method_label[m] = l
end

vars = Float16[0.00, 0.025, 0.05, 0.10, 0.20]
vars = Float16[0.00, 0.01]
vars = range(0.0, 0.5, length=3)

results = Dict()


#SAMIN_options = opt.Options(x_tol=10^-12, f_tol=10^-24, iterations=10^6)
#Grad_options = opt.Options(x_tol=10^-12, f_tol=10^-24, iterations=10^6)
t_limit = 60
x_tol = 10^-6
f_tol = 10^-12
iter = 10^5
SAMIN_options = opt.Options(x_tol=x_tol, f_tol=f_tol,
							iterations=iter, time_limit=t_limit)
Grad_options = opt.Options(x_tol=x_tol, f_tol=f_tol,
							iterations=iter, time_limit=t_limit, store_trace=true)
inner_optimizer = opt.LBFGS()

addprocs(2)
@everywhere begin
	using DiffEqParamEstim
	using DifferentialEquations

    function pdiff_calc(f::Function, u0::AbstractArray, t::StepRangeLen, actual::AbstractArray, forecast::AbstractArray)
        de_prob_actual = ODEProblem(f,u0,(t[1],t[end]),actual)
        de_sol_actual = DifferentialEquations.solve(de_prob_actual, AutoVern9(Rodas5()), saveat=t)
        data_actual = reduce(hcat, de_sol_actual.u)

        de_prob_forecast = ODEProblem(f,u0,(t[1],t[end]),forecast)
        de_sol_forecast = solve(de_prob_forecast, AutoVern9(Rodas5()), saveat=t)
        data_forecast = reduce(hcat, de_sol_forecast.u)

        #euc = euclidean(data_actual, data_forecast)/(size(data_actual)[1]*size(data_actual)[2])
        # Absolute Log Difference
        ldiff = maximum(abs.(log.(abs.(data_actual))-log.(abs.(data_forecast))))
        # Normalized Root Mean Squared Error
        #difference = sqrt(sum(abs2.((data_actual-data_forecast)/mean(data_actual))))/(size(data_actual)[1]*size(data_actual)[2])
        # Normalized Maximum Absolute Error
        #nmae = maximum(abs.((data_actual-data_forecast)./data_actual))
        # Maximum Absolute Error
        #mae = maximum(abs.((data_actual-data_forecast)))

        #=>
        p = plot(title="$(round(nmae,digits=3)) vs $(round(mae,digits=3)) vs  $(round(ldiff,digits=3))")
        plot!(p,t,data_actual')
        plot!(p,t,data_forecast',linestyle=:dashdot)
        display(p)
        sleep(5)
        <=#
        return ldiff
    end
end

function diff_states(f::Function, u0::AbstractArray, tspan::Tuple, actual::AbstractArray, forecast::AbstractArray; var=1.5)
    diff_arr = []
    t = range(tspan[1], stop=tspan[end], length=1000)
    runs = 20
    u0_arr = map(x -> abs.(add_noise(x,var)),[copy(u0) for _ in 1:runs])

    diff_arr = pmap(pdiff_calc,
					[f for _ in 1:runs],
					u0_arr,
					[t for _ in 1:runs],
					[actual for _ in 1:runs],
					[forecast for _ in 1:runs])

    return maximum(diff_arr)
end

#@suppress begin
@time for p in 1:8
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

	for sam in 5:45:50
		for var in vars
			res = Dict()
			for m in methods
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
	        loss = linear
	        #loss = soft_l1
	        #loss = huber

	        for rep in 1:25
	            data = copy(data_original)
				if var > 0.0
		            data = add_noise(data, var)
				end
	            p0 = rand_guess(bounds)

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
				function time_res!(f, obj, opt_type, p0)
					if opt_type == "G"
						timed = @elapsed res_obj = opt.optimize(f, lb, ub, p0, opt.SAMIN(verbosity=0), SAMIN_options)
					else
                    	#od = opt.OnceDifferentiable(adams_moulton_error, p0; autodiff = :forward)
                    	#timed = @elapsed res_obj = opt.optimize(od, lb, ub, p0, opt.Fminbox(inner_optimizer), Grad_options)
                    	#timed = @elapsed res_obj = opt.optimize(f, lb, ub, p0, opt.Fminbox(inner_optimizer), Grad_options; autodiff=:forward)
                    	#timed = @elapsed res_obj = opt.optimize(f, lb, ub, p0, opt.Fminbox(inner_optimizer), Grad_options)
                    	timed = @elapsed res_obj = opt.optimize(f, p0, opt.NelderMead(), Grad_options)
					end

	                dist = diff_states(fun, ini_cond, (t[1],t[end]), phi, res_obj.minimizer[1:length(phi)])
					#dist = euclidean(phi,res_obj.minimizer[1:length(phi)])
					#dist = abs(log_diff(phi,abs.(res_obj.minimizer[1:length(phi)])))
					# Normalized Mean Absolute Error
					#dist = mean(abs.((res_obj.minimizer[1:length(phi)]-phi)./phi))
					# Mean Absolute Error
					#dist = mean(abs.(res_obj.minimizer[1:length(phi)]-phi))
					trace = make_trace(opt.trace(res_obj))

	                push!(obj["error"], dist)
	                push!(obj["time"], timed)
	                push!(obj["trace"].time, trace.time)
	                push!(obj["trace"].eval, trace.eval)
					return res_obj.minimizer[1:length(phi)]
				end

	            # ----- Data Shooting -----

				if "DS" in methods
					#println("--- DS ---\n")
		            ds_err(p) = sum(abs2.(loss.(data_shooting(p, data, t, problem.fun; plot_estimated=false))))
					"""Local"""
						p0_ds = time_res!(ds_err, res["DS"], "L", p0)

			            # ----- Data to Single Shooting -----

						if "DSS" in methods
				            lsq_ss(p) = vec(single_shooting(p, data, t, problem.fun))
				            lsq_ss_sum(p) = sum(abs2.(loss.(lsq_ss(p))))
							#println("--- DSS ---\n")
							"""Local"""
				            try
								time_res!(lsq_ss_sum, res["DSS"], "L", p0_ds)
				            catch e
								println("DSS Error:")
				                @show e
				            end
						end
					"""Global"""
					#=>
					try
						time_res!(ds_err, res["ds_g"], "G")
					catch e
						println("G DS Error:")
						@show e
					end
					<=#
				end

	            # ----- Single Shooting -----

				if "SS" in methods
					#println("--- SS ---\n")
		            #lsq_ss(p) = vec(single_shooting_estimator(convert(Array{BigFloat}, p), convert(Array{BigFloat}, data), convert(Array{BigFloat}, t), problem.fun))
		            lsq_ss(p) = vec(single_shooting(p, data, t, problem.fun))
		            lsq_ss_sum(p) = sum(abs2.(loss.(lsq_ss(p))))

					"""Local"""
		            try
						time_res!(lsq_ss_sum, res["SS"], "L", p0)
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
				end
	        end
			results[prob_key][var] = res

			"""
			Plots
			Trace
			"""

			p = plot(xlabel="Time", ylabel="Function Evaluation")
			for m in methods
				p2 = plot(xlabel="Time", ylabel="Function Evaluation")
				trace = res[m]["trace"]
				if length(trace.eval) > 0
					trace = scale_eval(fill_trace(trace))
					#(best,med,worst) = get_range_traces(trace)

					plot!(p, mean(trace.time), log10.(mean(trace.eval)),
							label="Mean "*m, color=method_color[m])

					plot!(p2, mean(trace.time), log10.(mean(trace.eval)),
							label="Mean "*m, color=method_color[m])

					#=>
					plot!(p, mean(trace.time), log10.(mean(trace.eval)), label="Mean "*m,
							grid=true,
							ribbon=log10.(mean(std(trace.eval))),
							fillalpha=.4)

					plot!(p2, mean(trace.time), log10.(mean(trace.eval)), label="Mean "*m,
							grid=true,
							ribbon=log10.(mean(std(trace.eval))),
							fillalpha=.4)
					<=#

					#=>
					plot!(p, med.time, log10.(med.eval), label="Mean "*m,
							grid=true,
							ribbon=(log10.(abs.(med.eval-worst.eval)),
									log10.(abs.(best.eval-med.eval))),
							fillalpha=.4)

					plot!(p2, med.time, log10.(med.eval), label="Mean "*m,
							grid=true,
							ribbon=(log10.(abs.(med.eval-worst.eval)),
									log10.(abs.(best.eval-med.eval))),
							fillalpha=.4)
					<=#

					#=>
					plot!(p, best.time, log10.(best.eval), label="Best "*m)
					plot!(p, med.time, log10.(med.eval), label="Median "*m)
					plot!(p, mean(trace.time), log10.(mean(trace.eval)), label="Mean "*m)
					plot!(p, worst.time, log10.(worst.eval), label="Worst "*m)

					plot!(p2, best.time, log10.(best.eval), label="Best "*m)
					plot!(p2, med.time, log10.(med.eval), label="Median "*m)
					plot!(p2, mean(trace.time), log10.(mean(trace.eval)), label="Mean "*m)
					plot!(p2, worst.time, log10.(worst.eval), label="Worst "*m)
					<=#

					display(p2)
    				savefig(p2,"./trace_$(m)_$(sam)_$(replace(string(var),"."=>"")).svg")
				end
			end
			savefig(p,"./trace_all_$(sam)_$(replace(string(var),"."=>"")).svg")
	    end

		"""
		Error Plots
		xaxis: Noise Percentage
		yaxis: Mean Error
		"""
		plot_data = Dict()
		for m in methods
			plot_data[m] = Dict()
			plot_data[m]["error"] = []
			plot_data[m]["time"] = []
		end

		for var in vars
			for m in keys(results[prob_key][var])
				#error = filter_outlier(results[prob_key][var][m]["error"], p=5)
				error = results[prob_key][var][m]["error"]
				if length(error) > 0
					push!(plot_data[m]["error"], error)
				else
					push!(plot_data[m]["error"], [NaN])
				end
				trace = results[prob_key][var][m]["trace"]
				if length(trace.time) > 0
					push!(plot_data[m]["time"], map(t -> t[end], trace.time))
				else
					push!(plot_data[m]["time"], [NaN])
				end
			end
		end

		p = plot(x=vars, xlabel="Noise Percentage", ylabel="Median Error")
		ylim_arr = []
		for m in methods
			p2 = plot(x=vars, xlabel="Noise Percentage", ylabel="Median Error")
			error = plot_data[m]["error"]
			if !any(isnan.(vcat(error...)))
				qerror = hcat(box_scatter_plot.(error)...)
				plot!(p, vars, qerror[2,:],
							grid=true,
							ribbon=(qerror[1,:],
									qerror[3,:]),
							fillalpha=.5, label=method_label[m], color=method_color[m])
				push!(ylim_arr, ylims(p))

				plot!(p2, vars, qerror[2,:],
							grid=true,
							ribbon=(qerror[1,:],
									qerror[3,:]),
							fillalpha=.5, label=method_label[m], color=method_color[m])
				display(p2)

				savefig(p2,"./error_$(m)_$(sam)_$(replace(string(var),"."=>"")).svg")
				savefig(p,"./error_inter_$(m)_$(sam)_$(replace(string(var),"."=>"")).svg")
			end
		end
		if length(ylim_arr) > 1
			ylims!(p, (minimum([ylim_arr[1][1],ylim_arr[2][1]]),
						minimum([ylim_arr[1][2],ylim_arr[2][2]])))
		end
		display(p)

		savefig(p,"./error_all_$(sam)_$(replace(string(var),"."=>"")).svg")

		"""
		Error Plots - End
		"""

		"""
		Success vs Time Plots
		xaxis: Mean Computation Time
		yaxis: 1 / Success Rate
		"""

		p = scatter(xlabel="Mean Time", ylabel="Mean 1 / Success Rate")
		p2 = scatter(xlabel="Mean Time", ylabel="1 / Success Rate")
		ylim_arr = []
		for m in methods
			p3 = scatter(xlabel="Mean Time", ylabel="1 / Success Rate")

			sr = success_rate.(plot_data[m]["error"])
			isr = sr.^(-1)
			time = plot_data[m]["time"]
			if !any(isnan.(vcat(time...)))
				qtime = hcat(box_scatter_plot.(time)...)

				qqtime = box_scatter_plot(qtime[2,:])
				qisr = box_scatter_plot(isr)
				scatter!(p, (qqtime[2],qisr[2]),
							xerror=[(qqtime[1],qqtime[3])],
							yerror=[(qisr[1],qisr[3])],
							label=method_label[m], color=method_color[m])
				if ylims(p)[2] > 10
					ylims!(p, (-0.1,10.0))
				end
				if xlims(p)[2] > 10
					xlims!(p, (-0.1,10.0))
				end

				scatter!(p2, (qtime[2,:], isr),
							label=method_label[m], color=method_color[m])
				if ylims(p2)[2] > 10
					ylims!(p2, (-0.1,10.0))
				end
				if xlims(p2)[2] > 10
					xlims!(p2, (-0.1,10.0))
				end

				scatter!(p3, (qtime[2,:], isr),
							label=method_label[m], color=method_color[m],
							series_annotations = text.(vars, :top, 11))

				display(p3)

				savefig(p3,"./sr_$(m)_$(sam)_$(replace(string(var),"."=>"")).svg")
			end
		end
		display(p)
		display(p2)

		savefig(p, "./sr_all_median$(sam)_$(replace(string(var),"."=>"")).svg")
		savefig(p2, "./sr_all_$(sam)_$(replace(string(var),"."=>"")).svg")

		"""
		OE Plots - End
		"""


		"""
		Overall Efficiency (OE) Plots
		xaxis: Method
		yaxis: OE Score
		"""

		t_succ = []
		for m in methods
			error = plot_data[m]["error"]
			sr = success_rate.(error)
			time = plot_data[m]["time"]
			mean_time = mean.(time)
			t_succ_arr = mean_time./sr
			push!(t_succ, mean(t_succ_arr))
		end

		min_t_succ = minimum(t_succ)
		oe = map(x -> min_t_succ/x, t_succ)

		p = bar(xlabel="Method",
				ylabel="Overall Efficiency",
				legend=false)
		bar!(p, methods, oe, color=cur_colors[6])
		display(p)

		savefig(p, "./$(sam)_oe.svg")
		"""
		OE Plots - End
		"""
	end #samples loop
end #problem loop
#end
