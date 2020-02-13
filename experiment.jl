ENV["GKSwstype"]="nul"
using Transducers
using DiffEqParamEstim
using DifferentialEquations
using Statistics
using Random
using Distances
using Plots
using Plots.PlotMeasures
using PlotThemes
gr()

import Distributions: Uniform

using Optim

using Revise
includet("./problem_set.jl")
includet("./objective_function.jl")
includet("./utils.jl")
import .ProblemSet: get_problem, get_problem_key, DEProblem
import .ObjectiveFunction: data_shooting, single_shooting, soft_l1, huber
import .Utils: Trace, rand_guess, make_trace, get_range_traces, add_noise,
				filter_outlier, fill_trace, scale_eval,
				max_diff_states, diff_calc, step_success_rate, success_rate, box_data, box_scatter_plot,
				get_plot_data, oe_plots, sr_plots, error_plots

old_precision = precision(BigFloat)
new_precision = 1024
setprecision(new_precision)
#setprecision(old_precision)
const dp = Float64

theme(:default)
const cur_colors = get_color_palette(:auto, plot_color(:white), 17)

method_arr = ["DS","SS","DSS"]

const method_color = Dict()
for (m,c) in zip(method_arr,cur_colors)
	method_color[m] = c
end

const method_label = Dict()
m_labels = ["Data Shooting", "Single Shooting", "Data to Single Shooting"]
for (m,l) in zip(method_arr,m_labels)
	method_label[m] = l
end

linear(x) = x
const loss = linear
#loss = soft_l1
#loss = huber

#SAMIN_options = opt.Options(x_tol=10^-12, f_tol=10^-24, iterations=10^6)
#Grad_options = opt.Options(x_tol=10^-12, f_tol=10^-24, iterations=10^6)
const t_limit = 180
f_tol = 10^-12
x_tol = 10^-6
iter = 10^6

const SAMIN_options = Optim.Options(x_tol=x_tol, f_tol=f_tol,
							iterations=iter, time_limit=t_limit)
g_time_limit = 10
f_tol = 10^-12
x_tol = 10^-6
iter = 10^5

const Grad_options = Optim.Options(x_tol=x_tol, f_tol=f_tol,
							iterations=iter, time_limit=g_time_limit)

const inner_optimizer = Optim.LBFGS()

function optim_res(obj_fun::Function,
					problem::ProblemSet.DEProblem,
					p0::Array{T})::Array{Array{T}} where T
	lb = problem.bounds[1]
	ub = problem.bounds[2]
	timed = zero(1)

	timed += @elapsed res_obj = Optim.optimize(obj_fun,
								lb,ub,
								p0,
								Optim.SAMIN(verbosity=0, rt=0.65), SAMIN_options)

	timed += @elapsed res_obj = Optim.optimize(obj_fun,
								res_obj.minimizer[1:length(problem.phi)],
								Optim.NelderMead(), Grad_options)

    dist = max_diff_states(problem, res_obj.minimizer[1:length(problem.phi)], 1.5)
	#trace = make_trace(opt.trace(res_obj))

	#return [dist, timed, trace.time, trace.eval, res_obj.minimizer[1:length(phi)]]
	return [[dist], [timed], res_obj.minimizer[1:length(problem.phi)]]
end

function get_results(method_label::String,
					problem::ProblemSet.DEProblem,
	 				p0::Array{<:AbstractFloat})::Array{Array{<:AbstractFloat}}

	results = [[10_000], [t_limit], p0]
	p0_ds = p0
	# ----- Data Shooting -----
	if method_label == "DS" || method_label =="DSS"
	    ds_fun(x) = data_shooting(x, problem.data, problem.t, problem.fun)
		try
			results = optim_res(ds_fun, problem, p0)

			if method_label == "DS"
				return results
			end
			p0_ds = results[end]
	    catch e
			println("DS Error:")
            @show e
        end
        # ----- Data to Single Shooting -----
		if method_label == "DSS"
	    	dds_fun(x) = single_shooting(x, problem.data, problem.t, problem.fun)
			try
				"""
				If p0_ds is out of bounds, fix it
				"""
				@inbounds for i in 1:length(p0_ds)
					if p0_ds[i] < problem.bounds[1][i]
						p0_ds[i] = problem.bounds[1][i] + problem.bounds[1][i]*0.1
					elseif p0_ds[i] > problem.bounds[2][i]
						p0_ds[i] = problem.bounds[2][i] - problem.bounds[2][i]*0.1
					end
				end
				partial_res = optim_res(dds_fun, problem, p0_ds)
				partial_res[2] += results[2]
				results = partial_res
		    catch e
				println("DSS Error:")
	            @show e
	        end
		end
	# ----- Single Shooting -----
	elseif method_label == "SS"
    	ss_fun(x) = single_shooting(x, problem.data, problem.t, problem.fun)
		try
    		results = optim_res(ss_fun, problem, p0)
	    catch e
			println("SS Error:")
            @show e
        end
	end
	return results
end

#@suppress begin
function experiment(p_num::Int64,sams::AbstractArray{<:Int},
					vars::AbstractArray{<:AbstractFloat},
					method_arr::Array{<:String},
					dir::String,
					parallel::Bool)::Dict
	results::Dict = Dict()
	prob_key::String = get_problem_key(p_num)
	problem::ProblemSet.DEProblem = get_problem(prob_key)
	results[prob_key] = Dict()

	cd(dir)
	mkdir(prob_key)
	cd(prob_key)

    #print("\n----- Getting results for Flouda's Problem number $i -----\n")
    fun::Function = problem.fun
    phi::Array = problem.phi
    bounds::Vector = problem.bounds
    ini_cond::Array = problem.data[1]
    t::AbstractArray = problem.t

	# Minimum number of data points
	min_data = round(length(phi)/length(ini_cond),RoundUp)
	data_sams = convert.(eltype(sams[1]),sams.*min_data)
	epsilon = 10^-3

	for sam in data_sams
		for v in vars
	        # Artificial Data
            _t = range(t[1], stop=t[end], length=sam)
	        tspan = (t[1], t[end])
	        ode_prob = ODEProblem(fun, ini_cond, tspan, phi)
	        ode_sol  = solve(ode_prob, AutoTsit5(Rosenbrock23()), saveat=_t)
	        data = ode_sol.u
	        #data_plot = plot(t,data')
			#display(data_plot)

			reps = 100
			problem_arr = [ProblemSet.DEProblem(problem.fun, problem.phi,
								problem.bounds, add_noise(data,v,epsilon), _t)
				 			for _ in 1:reps]
			bounds_arr = [rand_guess(bounds) for _ in 1:reps]

			get_results_args = zip(problem_arr, bounds_arr)

			res = Dict()
			for m in method_arr
				if parallel
					res[m] = tcollect(Map(x->get_results(m,x...)),
								collect(get_results_args))
				else
					res[m] = [get_results(m,x...) for x in get_results_args]
				end
			end

			results[prob_key][v] = res
	    end

		plot_data = get_plot_data(results, prob_key, vars, method_arr)

		error_plots(plot_data,vars,method_arr,method_label,method_color,sam)

		sr_plots(plot_data,vars,method_arr,method_label,method_color,sam)

		oe_plots(plot_data,vars,method_arr,method_label,method_color,sam)

	end #samples loop
	return results
end

function problem_exp_loop(probs::AbstractArray{<:Int},
							sams::AbstractArray{<:Int},
							vars::AbstractArray{<:AbstractFloat},
							dir::String,
							parallel::Bool)
	for p in probs
		experiment(p,sams,vars,method_arr,dir,parallel)
	end
end

function main(args::Array{<:String})::Nothing

	dir = string(args[1])
	par = string(args[2]) == "true"

	probs = 1:10
	sams = [5,50,100]
	vars = range(0.0, 0.3, length=4)

	time_main = @time problem_exp_loop(probs,sams,vars,dir,par)
	println(time_main)
	nothing
end

main(ARGS)
