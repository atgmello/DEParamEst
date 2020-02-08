using Transducers
using DiffEqParamEstim
using DifferentialEquations
using LSODA
using Statistics
using Random
using Distances
using Plots
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

method_color = Dict()
for (m,c) in zip(method_arr,cur_colors)
	method_color[m] = c
end

method_label = Dict()
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
const t_limit = 240
x_tol = 10^-12
f_tol = 10^-24
iter = 10^8
const SAMIN_options = opt.Options(x_tol=x_tol, f_tol=f_tol,
							iterations=iter, time_limit=t_limit)
const Grad_options = opt.Options(x_tol=x_tol, f_tol=f_tol,
							iterations=iter, time_limit=t_limit)
const inner_optimizer = opt.LBFGS()

function optim_res(obj_fun::Function, de_fun::Function, p0::Array,
					phi::Array,
					ini_cond::Array,
					t::AbstractArray)::Array{Array{<:AbstractFloat}}
	timed = @elapsed res_obj = opt.optimize(obj_fun, p0, opt.LBFGS(), autodiff=:forward, Grad_options)
    dist = max_diff_states(de_fun, ini_cond, (t[1],t[end]), phi, res_obj.minimizer[1:length(phi)], 1.5)
	#trace = make_trace(opt.trace(res_obj))

	#return [dist, timed, trace.time, trace.eval, res_obj.minimizer[1:length(phi)]]
	return [[dist], [timed], res_obj.minimizer[1:length(phi)]]
end

function get_results(de_fun::Function,
					m::String,
	 				args::Tuple)::Array{Array{<:AbstractFloat}}

	data::Array{<:AbstractFloat},
	p0::Array,
	phi::Array,
	ini_cond::Array,
	t::AbstractArray{<:AbstractFloat} = args

	results = [[10_000], [t_limit], p0]
	p0_ds = p0
	# ----- Data Shooting -----
	if m == "DS" || m =="DSS"
	    ds_fun(p) = sum(abs2.(loss.(data_shooting(p, data, t, de_fun))))

        try
			results = optim_res(ds_fun, de_fun, p0,
	                            phi, ini_cond, t)

			if m == "DS"
				return results
			end
			p0_ds = results[end]

	    catch e
			println("DS Error:")
            @show e
        end
        # ----- Data to Single Shooting -----
		if m == "DSS"
            dds_fun(p) = sum(abs2.(loss.(single_shooting(p, data, t, de_fun))))
			try
				partial_res = optim_res(dds_fun, de_fun, p0_ds,
                                        phi, ini_cond, t)
				partial_res[2] += results[2]
				results = partial_res
		    catch e
				println("DSS Error:")
	            @show e
	        end
		end
	# ----- Single Shooting -----
	elseif m == "SS"
		ss_fun(p) = sum(abs2.(loss.(single_shooting(p, data, t, de_fun))))
		try
    		results = optim_res(ss_fun, de_fun, p0,
                                phi, ini_cond, t)
	    catch e
			println("SS Error:")
            @show e
        end
	end
	return results
end


#@suppress begin
function experiment(p::Int64,sams::AbstractArray{<:Int},
					vars::AbstractArray{<:AbstractFloat},
					method_arr::Array{<:String},
					parallel::Bool)::Dict
	results::Dict = Dict()
	prob_key::String = get_problem_key(p)
	problem::ProblemSet.DEProblem = get_problem(prob_key)
	results[prob_key] = Dict()

	cd("/home/andrew/git/ChemParamEst/plots/experiments/")
	mkdir(prob_key)
	cd(prob_key)

    #print("\n----- Getting results for Flouda's Problem number $i -----\n")
    fun::Function = problem.fun
    phi::Array = problem.phi
    bounds::Array{<:AbstractFloat} = problem.bounds
    ini_cond::Array = problem.data[:,1]
    lb::Array{<:AbstractFloat} = bounds[:,1]
    ub::Array{<:AbstractFloat} = bounds[:,2]
    t::AbstractArray = problem.t

	# Minimum number of data points
	min_data::Int64 = round(length(phi)/length(ini_cond),RoundUp)

	for sam in sams.*min_data
		for v in vars
	        # Artificial Data
	        if sam == 1
	            t = range(t[1], stop=t[end], length=length(t))
	        else
	            t = range(t[1], stop=t[end], length=sam)
	        end
	        tspan = (t[1], t[end])
	        ode_prob = ODEProblem(fun, ini_cond, tspan, phi)
	        ode_sol  = solve(ode_prob, lsoda(), saveat=reduce(vcat, t))
	        data = reduce(hcat, ode_sol.u)
	        #data_plot = plot(t,data')
			#display(data_plot)

			reps = 100
			if v > 0.0
				data_arr = [add_noise(data,v) for _ in 1:reps]
			else
				data_arr = [data for _ in 1:reps]
			end

			get_results_args = zip(data_arr,
									[rand_guess(bounds) for _ in 1:reps],
									[phi for _ in 1:reps],
									[ini_cond for _ in 1:reps],
									[t for _ in 1:reps]
									#[lb for _ in 1:reps],
									#[ub for _ in 1:reps]
									)

			res = Dict()
			for m in method_arr
				if parallel
					res[m] = tcollect(Map(x->get_results(fun,m,x)),collect(get_results_args))
				else
					res[m] = [get_results(fun,m,x) for x in get_results_args]
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
							parallel::Bool)
	for p in probs
		experiment(p,sams,vars,method_arr,parallel)
	end
end

probs = 1:10
sams = [5,10,50,100]
vars = range(0.0, 0.3, length=4)

@time problem_exp_loop(probs,sams,vars,true)

#@code_warntype experiment(vars)
