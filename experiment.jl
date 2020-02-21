ENV["GKSwstype"]="nul"
using Transducers
using DiffEqParamEstim
using DifferentialEquations
using Statistics
using StatsBase: sample
using Random
using Distances
using LinearAlgebra
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
import .ObjectiveFunction: data_shooting, single_shooting, tikhonov
import .Utils: Trace, rand_guess, make_trace, get_range_traces, add_noise,
				filter_outlier, fill_trace, scale_eval,
				max_diff_states, diff_calc, step_success_rate, success_rate, box_data, box_scatter_plot,
				get_plot_data, oe_plots, sr_plots, error_plots, nmse, plot_compare

#old_precision = precision(BigFloat)
#new_precision = 1024
#setprecision(new_precision)
#setprecision(old_precision)
const dp = Float64

theme(:default)
const cur_colors = get_color_palette(:auto, plot_color(:white), 17)

const method_arr = ["DS","SS"]

const method_color = Dict()
for (m,c) in zip(method_arr,cur_colors)
	method_color[m] = c
end

const method_label = Dict()
m_labels = ["DS", "SS", "DSS"]
for (m,l) in zip(method_arr,m_labels)
	method_label[m] = l
end

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
					testing_set::Vector{ProblemSet.DEProblem},
					p0::Vector{T})::Vector{Vector{T}} where T
	lb = testing_set[1].bounds[1]
	ub = testing_set[1].bounds[2]
	timed = zero(1)

	timed += @elapsed res_obj = Optim.optimize(obj_fun,
								lb,ub,
								p0,
								Optim.SAMIN(verbosity=0, rt=0.2), SAMIN_options)

	timed += @elapsed res_obj = Optim.optimize(obj_fun,
								lb,ub,
								res_obj.minimizer[1:length(p0)],
								Fminbox(inner_optimizer),
								Grad_options,
								autodiff = :forward)

	phi_est = res_obj.minimizer[1:length(p0)]

	p = testing_set[1]
    tspan = (p.t[1], p.t[end])
    ode_prob = ODEProblem(p.fun, p.data[1], tspan, phi_est)
    ode_sol  = solve(ode_prob, Tsit5(), saveat=p.t)
	data_est = reduce(vcat,ode_sol.u)

	# Normalized Root Mean Squared Error
    nrmse = mean([nmse(reduce(vcat,tp.data), data_est)
			for tp in testing_set]) |> x -> sqrt(x)

	"""
	Plotting
	p = testing_set[1]
	plot_compare(p.data, ode_sol.u)
	"""

	return [[nrmse], [timed], phi_est]
end

function get_results(method_label::String,
					training_set::Vector{ProblemSet.DEProblem},
					testing_set::Vector{ProblemSet.DEProblem},
	 				p0::Vector{T})::Array{Array{T}} where T

	results = [[10_000], [t_limit], p0]
	p0_ds = p0
	#lambda_arr = [1e-2,1e-1,0.0,1e0,1e1,1e2]
	lambda_arr = [1e0,0.0,1e-2,1e-4]
	best_lambda = lambda_arr[1]

	"""
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
	end
	"""
	if method_label == "SS"
		f = single_shooting
		phi_prior = zeros(T,length(p0))
		m = length(phi_prior)
		w = Matrix{T}(I,m,m)
	elseif method_label == "DS"
		f = data_shooting
		phi_prior = zeros(T,length(p0))
		m = length(phi_prior)
		w = Matrix{T}(I,m,m)
	end


	partial_res = [[10_000], [t_limit], p0]
	bounds = training_set[1].bounds
	oos_errors = fill(NaN64,length(lambda_arr))
	elapsed = zero(T)

	# Lamdba selection via
	# Cross Validation
	println("\n--- x ---")
	println(method_label)
	println("--- x ---\n")
	println("Initial guess:\n$(p0)")
	println("Starting Cross Validation!")
	println("...")

	# K-Fold Cross Validation
	# Fixed K = 4
	# TODO: make it variable

	cv_folds = sample(1:8, (4,2), replace=false)
	num_folds,num_elem_fold = size(cv_folds)

	for i in 1:length(lambda_arr)

		lambda = lambda_arr[i]
		println("\n--- x ---\n")
		println("Lambda = $(lambda)")
		oos_errors_partial = fill(NaN64,length(cv_folds))

		for j in 1:num_folds

			hold_out = cv_folds[j,:]
			slice = filter(x -> x != hold_out, 1:length(training_set))
			training_slice = training_set[slice]
			oos_problem = training_set[hold_out]

			obj_fun = function (x)
							total_error = zero(T)
							for p in training_slice
								total_error += f(x, p.data, p.t, p.fun)
							end
							total_error += tikhonov(lambda, x, phi_prior, w)
							return total_error
				    	end

				try
		    		partial_res = optim_res(obj_fun, oos_problem, p0)

					p0 = partial_res[3]
					"""
					If p0 becomes out of bounds, fix it
					"""
					@inbounds for k in 1:length(p0)
						if p0[k] < bounds[1][k]
							p0[k] = bounds[1][k] + bounds[1][k]*0.1
						elseif p0[k] > bounds[2][k]
							p0[k] = bounds[2][k] - bounds[2][k]*0.1
						end
					end

					elapsed += partial_res[2][1]
					oos_errors_partial[j] = partial_res[1][1]
					println("Error $(j):\n$(oos_errors_partial[j])")
				catch e
					println("Error suring CV fold!")
				    @show e
					if e == InterruptException
						break
					end
				end
		end
		oos_errors[i] = mean(oos_errors_partial[
								filter(x -> !isnan(oos_errors_partial[x]),
											1:length(oos_errors_partial))
											])
		println("Mean CV Error for lambda = $(lambda_arr[i]):\n$(oos_errors[i])\n\n")
	end

	elapsed /= length(lambda_arr)*length(cv_folds)
	best_lambda = lambda_arr[argmin(oos_errors)]

	println("Done!")
	println("Best lambda:\n$(best_lambda)")
	println("Optimizing on whole dataset.")
	println("...")

	final_obj_fun = function (x)
					total_error = zero(T)
					for p in training_set
						total_error += f(x, p.data, p.t, p.fun)
					end
					total_error += tikhonov(best_lambda, x, phi_prior, w)
					return total_error
				end

	try
		partial_res = optim_res(final_obj_fun, testing_set, p0)
		results = [partial_res[1], [elapsed], partial_res[3]]

		println("Done!")
		println("Test Error:\n$(results[1])")
		println("Estimated parameters:\n$(results[3])\n\n")
	catch e
		println("Error!")
        @show e
    end


	"""
	Plotting
	p = testing_set[1]
    tspan = (p.t[1], p.t[end])
	phi_est = results[3]
    ode_prob = ODEProblem(p.fun, p.data[1], tspan, phi_est)
    ode_sol  = solve(ode_prob, Tsit5(), saveat=p.t)
	data_est = ode_sol.u
	plot_compare(p.data, data_est)
	sleep(7)
	"""

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

			num_reps = 4
			guess_arr = [rand_guess(bounds) for _ in 1:num_reps]

			n_training = 8
			training_set_arr = [[ProblemSet.DEProblem(problem.fun, problem.phi,
								problem.bounds, add_noise(data,v), _t)
				 			for _ in 1:n_training] for _ in 1:num_reps]

			n_testing = 2
			testing_set_arr = [[ProblemSet.DEProblem(problem.fun, problem.phi,
								problem.bounds, add_noise(data,v), _t)
				 			for _ in 1:n_testing] for _ in 1:num_reps]

			res = Dict()
			@sync for m in method_arr
				@async res[m] = [get_results(m,
									training_set_arr[i],
									testing_set_arr[i],
									guess_arr[i])
							for i in 1:num_reps]
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
	probs = eval(Meta.parse(args[2]))
	sams = eval(Meta.parse(args[3]))
	vars = eval(Meta.parse(args[4]))
	par = string(args[5]) == "true"
	#vars = range(0.0, 0.3, length=4)
	time_main = @time problem_exp_loop(probs,sams,vars,dir,par)
	println(time_main)
	nothing
end

dir="/home/andrew/git/ChemParamEst/plots/experiments/tmp/"
p="1:10"
sams="[2,4]"
#vars="[0.05,0.1,0.15]"
vars="[0.05,0.1,0.15]"
par="true"
args = [dir,p,sams,vars,par]

main(args)

#TODO FIX TIKHONOV REGULARIZATION

#clearconsole()
