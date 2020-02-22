ENV["GKSwstype"]="nul"
using Distributed
using Transducers
using DiffEqParamEstim
using DifferentialEquations
using Statistics
using StatsBase: sample
using Random
Random.seed!(1234)
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

@assert Threads.nthreads() > 1

#old_precision = precision(BigFloat)
#new_precision = 1024
#setprecision(new_precision)
#setprecision(old_precision)
const dp = Float64

theme(:default)
const cur_colors = get_color_palette(:auto, plot_color(:white), 17)

const all_method_arr = ["DS","SS","DDS"]

const method_color = Dict()
for (m,c) in zip(all_method_arr,cur_colors)
	method_color[m] = c
end

const method_label = Dict()
for m in all_method_arr
	method_label[m] = m
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
	elapsed_time = zero(T)

	elapsed_time += @elapsed res_obj = Optim.optimize(obj_fun,
								lb,ub,
								p0,
								Optim.SAMIN(verbosity=0, rt=0.2), SAMIN_options)

	elapsed_time += @elapsed res_obj = Optim.optimize(obj_fun,
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

	return [[nrmse], [elapsed_time], phi_est]
end

function cv_optimize(training_set::Vector{ProblemSet.DEProblem},
					testing_set::Vector{ProblemSet.DEProblem},
					p0::Vector{T},
					k::Int64,
					f::Function,
					phi_prior::Vector{T},
					m::Int64,
					w::Array)::Array{Array{T}} where T

	results = [[10_000], [t_limit], p0]
	#lambda_arr = [1e-2,1e-1,0.0,1e0,1e1,1e2]
	lambda_arr = [1e0,0.0,1e-2,1e-4]
	num_lambdas = length(lambda_arr)
	best_lambda = lambda_arr[1]

	bounds = training_set[1].bounds

	# Lamdba selection via
	# Cross Validation
	#println("\n--- x ---")
	#println(method_label)
	#println("--- x ---\n")
	#println("Initial guess:\n$(p0)")
	#println("Starting Cross Validation!")
	#println("Preprocessing starting")
	#println("...")

	# K-Fold Cross Validation

	num_training_sets = length(training_set)
	num_elem_folds = convert(Int64, (num_training_sets/k))

	cv_folds = sample(1:num_training_sets, (k,num_elem_folds), replace=false)

	num_elem_train_slice = convert(Int64, (length(training_set) - num_elem_folds))

	# Pre - Allocate and prepare necessary data
	partial_res = [[10_000], [t_limit], p0]
	partial_res_arr = [[[Vector{Float64}(undef, 1)
							for _ in 1:3] for _ in 1:k]
							for _ in 1:num_lambdas]

	training_set_folds = [Vector{ProblemSet.DEProblem}(undef,num_elem_train_slice)
	 						for _ in 1:k]
	training_set_arr = [[Vector{ProblemSet.DEProblem}(undef,num_elem_train_slice)
	 						for _ in 1:k]
							for _ in 1:num_lambdas]

	hold_out_set_folds = [Vector{ProblemSet.DEProblem}(undef,num_elem_folds)
							for _ in 1:k]
	hold_out_set_arr = [[Vector{ProblemSet.DEProblem}(undef,num_elem_folds)
							for _ in 1:k]
							for _ in 1:num_lambdas]

	obj_fun_folds = Vector{Function}(undef,k)
	obj_fun_arr = [Vector{Function}(undef,k) for _ in 1:num_lambdas]

	for i in 1:num_lambdas
		for j in 1:k
			hold_out = cv_folds[j,:]
			slice = filter(x -> !(x in hold_out), 1:length(training_set))

			training_sliced = training_set[slice]
			training_set_folds[j] = training_sliced

			hold_out_set = training_set[hold_out]
			hold_out_set_folds[j] = hold_out_set

			obj_fun = function (x)
						total_error = zero(T)
						for p in training_sliced
							total_error += f(x, p.data, p.t, p.fun)
						end
						total_error += tikhonov(lambda_arr[i], x, phi_prior, w)
						return total_error
					end
			obj_fun_folds[j] = obj_fun
		end
		obj_fun_arr[i] = obj_fun_folds
		training_set_arr[i] = training_set_folds
		hold_out_set_arr[i] = hold_out_set_folds
	end

	#println("Preprocessing done!")
	#println("Async CV starting")
	#println("...")

	# Async CV
	@sync for i in 1:num_lambdas
		for j in 1:k
			#try
				Threads.@spawn partial_res_arr[i][j] = optim_res(obj_fun_arr[i][j],
														hold_out_set_arr[i][j],
														p0)
			#catch e
			#	println("Error suring CV fold!")
		    #	@show e
			#	if e == InterruptException
			#		break
			#	end
			#end
		end
	end

	# Post - Process results
	#println("Async CV done for $(method_label)!")
	#println("Evaluating results:")
	elapsed_time = zero(T)
	fold_error_mean = zero(T)
	lambda_hold_errors = Vector{Float64}(undef,num_lambdas)
	for i in 1:num_lambdas
		fold_error_mean = 0.0
		#println("\nResults for $(lambda_arr[i])")
		for j in 1:k
			pres = partial_res_arr[i][j]
			fold_error_mean += pres[1][1]
			elapsed_time += pres[2][1]
			#println("Error $(j):\n$(pres[1][1])")
		end
		fold_error_mean /= k
		#println("Mean error on $(lambda_arr[i]):\n$(fold_error_mean)\n")
		lambda_hold_errors[i] = fold_error_mean
	end

	elapsed_time /= num_lambdas*k
	best_lambda = lambda_arr[argmin(lambda_hold_errors)]

	#println("\nDone!")
	#println("Best lambda:\n$(best_lambda)")
	#println("Optimizing on whole dataset.")
	#println("...")

	final_obj_fun(x) = begin
						total_error = zero(T)
						for p in training_set
							total_error += f(x, p.data, p.t, p.fun)
						end
						total_error += tikhonov(best_lambda, x, phi_prior, w)
						return total_error
					end

	try
		partial_res = optim_res(final_obj_fun, testing_set, p0)
		results = [partial_res[1], [elapsed_time], partial_res[3]]

		#println("Done!")
		#println("Test Error:\n$(results[1])")
		#println("Estimated parameters:\n$(results[3])\n\n")
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

function get_results(method_label::String,
					training_set::Vector{ProblemSet.DEProblem},
					testing_set::Vector{ProblemSet.DEProblem},
	 				p0::Vector{T})::Array{Array{T}} where T

	k = convert(Int64, length(training_set)/2)

	# For DDS
	add_time = zero(T)

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
	elseif method_label == "DDS"
		f = data_shooting
		phi_prior = zeros(T,length(p0))
		m = length(phi_prior)
		w = Matrix{T}(I,m,m)
		partial_res = cv_optimize(training_set, testing_set,
										p0,k,f,phi_prior,m,w)
		add_time = partial_res[2][1]

		p0 = partial_res[3]
		f = single_shooting
		phi_prior = p0
		m = length(phi_prior)
		w = Matrix{T}(I,m,m)./p0
	end

	results = cv_optimize(training_set, testing_set,
							p0,k,f,phi_prior,m,w)
	results[2][1] += add_time

	return results
end

#@suppress begin
function experiment(p_num::Int64,sams::AbstractArray{<:Int},
					vars::AbstractArray{<:AbstractFloat},
					method_arr::Array{<:String},
					dir::String)::Dict
	results::Dict = Dict()
	prob_key::String = get_problem_key(p_num)
	problem::ProblemSet.DEProblem = get_problem(prob_key)

	full_path = dir*prob_key
	mkdir(full_path)

    fun::Function = problem.fun
    phi::Array = problem.phi
    bounds::Vector = problem.bounds
    ini_cond::Array = problem.data[1]
    t::AbstractArray = problem.t

	# Minimum number of data points
	min_data = round(length(phi)/length(ini_cond),RoundUp)
	data_sams = convert.(eltype(sams[1]),sams.*min_data)

	results_final = Dict()
	for sam in data_sams
		results_final[sam] = Dict()
	end

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

			num_reps = 10
			guess_arr = [rand_guess(bounds) for _ in 1:num_reps]

			n_training = 8
			training_set_arr = [[ProblemSet.DEProblem(problem.fun, problem.phi,
								problem.bounds, add_noise(data,v), _t)
				 			for _ in 1:n_training] for _ in 1:num_reps]

			n_testing = 2
			testing_set_arr = [[ProblemSet.DEProblem(problem.fun, problem.phi,
								problem.bounds, add_noise(data,v), _t)
				 			for _ in 1:n_testing] for _ in 1:num_reps]

			results_methods = Dict()
			for m in method_arr
				results_methods[m] = []
			end
			@sync for m in method_arr
				for i in 1:num_reps
					Threads.@spawn push!(results_methods[m], get_results(m,
										training_set_arr[i],
										testing_set_arr[i],
										guess_arr[i]))
				end
			end
			results_final[sam][v] = results_methods
	    end
	end #samples loop

	"""
	Plotting
	"""
	for sam in data_sams
		results = results_final[sam]
		plot_data = get_plot_data(results, vars, method_arr)

		error_plots(plot_data,vars,method_arr,method_label,method_color,sam,full_path)

		sr_plots(plot_data,vars,method_arr,method_label,method_color,sam,full_path)

		oe_plots(plot_data,vars,method_arr,method_label,method_color,sam,full_path)
	end
	return results_final
end

function problem_exp_loop(probs::Vector{<:Int},
							sams::Vector{<:Int},
							vars::Vector{<:AbstractFloat},
							method_arr::Vector{String},
							dir::String)
	for p in probs
		experiment(p,sams,vars,method_arr,dir)
	end
end

function main(args::Array{<:String})::Nothing
	dir = string(args[1])
	probs = eval(Meta.parse(args[2]))
	sams = eval(Meta.parse(args[3]))
	vars = eval(Meta.parse(args[4]))
	method_arr = ["DS","SS","DDS"]
	time_main = @time problem_exp_loop(probs,sams,vars,method_arr,dir)
	println(time_main)
	nothing
end

main(ARGS)
