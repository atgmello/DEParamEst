ENV["GKSwstype"]="nul"

using Pkg
Pkg.activate(".")
Pkg.instantiate()

using Distributed
using DiffEqParamEstim
using DifferentialEquations
using Statistics
using Random
import StatsBase: sample
import Distributions: Uniform
using Optim
using PlotThemes
using StatsPlots
using JLSO

using Revise
includet("./problem_set.jl")
includet("./objective_function.jl")
includet("./utils.jl")
import .ProblemSet: get_problem, get_problem_key, DEProblem
import .ObjectiveFunction: data_shooting, single_shooting, tikhonov
import .Utils: rand_guess, add_noise,
				scale_eval,
				diff_calc, step_success_rate, success_rate, box_data, box_scatter_plot,
				get_plot_data, oe_plots, sr_plots, error_plots,
				nmse, plot_compare, box_error_plots, parameter_plots

@assert Threads.nthreads() > 1

Random.seed!(1234)
gr()
theme(:vibrant)

#old_precision = precision(BigFloat)
#new_precision = 1024
#setprecision(new_precision)
#setprecision(old_precision)
#const dp = Float64

"""
Performs optimization and returns a vector with
information about test error (first vector position),
how much time was spent (second position) and
estimated parameters (third position)
"""
function optim_res(obj_fun::Function,
					testing_set::Vector{ProblemSet.DEProblem},
					p0::Vector{T})::Vector{Vector{T}} where T
	g_t_lim = 10
	f_tol = 10^-12
	x_tol = 10^-6
	iter = 10^8

	SAMIN_options = Optim.Options(x_tol=x_tol, f_tol=f_tol,
								iterations=iter, time_limit=g_t_lim)
	t_lim = 1

	Grad_options = Optim.Options(x_tol=x_tol, f_tol=f_tol,
								iterations=iter, time_limit=t_lim)

	inner_optimizer = Optim.LBFGS()

	lb = testing_set[1].bounds[1]
	ub = testing_set[1].bounds[2]
	elapsed_time = zero(T)
	phi_est = deepcopy(p0)
	nrmse = Inf64

	try
		elapsed_time += @elapsed res_obj = Optim.optimize(obj_fun,
									lb,ub,
									p0,
									Optim.SAMIN(verbosity=0, rt=0.15), SAMIN_options)

		#elapsed_time += @elapsed res_obj = Optim.optimize(obj_fun,
		#							lb,ub,
		#							res_obj.minimizer[1:length(p0)],
		#							Fminbox(inner_optimizer),
		#							Grad_options,
		#							autodiff = :forward)

		phi_est = res_obj.minimizer[1:length(p0)]

		p = testing_set[1]
	    tspan = (p.t[1], p.t[end])
	    ode_prob = ODEProblem(p.fun, p.data[1], tspan, phi_est)
	    ode_sol  = solve(ode_prob, OwrenZen3(), saveat=p.t)
		data_est = reduce(vcat,ode_sol.u)

		# Normalized Root Mean Squared Error
	    nrmse = mean([nmse(reduce(vcat,tp.data), data_est)
				for tp in testing_set]) |> x -> sqrt(x)

		"""
		Plotting
		p = testing_set[1]
		plot_compare(p.data, ode_sol.u)
		"""
	catch e
		println("Optim error!")
		@show e
	end

	if elapsed_time == zero(T)
		elapsed_time == NaN64
	end

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

	results = [[Inf64], [NaN64], p0]
	#lambda_arr = [1e-2,1e-1,0.0,1e0,1e1,1e2]
	lambda_arr = [1e0,0.0,1e-1,1e-2,1e-3]
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

	num_elem_train_slice = convert(Int64, (num_training_sets - num_elem_folds))

	# Pre - Allocate and prepare necessary data
	partial_res = [[Inf64], [NaN64], p0]
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
			slice = filter(x -> !(x in hold_out), 1:num_training_sets)

			training_sliced = training_set[slice]
			training_set_folds[j] = training_sliced

			hold_out_set = training_set[hold_out]
			hold_out_set_folds[j] = hold_out_set

			obj_fun = function (x)
						total_error = zero(T)
						@inbounds for i in 1:length(training_sliced)
							total_error += f(x,
											training_sliced[i].data,
											training_sliced[i].t,
											training_sliced[i].fun)
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
			Threads.@spawn partial_res_arr[i][j] = optim_res(obj_fun_arr[i][j],
														hold_out_set_arr[i][j],
														p0)
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
						@inbounds for i in 1:length(training_set)
							total_error += f(x,
											training_set[i].data,
											training_set[i].t,
											training_set[i].fun)
						end
						total_error += tikhonov(best_lambda, x, phi_prior, w)
						return total_error
					end

	try
		partial_res = optim_res(final_obj_fun, testing_set, p0)
		results = [partial_res[1],
					[(partial_res[2][1]+elapsed_time)/2.0],
					partial_res[3]]

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
    ode_sol  = solve(ode_prob, OwrenZen3(), saveat=p.t)
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

	k = convert(Int64, length(training_set)/1)

	# For DSS
	add_time = zero(T)

	if method_label == "SS"
		f = single_shooting
		phi_prior = zeros(T,length(p0))
		m = length(phi_prior)
		w = ones(T,m)

		partial_res = cv_optimize(training_set,testing_set,
										p0,k,f,phi_prior,m,w)
		add_time = partial_res[2][1]
		phi_prior = partial_res[3]
		p0 = phi_prior
	elseif method_label == "DS"
		f = data_shooting
		phi_prior = zeros(T,length(p0))
		m = length(phi_prior)
		w = ones(T,m)
	elseif method_label == "DSS"
		f = data_shooting
		phi_prior = zeros(T,length(p0))
		m = length(phi_prior)
		w = ones(T,m)
		partial_res = cv_optimize(training_set,testing_set,
										p0,k,f,phi_prior,m,w)
		add_time = partial_res[2][1]

		f = single_shooting
		phi_prior = partial_res[3]
		p0 = phi_prior
		if partial_res[1][1] < 0.3
			w = ones(T,m)./phi_prior
		else
			w = ones(T,m)
		end
	end

	results = cv_optimize(training_set,testing_set,
							p0,k,f,phi_prior,m,w)
	results[2][1] += add_time

	return results
end

function experiment(p_num::Int64,samples::AbstractArray{<:Int},
					noise_levels::AbstractArray{<:AbstractFloat},
					methods::Array{<:String},
					dir::String)::Dict

	all_methods = ["DS","SS","DSS"]

	results::Dict = Dict()
	prob_key::String = get_problem_key(p_num)
	problem::ProblemSet.DEProblem = get_problem(prob_key)

    fun::Function = problem.fun
    phi::Array = problem.phi
    bounds::Vector = problem.bounds
    ini_cond::Array = problem.data[1]
    t::AbstractArray = problem.t

	# Minimum number of data points
	#min_data = round(length(phi)/length(ini_cond),RoundUp)
	#data_sams = convert.(eltype(sams[1]),sams.*min_data)
	data_sams = samples

	num_reps = 10

	results_final = Dict()
	for sam in data_sams
		results_final[Symbol(sam)] = Dict()
		for noise in noise_levels
			results_final[Symbol(sam)][Symbol(noise)] = Dict()
			for m in methods
				results_final[Symbol(sam)][Symbol(noise)][Symbol(m)] =
					[[[NaN64] for _ in 1:3]
					for _ in 1:num_reps]
			end
		end
	end

	num_trainings = 5
	num_tests = 10

	training_set_dict = Dict()
	testing_set_dict = Dict()
	guess_dict = Dict()
	for sam in data_sams
		training_set_dict[sam] = Dict()
		testing_set_dict[sam] = Dict()
		guess_dict[sam] = Dict()
		for noise in noise_levels
			training_set_dict[sam][noise] = [Vector{ProblemSet.DEProblem}(undef,num_trainings)
											for _ in 1:num_reps]
			testing_set_dict[sam][noise] = [Vector{ProblemSet.DEProblem}(undef,num_tests)
											for _ in 1:num_reps]
			guess_dict[sam][noise] = [Vector{Float64}(undef,length(phi))
									for _ in 1:num_reps]
		end
	end

	var_ini_cond = 0.01

	for sam in data_sams
		for noise in noise_levels
	        # Artificial Data
            _t = range(t[1], stop=t[end], length=sam)
	        tspan = (t[1], t[end])
	        ode_prob = ODEProblem(fun, ini_cond, tspan, phi)
	        ode_sol  = solve(ode_prob, OwrenZen3(), saveat=_t)
	        data = ode_sol.u
	        #data_plot = plot(t,data')
			#display(data_plot)

			guess_arr = [rand_guess(bounds) for _ in 1:num_reps]
			guess_dict[sam][noise] .= guess_arr

			training_set_arr = [[ProblemSet.DEProblem(problem.fun, problem.phi,
								problem.bounds, add_noise(data,noise), _t)
				 			for _ in 1:num_trainings] for _ in 1:num_reps]
			training_set_dict[sam][noise] .= training_set_arr

			testing_set_arr = []
			for r in 1:num_reps
				testing_set_arr_partial = []
				for j in 1:num_tests
					_t = range(t[1], stop=t[end], length=sam)
			        tspan = (t[1], t[end])
			        ode_prob = ODEProblem(fun, add_noise(ini_cond,var_ini_cond), tspan, phi)
			        ode_sol  = solve(ode_prob, OwrenZen3(), saveat=_t)
			        data = ode_sol.u

					push!(testing_set_arr_partial,ProblemSet.DEProblem(problem.fun, problem.phi,
										problem.bounds, add_noise(data,noise), _t))
				end
				push!(testing_set_arr,testing_set_arr_partial)
			end
			testing_set_dict[sam][noise] .= testing_set_arr
	    end
	end #samples loop

	@sync for sam in data_sams
		for noise in noise_levels
			for m in methods
				for i in 1:num_reps
					Threads.@spawn results_final[Symbol(sam)][Symbol(noise)][Symbol(m)][i] =
											get_results(m,
											training_set_dict[sam][noise][i],
											testing_set_dict[sam][noise][i],
											guess_dict[sam][noise][i])
				end
			end
		end
	end

	return results_final
end

function run_experiments(problems::Vector{<:Int},
							samples::Vector{<:Int},
							noise_levels::Vector{<:AbstractFloat},
							methods::Vector{String},
							dir::String)::Nothing

	# Temporary directory for saving
	# serialized intermediary results
	mkdir(joinpath(dir,"tmp"))

	results = Vector(undef,length(problems))
	for i in 1:length(problems)
		p = problems[i]
		p_name = get_problem_key(p)
		result = experiment(p,samples,noise_levels,methods,dir)
		results[i] = Pair(Symbol(p_name), result)
		try
			JLSO.save(joinpath(dir,"tmp",p_name*".jlso"), results[i])
		catch e
			println("Error saving serialized $(results[i])!")
			@show e
		end
	end

	try
		JLSO.save(joinpath(dir,"experiment_results.jlso"), results...)
		rm(joinpath(dir,"tmp"), recursive=true)
	catch e
		println("Error saving serialized results!")
		@show e
	end

end

function main(args::Array{<:String})::Nothing
	path = string(args[1])
	problems = eval(Meta.parse(args[2]))
	samples = eval(Meta.parse(args[3]))
	noise_level = eval(Meta.parse(args[4]))
	methods = ["DS","SS","DSS"]
	time_main = @elapsed run_experiments(problems,samples,
										noise_level,methods,path)
	println(time_main)
end

main(ARGS)
