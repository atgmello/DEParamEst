using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using Distributed
using DiffEqParamEstim
using DifferentialEquations
using Statistics
using Random
using Optim
using JLSO
using Logging
using LoggingExtras
import StatsBase: sample
import Distributions: Uniform
import Dates: now

using Revise
includet("./problem_set.jl")
includet("./objective_function.jl")
includet("./utils.jl")
import .ProblemSet: get_problem, get_problem_key, DEProblem
import .ObjectiveFunction: data_shooting, single_shooting, tikhonov
import .Utils: rand_guess, add_noise, nmse, transpose_vector

Random.seed!(1234)

const RT = 0.95
const MAXT = 10^6
const M = 50
const MIN_LOG = Logging.Info

"""
Performs optimization and returns a vector with
information about test error (first vector position),
how much time was spent (second position) and
estimated parameters (third position)
"""
function optim_res(obj_fun::Function,
					testing_set::Vector{ProblemSet.DEProblem},
					p0::Vector{T})::Vector{Vector{T}} where T
	g_t_lim = MAXT
	f_tol = 10^-12
	x_tol = 10^-6
	iter = 10^8

	SAMIN_options = Optim.Options(x_tol=x_tol, f_tol=f_tol,
								iterations=iter, time_limit=g_t_lim)

	# Local method disabled since it does not improve accuracy
	# t_lim = 1

	# Grad_options = Optim.Options(x_tol=x_tol, f_tol=f_tol,
	# 							iterations=iter, time_limit=t_lim)

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
									Optim.SAMIN(verbosity=0, rt=RT), SAMIN_options)

		# Local method disabled since it does not improve accuracy
		#elapsed_time += @elapsed res_obj = Optim.optimize(obj_fun,
		#							lb,ub,
		#							res_obj.minimizer[1:length(p0)],
		#							Fminbox(inner_optimizer),
		#							Grad_options,
		#							autodiff = :forward)

		phi_est = res_obj.minimizer[1:length(p0)]
	catch e
		bt = backtrace()
		msg = sprint(showerror, e, bt)
		@warn """Error!
				   There was an error during the optimization step.""" msg
	end

	try
		generate_estimated_data = function (p)
			tspan = (p.t[1], p.t[end])
			ode_prob = ODEProblem(p.fun, p.data[1], tspan, phi_est)
			ode_sol  = solve(ode_prob, AutoVern7(Rodas5()), saveat=p.t)
			ode_sol.u
		end

		generate_then_transpose(p) = p |> generate_estimated_data |> transpose_vector
		data_estimated_set = map(generate_then_transpose, testing_set)

		get_then_transpose(p) = p.data |> transpose_vector
		data_training_set = map(get_then_transpose, testing_set)

		compare_testing_estimated = zip(data_training_set, data_estimated_set)

		nmse_each_state_vector(compare) =  map(vector -> nmse(vector[1], vector[2]),
												zip(compare[1], compare[2]))

		total_nmse = map(nmse_each_state_vector, compare_testing_estimated)
		# (mean) Normalized Root Mean Squared Error
		nrmse = mean(total_nmse) |> mean |> sqrt
	catch e
		bt = backtrace()
		msg = sprint(showerror, e, bt)
		@warn """Error!
				   There was an error during the NRMSE calculation step.""" msg
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
	lambda_arr = [0.0,1e-3,1e-2,1e-1,1e0]
	num_lambdas = length(lambda_arr)
	best_lambda = lambda_arr[1]

	bounds = training_set[1].bounds

	elapsed_time = zero(T)

	# Lamdba selection via
	# Cross Validation
	@debug """"$(method)
				Initial guess:\n$(p0)
				Starting Cross Validation!
				Preprocessing starting
				..."""

	if sum(w) != 0.0
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

		@debug """Preprocessing done!
					Async CV starting
					..."""

		# Async CV
		@sync for i in 1:num_lambdas
			for j in 1:k
				Threads.@spawn partial_res_arr[i][j] = optim_res(obj_fun_arr[i][j],
																 hold_out_set_arr[i][j],
																 p0)
			end
		end

		# Post - Process results
		@debug """Async CV done for $(method)!
					Evaluating results:"""
		fold_error_mean = zero(T)
		lambda_hold_errors = Vector{Float64}(undef,num_lambdas)
		for i in 1:num_lambdas
			fold_error_mean = 0.0
			@debug """Results for $(lambda_arr[i])"""
			for j in 1:k
				pres = partial_res_arr[i][j]
				fold_error_mean += pres[1][1]
				elapsed_time += pres[2][1]
				@debug """Error $(j)""" pres[1][1]
			end
			fold_error_mean /= k
			@debug """Mean error on $(lambda_arr[i])""" fold_error_mean
			lambda_hold_errors[i] = fold_error_mean
		end

		elapsed_time /= num_lambdas*k
		best_lambda = lambda_arr[argmin(lambda_hold_errors)]

		@debug """Done!
					Best lambda:\n$(best_lambda)
					Optimizing on whole dataset.
					..."""
	end

	final_obj_fun = function (x)
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
				   # [(partial_res[2][1]+elapsed_time)/2.0],
				   partial_res[2],
				   partial_res[3]]

		@debug """Done!
					Test Error:\n$(results[1])
					Estimated parameters:\n$(results[3])\n"""
	catch e
		bt = backtrace()
		msg = sprint(showerror, e, bt)
		@warn """Error!
				   There was an error during the call to optim_res.""" msg
	end

	@info """Results.
			  Problem:\t$(Symbol(training_set[1].fun))
			  Method:\t$(Symbol(f))
			  Lambdas:\t$(lambda_arr)
			  Initial phi:\t$(p0)
			  CV Errors:\t$(@isdefined(lambda_hold_errors) ? lambda_hold_errors : false)
			  Best lambda:\t$(best_lambda)
			  Final Errors:\t$(results[1][1])
			  Final phi:\t$(results[3])"""
	return results
end

function get_results(method::String,
					training_set::Vector{ProblemSet.DEProblem},
					testing_set::Vector{ProblemSet.DEProblem},
	 				p0::Vector{T})::Array{Array{T}} where T

	k = convert(Int64, length(training_set)/1)

	add_time = zero(T)

	if method == "SS"
		f = single_shooting
		phi_prior = zeros(T,length(p0))
		m = length(phi_prior)
		w = zeros(T,m)
	elseif method == "SSR"
		f = single_shooting
		phi_prior = zeros(T,length(p0))
		m = length(phi_prior)
		w = ones(T,m)

		partial_res = cv_optimize(training_set,testing_set,
										p0,k,f,phi_prior,m,w)
		add_time = partial_res[2][1]
		phi_prior = partial_res[3]
		p0 = phi_prior
	elseif method == "DS"
		f = data_shooting
		phi_prior = zeros(T,length(p0))
		m = length(phi_prior)
		w = zeros(T,m)
	elseif method == "DSS"
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

		# Check the model error.
		# If the model is good, then use the
		# best case strategy.
		# Otherwise, use medium case strategy.
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

	all_methods = ["DS","SS","SSR","DSS"]

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

	num_reps = M

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
			ode_sol  = solve(ode_prob, AutoVern7(Rodas5()), saveat=_t)
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
					ode_sol  = solve(ode_prob, AutoVern7(Rodas5()), saveat=_t)
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
			bt = backtrace()
			msg = sprint(showerror, e, bt)
			@warn """Error!
						Error during the call to JLSO.save for $(results[i])."""
		end
	end

	try
		JLSO.save(joinpath(dir,"experiment_results.jlso"), results...)
		rm(joinpath(dir,"tmp"), recursive=true)
	catch e
		bt = backtrace()
		msg = sprint(showerror, e, bt)
		@warn """Error!
				   Error when joining all the results.""" msg
	end

end

function main(args::Array{<:String})::Nothing
	path = string(args[1])
	problems = eval(Meta.parse(args[2]))
	samples = eval(Meta.parse(args[3]))
	noise_level = eval(Meta.parse(args[4]))

	methods = ["DS","SS","SSR","DSS"]

	main_file_logger = MinLevelLogger(FileLogger(joinpath(path,"general.log")),
									  MIN_LOG)
	console_logger = ConsoleLogger()
	main_logger = TeeLogger(console_logger, main_file_logger)
	global_logger(main_logger)

	@info "Program started." now()
	@info "Number of threads." Threads.nthreads()

	minor_logger = FileLogger(joinpath(path,"info.log"))
	minor_and_global_logger = TeeLogger(minor_logger, global_logger())

	with_logger(minor_and_global_logger) do
		@info "Arguments." args
		@info "Methods." methods
		@info "Cooling rate." RT
		@info "Maximum Time." MAXT
		@info "Number of runs." M
	end

	elapsed_time = @elapsed run_experiments(problems,samples,
										noise_level,methods,path)

	with_logger(minor_and_global_logger) do
		@info "Program finished." now()
		@info "Elapsed time." elapsed_time
	end

end

main(ARGS)
