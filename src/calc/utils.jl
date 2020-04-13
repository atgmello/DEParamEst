module Utils

import Distributions: Normal, Uniform
import Statistics: quantile
using Random
using Statistics
using Distances
using DifferentialEquations
using Revise
using ..ProblemSet: DEProblem

Random.seed!(1234)

"""
From bounds, generates a random startign point for
the optimization procedure.
"""
function rand_guess(bounds::Vector)::Vector
    [rand(Uniform(bounds[1][i], bounds[2][i]))
            for i in 1:length(bounds[1])]
end

"""
Adds noise to a Vector of Vector
Used for adding simulated noise to
the states of a system.
Assumes that the values can be negative.
"""
function add_noise(data::Vector{Vector{T}},
                    percent::T,positive::Bool=false)::Vector{Vector{T}} where T
    if percent == 0.0
        return data
    end

    noise_data = deepcopy(data)
    epsilon_arr = [0.01*mean(getindex.(data,i)) for i in 1:length(data[1])]
    sigma = zero(T)
    @inbounds for i in 1:length(noise_data)
        for j in 1:length(noise_data[1])
            sigma = abs(percent*noise_data[i][j] + epsilon_arr[j])
            d = Normal(0,sigma)
            noise_data[i][j] += rand(d)
            noise_data[i][j] = positive ? abs(noise_data[i][j]) : noise_data[i][j]
        end
    end
    return noise_data
end

"""
Adds noise to a Vector of T
Used for generating new reasonable
initial values to an IVP. Assumes
that the values should be positive.
"""
function add_noise(data::Vector{T},
                    percent::T,positive::Bool=true)::Vector{T} where T
    if percent == 0.0
        return data
    end

    noise_data = deepcopy(data)
    epsilon = 0.01 * mean(data)
    sigma = zero(T)
    @inbounds for i in 1:length(noise_data)
        sigma = abs(percent*noise_data[i] + epsilon)
        d = Normal(0,sigma)
        noise_data[i] += rand(d)
        noise_data[i] = positive ? abs(noise_data[i]) : noise_data[i]
    end
    return noise_data
end

function fae(actual::Array, forecast::Array)::Number
    n = length(actual)
    frac = 1/n
    e = abs.(actual-forecast)
    normalize = abs.(actual)+abs.(forecast)
    stat = frac*sum(2*e/normalize)
end

function smape(actual::Array, forecast::Array)::Float16
    100*fae(actual,forecast)
end

function mape(actual::Array, forecast::Array)::Float32
    n = length(actual)
    frac = 1/n
    e = abs.(actual-forecast)
    normalize = abs.(actual)
    stat = 100*frac*sum(e/normalize)
end

function mrae(actual::Array, forecast::Array)::Float32
    n = length(actual)
    e = abs.(actual-forecast)
    stat = (1/n)*sum(e./abs.(actual))
end

"""
Normalized Mean Square Error
"""
function nmse(data::Vector{T}, data_est::Vector{T})::T where T
    normalizer = abs2(maximum(data_est) - minimum(data_est))
    res = sum(abs2.(data-data_est))/normalizer
    res /= length(data)
    return res
end

function filter_outlier(arr::Array; p::Float64=2)::Array
    q = quantile(arr)
    iqr = q[3] - q[1]
    maxValue = q[3] + iqr * p
    minValue = q[1] - iqr * p

    return filter(x -> (x >= minValue) && (x <= maxValue), arr)
end

"""
A Trace object is intended to hold
multiple pairs of time and eval.
Therefore, the attributes time and
eval shall be array of arrays, that,
when zipped, recreate the pairs
mentioned above.
"""
struct Trace
    time::Array{<:Array{<:AbstractFloat},1}
    eval::Array{<:Array{<:AbstractFloat},1}
end

function make_trace(trace)::Trace
    time = []
    eval = []
    @inbounds for i in 1:length(trace)
        append!(time, parse(Float64, split(string(trace[i]))[end]))
        append!(eval, parse(Float64, split(string(trace[i]))[2]))
    end
    return Trace(time, eval)
end

"""
Given a trace, make sure that
both time and eval arrays have the same size.
This is done by repeating the last registered
value in each array.
"""
function fill_trace(t::Trace)::Trace
    max_t_array_len = maximum(map(x -> length(x), t.time))
    resized_t = map( x -> append!(x, fill(x[end], max_t_array_len - length(x))), t.time)

    max_eval_array_len = maximum(map(x -> length(x), t.eval))
    resized_eval = map( x -> append!(x, fill(x[end], max_eval_array_len - length(x))), t.eval)

    return Trace(resized_t, resized_eval)
end

function scale_eval(t::Trace)::Trace
    scaled_eval = [e/e[end] for e in t.eval]
    return Trace(t.time, scaled_eval)
end

"""
Given a Trace
select the best trace (the one with the least amount of
time spent to get to the optimal solution) and the worst
trace (the one with the most)
"""
function get_range_traces(trace::Trace)::Trace
    trace_times = [t[end] for t in trace.time]
    (_,min_idx) = findmin(trace_times)
    best = Trace(trace.time[min_idx], trace.eval[min_idx])

    (_,max_idx) = findmax(trace_times)
    worst = Trace(trace.time[max_idx], trace.eval[max_idx])

    if length(trace_times)%2 == 0
        append!(trace_times, 0.0)
        median_idx = findfirst(x -> x == median(trace_times), trace_times)
        median_idx -= 1
    else
        median_idx = findfirst(x -> x == median(trace_times), trace_times)
    end
    med = Trace(trace.time[median_idx], trace.eval[median_idx])

    return best, med, worst
end


function success_rate(x::T)::T where T
    return 2/(1+exp100(x))
end


function step_success_rate(x::T)::Int64 where T
    if x < 0.125
       return 1
    else
        return 0
    end
end


"""
Absolute Log Difference
Since the data is required to be positive,
find the minimum value from both actual and
forecast data and lift up all values to avoid
zeros and negative numbers.
"""
function abs_log_diff(data::Vector{T}, data_est::Vector{T})::T where T
    min_val = minimum(vcat(data,data_est))
    if min_val < 0.0
        data .= data .- (min_val-0.1)
        data_est .= data_est .- (min_val-0.1)
    end
    return abs.(log.(data)-log.(data_est))
end

function diff_calc(problem::DEProblem,
                    estimated::Array{<:AbstractFloat,1},
                    u0::Array{<:AbstractFloat},
                    f::Function)::Float64
    f = problem.fun
    t = problem.t
    known = problem.phi

    de_prob = ODEProblem(f,u0,(t[1],t[end]),known)
    de_sol = DifferentialEquations.solve(de_prob, Tsit5(), saveat=t)
    data = reduce(hcat, de_sol.u)

    de_prob_est = ODEProblem(f,u0,(t[1],t[end]),estimated)
    de_sol_est = DifferentialEquations.solve(de_prob_est, Tsit5(), saveat=t)
    data_est = reduce(hcat, de_sol_est.u)


    fdiff = mean(f(data, data_est))

    return fdiff
end

function max_diff_states(problem::DEProblem,
                        estimated::Array{<:AbstractFloat,1},
                        variance::Float64)::Float64

    len = round(Int64, 1e3*(problem.t[end]-problem.t[1]))
    _t = range(problem.t[1], stop=problem.t[end], length=len)
    _problem = DEProblem(problem.fun, problem.phi,
                        problem.bounds, problem.data, _t)

    u0 = problem.data[1]
    reps = 5
    #u0_arr = collect(Map(x -> abs.(add_noise(x,variance))),eduction(u0 for _ in 1:reps))
    u0_arr = [abs.(add_noise(u0,variance)) for _ in 1:reps]

    median_max_error = mean([diff_calc(_problem,estimated,x) for x in u0_arr])

    return median_max_error
end

function log_scale_time(time::Array{T})::Array{T} where T<:AbstractFloat
	_arr = deepcopy(arr)
	while any(_arr .< 1.0)
		_arr .= _arr .* 10.0
    end
	return _arr
end

end
