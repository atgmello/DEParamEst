module Utils

import Distributions: Uniform
import Statistics: quantile
using Random
using Statistics
using Distances
using DifferentialEquations
using Plots
using Revise
using ..ProblemSet: DEProblem

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
"""
function add_noise(data::Vector{Vector{T}},
                    variance::T, epsilon::T)::Vector{Vector{T}} where T
    if variance == 0.0
        return data
    end

    noise_data = deepcopy(data)
    @inbounds for i in 1:length(noise_data)
        for j in 1:length(noise_data[1])
            if noise_data[i][j] == 0.0
                noise_data[i][j] += 10^-3
            end
            if -variance*noise_data[i][j] < variance*noise_data[i][j]
                a = -variance*noise_data[i][j]
                b = variance*noise_data[i][j]
            else
                b = -variance*noise_data[i][j]
                a = variance*noise_data[i][j]
            end
            noise_data[i][j] += rand(Uniform(a, b)) + epsilon
        end
    end
    return noise_data
end

"""
Adds noise to a Vector of T
"""
function add_noise(data::Vector{T},
                    variance::T, epsilon::T)::Vector{T} where T
    if variance == 0.0
        return data
    end

    noise_data = deepcopy(data)
    @inbounds for i in 1:length(noise_data)
        if noise_data[i] == 0.0
            noise_data[i] += 10^-3
        end
        if -variance*noise_data[i] < variance*noise_data[i]
            a = -variance*noise_data[i]
            b = variance*noise_data[i]
        else
            b = -variance*noise_data[i]
            a = variance*noise_data[i]
        end
        noise_data[i] += rand(Uniform(a, b)) + epsilon
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

function success_rate(x::T)::T where T<:AbstractFloat
    return 2/(1+exp(x))
end

function step_success_rate(x::AbstractFloat)::Int64
    y = 11/(10+exp10(x))
    if y > 0.5
       return 1
    else
        return 0
    end
end

function box_data(arr::Array{<:AbstractFloat})::Array{<:AbstractFloat}
    q = quantile(arr,[0.25,0.5,0.75])
    iqr = q[3] - q[1]
    min_val = q[1] - 1.5*iqr
    max_val = q[3] + 1.5*iqr
    return [maximum([min_val,minimum(arr)]),
            q[1],q[2],q[3],
            minimum([max_val,maximum(arr)])]
end

function box_scatter_plot(arr::Array{<:AbstractFloat})::Array{<:AbstractFloat}
    qarr = box_data(arr)
    return [qarr[3]-qarr[1],qarr[3],qarr[5]-qarr[3]]
end

function diff_calc(problem::DEProblem,
                    estimated::Array{<:AbstractFloat,1},
                    u0::Array{<:AbstractFloat})::Float64
    f = problem.fun
    t = problem.t
    known = problem.phi

    de_prob = ODEProblem(f,u0,(t[1],t[end]),known)
    de_sol = DifferentialEquations.solve(de_prob, AutoTsit5(Rosenbrock23()), saveat=t)
    data = reduce(hcat, de_sol.u)

    de_prob_est = ODEProblem(f,u0,(t[1],t[end]),estimated)
    de_sol_est = DifferentialEquations.solve(de_prob_est, AutoTsit5(Rosenbrock23()), saveat=t)
    data_est = reduce(hcat, de_sol_est.u)

    """
    Absolute Log Difference
    Since the data is required to be positive,
    find the minimum value from both actual and
    forecast data and lift up all values to avoid
    zeros and negative numbers.
    """
    min_val = minimum(vcat(data,data_est))
    if min_val < 0.0
        data .= data .- (min_val-0.1)
        data_est .= data_est .- (min_val-0.1)
    end

    ldiff = maximum(abs.(log.(data)-log.(data_est)))

    #=>
    p = plot(title="$(round(ldiff,digits=3))")
    plot!(p,t,data_actual')
    plot!(p,t,data_forecast',linestyle=:dashdot)
    display(p)
    sleep(2)
    <=#

    return ldiff
end

function max_diff_states(problem::DEProblem,
                        estimated::Array{<:AbstractFloat,1},
                        variance::Float64)::Float64

    _t = range(problem.t[1], stop=problem.t[end], length=1000)
    _problem = DEProblem(problem.fun, problem.phi,
                        problem.bounds, problem.data, _t)

    u0 = problem.data[1]

    epsilon = 10^-3
    reps = 10
    #u0_arr = collect(Map(x -> abs.(add_noise(x,variance))),eduction(u0 for _ in 1:reps))
    u0_arr = [abs.(add_noise(u0,variance,epsilon)) for _ in 1:reps]

    median_max_error = median([diff_calc(_problem,estimated,x) for x in u0_arr])

    return median_max_error
end

function log_scale_time(time::Array{T})::Array{T} where T<:AbstractFloat
	_arr = deepcopy(arr)
	while any(_arr .< 1.0)
		_arr .= _arr .* 10.0
    end
	return _arr
end


"""
Plotting functions
"""

function get_plot_data(results::Dict,prob_key::String,
                        vars::AbstractArray{<:AbstractFloat},
                        method_arr::Array{<:String,1})::Dict
    plot_data = Dict()
    for m in method_arr
        plot_data[m] = Dict()
        plot_data[m]["error"] = []
        plot_data[m]["time"] = []
    end

    for v in vars
        for m in method_arr
            error = [e[1][1] for e in results[prob_key][v][m]]
            if length(error) > 0
                push!(plot_data[m]["error"], error)
            else
                push!(plot_data[m]["error"], [NaN])
            end
            time = [e[2][1] for e in results[prob_key][v][m]]
            if length(time) > 0
                push!(plot_data[m]["time"], time)
            else
                push!(plot_data[m]["time"], [NaN])
            end
        end
    end
    return plot_data
end

function error_plots(plot_data::Dict,
                    vars::AbstractArray{<:AbstractFloat},
                    method_arr::Array,
                    method_label::Dict,
                    method_color::Dict,
                    sam::Int)::Nothing
    """
    Error Plots
    xaxis: Noise Percentage
    yaxis: Mean Error
    """
    p = plot(x=vars, xlabel="Noise Percentage", ylabel="Error")
    ylim_arr = []
    for m in method_arr
        p2 = plot(x=vars, xlabel="Noise Percentage", ylabel="Error")
        error = plot_data[m]["error"]
        # Proceed only if there are no NaN
        if !any(isnan.(vcat(error...)))
            qerror = hcat(box_scatter_plot.(error)...)
            # Don't append to plot p if data is too far
            # from the expected values (i.e. plot only if
            # data < 20)
            if !any(qerror[3,:] .> 20.0)
                plot!(p, vars, qerror[2,:],
                            grid=true,
                            ribbon=(qerror[1,:],
                                    qerror[3,:]),
                            fillalpha=.5, label=method_label[m], color=method_color[m])
                push!(ylim_arr, ylims(p))
            end

            plot!(p2, vars, qerror[2,:],
                        grid=true,
                        ribbon=(qerror[1,:],
                                qerror[3,:]),
                        fillalpha=.5, label=method_label[m], color=method_color[m])
            #display(p2)

            savefig(p,"./error_inter_$(m)_$(sam).svg")
            savefig(p2,"./error_$(m)_$(sam).svg")
        end
    end
    #=>
    if length(ylim_arr) > 1
        ylims!(p, (minimum(abs.([yl[1] for yl in ylim_arr])),
                    minimum(abs.([yl[2] for yl in ylim_arr]))))
    end
    <=#
    #display(p)

    savefig(p,"./error_all_$(sam).svg")
    nothing
end

function sr_plots(plot_data::Dict,
                    vars::AbstractArray{<:AbstractFloat},
                    method_arr::Array{<:String,1},
                    method_label::Dict,
                    method_color::Dict,
                    sam::Int)::Nothing
    """
    Success Rate vs Time Plots
    xaxis: Mean Computation Time
    yaxis: 1 / Success Rate
    """

    p = scatter(xlabel="Time", ylabel="1 / Success Rate")
    p2 = scatter(xlabel="Time", ylabel="1 / Success Rate")
    ylim_arr = []
    for m in method_arr
        p3 = scatter(xlabel="Time", ylabel="1 / Success Rate")

        sr = mean.([step_success_rate.(e) for e in plot_data[m]["error"]])
        isr = sr.^(-1)
        time = plot_data[m]["time"]
        if !any(isnan.(vcat(time...)))
            qtime = hcat(box_scatter_plot.(time)...)
            qqtime = box_scatter_plot(qtime[2,:])
            qisr = box_scatter_plot(isr)
            if qisr[2] < 20.0 || qqtime[2] < 20.0
                scatter!(p, (qqtime[2],qisr[2]),
                            xerror=[(qqtime[1],qqtime[3])],
                            yerror=[(qisr[1],qisr[3])],
                            label=method_label[m], color=method_color[m])
            end

            scatter!(p2, (qtime[2,:], isr),
                        label=method_label[m], color=method_color[m])

            scatter!(p3, (qtime[2,:], isr),
                        label=method_label[m], color=method_color[m],
                        series_annotations = text.(vars, :top, 11))

            #display(p3)

            savefig(p3,"./sr_$(m)_$(sam).svg")
        end
    end
    #display(p)
    #display(p2)

    savefig(p, "./sr_all_medians_$(sam).svg")
    savefig(p2, "./sr_all_$(sam).svg")
    nothing
end

function oe_plots(plot_data::Dict,
                    vars::AbstractArray{<:AbstractFloat},
                    method_arr::Array{<:String,1},
                    method_label::Dict,
                    method_color::Dict,
                    sam::Int)::Nothing
    """
    Overall Efficiency (OE) Plots
    xaxis: Method
    yaxis: OE Score
    """

    t_succ = zeros(length(method_arr))
    @inbounds for i in 1:length(method_arr)
        m = method_arr[i]
        error = plot_data[m]["error"]
        sr = mean.([step_success_rate.(e) for e in error])
        time = plot_data[m]["time"]
        mean_time = mean.(time)
        t_succ_arr = mean_time./sr
        t_succ[i] = median(t_succ_arr)
    end

    oe = minimum(t_succ)./t_succ

    p = bar(xlabel="Method",
            ylabel="Overall Efficiency",
            legend=false)
    bar!(p, method_arr, oe,
            color=[method_color[m] for m in method_arr])
    #display(p)

    savefig(p, "./$(sam)_oe.svg")
    nothing
end

end
