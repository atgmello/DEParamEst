module Utils

import Distributions: Normal, Uniform
import Statistics: quantile
using Random
using Statistics
using Distances
using DifferentialEquations
using StatsPlots
using PlotThemes
using Revise
using ..ProblemSet: DEProblem

gr()
theme(:vibrant)
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

    #=>
    p = plot(title="$(round(ldiff,digits=3))")
    plot!(p,t,data_actual')
    plot!(p,t,data_forecast',linestyle=:dashdot)
    display(p)
    sleep(2)
    <=#

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


"""
Plotting functions
"""

function get_plot_data(results::Dict,
                        noise_level::AbstractArray{Symbol},
                        methods::Vector{Symbol})::Dict
    plot_data = Dict()
    for m in methods
        plot_data[m] = Dict()
        plot_data[m]["error"] = []
        plot_data[m]["time"] = []
        plot_data[m]["est"] = []
    end
    for v in noise_level
        for m in methods
            error = [e[1][1] for e in results[v][m]]
            if length(error) > 0
                push!(plot_data[m]["error"], error)
            else
                push!(plot_data[m]["error"], [NaN])
            end
            time = [e[2][1] for e in results[v][m]]
            if length(time) > 0
                push!(plot_data[m]["time"], time)
            else
                push!(plot_data[m]["time"], [NaN])
            end
            est = [e[3] for e in results[v][m]]
            if length(time) > 0
                push!(plot_data[m]["est"], est)
            else
                push!(plot_data[m]["est"], [NaN])
            end
        end
    end
    return plot_data
end

function error_plots(plot_data::Dict,
                    noise_level::AbstractArray{<:AbstractFloat},
                    methods::Array,
                    method_label::Dict,
                    method_color::Dict,
                    sam::Int,
                    path::String)::Nothing
    """
    Error Plots
    xaxis: Noise Percentage
    yaxis: Mean Error
    """
    p = plot(x=noise_level, xlabel="Noise Percentage", ylabel="Error", legend=:outertopright)
    ylim_arr = []
    for m in methods
        p2 = plot(x=noise_level, xlabel="Noise Percentage", ylabel="Error", legend=:outertopright)
        error = plot_data[m]["error"]
        # Proceed only if there are no NaN
        if !any(isnan.(vcat(error...)))
            qerror = hcat(box_scatter_plot.(error)...)
            # Don't append to plot p if data is too far
            # from the expected values (i.e. plot only if
            # data < 5)
            if !any(qerror[3,:] .> 5.0)
                plot!(p, noise_level, qerror[2,:],
                            grid=true,
                            ribbon=(qerror[1,:],
                                    qerror[3,:]),
                            fillalpha=.5, label=method_label[m], color=method_color[m])
                push!(ylim_arr, ylims(p))
            end

            plot!(p2, noise_level, qerror[2,:],
                        grid=true,
                        ribbon=(qerror[1,:],
                                qerror[3,:]),
                        fillalpha=.5, label=method_label[m], color=method_color[m])
            #display(p2)

            savefig(p,path*"/error_inter_$(m)_$(sam).pdf")
            savefig(p2,path*"/error_$(m)_$(sam).pdf")
        end
    end
    #=>
    if length(ylim_arr) > 1
        ylims!(p, (minimum(abs.([yl[1] for yl in ylim_arr])),
                    minimum(abs.([yl[2] for yl in ylim_arr]))))
    end
    <=#
    #display(p)

    savefig(p,path*"/error_all_$(sam).pdf")
    nothing
end

"""
Box Error Plots
xaxis: Method
yaxis: Error Distribution
"""
function box_error_plots(plot_data::Dict,
                    var::Symbol,
                    methods::Vector{Symbol},
                    method_label::Dict,
                    method_color::Dict,
                    sam::Symbol,
                    path::String)::Nothing

    p = plot(legend=false, ylabel="Error", xlabel="Method")
    for m in methods
        # Substitute infinite for missing
        # so that boxplot can still work
        # with the data
        data = [ifelse(isinf(x),missing,x) for x in plot_data[m]["error"][1]]
        boxplot!(p, [method_label[m]], data, color=method_color[m])
    end
    savefig(p,joinpath(path,replace("box_$(sam)_$(var)","."=>"")*".pdf"))
end

"""
Parameter Distribution Plots
xaxis: Parameter
yaxis: Value Distribution
"""
function parameter_plots(plot_data::Dict,
                    var::Symbol,
                    methods::Array,
                    method_label::Dict,
                    method_color::Dict,
                    sam::Symbol,
                    path::String)::Nothing
    num_pars = length(plot_data[methods[1]]["est"][1][1])
    p_arr = []
    for m in methods
        p = plot(legend=:outertopright, ylabel="Value", xlabel="Parameter")
    	for i in 1:num_pars
            data = getindex.(plot_data[m]["est"][1],i)
    		if i == 1
    			boxplot!(p, [string(i)], log10.(data), color=method_color[m], label=m)
    		else
    			boxplot!(p, [string(i)], log10.(data), color=method_color[m], label="")
    		end
    	end
        savefig(p,joinpath(path,replace("par_$(m)_$(sam)_$(var)","."=>"")*".pdf"))
    	push!(p_arr,p)
    end

    p = plot(p_arr...,layout=(length(methods),1))
    savefig(p,joinpath(path,replace("par_all_$(sam)_$(var)","."=>"")*".pdf"))
end

"""
Success Rate vs Time Plots
xaxis: Mean Computation Time
yaxis: 1 / Success Rate
"""
function sr_plots(plot_data::Dict,
                    noise_level::AbstractArray{Symbol},
                    methods::Vector{Symbol},
                    method_label::Dict,
                    method_color::Dict,
                    sam::Symbol,
                    path::String)::Nothing

    p = scatter(xlabel="Time", ylabel="1 / Success Rate", legend=:outertopright)
    p2 = scatter(xlabel="Time", ylabel="1 / Success Rate", legend=:outertopright)
    ylim_arr = []
    for m in methods
        p3 = scatter(xlabel="Time", ylabel="1 / Success Rate", legend=:outertopright)

        sr = mean.([step_success_rate.(e) for e in plot_data[m]["error"]])
        isr = 1.0./sr
        timed = plot_data[m]["time"]
        if !any(isnan.(vcat(timed...)))
            qtime = hcat(box_scatter_plot.(timed)...)
            qqtime = box_scatter_plot(qtime[2,:])
            qisr = box_scatter_plot(isr)
            if qisr[2] < 10.0
                scatter!(p, (qqtime[2],qisr[2]),
                            xerror=[(qqtime[1],qqtime[3])],
                            yerror=[(qisr[1],qisr[3])],
                            label=method_label[m], color=method_color[m])
            end

            scatter!(p2, (qtime[2,:], isr),
                        label=method_label[m], color=method_color[m])

            scatter!(p3, (qtime[2,:], isr),
                        label=method_label[m], color=method_color[m],
                        series_annotations = text.(noise_level, :top, 11))

            #display(p3)

            savefig(p3,path*"/sr_$(m)_$(sam).pdf")
        end
    end
    #display(p)
    #display(p2)

    savefig(p, path*"/sr_all_medians_$(sam).pdf")
    savefig(p2, path*"/sr_all_$(sam).pdf")
    nothing
end
function sr_plots(plot_data::Dict,
                    var::Symbol,
                    methods::Vector{Symbol},
                    method_label::Dict,
                    method_color::Dict,
                    sam::Symbol,
                    path::String)::Nothing

    p = scatter(xlabel="Time", ylabel="1 / Success Rate", legend=:outertopright)
    ylim_arr = []
    for m in methods
        sr = mean([step_success_rate(e) for e in plot_data[m]["error"][1]])
        isr = [1.0./sr]
        timed = plot_data[m]["time"][1]

        if !any(isnan.(vcat(timed...)))
            qtime = box_scatter_plot(timed)
            qisr = box_scatter_plot(isr)
            if qisr[2] < 20.0 && qtime[2] < 20.0
                scatter!(p, (qtime[2],qisr[2]),
                            xerror=[(qtime[1],qtime[3])],
                            yerror=[(qisr[1],qisr[3])],
                            label=method_label[m], color=method_color[m])
            end
        end
    end

    #display(p)
    savefig(p,joinpath(path,replace("sr_all_medians_$(sam)_$(var)","."=>"")*".pdf"))
    nothing
end


function plot_compare(data::Vector, data_est::Vector)
    alphabet='A':'Z'
    label_n=reshape(["$i (Nominal)" for i in alphabet[1:length(data[1])]],(1,length(data[1])))
    label_e=reshape(["$i (Estimated)" for i in alphabet[1:length(data[1])]],(1,length(data[1])))
    err = sqrt(nmse(reduce(vcat,data), reduce(vcat,data_est)))
    p = plot(title = "RMSE = $(err)")
    plot!(p, reduce(hcat,data)', label=label_n, markershape=:circle, linestyle=:solid)
    plot!(p, reduce(hcat,data_est)', label=label_e, markershape=:cross, linestyle=:dash)
    display(p)
    return p
end

"""
Overall Efficiency (OE) Plots
xaxis: Method
yaxis: OE Score
"""
function oe_plots(plot_data::Dict,
                    noise_level::AbstractArray{<:AbstractFloat},
                    methods::Array{<:String,1},
                    method_label::Dict,
                    method_color::Dict,
                    sam::Int,
                    path::String)::Nothing

    t_succ = zeros(length(methods))
    @inbounds for i in 1:length(methods)
        m = methods[i]
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
    bar!(p, methods, oe,
            color=[method_color[m] for m in methods])
    #display(p)

    savefig(p, path*"/$(sam)_oe.pdf")
    nothing
end

"""
Load serialized results and save the
most relevant plots
"""
function plot_main_results(results::Dict,
                            path::String)::Nothing

    first_key = collect(keys(results))[1]
    first_result = results[first_key]
    samples = collect(keys(first_result))
    noise_levels = Array{Symbol}(collect(keys(first_result[samples[1]])))
    #methods = Array{Symbol}(collect(keys(first_result[samples[1]][noise_levels[1]])))
    methods = [:SS,:DSS]

    cur_colors = get_color_palette(:lighttest, plot_color(:white), 10)

    method_label = Dict()
    for m in methods
    	method_label[m] = String(m)
    end
    method_label

    method_color = Dict()
    for (m,c) in zip([:DS,:SS,:DSS],cur_colors)
    	method_color[m] = c
    end

    for (method,result) in results
        dir_path = joinpath(path,string(method))
    	mkdir(dir_path)

        for sam in samples
            res = result[sam]
        	plot_data = get_plot_data(res,noise_levels,methods)
        	sr_plots(plot_data,noise_levels,methods,
                    method_label,method_color,sam,dir_path)

        	for noise in noise_levels
        		plot_data = get_plot_data(res,[noise],methods)

        		box_error_plots(plot_data,noise,methods,
                                method_label,method_color,sam,dir_path)

        		parameter_plots(plot_data,noise,methods,
                                method_label,method_color,sam,dir_path)

        		sr_plots(plot_data,noise,methods,
                        method_label,method_color,sam,dir_path)
        	end
        end
    end

end

end
