module Utils

import Distributions: Uniform
import Statistics: quantile
using Random
using Statistics
using Distances
using DifferentialEquations
using Plots

"""
From bounds, generates a random startign point for
the optimization procedure.
"""
function rand_guess(bounds::Array{<:AbstractFloat})::Array{<:AbstractFloat,1}
    vcat([rand(Uniform(bounds[i,1], bounds[i,2]))
            for i in 1:size(bounds)[1]])
end

function add_noise(data::Array, variance::Float64)::Array
    noise_data = copy(data)
    m = length(noise_data)
    @inbounds for i in 1:m
        if noise_data[i] == 0.
            noise_data[i] = mean(noise_data)+10^(-3)
        end
        if -variance*noise_data[i] < variance*noise_data[i]
            a = -variance*noise_data[i]
            b = variance*noise_data[i]
        else
            b = -variance*noise_data[i]
            a = variance*noise_data[i]
        end
        noise_data[i] += rand(Uniform(a, b))
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

function diff_calc(f::Function, u0::Array, t::AbstractArray,
                    actual::Array{<:AbstractFloat,1}, forecast::Array{<:AbstractFloat,1})::Float64
    de_prob_actual = ODEProblem(f,u0,(t[1],t[end]),actual)
    de_sol_actual = DifferentialEquations.solve(de_prob_actual, AutoTsit5(Rosenbrock23()), saveat=t)
    data_actual = reduce(hcat, de_sol_actual.u)

    de_prob_forecast = ODEProblem(f,u0,(t[1],t[end]),forecast)
    de_sol_forecast = DifferentialEquations.solve(de_prob_forecast, AutoTsit5(Rosenbrock23()), saveat=t)
    data_forecast = reduce(hcat, de_sol_forecast.u)

    """
    Absolute Log Difference
    Since the data is required to be positive,
    find the minimum value from both actual and
    forecast data and lift them up to avoid
    zeros and negative numbers.
    """

    min_val = minimum(vcat(data_actual,data_forecast))
    if min_val < 0.0
        data_actual .= data_actual .- (min_val-0.1)
        data_forecast .= data_forecast .- (min_val-0.1)
    end

    ldiff = maximum(abs.(log.(data_actual)-log.(data_forecast)))

    #=>
    p = plot(title="$(round(ldiff,digits=3))")
    plot!(p,t,data_actual')
    plot!(p,t,data_forecast',linestyle=:dashdot)
    display(p)
    sleep(2)
    <=#

    return ldiff
end

function max_diff_states(f::Function, u0::Array, tspan::Tuple,
                    actual::Array{<:AbstractFloat,1}, forecast::Array{<:AbstractFloat,1},
                    variance::Float64)::Float64
    t = range(tspan[1], stop=tspan[end], length=1000)
    reps = 5
    #u0_arr = collect(Map(x -> abs.(add_noise(x,variance))),eduction(u0 for _ in 1:reps))
    u0_arr = [abs.(add_noise(u0,variance)) for _ in 1:reps]

    max_error = maximum([diff_calc(f,x,t,actual,forecast) for x in u0_arr])

    return max_error
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

function error_plots(plot_data::Dict, vars::AbstractArray{<:AbstractFloat},
                    method_arr::Array,
                    method_label::Dict,
                    method_color::Dict,
                    sam::Int)::Nothing
    """
    Error Plots
    xaxis: Noise Percentage
    yaxis: Mean Error
    """
    p = plot(x=vars, xlabel="Noise Percentage", ylabel="Median Error")
    ylim_arr = []
    for m in method_arr
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

            savefig(p2,"./error_$(m)_$(sam).svg")
            savefig(p,"./error_inter_$(m)_$(sam).svg")
        end
    end
    if length(ylim_arr) > 1
        ylims!(p, (minimum(abs.([yl[1] for yl in ylim_arr])),
                    minimum(abs.([yl[2] for yl in ylim_arr]))))
    end
    display(p)

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
            if qisr[2] < 10.0 || qqtime[2] < 10
                scatter!(p, (qqtime[2],qisr[2]),
                            xerror=[(qqtime[1],qqtime[3])],
                            yerror=[(qisr[1],qisr[3])],
                            label=method_label[m], color=method_color[m])
            end
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

            savefig(p3,"./sr_$(m)_$(sam).svg")
        end
    end
    display(p)
    display(p2)

    savefig(p, "./sr_all_median$(sam).svg")
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
    display(p)

    savefig(p, "./$(sam)_oe.svg")
    nothing
end

end
