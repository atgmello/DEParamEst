module Utils

using Distributed
import Distributions: Uniform
import Statistics: quantile
using Random
using Statistics
using Distances
using DifferentialEquations
using Plots

function rand_guess(bounds)
    [rand(Uniform(bounds[1][i], bounds[end][i]))
        for i in 1:length(bounds[1])]
end

function add_noise(data, var)
    if length(size(data)) == 2
        m,n = size(data)
        for i in 1:m
            for j in 1:n
                if data[i,j] == 0.
                    data[i,j] = mean(data)+10^(-3)
                end
                if -var*data[i,j] < var*data[i,j]
                    a = -var*data[i,j]
                    b = var*data[i,j]
                else
                    b = -var*data[i,j]
                    a = var*data[i,j]
                end
                data[i,j] += rand(Uniform(a, b))
            end
        end
    elseif length(size(data)) == 1
        m = length(data)
        for i in 1:m
            if data[i] == 0.
                data[i] = mean(data)+10^(-3)
            end
            if -var*data[i] < var*data[i]
                a = -var*data[i]
                b = var*data[i]
            else
                b = -var*data[i]
                a = var*data[i]
            end
            data[i] += rand(Uniform(a, b))
        end
    end
    return data
end

function fae(actual::AbstractArray, forecast::AbstractArray)
    n = length(actual)
    frac = 1/n
    e = abs.(actual-forecast)
    normalize = abs.(actual)+abs.(forecast)
    stat = frac*sum(2*e/normalize)
end

function smape(actual::AbstractArray, forecast::AbstractArray)
    100*fae(actual,forecast)
end

function mape(actual::AbstractArray, forecast::AbstractArray)
    n = length(actual)
    frac = 1/n
    e = abs.(actual-forecast)
    normalize = abs.(actual)
    stat = 100*frac*sum(e/normalize)
end

function mrae(actual::AbstractArray, forecast::AbstractArray)
    n = length(actual)
    e = abs.(actual-forecast)
    stat = (1/n)*sum(e./abs.(actual))
end

function filter_outlier(arr; p=2)
    q = quantile(arr)
    iqr = q[3] - q[1]
    maxValue = q[3] + iqr * p
    minValue = q[1] - iqr * p

    return filter(x -> (x >= minValue) && (x <= maxValue), arr)
end

struct Trace
    """
    A Trace object is intended to hold
    multiple pairs of time and eval.
    Therefore, the attributes time and
    eval shall be array of arrays, that,
    when zipped, recreate the pairs
    mentioned above.
    """
    time::AbstractArray
    eval::AbstractArray
end

function make_trace(trace)
    time = []
    eval = []
    for i in 1:length(trace)
        append!(time, parse(Float64, split(string(trace[i]))[end]))
        append!(eval, parse(Float64, split(string(trace[i]))[2]))
    end
    return Trace(time, eval)
end

function fill_trace(t::Trace)
    """
    Given a trace, make sure that
    both time and eval arrays have the same size.
    This is done by repeating the last registered
    value in each array.
    """

    max_t_array_len = maximum(map(x -> length(x), t.time))
    resized_t = map( x -> append!(x, fill(x[end], max_t_array_len - length(x))), t.time)

    max_eval_array_len = maximum(map(x -> length(x), t.eval))
    resized_eval = map( x -> append!(x, fill(x[end], max_eval_array_len - length(x))), t.eval)

    return Trace(resized_t, resized_eval)
end

function scale_eval(t::Trace)
    scaled_eval = map(e -> e/e[end], t.eval)
    return Trace(t.time, scaled_eval)
end

function get_range_traces(trace::Trace)
    """
    Given a Trace
    select the best trace (the one with the least amount of
    time spent to get to the optimal solution) and the worst
    trace (the one with the most)
    """
    trace_times = map(t -> t[end], trace.time)
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

function success_rate(errors::AbstractArray)
    f = x -> (11)/(10+exp10(x))
    mean(map(x ->  begin
                        if x > 0.5
                            1
                        else
                            0
                        end
                    end,
                f.(errors)))
end

function box_data(arr::AbstractArray)
    q = quantile(arr,[0.25,0.5,0.75])
    iqr = q[3] - q[1]
    min_val = q[1] - 1.5*iqr
    max_val = q[3] + 1.5*iqr
    return [maximum([min_val,minimum(arr)]),
            q[1],q[2],q[3],
            minimum([max_val,maximum(arr)])]
end

function box_scatter_plot(arr::AbstractArray)
    qarr = box_data(arr)
    return [qarr[3]-qarr[1],qarr[3],qarr[5]-qarr[3]]
end

end
