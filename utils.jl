module Utils

import Distributions: Uniform
using Random

struct ErrorTimeData
    error::AbstractArray
    time::AbstractArray
end

function rand_guess(bounds)
    [rand(Uniform(bounds[1][i], bounds[end][i]))
        for i in 1:length(bounds[1])]
end

function add_noise!(data, var)
    if length(size(data)) == 2
        m,n = size(data)
        for i in 1:m
            for j in 1:n
                if data[i,j] != 0.
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
        end
    elseif length(size(data)) == 1
        m = length(data)
        for i in 1:m
            if data[i] != 0.
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
    end
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

function diff_states(f, u0::AbstractArray, tspan::Tuple, actual::AbstractArray, forecast::AbstractArray; var=1.5)
    dist = 0.0
    t = range(tspan[1], stop=tspan[end], length=1000)
    runs = 50
    for i in 1:runs
        u0_perturb = copy(u0)
        add_noise!(u0_perturb, var)
        for j in 1:length(u0_perturb)
            if u0_perturb[j] < 0
                u0_perturb[j] = 0
            end
        end
        de_prob_actual = ODEProblem(f,u0_perturb,tspan,actual)
        de_sol_actual = DifferentialEquations.solve(de_prob_actual, AutoVern9(Rodas5()), saveat=t)
        data_actual = reduce(hcat, de_sol_actual.u)

        de_prob_forecast = ODEProblem(f,u0_perturb,tspan,forecast)
        de_sol_forecast = solve(de_prob_forecast, AutoVern9(Rodas5()), saveat=t)
        data_forecast = reduce(hcat, de_sol_forecast.u)

        dist += euclidean(data_actual, data_forecast)/(size(data_actual)[1]*size(data_actual)[2])
    end
    dist /= runs
end

#=>
a = [1. 1. 1.]
b = [2. 2. 2.]
c = [100. 10. 33.]
mape(b,a)
mape(a,b)
mape(a,c)
smape(b,a)
smape(a,b)
smape(a,c)
fae(a,b)
fae(b,a)
mrae(a,b)
mrae(b,a)
mrae(a,c)
mrae(c,a)
euclidean(a,b)
euclidean(a,c)
<=#

end
