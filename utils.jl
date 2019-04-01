function rand_guess(bounds)
    [rand(Uniform(bounds[1][i], bounds[end][i]))
        for i in 1:length(bounds[1])]
end

function add_noise!(data, var)
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
