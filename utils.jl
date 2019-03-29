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
