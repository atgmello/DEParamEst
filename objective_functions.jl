function adams_moulton_estimator(phi, data, time_array, ode_fun; plot_estimated=false, return_estimated=false)
    num_state_variables, num_samples = size(data)

    estimated = zeros(promote_type(eltype(phi),eltype(data)), num_samples*num_state_variables)
    estimated = reshape(estimated, (num_state_variables, num_samples))
    estimated[:, 1] = data[:,1] #Initial conditions are stored at x_dot_num's first column

    for i in range(1, stop=num_samples-1)
        delta_t = time_array[i+1] - time_array[i]
        x_k_0 = data[:, i]
        x_k_1 = data[:, i+1]

        f_eval_0 = zeros(promote_type(eltype(phi),eltype(data)), num_state_variables)
        ode_fun(f_eval_0, x_k_0, phi, 0)
        f_eval_1 = zeros(promote_type(eltype(phi),eltype(data)), num_state_variables)
        ode_fun(f_eval_1, x_k_1, phi, 0)

        x_k_1_est = x_k_0 + (1/2)*delta_t*(f_eval_0+f_eval_1)
        estimated[:, i+1] = x_k_1_est
    end

    if plot_estimated
        p_data = scatter(transpose(estimated))
        scatter!(p_data, transpose(data))
        display(p_data)
        println("Plot for\n$phi\n")
    end

    weight = abs2(1/findmax(data)[1])
    residuals = weight .* (data-estimated)
    #=>
    if findmin(estimated)[1] < 0
        for i in 1:length(residuals)
            residuals[i] = 10^10
        end
    end
    <=#
    if return_estimated
        return estimated
    end
    return residuals
end

function single_multiple_adams_shooting(phi, data, time_array, ode_fun; plot_estimated=false, return_estimated=false)
    partial_estimate = single_shooting_estimator(phi, data, time_array, ode_fun; return_estimated=true)
    num_state_variables, num_samples = size(partial_estimate)

    estimated = zeros(promote_type(eltype(phi),eltype(partial_estimate)), num_samples*num_state_variables)
    estimated = reshape(estimated, (num_state_variables, num_samples))
    estimated[:, 1] = partial_estimate[:,1] #Initial conditions are stored at x_dot_num's first column

    for i in range(1, stop=num_samples-1)
        delta_t = time_array[i+1] - time_array[i]
        x_k_0 = partial_estimate[:, i]
        x_k_1 = partial_estimate[:, i+1]

        f_eval_0 = zeros(promote_type(eltype(phi),eltype(partial_estimate)), num_state_variables)
        ode_fun(f_eval_0, x_k_0, phi, 0)
        f_eval_1 = zeros(promote_type(eltype(phi),eltype(partial_estimate)), num_state_variables)
        ode_fun(f_eval_1, x_k_1, phi, 0)

        x_k_1_est = x_k_0 + (1/2)*delta_t*(f_eval_0+f_eval_1)
        estimated[:, i+1] = x_k_1_est
    end

    if plot_estimated
        p_data = scatter(transpose(estimated))
        scatter!(p_data, transpose(data))
        display(p_data)
        println("Plot for\n$phi\n")
    end

    weight = abs2(1/findmax(data)[1])
    residuals = weight .* (data-estimated)
    #=>
    if findmin(estimated)[1] < 0
        for i in 1:length(residuals)
            residuals[i] = 10^10
        end
    end
    <=#
    if return_estimated
        return estimated
    end
    return residuals
end

function sm_mean_shooting(phi, data, time_array, ode_fun; plot_estimated=false, return_estimated=false)
    partial_estimate = single_shooting_estimator(phi, data, time_array, ode_fun; return_estimated=true)
    partial_estimate = (partial_estimate+data)*(1/2)
    num_state_variables, num_samples = size(partial_estimate)

    estimated = zeros(promote_type(eltype(phi),eltype(partial_estimate)), num_samples*num_state_variables)
    estimated = reshape(estimated, (num_state_variables, num_samples))
    estimated[:, 1] = partial_estimate[:,1] #Initial conditions are stored at x_dot_num's first column

    for i in range(1, stop=num_samples-1)
        delta_t = time_array[i+1] - time_array[i]
        x_k_0 = partial_estimate[:, i]
        x_k_1 = partial_estimate[:, i+1]

        f_eval_0 = zeros(promote_type(eltype(phi),eltype(partial_estimate)), num_state_variables)
        ode_fun(f_eval_0, x_k_0, phi, 0)
        f_eval_1 = zeros(promote_type(eltype(phi),eltype(partial_estimate)), num_state_variables)
        ode_fun(f_eval_1, x_k_1, phi, 0)

        x_k_1_est = x_k_0 + (1/2)*delta_t*(f_eval_0+f_eval_1)
        estimated[:, i+1] = x_k_1_est
    end

    if plot_estimated
        p_data = scatter(transpose(estimated))
        scatter!(p_data, transpose(data))
        display(p_data)
        println("Plot for\n$phi\n")
    end

    weight = abs2(1/findmax(data)[1])
    residuals = weight .* (data-estimated)
    #=>
    if findmin(estimated)[1] < 0
        for i in 1:length(residuals)
            residuals[i] = 10^10
        end
    end
    <=#
    if return_estimated
        return estimated
    end
    return residuals
end

function adams_moulton_fourth_estimator(phi, data, time_array, ode_fun; plot_estimated=false, return_estimated=false)
    num_state_variables, num_samples = size(data)

    estimated = zeros(promote_type(eltype(phi),eltype(data)), num_samples*num_state_variables)
    estimated = reshape(estimated, (num_state_variables, num_samples))
    estimated[:, 1] = data[:,1] #Initial conditions are stored at x_dot_num's first column
    estimated[:, 2] = data[:,2] #Initial conditions are stored at x_dot_num's first column
    estimated[:, 3] = data[:,3] #Initial conditions are stored at x_dot_num's first column
    estimated[:, 4] = data[:,4] #Initial conditions are stored at x_dot_num's first column

    for i in range(1, stop=num_samples-3)
        delta_t = []
        push!(delta_t, time_array[i+1] - time_array[i])
        push!(delta_t, time_array[i+2] - time_array[i+1])
        push!(delta_t, time_array[i+3] - time_array[i+2])
        push!(delta_t, time_array[i+1] - time_array[i])

        x_k = []
        push!(x_k, data[:, i])
        push!(x_k, data[:, i+1])
        push!(x_k, data[:, i+2])
        push!(x_k, data[:, i+3])

        f_eval_1 = zeros(promote_type(eltype(phi),eltype(data)), num_state_variables)
        ode_fun(f_eval_1, x_k[1], phi, 0)
        f_eval_2 = zeros(promote_type(eltype(phi),eltype(data)), num_state_variables)
        ode_fun(f_eval_2, x_k[2], phi, 0)
        f_eval_3 = zeros(promote_type(eltype(phi),eltype(data)), num_state_variables)
        ode_fun(f_eval_3, x_k[3], phi, 0)
        f_eval_4 = zeros(promote_type(eltype(phi),eltype(data)), num_state_variables)
        ode_fun(f_eval_4, x_k[4], phi, 0)

        x_k_est = x_k[3] + (delta_t[1]*(9/24)*f_eval_4+delta_t[1]*(19/24)*f_eval_3-delta_t[1]*(5/24)f_eval_2+delta_t[1]*(1/24)*f_eval_1)
        estimated[:, i+3] = x_k_est
    end

    if plot_estimated
        p_data = scatter(transpose(estimated))
        scatter!(p_data, transpose(data))
        display(p_data)
        println("Plot for\n$phi\n")
    end

    weight = abs2(1/findmax(data)[1])
    residuals = weight .* (data-estimated)
    #=>
    if findmin(estimated)[1] < 0
        for i in 1:length(residuals)
            residuals[i] = 10^10
        end
    end
    <=#
    if return_estimated
        return estimated
    end
    return residuals
end


function data_shooting_estimator(phi, data, t, ode_fun; steps=1, plot_estimated=false)
    num_state_variables, num_samples = size(data)

    estimated = zeros(promote_type(eltype(phi),eltype(data)), num_samples*num_state_variables)
    estimated = reshape(estimated, (num_state_variables, num_samples))
    estimated[:, 1] = data[:,1] #Initial conditions are stored at x_dot_num's first column

    for i in range(1, stop=num_samples-1)
        t_1 = t[i+1]
        t_0 = t[i]
        delta_t = t_1 - t_0

        x_k_0 = data[:, i]

        for i in 1:steps
            tspan = (t_0, t_1)
            oprob = ODEProblem(ode_fun, x_k_0, tspan, phi)
            osol  = solve(oprob, lsoda(), saveat=reduce(vcat, tspan))

            x_k_1 = x_k_0 + delta_t*(osol.u[end])
            x_k_0 = x_k_1
        end
        estimated[:, i+1] = x_k_1
    end

    if plot_estimated
        p = scatter(transpose(estimated))
        display(p)
        println("Plot for\n$phi\n")
    end

    residuals = (data-estimated)
    return residuals
end

function data_shooting_estimator_node(phi, data, t, ode_fun; steps=1)
    num_state_variables, num_samples  = size(data)

    estimated = zeros(promote_type(eltype(data),eltype(phi)), num_samples*num_state_variables)
    estimated = reshape(estimated, (num_state_variables, num_samples))
    estimated[:,1] = data[:,1]

    for i in 1:num_samples-1
        t_1 = t[i+1]
        t_0 = t[i]
        delta_t = (t_1 - t_0)/steps

        x_k_0 = data[:, i]

        x_k_1 = 0
        for j in 1:steps
            f_eval = zeros(promote_type(eltype(data),eltype(phi)), num_state_variables)
            ode_fun(f_eval, x_k_0, phi, 0)
            x_k_1 = x_k_0 + delta_t.*f_eval
            x_k_0 = x_k_1
        end
        estimated[:, i+1] = x_k_1
    end
    estimated
end

function single_shooting_estimator(phi, data, t, ode_fun; plot_estimated=false, return_estimated=false)
    tspan = (t[1], t[end])
    ini_cond = data[:,1]
    oprob = ODEProblem(ode_fun, ini_cond, tspan, phi)
    osol  = solve(oprob, AutoTsit5(Rosenbrock23()), saveat=t)
    estimated = reduce(hcat, osol.u)

    if plot_estimated
        p = scatter(transpose(estimated))
        display(p)
        println("Plot for\n$phi\n")
    end

    if return_estimated
        return estimated
    end
    weight = abs2(1/findmax(data)[1])
    residuals = weight .* (data-estimated)
    return  residuals
end

soft_l1(z) = (2 * ((1 + z)^0.5 - 1))
