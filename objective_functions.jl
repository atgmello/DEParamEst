function adams_moulton_estimator(phi, data, time_array, ode_fun)
    num_state_variables, num_samples = size(data)

    estimated = zeros(num_samples*num_state_variables)
    estimated = reshape(estimated, (num_state_variables, num_samples))
    estimated[:, 1] = data[:,1] #Initial conditions are stored at x_dot_num's first column

    for i in range(1, stop=num_samples-1)
        delta_t = time_array[i+1] - time_array[i]
        x_k_0 = data[:, i]
        x_k_1 = data[:, i+1]

        f_eval_0 = zeros(num_state_variables)
        ode_fun(f_eval_0, x_k_0, phi, 0)
        f_eval_1 = zeros(num_state_variables)
        ode_fun(f_eval_1, x_k_0, phi, 0)

        x_k_1 = x_k_0 + (1/2)*delta_t*(f_eval_0+f_eval_1)
        estimated[:, i+1] = x_k_1
    end

    #=>
    println("Plot of $phi")
    p = scatter(transpose(estimated))
    display(p)
    <=#
    residuals = (data-estimated)
    return sum(residuals.^2)
end


function data_shooting_estimator(phi, data, t, ode_fun)
    num_state_variables, num_samples = size(data)

    estimated = zeros(num_samples*num_state_variables)
    estimated = reshape(estimated, (num_state_variables, num_samples))
    estimated[:, 1] = data[:,1] #Initial conditions are stored at x_dot_num's first column

    for i in range(1, stop=num_samples-1)
        t_1 = t[i+1]
        t_0 = t[i]
        delta_t = t_1 - t_0

        x_k_0 = data[:, i]

        tspan = (t_0, t_1)
        oprob = ODEProblem(ode_fun, x_k_0, tspan, phi)
        osol  = solve(oprob, Tsit5(), saveat=tspan)

        x_k_1 = x_k_0 + delta_t*(osol.u[end])
        estimated[:, i+1] = x_k_1
    end

    #=>
    println("Plot of $phi")
    p = scatter(transpose(estimated))
    display(p)
    <=#
    residuals = (data-estimated)
    return sum(residuals.^2)
end

function single_shooting_estimator(phi, data, t, ode_fun)
    tspan = (t[1], t[end])
    ini_cond = data[:,1]
    oprob = ODEProblem(ode_fun, ini_cond, tspan, phi)
    osol  = solve(oprob, Tsit5(), saveat=t)
    estimated = reduce(hcat, osol.u)

    #=>
    println("Plot of $phi")end
    p = scatter(transpose(estimated))
    display(p)
    <=#
    residuals = (data-estimated)
    return sum(residuals.^2)
end
