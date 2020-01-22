module ObjectiveFunctions

using LSODA
using DifferentialEquations
using Plots

export data_shooting, single_shooting

function data_shooting(params::AbstractArray,
                                data::Matrix,
                                time_array::AbstractArray,
                                f::Function;
                                unknown_vars=[],
                                differential_vars=[],
                                plot_estimated=false)
    """
    params: Parameters to be optimized. Can be comprized of rate constants and initial values.
    data: Observed data, used for calculating the residuals.
    time_array: Time intervals in which the data has been colected.
    f: DE that describes the phenomena.
    """

    num_state_vars, num_samples = size(data)

    phi = params[1:length(params)-length(unknown_vars)]

    data = convert(Array{eltype(phi)}, data)
    time_array = convert(Array{eltype(phi)}, time_array)

    # If the initial values of some states are not known
    if unknown_vars != []
        known_vars = collect(1:num_state_vars+length(unknown_vars))
        setdiff!(known_vars,unknown_vars)

        num_state_vars += length(unknown_vars)
        estimated = zeros(
                        promote_type(
                            eltype(phi),eltype(data)
                        ),
                        num_state_vars,num_samples
                    )

        # Initial conditions are stored at x_dot_num's first column
        i = 1
        for k in known_vars
            estimated[k,1] = data[i,1]
            i+=1
        end
        i = 1
        for k in unknown_vars
            estimated[k,1] = params[length(params)-length(unknown_vars)+i]
            i+=1
        end

        # Numerical method
        tspan = (time_array[1], time_array[end])

        ini_cond = estimated[:,1]

        oprob = ODEProblem(f, ini_cond, tspan, phi)
        osol  = solve(oprob, Rodas5(), saveat=time_array)
        partial_estimate = reduce(hcat, osol.u)

        # Now make an array comprised of all the states:
        # the ones calculated by the numerical method above
        # and the known ones, from the data
        partial_data = Matrix{Float64}(undef,0,length(time_array))
        i = 1
        j = 1

        state_vars = collect(1:num_state_vars)
        for k in state_vars
            if k in known_vars
                partial_data = vcat(partial_data,(data[i,:]'))
                i+=1
            else
                partial_data = vcat(partial_data,(partial_estimate[j,:]'))
                j+=1
            end
        end
        data = partial_data
        # TODO
        # Try to avoid so much array transpositions

    else
        estimated = zeros(
                        promote_type(
                            eltype(phi),eltype(data)
                        ),
                        num_state_vars,num_samples
                    )
        # Initial conditions are stored at x_dot_num's first column
        estimated[:,1] = data[:,1]
    end

    for i in 1:num_samples-1
        delta_t = time_array[i+1] - time_array[i]
        delta_t = convert(eltype(phi), delta_t)

        x_0 = data[:, i]
        x_1 = data[:, i+1]

        f_0 = zeros(num_state_vars)
        f(f_0, x_0, phi, 0)
        f_1 = zeros(num_state_vars)
        f(f_1, x_1, phi, 0)

        if differential_vars != []
            x_est = zeros(num_state_vars)
            x_est[differential_vars] .= x_0[differential_vars] +
                                            (1/2)*delta_t*(
                                                f_0[differential_vars]+
                                                f_1[differential_vars]
                                            )
            x_est[.~differential_vars] .= f_1[.~differential_vars]
        else
            x_est = x_0 + (1/2)*delta_t*(f_0+f_1)
        end

        estimated[:, i+1] = x_est
    end

    if plot_estimated
        p_data = plot(transpose(estimated), linewidth=2)
        plot!(p_data, transpose(data))
        display(p_data)
        println("Plot for\n$params\n")
    end

    states_to_filter = []
    if unknown_vars != []
        states_to_filter = map(x -> if x in unknown_vars false else true end, state_vars)
    elseif differential_vars != []
        states_to_filter = .~differential_vars
    end

    # TODO
    # Fix weighting
    if states_to_filter != []
        data = data[states_to_filter,:]
        estimated = estimated[states_to_filter,:]
        weight = abs2(1/findmax(data)[1])
        residuals = weight .* (data-estimated)
    else
        weight = abs2(1/findmax(data)[1])
        residuals = weight .* (data-estimated)
    end
    #=>
    if findmin(estimated)[1] < 0
        for i in 1:length(residuals)
            residuals[i] = 10^10
        end
    end
    <=#
    return residuals
end

function single_shooting(params::AbstractArray,
                                    data::AbstractArray,
                                    t::AbstractArray,
                                    f::Function;
                                    unknown_vars = [],
                                    plot_estimated=false, return_estimated=false)

    data = convert(Array{eltype(params)}, data)
    t = convert(Array{eltype(params)}, t)

    num_state_vars, num_samples = size(data)
    state_vars = collect(1:num_state_vars)
    # Check whether all state variables are known
    if unknown_vars == []
        known_vars = 1:num_state_vars
        # If there are no unknown states, then there are no initial condition
        # values to be optimized among the "params". This way, "params"
        # stores only to the rate constants to be found.
        ini_cond = data[:,1]
        phi = params
    else
        ini_cond = data[:,1]
        for (idx,val) in zip(unknown_vars,params)
            insert!(ini_cond, idx, val)
        end

        num_state_vars += length(unknown_vars)
        known_vars = collect(1:num_state_vars)
        setdiff!(known_vars,unknown_vars)
        phi = params[1:length(params)-length(unknown_vars)]
    end

    tspan = (t[1], t[end])
    oprob = ODEProblem(f, ini_cond, tspan, phi)
    osol  = solve(oprob, Rodas5(), saveat=t, save_idxs=known_vars)
    estimated = reduce(hcat, osol.u)

    if plot_estimated
        p_data = plot(transpose(estimated), linewidth=2)
        plot!(p_data, transpose(data))
        display(p_data)
        println("Plot for\n$params\n")
    end

    if return_estimated
        return estimated
    end

    # TODO
    # Fix weighting

    weight = abs2(1/findmax(data)[1])
    residuals = weight .* (data-estimated)

    return  residuals
end

function adams_moulton_estimator(phi::AbstractArray, data::AbstractArray, time_array::AbstractArray, f::Function; plot_estimated=false, return_estimated=false)
    num_state_vars, num_samples = size(data)
    data = convert(Array{eltype(phi)}, data)
    time_array = convert(Array{eltype(phi)}, time_array)

    estimated = zeros(promote_type(eltype(phi),eltype(data)), num_samples*num_state_vars)
    estimated = reshape(estimated, (num_state_vars, num_samples))
    estimated[:, 1] = data[:,1] #Initial conditions are stored at x_dot_num's first column

    for i in range(1, stop=num_samples-1)
        delta_t = time_array[i+1] - time_array[i]
        delta_t = convert(eltype(phi), delta_t)

        x_k_0 = data[:, i]
        x_k_1 = data[:, i+1]

        f_eval_0 = zeros(promote_type(eltype(phi),eltype(data)), num_state_vars)
        f(f_eval_0, x_k_0, phi, 0)
        f_eval_1 = zeros(promote_type(eltype(phi),eltype(data)), num_state_vars)
        f(f_eval_1, x_k_1, phi, 0)

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

function sm_mean_shooting(phi, data, time_array, f; plot_estimated=false, return_estimated=false)
    partial_estimate = single_shooting_estimator(phi, data, time_array, f; return_estimated=true)
    partial_estimate = (partial_estimate+data)*(1/2)
    num_state_vars, num_samples = size(partial_estimate)

    estimated = zeros(promote_type(eltype(phi),eltype(partial_estimate)), num_samples*num_state_vars)
    estimated = reshape(estimated, (num_state_vars, num_samples))
    estimated[:, 1] = partial_estimate[:,1] #Initial conditions are stored at x_dot_num's first column

    for i in range(1, stop=num_samples-1)
        delta_t = time_array[i+1] - time_array[i]
        x_k_0 = partial_estimate[:, i]
        x_k_1 = partial_estimate[:, i+1]

        f_eval_0 = zeros(promote_type(eltype(phi),eltype(partial_estimate)), num_state_vars)
        f(f_eval_0, x_k_0, phi, 0)
        f_eval_1 = zeros(promote_type(eltype(phi),eltype(partial_estimate)), num_state_vars)
        f(f_eval_1, x_k_1, phi, 0)

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

function adams_moulton_fourth_estimator(phi, data, time_array, f; plot_estimated=false, return_estimated=false)
    num_state_vars, num_samples = size(data)

    estimated = zeros(promote_type(eltype(phi),eltype(data)), num_samples*num_state_vars)
    estimated = reshape(estimated, (num_state_vars, num_samples))
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

        f_eval_1 = zeros(promote_type(eltype(phi),eltype(data)), num_state_vars)
        f(f_eval_1, x_k[1], phi, 0)
        f_eval_2 = zeros(promote_type(eltype(phi),eltype(data)), num_state_vars)
        f(f_eval_2, x_k[2], phi, 0)
        f_eval_3 = zeros(promote_type(eltype(phi),eltype(data)), num_state_vars)
        f(f_eval_3, x_k[3], phi, 0)
        f_eval_4 = zeros(promote_type(eltype(phi),eltype(data)), num_state_vars)
        f(f_eval_4, x_k[4], phi, 0)

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

soft_l1(z) = (2 * ((1 + z)^0.5 - 1))

end
