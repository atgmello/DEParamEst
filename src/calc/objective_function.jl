module ObjectiveFunction

using DifferentialEquations
using LinearAlgebra

export data_shooting, single_shooting, tikhonov

"""
Tikhonov Regularization
"""
function tikhonov(lambda::T, phi::Vector{T}, phi_ref::Vector{T},
                  w::Vector{T})::T where T
    res = zero(T)
    @simd for i in 1:length(phi)
        @inbounds res += abs2((phi[i]-phi_ref[i])*w[i])
    end
    res *= lambda
    return res
end
function tikhonov(lambda::T, phi::Vector{D}, phi_ref::Vector{T},
                  w::Vector{T})::D where T where D
    res = zero(D)
    @simd for i in 1:length(phi)
        @inbounds res += abs2((phi[i]-phi_ref[i])*w[i])
    end
    res *= lambda
    return res
end

"""
Sum of Squared Errors
"""
function sse(a::Vector{Vector{T}},b::Vector{Vector{T}})::T where T<:AbstractFloat
    sum = zero(T)

    if length(a) != length(b) || length(a[1]) != length(b[1])
        return Inf64
    end

    @simd for i in 1:length(a)
        for j in 1:length(a[1])
            @inbounds sum += abs2(a[i][j]-b[i][j])
        end
    end
    return sum
end
function sse(a::Vector,b::Vector)
    T = promote_type(eltype(a[1]),eltype(b[1]))
    sum = zero(T)

    if length(a) != length(b) || length(a[1]) != length(b[1])
        return Inf64
    end

    for i in 1:length(a)
        for j in 1:length(a[1])
            @inbounds sum += abs2(a[i][j]-b[i][j])
        end
    end
    return sum
end

"""
phi: Parameters to be optimized. Can be comprized of rate constants.
data: Observed data, used for calculating the residuals.
time_array: Time intervals in which the data has been colected.
f: DE that describes the phenomena.
"""
function data_shooting(phi::Vector{T},
                        data::Vector,
                        time_array::AbstractArray,
                        f::Function)::T where T

    num_samples = length(data)
    num_state_vars = length(data[1])

    estimated = zeros(T,num_state_vars)
    sum_error = zero(T)
    f_0 = zeros(T,num_state_vars)
    f_1 = zeros(T,num_state_vars)
    delta_t = zero(eltype(time_array))

    @inbounds for i in 1:num_samples-1
        delta_t = time_array[i+1] - time_array[i]

        x_0 = data[i]
        x_1 = data[i+1]

        f(f_0, x_0, phi, 0)
        f(f_1, x_1, phi, 0)

        for j in 1:num_state_vars
            estimated[j] = x_0[j] + 0.5*delta_t*(f_0[j]+f_1[j])
            sum_error += abs2(estimated[j]-data[i+1][j])
        end
    end
    return sum_error
end

function single_shooting(phi::Vector{T},
                        data::Vector,
                        t::AbstractArray,
                        f::Function)::T where T
    ini_cond = convert.(eltype(phi),data[1])
    tspan = (t[1], t[end])

    prob = ODEProblem(f, ini_cond, tspan, phi,
                      isoutofdomain=(u,p,t) -> any(x -> (x < 0), p))
    osol  = solve(prob, saveat=t)

    return sse(data,osol.u)
end

"""
params: Parameters to be optimized. Can be comprized of rate constants and initial values.
data: Observed data, used for calculating the residuals.
time_array: Time intervals in which the data has been colected.
f: DE that describes the phenomena.
"""
function data_shooting_exp(params::Array{<:AbstractFloat,1},
                        data::Array{<:AbstractFloat},
                        time_array::StepRangeLen,
                        f::Function;
                        unknown_vars::Array{<:Float64,1}=[],
                        differential_vars::Array{<:Float64,1}=[])::Array{<:AbstractFloat,1}

    num_state_vars, num_samples = size(data)

    phi = params[1:length(params)-length(unknown_vars)]

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
        partial_data = Array{Float64}(undef,0,length(time_array))
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

    @inbounds for i in 1:num_samples-1
        delta_t = time_array[i+1] - time_array[i]

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


function single_shooting_exp(params::Array{<:AbstractFloat},
                        data::Array{<:AbstractFloat},
                        t::StepRangeLen,
                        f::Function;
                        unknown_vars::Array{<:AbstractFloat} = [],
                        return_estimated::Bool=false)::Array{<:AbstractFloat}

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

    if return_estimated
        return estimated
    end

    # TODO
    # Fix weighting

    weight = abs2(1/findmax(data)[1])
    residuals = weight .* (data-estimated)

    return residuals
end

function soft_l1(z::T)::T where T<:Number
    sz = (2 * ((1 + z)^0.5 - 1))
end

function huber(z::T)::T where T<:Number
    M = 1.35
    if z < M
        return abs2(z)
    else
        return 2.0*M*abs(z) - M^2
    end
end

end
