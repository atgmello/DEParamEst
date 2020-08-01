using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

 
using Logging
using Statistics
using Gadfly
using Cairo
using DataFrames
using DifferentialEquations
import Distributions: Normal, Uniform


struct IVP
    f::Function
    ini::Vector
    t::AbstractArray
    phi::Vector
    data::Vector
    IVP(f, ini, t, phi, data) = new(f, ini, t, phi, data)
    IVP(f, ini, t, phi) = new(f, ini, t, phi,
                              (solve_ivp(new(f, ini, t, phi, []))
                               |> transpose_vector))
end


function get_ivps()::Dict
    var_ini = 0.01
    ivps = Dict()

    function fic(dz_dt, z, phi, t)
        r_1 = phi[1]*z[1]
        r_2 = phi[2]*z[2]

        dz_dt[1] = -r_1
        dz_dt[2] = r_1 - r_2
    end

    ivps[:fic] = Dict()
    ini = [1., 0.]
    t = range(0.0, 1.0, length=10)
    phi = [5.0035, 1.0]

    ivps[:fic][:nom] = IVP(fic, ini, t, phi)
    ivps_test = IVP(fic, add_noise(ini, var_ini), t, phi)
    ivps[:fic][:test] = IVP(fic, ivps_test.ini, t, phi,
                            add_noise(ivps_test.data, 0.1))

    ivps[:fic][:ss] = IVP(fic, ini, t,
                          [5.011535232847903,
                           0.9490955193480339], [])
    ivps[:fic][:ssr] = IVP(fic, ini, t,
                           [5.662374165061859,
                            4.600853424625171], [])
    ivps[:fic][:ds] = IVP(fic, ini, t,
                          [4.767811504376019,
                           0.9620285702765027], [])
    ivps[:fic][:dss] = IVP(fic, ini, t,
                           [5.057700471184158,
                            1.1660869766794006], [])



    function ccgo(dz_dt, z, phi, t)
        r_1 = phi[1]*z[1]^2
        r_2 = phi[2]*z[2]
        r_3 = phi[3]*z[1]^2

        dz_dt[1] = - r_1 - r_3
        dz_dt[2] = r_1 - r_2
    end

    ivps[:ccgo] = Dict()
    ini = [1., 0.]
    t = range(0.0, 0.95, length=10)
    phi = [12.214, 7.9798, 2.2216]

    ivps[:ccgo][:nom] = IVP(ccgo, ini, t, phi)
    ivps_test = IVP(ccgo, add_noise(ini, var_ini), t, phi)
    ivps[:ccgo][:test] = IVP(ccgo, ivps_test.ini, t, phi,
                            add_noise(ivps_test.data, 0.1))

    ivps[:ccgo][:ss] = IVP(ccgo, ini, t,
                           [10.601173576959617,
                            7.024603700551421,
                            5.296705518161371], [])
    ivps[:ccgo][:ssr] = IVP(ccgo, ini, t,
                            [12.694091102822359,
                             7.7461038473135275,
                             2.3847262326540015], [])
    ivps[:ccgo][:ds] = IVP(ccgo, ini, t,
                           [3.261725472047141,
                            9.713554040149868,
                            15.845961062140784], [])
    ivps[:ccgo][:dss] = IVP(ccgo, ini, t,
                            [11.701892961901253,
                             7.718104344633692,
                             2.748291378149207], [])


    function fhn(dz_dt, z, phi, t)
        a, b, c = phi
        V, R = z
        u = 0
        dz_dt[1] = c*(V - (V^3)/3 + R) + u
        dz_dt[2] = -(1/c)*(V - a + b*R)
    end

    ivps[:fhn] = Dict()
    ini = [-1.0, 1.0]
    t = range(0.0, 20.0, length=10)
    phi = [0.2, 0.2, 3.0]

    ivps[:fhn][:nom] = IVP(fhn, ini, t, phi)
    ivps_test = IVP(fhn, add_noise(ini, var_ini, false), t, phi)
    ivps[:fhn][:test] = IVP(fhn, ivps_test.ini, t, phi,
                            add_noise(ivps_test.data, 0.01, false))

    ivps[:fhn][:ss] = IVP(fhn, ini, t,
                          [1.0006533637635783e-5,
                           36762.161906015695,
                           29720.269687392283], [])
    ivps[:fhn][:ssr] = IVP(fhn, ini, t,
                           [0.1973538954650349,
                            0.11390293281470151,
                            3.0111993244825737], [])
    ivps[:fhn][:ds] = IVP(fhn, ini, t,
                          [0.06942005151231272,
                           0.4361123400524667,
                           1.696854260779007], [])
    ivps[:fhn][:dss] = IVP(fhn, ini, t,
                           [0.17695871191283072,
                            0.18851523532377604,
                            3.0158958746888795], [])
    return ivps
end


function transpose_vector(v::Vector{Vector{T}})::Vector{Vector{T}} where T
    [[v[i][j] for i in 1:length(v)] for j in 1:length(v[1])]
end


function solve_ivp(ivp::IVP)::Vector
    prob = ODEProblem(ivp.f, ivp.ini,
                      (ivp.t[1], ivp.t[end]), ivp.phi)
    osol  = solve(prob, AutoVern7(Rodas5()), saveat=ivp.t)
    return osol.u
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
    @inbounds for i in 2:length(noise_data)
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


"""
Normalized Mean Square Error
"""
function nmse(data::Vector{T}, data_est::Vector{T})::T where T
    normalizer = abs2(maximum(data) - minimum(data))
    res = mean(abs2.(data_est-data))/normalizer
    return res
end

"""
Normalized Root Mean Square Error
"""
function nrmse(data_nominal::Vector, data_estimated::Vector)::Float64
    state_pairs = zip(data_nominal, data_estimated)
    total_nmse = map(pair -> nmse(pair...),
                     state_pairs)
    # (mean) Normalized Root Mean Squared Error
    nrmse = mean(total_nmse) |> mean |> sqrt
    return nrmse
end


function gen_plt_err(ivp_comp::IVP,
                     ivp_est::IVP)::NamedTuple

    # Generate estimated values using ivp_comp.ini as initial point
    ivp_est = IVP(ivp_est.f, ivp_comp.ini, ivp_est.t, ivp_est.phi)

    df_est = map(data -> DataFrame(x=ivp_est.t,
                                   y=data,
                                   Legend=repeat(["Estimated"],
                                                 length(ivp_est.t))),
                 ivp_est.data)

    df_comp = map(data -> DataFrame(x=ivp_comp.t,
                                   y=data,
                                   Legend=repeat(["Data"],
                                                 length(ivp_comp.t))),
                 ivp_comp.data)

    df_pairs = zip(df_comp, df_est)

    gen_layer(df) = layer(vcat(df...), x=:x, y=:y,
                          color=:Legend,
                          linestyle=:Legend,
                          Geom.line, Geom.point)

    layers = map(gen_layer, df_pairs)

    gen_plot(layers) = plot((layers...),
                            Guide.XLabel("t"),
                            Guide.YLabel("x"))
    plt = gen_plot(layers)

    nrmse_result = nrmse(ivp_comp.data, ivp_est.data)

    return (plt=plt, err=nrmse_result)
end


function get_res_from_dict(ivps::Dict)::Vector{NamedTuple}
    # Unpack first Dictionary layer into Vector of NamedTuple
    isest(x) = x != :nom && x != :test

    est_ivps = map(k -> (ivp=ivps[k], method=string(k)),
                   (keys(ivps) |> collect |> x -> filter(isest, x)))

    nom_test_ivps = map(k -> (ivp=ivps[k], method=string(k)),
                        (keys(ivps) |> collect |> x -> filter(!isest, x)))

    compare_pairs = [(i, j) for i in nom_test_ivps for j in est_ivps]

    results_list = map(x -> (gen_plt_err(x[1].ivp, x[2].ivp)
                             |> res -> (plt=res.plt,
                                        err=res.err,
                                        method=x[2].method,
                                        plt_type=x[1].method)),
                             compare_pairs)
    return results_list
end


function save_results(results::NamedTuple, path::AbstractString)::String
    base_path = joinpath(path, results.model)

    # Save plots
    map(x ->
        draw(PDF(joinpath(base_path, "$(x.plt_type)_$(x.method).pdf")), x.plt),
        results.res)

    # Save info
    io = joinpath(path, "info.log") |> f -> open(f, "a+")
    with_logger(SimpleLogger(io)) do
        map(x ->
            @info("NRMSE", x.plt_type, results.model, x.method, x.err),
            results.res)
    end
    close(io)
    return base_path
end


function main(args::Array{<:String})::Nothing
    path = string(args[1])

    str_mkdir(x) = (x
                    |> string
                    |> y -> joinpath(path, y)
                    |> z -> (isdir(z) ?
                             z : mkdir(z)))

    ivps = get_ivps()

    folders = map(str_mkdir,
                  (keys(ivps) |> collect))

    res_list = map(k -> (res=get_res_from_dict(ivps[k]),
                         model=string(k)),
                   (keys(ivps) |> collect))

    saved_files = map(r -> save_results(r, path), res_list)

    with_logger(ConsoleLogger()) do
        @debug res_list
        @debug saved_files
    end
    nothing
end

main(ARGS)
