using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using Logging
using Gadfly
using DifferentialEquations
using ColorSchemes
using LaTeXStrings
import Plots
import Cairo,
      Fontconfig

include("../calc/problem_set.jl")
include("../calc/objective_function.jl")
include("./utils.jl")
import .ProblemSet: get_problem, DEProblem
import .ObjectiveFunction: data_shooting, single_shooting, tikhonov
import .PlottingUtils: heatmap, problem_plot, add_noise

const PATH = "./data/contours/"
const GADFLY_FONT = "Arial"

Plots.pyplot()
Plots.theme(:default)

function contour_3d_plots(x, y, z, par, save_path; method="", save_name="", cam=(-60,30))
    title = Dict("ss"=>"Single Shooting", "ds"=>"Data Shooting")

    if length(par) > 1
        cont = PlottingUtils.heatmap(x, y, vec(z'),
                                     xlabel="θ₁", ylabel="θ₂",
                                     zlabel="Objective\nFunction",
                                     title=title[method],
                                     fillcolor=ColorSchemes.vik)
    else
        cont = plot(x=x, y=vec(z), Geom.line, Guide.title(title[method]),
                    Gadfly.Coord.cartesian(yflip=false,
                                            fixed=false,
                                            xmin=minimum(x),
                                            xmax=maximum(x),
                                            ymin=minimum(vec(z)),
                                            ymax=maximum(vec(z))),
                    Gadfly.Theme(minor_label_font=GADFLY_FONT,
                                    major_label_font=GADFLY_FONT,
                                    key_title_font=GADFLY_FONT,
                                    key_label_font=GADFLY_FONT),
                    Guide.xlabel("θ₁"), Guide.ylabel("Objective Function"))
    end

    if par[1] > x[1] && par[1] < x[end]
        pushfirst!(cont.layers,
                    layer(xintercept=[par[1]],
                            Geom.vline(color="magenta"))[1])
    end

    if length(par) > 1
        if par[2] > y[1] && par[2] < y[end]
            pushfirst!(cont.layers,
                        layer(yintercept=[par[2]],
                                Geom.hline(color="cyan"))[1])
        end


        if occursin("bep", save_name)
            zlabel = "\n\n\nObjective Function"
            xlabel="\n\n\n"*L"θ_1"
            ylabel="\n"*L"θ_2"
            zrotation=-35
            xformatter=:scientific
        else
            zlabel = "\nObjective Function"
            xlabel="\n"*L"θ_1"
            ylabel="\n"*L"θ_2"
            zrotation=nothing
            xformatter=:auto
        end
        # "Arial" font does not have the required glyphs under PyPlot
        three_dim = Plots.surface(x, y, z,
                                  fillcolor=:vik, alpha=0.7,
                                  title=title[method],
                                  # xrotation=xrotation,
                                  # yrotation=360-45,
                                  zrotation=zrotation,
                                  xformatter=xformatter,
                                  # yformatter=yformatter,
                                  # zformatter=zformatter,
                                  xlabel=xlabel,
                                  ylabel=ylabel,
                                  zlabel=zlabel,
                                  xtickfont=Plots.font("DejaVu Sans", 10, "#6c606b"),
                                  ytickfont=Plots.font("DejaVu Sans", 10, "#6c606b"),
                                  ztickfont=Plots.font("DejaVu Sans", 10, "#6c606b"),
                                  titlefont=Plots.font("DejaVu Sans", 12, "#564a55"),
                                  legendfont=Plots.font("DejaVu Sans", 10, "#6c606b"),
                                  guidefont=Plots.font("DejaVu Sans", 12, "#564a55"),
                                  colorbar=false, right_margin=30px,
                                  camera=cam
                                  )

        Plots.savefig(three_dim,
            joinpath(save_path,
            "./3d_$(method)_$(save_name).pdf"))
    end

    Gadfly.draw(PDF(joinpath(save_path,
                            "./cont_$(method)_$(save_name).pdf")),
                    cont)
end

function calc_z_plot(x::AbstractArray, y::AbstractFloat,
                     data, ode_fun, t, obj_fun;
                     unknown_states=[],fixed_pars=[-1.0,-1.0])::AbstractArray
    z = zeros(length(x))
    par_eval = copy(fixed_pars)
    idx = 1

    for k in x
        par_eval[fixed_pars .< 0.0] .= [k]
        z[idx] = obj_fun(par_eval, data, t, ode_fun)
        @debug "Eval" k par_eval z[idx]
        idx += 1
    end

    z = reshape(z, (length(y), length(x)))

    return z
end
function calc_z_plot(x::AbstractArray, y::AbstractArray,
                     data, ode_fun, t, obj_fun;
                     unknown_states=[],fixed_pars=[-1.0,-1.0])::AbstractArray
    z = zeros(length(y)*length(x))
    par_eval = copy(fixed_pars)
    idx = 1

    for i in x
        for j in y
            par_eval[fixed_pars .< 0.0] .= [i, j]
            # obj_eval = (obj_fun(par_eval, data, t, ode_fun)
            #             +tikhonov(0.01, par_eval,
            #                       zeros(length(par_eval)),
            #                       ones(length(par_eval))))
            z[idx] = obj_fun(par_eval, data, t, ode_fun)
            @debug "Eval" i j par_eval z[idx]
            idx += 1
        end
    end

    z = reshape(z, (length(y), length(x)))

    return z
end

function experiment_contour(exp::Dict, sample_range::AbstractArray)::Nothing
    ode_problem = exp["ode_problem"]
    ode_fun = ode_problem.fun
    t = ode_problem.t
    phi = ode_problem.phi
    bounds = ode_problem.bounds
    ini_cond = ode_problem.data[1]
    rand_range = ode_problem.bounds

    min_range = rand_range[1][1]
    max_range = rand_range[end][end]

    delta_t = 0.1

    dir = exp["ode_problem"].fun |> Symbol |> String
    save_path = joinpath(PATH,dir)
    mkdir(joinpath(PATH,dir))

    cam = (60,30)
    if Symbol(ode_fun) == :bep
        cam = (-100,35)
    end

    # --- Fine Tune This Part ---
    fixed_pars = exp["fixed_pars"]

    min_range = exp["min_range"]
    max_range = exp["max_range"]

    states = exp["states"]
    unknown_states = exp["unknown_states"]

    true_par = exp["true_par"]

    r = exp["reduced_raius"]
    s = exp["reduced_step"]/2.0

    noise = exp["noise"]
    # -- Fine Tune This Part ---

    for num_samples in sample_range
        tspan = (t[1], t[end])
        oprob = ODEProblem(ode_fun, ini_cond, tspan, phi)
        #saveat_t = range(t[1], stop=t[end], length=num_samples)
        #saveat_t = collect(t[1]:((t[end]-t[1])/(num_samples-1)):t[end])
        saveat_t = range(t[1], stop=t[end], length=num_samples)
        osol  = solve(oprob, saveat=saveat_t)
        Plots.plot(osol)
        data = osol.u
        known_states = setdiff(1:length(states),unknown_states)
        #data = data[filter(x -> x in known_states, states),:]
        if noise > 0.0
            data = add_noise(data, noise)
        end

        plot_data = problem_plot(reduce(hcat, data), collect(saveat_t),
                                "scatter_line")

        Gadfly.draw(PDF(joinpath(save_path,
                        "./data_$(length(saveat_t)*length(known_states)).pdf")),
                        plot_data)

        x = range(min_range, max_range, step=(max_range-min_range)/500)
        if length(true_par) == 1
            y = 0.0
        else
            y = range(min_range, max_range, step=(max_range-min_range)/500)
        end

        @info """Data Shooting Estimator
                   Calculating z...""" Symbol(ode_fun)
        z = calc_z_plot(x, y, data, ode_fun, saveat_t,
                        data_shooting,
                        unknown_states=unknown_states,
                        fixed_pars=fixed_pars)

        @info """Data Shooting Estimator
                   Plotting...""" Symbol(ode_fun)
        contour_3d_plots(x, y, z,
                         true_par, save_path,
                         method="ds",
                         save_name="$(dir)_$(length(data))_big",
                         cam=cam)

        @info """Data Shooting Estimator
                   Reducing range...""" Symbol(ode_fun)
        f_min,f_min_idx  = findmin(z)

        x_reduced = range(x[f_min_idx[2]]-r, x[f_min_idx[2]]+r, step=s)
        y_reduced = range(y[f_min_idx[1]]-r, y[f_min_idx[1]]+r, step=s)
        try
            @info """Data Shooting Estimator
                        Calculating z...""" Symbol(ode_fun)
            z = calc_z_plot(x_reduced, y_reduced, data, ode_fun, saveat_t,
                            data_shooting,
                            unknown_states=unknown_states,
                            fixed_pars=fixed_pars)

            @info """Data Shooting Estimator
                        Plotting...""" Symbol(ode_fun)
            contour_3d_plots(x_reduced, y_reduced, z,
                             true_par, save_path,
                             method="ds",
                             save_name="$(dir)_$(length(data))_small",
                             cam=cam)
        catch e
            bt = backtrace()
            msg = sprint(showerror, e, bt)
            @warn """Error!
                        Error in small grid.""" msg
        end

        try
            @info """Single Shooting Estimator
                        Calculating z...""" Symbol(ode_fun)
            z = calc_z_plot(x, y, data, ode_fun, saveat_t,
                            single_shooting,
                            unknown_states=unknown_states,
                            fixed_pars=fixed_pars)

            @info """Single Shooting Estimator
                        Plotting...""" Symbol(ode_fun)
            contour_3d_plots(x, y, z,
                             true_par, save_path,
                             method="ss",
                             save_name="$(dir)_$(length(data))_big",
                             cam=cam)

            @info """Single Shooting Estimator
                        Reducing range...""" Symbol(ode_fun)
            f_min,f_min_idx  = findmin(z)
            x_reduced = range(x[f_min_idx[2]]-r, x[f_min_idx[2]]+r, step=s)
            y_reduced = range(y[f_min_idx[1]]-r, y[f_min_idx[1]]+r, step=s)
            try
                @info """Single Shooting Estimator
                             Calculating z...""" Symbol(ode_fun)
                z = calc_z_plot(x_reduced, y_reduced, data, ode_fun, saveat_t,
                                single_shooting,
                                unknown_states=unknown_states,
                                fixed_pars=fixed_pars)

                @info """Single Shooting Estimator
                             Plotting...""" Symbol(ode_fun)
                contour_3d_plots(x_reduced, y_reduced, z,
                                 true_par, save_path,
                                 method="ss",
                                 save_name="$(dir)_$(length(data))_small",
                                 cam=cam)
            catch e
                bt = backtrace()
                msg = sprint(showerror, e, bt)
                @warn """Error!
                             Error in small grid.""" msg
            end
        catch e
            bt = backtrace()
            msg = sprint(showerror, e, bt)
            @warn """Error!
                        Error in small grid.""" msg
        end
    end
end

function get_experiments()::Vector{Dict}
    experiments = Dict[]
    # --- Interesting Experiments ---

    # No missing states

    # 0 - exponential
    expo = Dict(
        "ode_problem" => get_problem("exponential"),
        "fixed_pars" => [-1.0],
        "min_range" => -1.1,
        "max_range" => 1.15,
        "reduced_raius" => 1.0,
        "reduced_step" => 0.02,
        "noise" => 0.0,
        "states" => [1],
        "unknown_states" => [],
        "true_par" => [1.0])
    push!(experiments, expo)

    # 2 - floudas_1
    fic = Dict(
        "ode_problem" => get_problem("fic"),
        "fixed_pars" => [-1.0,-1.0],
        "min_range" => 0.0,
        "max_range" => 10.0,
        "reduced_raius" => 0.5,
        "reduced_step" => 0.01,
        "noise" => 0.0,
        "states" => [1,2],
        "unknown_states" => [],
        "true_par" => [5.0035,1.0])
    push!(experiments, fic)


    # 1 - floudas_6
    lvp = Dict(
        "ode_problem" => get_problem("lvp"),
        "fixed_pars" => [-1.0,-1.0],
        "min_range" => 0.5,
        "max_range" => 5.0,
        "reduced_raius" => 1.0,
        "reduced_step" => 0.02,
        "noise" => 0.0,
        "states" => [1,2],
        "unknown_states" => [],
        "true_par" => [3.2434,0.9209])
    push!(experiments, lvp)

    # 2 - fhn
    fhn = Dict(
        "ode_problem" => get_problem("fhn"),
        "fixed_pars" => [-1.0,-1.0, 3.0],
        "min_range" => 0.0,
        "max_range" => 1.50,
        "reduced_raius" => 0.91,
        "reduced_step" => 0.005,
        "noise" => 0.0,
        "states" => [1,2],
        "unknown_states" => [],
        "true_par" => [0.2,0.2])
    push!(experiments, fhn)


    # 2 - floudas_4
    bep = Dict(
        "ode_problem" => get_problem("bep"),
        "fixed_pars" => [-1.0,-1.0],
        "min_range" => 0.0,
        "max_range" => 0.01,
        "reduced_raius" => 1e-4,
        "reduced_step" => 1e-6,
        "noise" => 0.0,
        "states" => [1],
        "unknown_states" => [],
        "true_par" => [4.5704*10^(-6),2.7845*10^(-4)])
    push!(experiments, bep)


    # 2 - floudas_5
    mhp = Dict(
        "ode_problem" => get_problem("mhp"),
        "fixed_pars" => [-1.0,-1.0,1e-6,1e-6,1e-6],
        "min_range" => 0.0,
        "max_range" => 10.0,
        "reduced_raius" => 0.8,
        "reduced_step" => 0.01,
        "noise" => 0.0,
        "noise" => 0.0,
        "states" => [1,2,3],
        "unknown_states" => [],
        "true_par" => [5.2407,1.2176])
    push!(experiments, mhp)


    #TODO
    # Up to Here
    # 2 - bbg
    #phi = [0.4, 5, 0.05, 0.5]
    experiment = Dict(
        "ode_problem" => get_problem("bbg"),
        "fixed_pars" => [-1.0,5.0,-1.0,0.5],
        "min_range" => 0.0,
        "max_range" => 5.0,
        "reduced_raius" => 0.8,
        "reduced_step" => 0.01,
        "noise" => 0.0,
        "states" => [1,2],
        "unknown_states" => [],
        "true_par" => [0.4,0.05])
    # push!(experiments, experiment)

    experiment = Dict(
        "ode_problem" => get_problem("bbg"),
        "fixed_pars" => [0.4,-1.0,0.05,-1.0],
        "min_range" => 0.5,
        "max_range" => 8.0,
        "reduced_raius" => 0.1,
        "reduced_step" => 0.005,
        "noise" => 0.0,
        "states" => [1,2],
        "unknown_states" => [],
        "true_par" => [5.0,0.5])
    # push!(experiments, experiment)

    # 1 - floudas_6
    # Not so interesting
    experiment = Dict(
        "ode_problem" => get_problem("lvp"),
        "fixed_pars" => [-1.0,0.9209,-1.0],
        "min_range" => 0.5,
        "max_range" => 3.5,
        "reduced_raius" => 0.5,
        "reduced_step" => 0.05,
        "noise" => 0.0,
        "states" => [1,2],
        "unknown_states" => [2],
        "true_par" => [3.2434,1.1])
    # push!(experiments, experiment)

    # 1 - floudas_6
    # Better
    experiment_1 = Dict(
        "ode_problem" => get_problem("lvp"),
        "fixed_pars" => [-1.0,0.9209,-1.0],
        "min_range" => 0.5,
        "max_range" => 5.0,
        "reduced_raius" => 1.0,
        "reduced_step" => 0.05,
        "noise" => 0.0,
        "states" => [1,2],
        "unknown_states" => [1],
        "true_par" => [3.2434,1.2])
    # push!(experiments, experiment_1)

    # 2 - floudas_1
    experiment = Dict(
        "ode_problem" => get_problem("fic"),
        "fixed_pars" => [5.0035,-1.0,-1.0],
        "min_range" => 0.0,
        "max_range" => 5.0,
        "reduced_raius" => 0.5,
        "reduced_step" => 0.01,
        "noise" => 0.0,
        "states" => [1,2],
        "unknown_states" => [1],
        "true_par" => [1.0,1.0])
    # push!(experiments, experiment)

    # 2 - floudas_1
    # Not so interesting
    experiment = Dict(
        "ode_problem" => get_problem("fic"),
        "fixed_pars" => [-1.0,1.0,-1.0],
        "min_range" => -1.0,
        "max_range" => 7.0,
        "reduced_raius" => 0.5,
        "reduced_step" => 0.01,
        "noise" => 0.0,
        "states" => [1,2],
        "unknown_states" => [2],
        "true_par" => [5.0035,0.0])
    # push!(experiments, experiment)
    return experiments
end

function main()
    experiments = get_experiments()
    experiment_contour_10_50 = e -> experiment_contour(e, 10:20:50)
    for e in experiments
        experiment_contour_10_50(e)
    end
end

main()
