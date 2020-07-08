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
Plots.pyplot()
Plots.theme(:default)

include("../calc/problem_set.jl")
include("../calc/objective_function.jl")
include("./utils.jl")
import .ProblemSet: get_problem, DEProblem
import .ObjectiveFunction: data_shooting, single_shooting, tikhonov
import .PlottingUtils: heatmap, problem_plot, add_noise

const PATH = "./data/contours/"
const PLOT_FONT = "Arial"


function contour_3d_plots(x, y, z, par, save_path; method="", save_name="")
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
                    Gadfly.Theme(minor_label_font=PLOT_FONT,
                                    major_label_font=PLOT_FONT,
                                    key_title_font=PLOT_FONT,
                                    key_label_font=PLOT_FONT),
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


        three_dim = Plots.surface(
                        x, y, z, fillcolor=:vik, alpha=0.7,
                        title=title[method],
                        #xrotation=45,yrotation=360-45,
                        xlabel=L"θ_1", ylabel=L"θ_2",
                        zlabel="Objective Function",
                        xtickfont=Plots.font("Arial", 10, "#6c606b"),
                        ytickfont=Plots.font("Arial", 10, "#6c606b"),
                        ztickfont=Plots.font("Arial", 10, "#6c606b"),
                        titlefont=Plots.font("Arial", 12, "#564a55"),
                        legendfont=Plots.font("Arial", 10, "#6c606b"),
                        guidefont=Plots.font("Arial", 12, "#564a55"),
                        colorbar=false, right_margin=30px)

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

    # --- Fine Tune This Part ---
    fixed_pars = exp["fixed_pars"]

    min_range = exp["min_range"]
    max_range = exp["max_range"]

    states = exp["states"]
    unknown_states = exp["unknown_states"]

    true_par = exp["true_par"]

    r = exp["reduced_raius"]
    s = exp["reduced_step"]

    noise = exp["noise"]
    # -- Fine Tune This Part ---

    for num_samples in sample_range
        tspan = (t[1], t[end])
        oprob = ODEProblem(ode_fun, ini_cond, tspan, phi)
        #saveat_t = range(t[1], stop=t[end], length=num_samples)
        #saveat_t = collect(t[1]:((t[end]-t[1])/(num_samples-1)):t[end])
        saveat_t = range(t[1], stop=t[end], length=num_samples)
        osol  = solve(oprob, OwrenZen3(), saveat=saveat_t)
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

        x = range(min_range, max_range, step=(max_range-min_range)/100)
        if length(states) == 1
            y = 0.0
        else
            y = range(min_range, max_range, step=(max_range-min_range)/100)
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
                         save_name="$(dir)_$(length(data))_big")

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
                             save_name="$(dir)_$(length(data))_small")
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
                             save_name="$(dir)_$(length(data))_big")

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
                                 save_name="$(dir)_$(length(data))_small")
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

# --- Interesting Experiments ---

# No missing states

# 0 - exponential
expo_1 = Dict(
    "problem_name" => "exponential",
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


# 2 - floudas_1
floudas_1 = Dict(
    "problem_name" => "floudas_1",
    "ode_problem" => get_problem("floudas_1"),
    "fixed_pars" => [-1.0,-1.0],
    "min_range" => 0.0,
    "max_range" => 10.0,
    "reduced_raius" => 0.5,
    "reduced_step" => 0.01,
    "noise" => 0.0,
    "states" => [1,2],
    "unknown_states" => [],
    "true_par" => [5.0035,1.0])


# 1 - floudas_6
floudas_6 = Dict(
    "problem_name" => "floudas_6",
    "ode_problem" => get_problem("floudas_6"),
    "fixed_pars" => [-1.0,-1.0],
    "min_range" => 0.5,
    "max_range" => 5.0,
    "reduced_raius" => 1.0,
    "reduced_step" => 0.02,
    "noise" => 0.0,
    "states" => [1,2],
    "unknown_states" => [],
    "true_par" => [3.2434,0.9209])

# 2 - fhn
fhn = Dict(
    "problem_name" => "fhn",
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


# 2 - floudas_4
floudas_4 = Dict(
    "problem_name" => "floudas_4",
    "ode_problem" => get_problem("floudas_4"),
    "fixed_pars" => [-1.0,-1.0],
    "min_range" => 0.0,
    "max_range" => 0.01,
    "reduced_raius" => 1e-4,
    "reduced_step" => 1e-6,
    "noise" => 0.0,
    "states" => [1],
    "unknown_states" => [],
    "true_par" => [4.5704*10^(-6),2.7845*10^(-4)])


# 2 - floudas_5
floudas_5 = Dict(
    "problem_name" => "floudas_5",
    "ode_problem" => get_problem("floudas_5"),
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


#TODO
# Up to Here
# 2 - bbg
#phi = [0.4, 5, 0.05, 0.5]
experiment = Dict(
    "problem_name" => "bbg_1",
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

# experiment_countour(experiment, 10:20:50)

experiment = Dict(
    "problem_name" => "bbg_2",
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

# experiment_countour(experiment, 10:20:50)

# 1 - floudas_6
# Not so interesting
experiment = Dict(
    "problem_name" => "floudas_6_u2",
    "ode_problem" => get_problem("floudas_6"),
    "fixed_pars" => [-1.0,0.9209,-1.0],
    "min_range" => 0.5,
    "max_range" => 3.5,
    "reduced_raius" => 0.5,
    "reduced_step" => 0.05,
    "noise" => 0.0,
    "states" => [1,2],
    "unknown_states" => [2],
    "true_par" => [3.2434,1.1])

# experiment_countour(experiment, 10:20:50)

# 1 - floudas_6
# Better
experiment_1 = Dict(
    "problem_name" => "floudas_6_u1",
    "ode_problem" => get_problem("floudas_6"),
    "fixed_pars" => [-1.0,0.9209,-1.0],
    "min_range" => 0.5,
    "max_range" => 5.0,
    "reduced_raius" => 1.0,
    "reduced_step" => 0.05,
    "noise" => 0.0,
    "states" => [1,2],
    "unknown_states" => [1],
    "true_par" => [3.2434,1.2])

# experiment_countour(experiment_1, 10:20:50)

# 2 - floudas_1
experiment = Dict(
    "problem_name" => "floudas_1_u1",
    "ode_problem" => get_problem("floudas_1"),
    "fixed_pars" => [5.0035,-1.0,-1.0],
    "min_range" => 0.0,
    "max_range" => 5.0,
    "reduced_raius" => 0.5,
    "reduced_step" => 0.01,
    "noise" => 0.0,
    "states" => [1,2],
    "unknown_states" => [1],
    "true_par" => [1.0,1.0])

# experiment_countour(experiment, 10:20:50)

# 2 - floudas_1
# Not so interesting
experiment = Dict(
    "problem_name" => "floudas_1_u2",
    "ode_problem" => get_problem("floudas_1"),
    "fixed_pars" => [-1.0,1.0,-1.0],
    "min_range" => -1.0,
    "max_range" => 7.0,
    "reduced_raius" => 0.5,
    "reduced_step" => 0.01,
    "noise" => 0.0,
    "states" => [1,2],
    "unknown_states" => [2],
    "true_par" => [5.0035,0.0])

# experiment_countour(experiment, 10:20:50)

function main()
    # experiment_countour(floudas_5, 10:20:50)
    # experiment_countour(floudas_4, 10:20:50)
    experiment_countour(fhn, 10:20:50)
    # experiment_countour(floudas_6, 10:20:50)
    # experiment_countour(floudas_1, 10:20:50)
    # experiment_countour(expo_1, 10:60:50)
end

main()
