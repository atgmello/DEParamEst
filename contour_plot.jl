using DifferentialEquations
using Plots
gr()
theme(:vibrant)

using Revise
includet("./utils.jl")
includet("./problem_set.jl")
includet("./objective_function.jl")
import .ProblemSet: get_problem
import .ObjectiveFunction: data_shooting, single_shooting, tikhonov
import .Utils: add_noise

const PATH = "/home/andrew/git/DEParamEst/results/local/plots/contours/unregularized/"

function contour_3d_plots(x, y, z, par; title="", save_name="")
    if length(par) > 1
        cont = contourf(x, y, z, cbar=true, title=title,
                    xlabel="k1", ylabel="k2")
    else
        cont = plot(x, vec(z), title=title, legend=false,
                    xlabel="k1", ylabel="Cost Function")
    end

    if par[1] > x[1] && par[1] < x[end]
        vline!(cont, [par[1]], label="True k1", color=:cyan)
    end

    if length(par) > 1
        if par[2] > y[1] && par[2] < y[end]
            hline!(cont, [par[2]], label="True k2", color=:magenta)
        end

        three_dim = plot(title=title,
                            #xrotation=45,yrotation=360-45,
                            xlabel="k1", ylabel="k2", zlabel="Cost function")
        surface!(three_dim, x, y, z)
        plot!(three_dim, xrotation=45, yrotation=45)


        #display(three_dim)
        savefig(three_dim,"./3d_$(lowercase(replace(title," " => "_")))_$(save_name).pdf")
    end

    savefig(cont,"./cont_$(lowercase(replace(title," " => "_")))_$(save_name).pdf")
    #display(cont)

end


function calc_z_plot(x, y, data, ode_fun, t, obj_fun;
                        unknown_states=[],fixed_pars=[-1.0,-1.0])
    z = []

    if length(y) > 1
        #2D case
        for i in x
            for j in y
                par_eval = copy(fixed_pars)
                par_eval[par_eval .< 0.0] = [i, j]
                #obj_eval = obj_fun(par_eval, data, t, ode_fun)+tikhonov(0.01,par_eval,zeros(length(par_eval)),ones(length(par_eval)))
                obj_eval = obj_fun(par_eval, data, t, ode_fun)
                push!(z, obj_eval)
                #println("$i,$j\nPhi: $phi_eval\nRes: $obj_eval")
            end
        end
    else
        #1D case
        for i in x
            par_eval = copy(fixed_pars)
            par_eval[par_eval .< 0.0] = [i]
            obj_eval = obj_fun(par_eval, data, t, ode_fun)
            push!(z, obj_eval)
            #println("$i,$j\nPhi: $phi_eval\nRes: $obj_eval")
        end
    end

    z = reshape(z, (length(y), length(x)))

    return z
end

function experiment_countour(exp, sample_range)
    cd(PATH)
    dir = exp["problem_name"]
    mkdir(dir)
    cd(dir)

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
        plot(osol)
        data = osol.u
        known_states = setdiff(1:length(states),unknown_states)
        #data = data[filter(x -> x in known_states, states),:]
        if noise > 0.0
            data = add_noise(data, noise)
        end

        alphabet='A':'Z'
        label=reshape(["$i" for i in alphabet[1:length(known_states)]],(1,length(known_states)))

        data_plot = reduce(hcat, osol.u)
        plot_data = scatter(saveat_t, transpose(data_plot),
                            label=label, xlabel="Time", ylabel="State")
        display(plot_data)
        savefig(plot_data,"./data_$(length(saveat_t)*length(known_states)).pdf")

        x = range(min_range, max_range, step=(max_range-min_range)/100)
        if length(states) == 1
            y = 0.0
        else
            y = range(min_range, max_range, step=(max_range-min_range)/100)
        end

        println("\n----- Data Shooting Estimator -----")
            z = calc_z_plot(x, y, data, ode_fun, saveat_t,
                                        data_shooting,
                                        unknown_states=unknown_states,
                                        fixed_pars=fixed_pars)

            contour_3d_plots(x, y, z, true_par, title="Data Shooting", save_name="$(length(data))_big")
            f_min,f_min_idx  = findmin(z)

            x_reduced = range(x[f_min_idx[2]]-r, x[f_min_idx[2]]+r, step=s)
            y_reduced = range(y[f_min_idx[1]]-r, y[f_min_idx[1]]+r, step=s)
            try
                z = calc_z_plot(x_reduced, y_reduced, data, ode_fun, saveat_t,
                                            data_shooting,
                                            unknown_states=unknown_states,
                                            fixed_pars=fixed_pars)

                contour_3d_plots(x_reduced, y_reduced, z, true_par, title="Data Shooting", save_name="$(length(data))_small")
            catch e
                println("Error on small grid.\n$e")
            end

        println("\n----- Classic Estimator -----")
        try
            z = calc_z_plot(x, y, data, ode_fun, saveat_t,
                                        single_shooting,
                                        unknown_states=unknown_states,
                                        fixed_pars=fixed_pars)
            contour_3d_plots(x, y, z, true_par, title="Single Shooting", save_name="$(length(data))_big")
            f_min,f_min_idx  = findmin(z)
            x_reduced = range(x[f_min_idx[2]]-r, x[f_min_idx[2]]+r, step=s)
            y_reduced = range(y[f_min_idx[1]]-r, y[f_min_idx[1]]+r, step=s)
            try
                z = calc_z_plot(x_reduced, y_reduced, data, ode_fun, saveat_t,
                                            single_shooting,
                                            unknown_states=unknown_states,
                                            fixed_pars=fixed_pars)
                contour_3d_plots(x_reduced, y_reduced, z, true_par, title="Single Shooting", save_name="$(length(data))_small")
            catch e
                println("Error on small grid.\n$e")
            end
        catch e
            println("Error on big grid.\n$e")
        end
    end
end

# --- Interesting Experiments ---

# No missing states

# 0 - exponential
experiment_1 = Dict(
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

experiment_countour(experiment_1, 10:60:50)

# 2 - floudas_1
experiment = Dict(
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

experiment_countour(experiment, 10:20:50)

# 1 - floudas_6
experiment_1 = Dict(
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

experiment_countour(experiment_1, 10:20:50)

# 2 - fhn
experiment = Dict(
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

experiment_countour(experiment, 10:20:50)

# 2 - floudas_4
experiment = Dict(
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

experiment_countour(experiment, 10:20:50)

# 2 - floudas_5
experiment = Dict(
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

experiment_countour(experiment, 10:20:50)

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

experiment_countour(experiment, 10:20:50)

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

experiment_countour(experiment, 10:20:50)

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

experiment_countour(experiment, 10:20:50)

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

experiment_countour(experiment_1, 10:20:50)

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

experiment_countour(experiment, 10:20:50)

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

experiment_countour(experiment, 10:20:50)
