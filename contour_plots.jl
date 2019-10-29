linear(x) = x
#loss = soft_l1
loss = linear
#plotlyjs()
gr()

desired_precision = Float64

function contour_3d_plots(x, y, z, par; title="", reduced=false)
    cont = contour(x, y, z, fill=true, title=title)
    if reduced == false
        vline!(cont, [par[1]])
        hline!(cont, [par[2]])
    end
    display(cont)

    three_dim = surface(x,y,z, cbar=true, title=title)
    display(three_dim)
end


function calc_z_plot(x, y, data, ode_fun, true_par, t, obj_fun;
                        reduced=false, unknown_states=[],fixed_pars=[-1.0,-1.0])
    z = desired_precision[]

    for i in x
        for j in y
            par_eval = copy(fixed_pars)
            par_eval[par_eval .< 0.0] = [i, j]
            obj_eval = sum(loss.(abs2.(obj_fun(par_eval, data, t, ode_fun,
                                        unknown_states=unknown_states))))
            push!(z, obj_eval)
            #println("$i,$j\nPhi: $phi_eval\nRes: $obj_eval")
        end
    end
    z = reshape(z, (length(y), length(x)))
    num_samples = length(data)
    contour_3d_plots(x, y, z, true_par, title="for num_samples=$num_samples", reduced=reduced)

    a = findmin(z)
    return a
end

for i in ["floudas_1"]
    println("\n----- Plots for problem $i -----\n")
    p_solve = get_ode_problem(i)
    ode_fun = p_solve.fun
    t = p_solve.t
    phi = p_solve.phi
    bounds = p_solve.bounds
    ini_cond = p_solve.data[:,1]
    rand_range = p_solve.bounds

    min_range = rand_range[1][1]
    max_range = rand_range[end][end]

    delta_t = desired_precision(.1)

    fixed_pars=[1.0,-1.0,-1.0]

    min_range = 0.0
    max_range = 10.0

    states = [1,2]
    unknown_states=[1]

    true_par = [5.0035, 1.0]

    for num_samples in 10:45:55
        tspan = (t[1], t[end])
        oprob = ODEProblem(ode_fun, ini_cond, tspan, phi)
        #saveat_t = range(t[1], stop=t[end], length=num_samples)
        saveat_t = collect(t[1]:((t[end]-t[1])/num_samples):t[end])
        osol  = solve(oprob, Tsit5(), saveat=saveat_t)
        plot(osol)
        data = reduce(hcat, osol.u)
        known_states = setdiff(1:length(states),unknown_states)
        data = data[filter(x -> x in known_states, states),:]
        #add_noise!(data, 0.1)
        data_plot = scatter(saveat_t, transpose(data))
        display(data_plot)

        x = range(min_range, max_range, step=.1)
        y = range(min_range, max_range, step=.1)

        println("\n----- Adams-Moulton Estimator -----")
        try
            f_min,f_min_idx = calc_z_plot(x, y, data, ode_fun, true_par, t,
                                        adams_moulton_estimator_x,
                        reduced=false, unknown_states=unknown_states,
                        fixed_pars=fixed_pars)

            x_reduced = range(x[f_min_idx[2]]-.15, x[f_min_idx[2]]+.15, step=.01)
            y_reduced = range(y[f_min_idx[1]]-.15, y[f_min_idx[1]]+.15, step=.01)
            try
                calc_z_plot(x_reduced, y_reduced, data, ode_fun, true_par, t,
                                        adams_moulton_estimator_x,
                        reduced=true, unknown_states=unknown_states,
                        fixed_pars=fixed_pars)
            catch e
                println("Error on small grid.\n$e")
            end
        catch e
            println("Error on big grid.\n$e")
        end

        println("\n----- Classic Estimator -----")
        try
            f_min,f_min_idx = calc_z_plot(x, y, data, ode_fun, true_par, t,
                                        single_shooting_estimator,
                        reduced=false, unknown_states=unknown_states,
                        fixed_pars=fixed_pars)
            x_reduced = range(x[f_min_idx[2]]-.15, x[f_min_idx[2]]+.15, step=.01)
            y_reduced = range(y[f_min_idx[1]]-.15, y[f_min_idx[1]]+.15, step=.01)
            try
                calc_z_plot(x_reduced, y_reduced, data, ode_fun, true_par, t,
                                        single_shooting_estimator,
                        reduced=true, unknown_states=unknown_states,
                        fixed_pars=fixed_pars)
            catch e
                println("Error on small grid.\n$e")
            end
        catch e
            println("Error on big grid.\n$e")
        end
    end
end
