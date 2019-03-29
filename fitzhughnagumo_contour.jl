linear(x) = x
#loss = soft_l1
loss = linear

desired_precision = Float64
plotlyjs()
plotly()
gr()

function contour_3d_plots(x, y, z, phi; title="", reduced=false)
    cont = contour(x, y, z, fill=true, title=title)
    if reduced == false
        vline!(cont, [phi[1]])
        hline!(cont, [phi[2]])
    end
    display(cont)

    three_dim = surface(x,y,z, cbar=true, title=title)
    display(three_dim)
end


function calc_z_plot(x, y, data, ode_fun, phi, t; fun="am", reduced=false)
    z = desired_precision[]
    if fun=="am"
        obj_fun = adams_moulton_estimator
    elseif fun=="cla"
        obj_fun = single_shooting_estimator
    elseif fun=="SMS"
        obj_fun = single_multiple_adams_shooting
    elseif fun=="SMSM"
        obj_fun = sm_mean_shooting
    elseif fun=="Adapt"
        obj_fun = sm_adaptative_shooting
    elseif fun=="AdaptH"
        obj_fun = sm_adaptative_hard_shooting
    end

    for i in range(1, stop=length(x))
        for j in range(1, stop=length(y))
            phi_eval = [x[i], y[j], 3.0]
            if fun == "am" || fun == "cla" || fun == "SMS" || fun=="SMSM" || fun=="Adapt" || fun=="AdaptH"
                obj_eval = sum(loss.(abs2.(obj_fun(convert(Array{desired_precision}, phi_eval),
                                convert(Array{desired_precision}, data),
                                convert(Array{desired_precision}, t),
                                ode_fun))))
            else
                t_span = (t[1], t[end])
                de_prob = ODEProblem(ode_fun, data[:,1], t_span, phi_eval)
                if fun == "DiffEqCla"
                    f = build_loss_objective(de_prob,Tsit5(),L2Loss(t,data))
                    obj_eval = f(phi_eval)
                elseif fun == "DiffEqHeu"
                    f = two_stage_method(de_prob,t,data)
                    obj_eval = f(phi_eval)
                elseif fun == "DiffEqMS"
                    f = multiple_shooting_objective(de_prob,Tsit5(),L2Loss(t,data))
                    obj_eval = f(phi_eval)
                end
            end
            push!(z, obj_eval)
            #println("$i,$j\nPhi: $phi_eval\nRes: $obj_eval")
        end
    end
    z = reshape(z, (length(y), length(x)))
    num_samples = length(data)
    contour_3d_plots(x, y, z, phi, title="$fun for num_samples=$num_samples", reduced=reduced)

    a = findmin(z)
    return a
end

for i in [9]
    println("\n----- Plots for problem $i -----\n")
    p_solve = problem_set[i]
    ode_fun = p_solve.fun
    t = p_solve.t
    phi = p_solve.phi
    bounds = p_solve.bounds
    ini_cond = p_solve.data[:,1]
    rand_range = [[-0.75, -0.75], [1.1, 1.1]]
    min_range = rand_range[1][1]
    max_range = rand_range[end][end]
    delta_t = desired_precision(.1)

    for num_samples in 50:1:50
        tspan = (t[1], t[end])
        oprob = ODEProblem(ode_fun, ini_cond, tspan, phi)
        #saveat_t = range(t[1], stop=t[end], length=num_samples)
        saveat_t = t[1]:((t[end]-t[1])/num_samples):t[end]
        osol  = solve(oprob, lsoda(), saveat=saveat_t)
        plot(osol)
        data = reduce(hcat, osol.u)
        data_plot = scatter(saveat_t, transpose(data))
        display(data_plot)

        x = range(min_range, max_range, step=.01)
        y = range(min_range, max_range, step=.01)

        println("\n----- Adams-Moulton Estimator -----")
        try
            f_min,f_min_idx = calc_z_plot(x, y, data, ode_fun, phi, saveat_t, fun="am")
            #x_reduced = range(x[f_min_idx[2]]-0.9, x[f_min_idx[2]]+0.9, step=.005)
            #y_reduced = range(y[f_min_idx[1]]-0.9, y[f_min_idx[1]]+0.9, step=.005)
            try
                calc_z_plot(x_reduced, y_reduced, data, ode_fun, phi, saveat_t, fun="am", reduced=true)
            catch e
                println("Error on small grid.\n$e")
            end
        catch e
            println("Error on big grid.\n$e")
        end

        #=>
        println("\n----- SMS Estimator -----")
        try
            f_min,f_min_idx = calc_z_plot(x, y, data, ode_fun, phi, saveat_t, fun="SMS")
            x_reduced = range(x[f_min_idx[2]]-.1, x[f_min_idx[2]]+.1, step=.01)
            y_reduced = range(y[f_min_idx[1]]-.1, y[f_min_idx[1]]+.1, step=.01)
            try
                calc_z_plot(x_reduced, y_reduced, data, ode_fun, phi, saveat_t, fun="SMS", reduced=true)
            catch e
                println("Error on small grid.\n$e")
            end
        catch e
                println("Error on big grid.\n$e")
        end

        println("\n----- Adapt Estimator -----")
        try
            f_min,f_min_idx = calc_z_plot(x, y, data, ode_fun, phi, saveat_t, fun="Adapt")
            x_reduced = range(x[f_min_idx[2]]-.1, x[f_min_idx[2]]+.1, step=.01)
            y_reduced = range(y[f_min_idx[1]]-.1, y[f_min_idx[1]]+.1, step=.01)
            try
                calc_z_plot(x_reduced, y_reduced, data, ode_fun, phi, saveat_t, fun="Adapt", reduced=true)
            catch e
                println("Error on small grid.\n$e")
            end
        catch e
                println("Error on big grid.\n$e")
        end

        println("\n----- Adapt Hard Estimator -----")
        try
            f_min,f_min_idx = calc_z_plot(x, y, data, ode_fun, phi, saveat_t, fun="AdaptH")
            x_reduced = range(x[f_min_idx[2]]-.1, x[f_min_idx[2]]+.1, step=.01)
            y_reduced = range(y[f_min_idx[1]]-.1, y[f_min_idx[1]]+.1, step=.01)
            try
                calc_z_plot(x_reduced, y_reduced, data, ode_fun, phi, saveat_t, fun="AdaptH", reduced=true)
            catch e
                println("Error on small grid.\n$e")
            end
        catch e
                println("Error on big grid.\n$e")
        end
        <=#

        println("\n----- Classic Estimator -----")
        try
            f_min,f_min_idx = calc_z_plot(x, y, data, ode_fun, phi, saveat_t, fun="cla")
            #x_reduced = range(x[f_min_idx[2]]-0.9, x[f_min_idx[2]]+0.9, step=.005)
            #y_reduced = range(y[f_min_idx[1]]-0.9, y[f_min_idx[1]]+0.9, step=.005)
            try
                calc_z_plot(x_reduced, y_reduced, data, ode_fun, phi, saveat_t, fun="cla", reduced=true)
            catch e
                println("Error on small grid.\n$e")
            end
        catch e
                println("Error on big grid.\n$e")
        end

        #=>
        println("\n----- DiffEq Two Stage Estimator -----")
        try
            f_min,f_min_idx = calc_z_plot(x, y, data, ode_fun, phi, saveat_t, fun="DiffEqHeu")
            x_reduced = range(x[f_min_idx[2]]-.5, x[f_min_idx[2]]+.5, step=.05)
            y_reduced = range(y[f_min_idx[1]]-.5, y[f_min_idx[1]]+.5, step=.05)
            try
                calc_z_plot(x_reduced, y_reduced, data, ode_fun, phi, saveat_t, fun="DiffEqHeu", reduced=true)
            catch e
                println("Error on small grid.\n$e")
            end
        catch e
            println("Error on big grid.\n$e")
        end

        println("\n----- DiffEq Classic Estimator -----")
        try
            f_min,f_min_idx = calc_z_plot(x, y, data, ode_fun, phi, saveat_t, fun="DiffEqCla")
            x_reduced = range(x[f_min_idx[2]]-0.8, x[f_min_idx[2]]+0.8, step=.01)
            y_reduced = range(y[f_min_idx[1]]-0.8, y[f_min_idx[1]]+0.8, step=.01)
            try
                calc_z_plot(x_reduced, y_reduced, data, ode_fun, phi, saveat_t, fun="DiffEqCla", reduced=true)
            catch e
                println("Error on small grid.\n$e")
            end
        catch e
            println("Error on big grid.\n$e")
        end
        <=#

        #println("\n----- DiffEq Multi Shooting Estimator -----")
        #f_min,f_min_idx = calc_z_plot(x, y, data, ode_fun, phi, saveat_t, fun="DiffEqMS")
        #x_reduced = range(x[f_min_idx[2]]-.5, x[f_min_idx[2]]+.5, step=.05)
        #y_reduced = range(y[f_min_idx[1]]-.5, y[f_min_idx[1]]+.5, step=.05)
        #calc_z_plot(x_reduced, y_reduced, data, ode_fun, phi, saveat_t, fun="DiffEqMS", reduced=true)
    end
end
