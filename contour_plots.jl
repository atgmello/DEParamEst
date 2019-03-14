linear(x) = x
#loss = soft_l1
loss = linear
desired_precision = Float64

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
    end

    for i in range(1, stop=length(x))
        for j in range(1, stop=length(y))
            phi_eval = [x[i], y[j]]
            obj_eval = sum(loss.(abs2.(obj_fun(convert(Array{desired_precision}, phi_eval),
                                convert(Array{desired_precision}, data),
                                convert(Array{desired_precision}, t),
                                ode_fun))))
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

for i in [6]
    println("\n----- Plots for problem $i -----\n")
    p_solve = problem_set[i]
    ode_fun = p_solve.fun
    t = p_solve.t
    phi = p_solve.phi
    bounds = p_solve.bounds
    ini_cond = p_solve.data[:,1]
    rand_range = p_solve.bounds
    min_range = rand_range[1][1]
    max_range = rand_range[end][end]
    delta_t = desired_precision(.1)

    for num_samples in 3:20:23
        tspan = (t[1], t[end])
        oprob = ODEProblem(ode_fun, ini_cond, tspan, phi)
        #saveat_t = range(t[1], stop=t[end], length=num_samples)
        saveat_t = t[1]:((t[end]-t[1])/num_samples):t[end]
        osol  = solve(oprob, Tsit5(), saveat=saveat_t)
        plot(osol)
        data = reduce(hcat, osol.u)
        data_plot = scatter(saveat_t, transpose(data))
        display(data_plot)

        x = range(min_range, max_range, step=.05)
        y = range(min_range, max_range, step=.05)

        println("\n----- Adams-Moulton Estimator -----")
        f_min,f_min_idx = calc_z_plot(x, y, data, ode_fun, phi, saveat_t, fun="am")
        x_reduced = range(x[f_min_idx[2]]-.5, x[f_min_idx[2]]+.5, step=.05)
        y_reduced = range(y[f_min_idx[1]]-.5, y[f_min_idx[1]]+.5, step=.05)
        calc_z_plot(x_reduced, y_reduced, data, ode_fun, phi, saveat_t, fun="am", reduced=true)

        println("\n----- Classic Estimator -----")
        f_min,f_min_idx = calc_z_plot(x, y, data, ode_fun, phi, saveat_t, fun="cla")
        x_reduced = range(x[f_min_idx[2]]-.5, x[f_min_idx[2]]+.5, step=.05)
        y_reduced = range(y[f_min_idx[1]]-.5, y[f_min_idx[1]]+.5, step=.05)
        calc_z_plot(x_reduced, y_reduced, data, ode_fun, phi, saveat_t, fun="cla", reduced=true)
    end
end
