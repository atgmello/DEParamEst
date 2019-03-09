using Plots
gui()

plotly()

gr()
#delta_t = desired_precision(.1)


function contour_3d_plots(x, y, z, phi; title="")
    cont = contour(x, y, z, fill=true, title=title)
    vline!(cont, [phi[1]])
    hline!(cont, [phi[2]])
    display(cont)

    three_dim = surface(x,y,z, cbar=true, title=title)
    display(three_dim)
end


function calc_z_plot(x, y, data, ode_fun, phi, t; fun="am")
    z = desired_precision[]
    if fun=="am"
        obj_fun = adams_moulton_estimator
    elseif fun=="cla"
        obj_fun = single_shooting_estimator
    end

    for i in range(1, stop=length(x))
        for j in range(1, stop=length(y))
            phi_eval = [x[i], y[j]]
            obj_eval = sum_square(soft_l1(obj_fun(phi_eval, data, t, ode_fun)))
            push!(z, obj_eval)
        end
    end
    z = reshape(z, (length(y), length(x)))
    num_samples = length(data)
    contour_3d_plots(x, y, z, phi, title="$fun for num_samples=$num_samples")
end

for i in [6]
    println("\n----- Plots for problem $i -----\n")
    ode_fun = ode_fun_array[i]
    t = t_array[i]
    phi = phi_array[i]
    bounds = bounds_array[i]
    ini_cond = ini_cond_array[i]
    rand_range = rand_range_array[i]
    min_range = rand_range[1]
    max_range = rand_range[end]
    delta_t = desired_precision(.1)

    for num_samples in 3:20:23
        tspan = (t[1], t[end])
        oprob = ODEProblem(ode_fun, ini_cond, tspan, phi)
        #saveat_t = range(t[1], stop=t[end], length=num_samples)
        saveat_t = t[1]:((t[end]-t[1])/num_samples):t[end]
        osol  = solve(oprob, Tsit5(), saveat=saveat_t)
        plot(osol)
        println(osol.t)
        println(osol.u)
        data = reduce(hcat, osol.u)
        println(data)
        data_plot = scatter(saveat_t, transpose(data))
        display(data_plot)

        x = range(min_range, max_range, step=.05)
        y = range(min_range, max_range, step=.05)

        x_reduced = range(phi[1]-.05, phi[1]+.05, step=.001)
        y_reduced = range(phi[2]-.05, phi[2]+.05, step=.001)

        println("\n----- Adams-Moulton Estimator -----")
        calc_z_plot(x, y, data, ode_fun, phi, saveat_t, fun="am")
        calc_z_plot(x_reduced, y_reduced, data, ode_fun, phi, saveat_t, fun="am")

        println("\n----- Classic Estimator -----")
        calc_z_plot(x, y, data, ode_fun, phi, saveat_t, fun="cla")
        calc_z_plot(x_reduced, y_reduced, data, ode_fun, phi, saveat_t, fun="cla")
    end
end

x = range(1, 10, step=1)
y = range(1, 4, step=1)

z = []
for i in x
    for j in y
        push!(z, i*j)
    end
end
cont = contour(x, y, z, aspect_ratio=1, fill=true)
print(z)
z = reshape(z,(length(y),length(x)))
cont = contour(x, y, z, aspect_ratio=1, fill=true)
vline!(cont, [3])
hline!(cont, [3])
print(z)

z = float((1:4) * reshape(1:10, 1, :))
cont = contour(x, y, z, aspect_ratio=1, fill=true)
print(z)
