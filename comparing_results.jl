using Optim
using DifferentialEquations
using DiffEqParamEstim
import Distributions: Uniform
using NLsolve
import LeastSquaresOptim
const lso = LeastSquaresOptim
import LsqFit
const lsf = LsqFit
loss = soft_l1


for i in [1,2,3,4,5,6] #range(1, stop=2)
    for sample_size in 11:1:11
        println("\n----- Solving problem $i with $sample_size samples -----\n")
        ode_fun = ode_fun_array[i]
        t = t_array[i]
        phi = phi_array[i]
        ini_cond = ini_cond_array[i]

        sample_t = range(t[1], stop=t[end], length=sample_size)
        tspan = (t[1], t[end])
        oprob = ODEProblem(ode_fun, ini_cond, tspan, phi)
        osol  = solve(oprob, Tsit5(), saveat=sample_t)
        osol_plot = scatter(osol)
        display(osol_plot)
        data = reduce(hcat, osol.u)
        bounds = bounds_array[i]
        lb = vec([bounds[1] for i in 1:length(phi)])
        ub = vec([bounds[end] for i in 1:length(phi)])

        #data = floudas_samples_array[i]
        #sample_t = floudas_samples_times_array[i]

        rand_range = rand_range_array[i]

        initial_guess = desired_precision[]
        for i in range(1, stop=length(phi))
            rand_num = rand(Uniform(rand_range[1], rand_range[end]))
            push!(initial_guess, rand_num)
        end

        #initial_guess = phi

        print("Initial guess:\n$initial_guess\n")

        lower = desired_precision[]
        upper = desired_precision[]

        for i in 1:length(phi)
            push!(lower, bounds[1])
            push!(upper, bounds[2])
        end


        print("\n----- Adams-Moulton Estimator -----\n")
        am_optim(p) = sum(loss.(abs2.(adams_moulton_estimator(p, data, sample_t, ode_fun))))
        res_am = optimize(am_optim, lower, upper, initial_guess, Fminbox(NelderMead()))
        println("\nReal:\n$phi\nEstimated:\n$(res_am.minimizer)\n")
        print(res_am)

        #initial_guess = res_am.minimizer

        #=>
        oprob = ODEProblem(ode_fun, ini_cond, tspan, res_am.minimizer)
        osol  = solve(oprob, Tsit5())
        plot!(osol_plot, osol, label="AM", color="red")
        <=#

        print("\n----- Adams-Moulton Residual Estimator -----\n")
        lsq_am(p) = vec(loss.(abs2.(adams_moulton_estimator(p, data, sample_t, ode_fun))))
        float64_guess = [Float64(ini) for ini in initial_guess]
        res_am_resid = lso.optimize(lsq_am, float64_guess,
                                    lower=lb, upper=ub,
                                    lso.Dogleg())
        println("\nReal:\n$phi\nEstimated:\n$(res_am_resid.minimizer)")
        print(res_am_resid)
        initial_guess = res_am_resid.minimizer

        print("\n----- Single Shooting Estimator -----\n")
        ss_optim(p) = sum(loss.(abs2.(single_shooting_estimator(p, data, sample_t, ode_fun))))
        res_ss = optimize(ss_optim, lower, upper, initial_guess, Fminbox(NelderMead()))
        println("\nReal:\n$phi\nEstimated:\n$(res_ss.minimizer)\n")
        print(res_ss)

        print("\n----- Single Shooting Residual Estimator -----\n")
        lsq_ss(p) = vec(loss.(abs2.(single_shooting_estimator(p, data, sample_t, ode_fun))))
        float64_guess = [Float64(ini) for ini in initial_guess]
        res_ss_resid = lso.optimize(lsq_ss, float64_guess,
                                    lower=lb, upper=ub,
                                    lso.Dogleg())
        println("\nReal:\n$phi\nEstimated:\n$(res_ss_resid.minimizer)")
        print(lsq_ss)

        print("\n----- Classic Estimator -----\n")
        cost_function = build_loss_objective(oprob,Tsit5(),L2Loss(sample_t,data),
                                     maxiters=10000,verbose=false)
        res_cla = optimize(cost_function,
                            lower, upper,
                            initial_guess)
        println("\nReal:\n$phi\nEstimated:\n$(res_cla.minimizer)")
        println(res_cla)

        #=>
        print("\n----- Lsq Adams-Moulton Estimator -----\n")
        function lsq_dsam_estimator(time_array, phi)
            num_state_variables, num_samples = size(data)

            estimated = zeros(num_samples*num_state_variables)
            estimated = reshape(estimated, (num_state_variables, num_samples))
            estimated[:, 1] = data[:,1] #Initial conditions are stored at x_dot_num's first column

            for i in range(1, stop=num_samples-1)
                delta_t = time_array[i+1] - time_array[i]phi
                x_k_0 = data[:, i]
                x_k_1 = data[:, i+1]

                f_eval_0 = zeros(num_state_variables)
                ode_fun(f_eval_0, x_k_0, phi, 0)
                f_eval_1 = zeros(num_state_variables)
                ode_fun(f_eval_1, x_k_0, phi, 0)

                x_k_1 = x_k_0 + (1/2)*delta_t*(f_eval_0+f_eval_1)
                estimated[:, i+1] = x_k_1
            end
            return estimated
        end

        fit = curve_fit(lsq_dsam_estimator, t, data, res_am.minimizer)

        println("\nReal:\n$phi\nEstimated:\n$(fit.param)")

        oprob = ODEProblem(ode_fun, ini_cond, tspan, fit.param)
        osol  = solve(oprob, Tsit5(), saveat=t)
        plot!(osol_plot, osol)

        print("\n----- Lsq Single Shooting Estimator -----\n")
        function lsq_ss_estimator(time_array, phi)
            tspan = (t[1], t[end])
            ini_cond = data[:,1]
            oprob = ODEProblem(ode_fun, ini_cond, tspan, phi)
            osol  = solve(oprob, Tsi)t5(), saveat=t)
            estimated = reduce(hcat, osol.u)
        end

        fit = curve_fit(lsq_ss_estimator, t, data, res_am.minimizer)

        println("\nReal:\n$phi\nEstimated:\n$(fit.param)")

        oprob = ODEProblem(ode_fun, ini_cond, tspan, fit.param)
        osol  = solve(oprob, Tsit5(), saveat=t)
        plot!(osol_plot, osol)
        <=#end

        #display(osol_plot)
    end
end

#=>
    res_ds = optimize(p -> data_shooting_estimator(p, data, t, ode_fun), lower,
                    upper, initial_guess, Fminbox(LBFGS()))
    println(res)
    println("\nEstimated:\n$(res.minimizer)\nReal:\n$phi\n")
    res_ss = optimize(p -> single_shooting_estimator(p, data, t, ode_fun), lower,
                    upper, res_am.minimizer, Fminbox(NelderMead()))
    #println(res)
    println("\nEstimated:\n$(res_ss.minimizer)\nReal:\n$phi\n")
<=#
