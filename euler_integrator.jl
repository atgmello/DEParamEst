function euler_method(sol, ode_fun, phi, x0, delta_t, m)
    sol[:,1] = x0
    f_eval = Float64[0,0]
    for i in range(1, stop=m-1)
        ode_fun(f_eval, sol[:,i], phi, 0)
        result = f_eval*delta_t + sol[:,i]
        sol[:,i+1] = result
    end
end
