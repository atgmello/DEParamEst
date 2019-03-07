using DifferentialEquations
using ParameterizedFunctions
using DifferentialEquations
using Plots
gr()

#desired_precision = Float64
desired_precision = BigFloat

ode_fun_array = []
bounds_array = Vector{desired_precision}[]
rand_range_array = Vector{desired_precision}[]
phi_array = Vector{desired_precision}[]
ini_cond_array = Vector{desired_precision}[]
#x_t_array = Vector{desired_precision}[]
#delta_t_array = Vector{desired_precision}[]
t_array = Vector{desired_precision}[]
floudas_samples_array = Vector{desired_precision}[]
floudas_samples_times_array = Vector{desired_precision}[]

# ----- Problem 1 -----

function floudas_one(dz_dt, z, phi, t)
    r_1 = phi[1]*z[1]
    r_2 = phi[2]*z[2]

    dz_dt[1] = -r_1
    dz_dt[2] = r_1 - r_2
end

#==>
floudas_one_lambda = @ode_def begin
    r_1 = α*x
    r_2 = β*y

    dx = - r_1
    dy = r_1 - r_2
end α β
<=#

push!(ode_fun_array, floudas_one)
k1 = 5.0035
k2 = 1.
phi = [k1, k2]
push!(phi_array, phi)
push!(bounds_array, [0.,10.])
push!(rand_range_array, [0.,10.])
ini_cond = [1., 0.]
push!(ini_cond_array, ini_cond)
t = range(0, 1, length=10)
push!(t_array, t)

tspan = (0., 1.)
oprob = ODEProblem(floudas_one, ini_cond_array[1], tspan, phi_array[1])
osol  = solve(oprob, Tsit5(), saveat=t_array[1])
print(osol.u)
sol = reduce(hcat, osol.u)
plot(transpose(sol))

# ----- Problem 2 -----

function floudas_two(dz_dt, z, phi, t)
    r_1 = phi[1]*z[1]
    r_2 = phi[2]*z[2]
    r_3 = phi[3]*z[2]
    r_4 = phi[4]*z[3]

    dz_dt[1] = - r_1 + r_2
    dz_dt[2] = r_1 - r_2 - r_3 + r_4
    dz_dt[3] = - r_4 + r_3
end

push!(ode_fun_array, floudas_two)
k1 = 4.
k2 = 2.
k3 = 40.013
k4 = 20.007
phi = [k1, k2, k3, k4]
push!(phi_array, phi)
push!(bounds_array, [0.,50.])
push!(rand_range_array, [0.,50.])
ini_cond = [1., 0., 0.]
push!(ini_cond_array, ini_cond)
t = range(0, 1, length=10)
push!(t_array, t)

tspan = (0., 1.)
oprob = ODEProblem(floudas_two, ini_cond_array[2], tspan, phi_array[2])
osol  = solve(oprob, Tsit5())
plot(osol)

# ----- Problem 3 -----

function floudas_three(dz_dt, z, phi, t)
    r_1 = phi[1]*z[1]^2
    r_2 = phi[2]*z[2]
    r_3 = phi[3]*z[1]^2

    dz_dt[1] = - r_1 - r_3
    dz_dt[2] = r_1 - r_2
end

push!(ode_fun_array, floudas_three)
k1 = 12.214
k2 = 7.9798
k3 = 2.2216
phi = [k1, k2, k3]
push!(phi_array, phi)
push!(bounds_array, [0.,20.])
push!(rand_range_array, [0.,20.])
ini_cond = [1., 0.]
push!(ini_cond_array, ini_cond)
t = range(0, 1, length=10)
push!(t_array, t)

tspan = (0., 1.)
oprob = ODEProblem(floudas_three, ini_cond_array[3], tspan, phi_array[3])
osol  = solve(oprob, Tsit5())
plot(osol)

# ----- Problem 4 -----

function floudas_four(dz_dt, z, phi, t)
    dz_dt[1] = phi[1]*(126.2 - z[1])*(91.9 - z[1])^2 - phi[2]*z[1]^2
end

push!(ode_fun_array, floudas_four)
k1 = 4.5704*10^(-6)
k2 = 2.7845*10^(-4)
phi = [k1, k2]
push!(phi_array, phi)
push!(bounds_array, [0,.1])
push!(rand_range_array, [0,.1])
ini_cond = [0.]
push!(ini_cond_array, ini_cond)
t = range(0, 40, length=10)
push!(t_array, t)

tspan = (0., 40.)
oprob = ODEProblem(floudas_four, ini_cond_array[4], tspan, phi_array[4])
osol  = solve(oprob, Tsit5())
plot(osol)

# ----- Problem 5 -----

function floudas_five(dz_dt, z, phi, t)
    dz_dt[1] = - (2*phi[1] - ((phi[1]*z[2])/((phi[2] + phi[5])*z[1] + z[2])) + phi[3] + phi[4])*z[1]
    dz_dt[2] = ((phi[1]*z[1])*(phi[2]*z[1] - z[2]))/((phi[2] + phi[5])*z[1] + z[2]) + phi[3]*z[1]
    dz_dt[3] = ((phi[1]*z[1])*(z[2] + phi[5]*z[1]))/((phi[2] + phi[5])*z[1] + z[2]) + phi[4]*z[1]
end

push!(ode_fun_array, floudas_five)
k1 = 5.2407
k2 = 1.2176
k3 = 0.
k4 = 0.
k5 = 0.
phi = [k1, k2, k3, k4, k5]
push!(phi_array, phi)
push!(bounds_array, [0.,100.])
push!(rand_range_array, [0.,100.])
ini_cond = [1., 0., 0.]
push!(ini_cond_array, ini_cond)
t = range(0, 1.2, length=10)
push!(t_array, t)

tspan = (0., 1.2)
oprob = ODEProblem(floudas_five, ini_cond_array[5], tspan, phi_array[5])
osol  = solve(oprob, Tsit5())
plot(osol)

# ----- Problem 6 -----

function floudas_six(dz_dt, z, phi, t)
    dz_dt[1] = phi[1]*z[1]*(1-z[2])
    dz_dt[2] = phi[2]*z[2]*(z[1]-1)
end

push!(ode_fun_array, floudas_six)
k1 = 3.2434
k2 = 0.9209
phi = [k1, k2]
push!(phi_array, phi)
push!(bounds_array, [0.,10.])
push!(rand_range_array, [0.,10.])
ini_cond = [1.2, 1.1]
push!(ini_cond_array, ini_cond)
t = range(0, 10, length=10)
push!(t_array, t)

tspan = (0., 10)
oprob = ODEProblem(floudas_six, ini_cond_array[6], tspan, phi_array[6])
osol  = solve(oprob, Tsit5())
plot(osol)
