using Plots
using DifferentialEquations
plotly()

# ----- Plotting -----
for i in 1:6
    p = get_ode_problem(get_problem_keys(i))
    plot_canvas = scatter(p.t',p.data', title="Problem $i")
    display(plot_canvas)
end

for i in 7:12
    p = problem_set[i]
    plot_canvas = scatter(p.t,p.data', title="Problem $i")
    display(plot_canvas)
end

phi = BigFloat[5.0035, 1]
ini_cond = BigFloat[1,0]
delta_t = BigFloat(.1)
m = 50
t_span = range(0, length=m+1, stop=m*delta_t)
sol_two = ones( BigFloat, length(phi), m+1)
euler_method(sol_two, floudas_one, phi, ini_cond, delta_t, m)

plot(t_span,transpose(sol_two))

tspan = (0., 5.)
oprob = ODEProblem(floudas_one, ini_cond, tspan, phi)
osol  = solve(oprob, Tsit5())
plot(osol)
print(osol.u[end])

# Comparando Precisão!

using JuMP
using Ipopt
using Optim

my_square(x) = x^2
my_f_JuMP(x,y) = (x - 1)^2 + (y - 2)^2

model = Model(with_optimizer(Ipopt.Optimizer))

register(model, :my_f, 2, my_f, autodiff=true)

@variable(model, x[1:2])
@NLobjective(model, Min, my_f(x[1], x[2]))

result_Ipopt = optimize!(model)
print("x: ", value(x[1]), " y: ", value(x[2]))

using Optim

rosenbrock(x) =  (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
my_f(x) = (x[1] - 1)^2 + (x[2] - 2)^2

x0 = [BigFloat(0), BigFloat(0)]
x0_bf = zeros(2)

result = optimize(my_f, x0, BFGS())
println(result,"\n")

result_bf = optimize(my_f, x0_bf, BFGS())
println(result_bf,"\n")


# Testando Argumentos

function my_new_f(x, y)
    return (x - 1)^2 + (y - 2)^2
end

my_new_f(0,2)
function my_new_square(x)
    return x^2
end

my_new_square(2)
my_new_f(2, my_new_square(2))

x0 = [BigFloat(0), BigFloat(0)]

res = optimize(b -> my_new_f(b, 2), .0)
println(result,"\n")


x = [1.0, 2.0, 3.0]
y = 2 * x + [-0.3, 0.3, -0.1]
function sqerror(betas, X, Y)
    err = 0.0
    for i in 1:length(X)
        pred_i = betas[1] + betas[2] * X[i]
        err += (Y[i] - pred_i)^2
    end
    return err
end
sqerror([1, 1], x, y)
res = optimize(b -> sqerror(b, x, y), [0.0, 0.0])

# ----- Testando estrutura de dados -----

using RecursiveArrayTools # for VectorOfArray

desired_precision = BigFloat
a = Vector{desired_precision}[]
b = BigFloat(1.4)
c = 1.3
push!(a, [BigFloat(b),BigFloat(b)])
print(a)
push!(a, [c,c])
push!(a, [b,c])
print(a)
typeof(a[3][2])

ode_fun = ode_fun_array[1]
t = t_array[1]
phi = phi_array[1]
ini_cond = ini_cond_array[1]

tspan = (t[1], t[end])
oprob = ODEProblem(ode_fun, ini_cond, tspan, phi)
osol  = solve(oprob, Tsit5(), saveat=t)
randomized = VectorOfArray([(osol(t[i]) + .01randn(2)) for i in 1:length(t)])
print(randomized)
data = convert(Array,randomized)
print(data)
print(data[1][1])
print(reshape(data,(1,length(ini_cond)*length(t))))

# ----- Testing LsqFit -----

using LsqFit

# a two-parameter exponential model
# x: array of independent variables
# p: array of model parameters
function model(x, p)
    res = []
    for i in x
        push!(res, p[1]*exp(-i.*p[2]))
    end
    return res
end

# some example data
# xdata: independent variables
# ydata: dependent variable
xdata = range(0, stop=10, length=20)
#ydata = model(xdata, [1.0 2.0]) + 0.01*randn(length(xdata))
ydata = model(xdata, [1.0 2.0])
plot(ydata)
fit = curve_fit(model, xdata, ydata, [0.5, 0.5])
print(fit.param)
# fit is a composite type (LsqFitResult), with some interesting values:
#   fit.dof: degrees of freedom
#   fit.param: best fit parameters
#   fit.resid: residuals = vector of residuals
#   fit.jacobian: estimated Jacobian at solution

# We can estimate errors on the fit parameters,
# to get 95% confidence error bars:
errors = margin_error(fit, 0.95)
print(errors)

# And another one

xdata = [[42.7908, 240.168, 32.799],
 [43.262, 346.383, -0.296673],
 [43.842, 332.0, -32.1631]   ,
 [44.184, 324.176, 35.1956]  ,
 [44.8529, 270.56, -21.4661] ,
 [44.8649, 184.976, 11.9061] ,
 [45.1269, 328.77, -10.8362] ,
 [45.4122, 248.579, 37.3079] ,
 [45.4975, 156.928, -76.5363],
 [45.8181, 347.873, -9.41383]];

ydata = [42.37397477812974,
 43.644649744028534,
 43.54054864852959,
 43.93823563647773,
 44.362474498426906,
 45.063235860732235,
 44.30769385086685,
 44.3473803820133,
 45.76719596019439,
 45.74372210459567];

function angular_sepration(l2,b2)
    l1,b1 = 310,-10
    l1 = deg2rad(l1)
    b1 = deg2rad(b1)
    l2 = deg2rad(l2)
    b2 = deg2rad(b2)
    b11 = (pi/2) - b1
    b22 = (pi/2) - b2
    sep = acos( cos(b11) * cos(b22) + sin(b11) * sin(b22) * cos(l1 - l2))
    return rad2deg(sep)
end;

function fit_func(xdatas,p)
    fit_d,fit_l,fit_b = p

    return_list = []

    for parameter in xdatas
        mu = parameter[1]
        l = parameter[2] # degree
        b = parameter[3] # degree

        angular = angular_sepration(l,b)
        cos_theta = cosd(angular)

        mu_diople = mu * (1 - fit_d * cos_theta)
        append!(return_list,mu_diople)
    end
    return return_list
end

p0 = [0.001,310,-10]
p0 = [0.01,210,-1]
fit = curve_fit(fit_func,xdata,ydata,p0)
print(fit.param)

# And another one

x = -16:0.5:35
y = 19.4./((x .- 7).^2 .+ 15.8) + randn(size(x))./10;

model(x, p) = p[1]./((x.-p[2]).^2 .+ p[3])

fit = curve_fit(model, x, y, [25.,10.,10.])
p = fit.param
f = p[1]./((x.-p[2]).^2 .+ p[3])
p = plot(x, y, marker="d")
plot!(p, x, f, marker="-")

# And another one

x = -16:0.5:35
y = [19.4./((x .- 7).^2 .+ 15.8) + randn(size(x))./10, 2*19.4./((x .- 7).^2 .+ 3*15.8) + randn(size(x))./10];

#y_flat = hcat(transpose(y)...)
#print(y_flat)

#model(x, p) = hcat(transpose([p[1]./((x.-p[2]).^2 .+ p[3]), 2*p[1]./((x.-p[2]).^2 .+ 3*p[3])])...)
model(x, p) = [p[1]./((x.-p[2]).^2 .+ p[3]), p[1]./((x.-p[2]).^2 .+ p[3])]


fit = curve_fit(model, x, y, [25.,10.,10.])
p = fit.param
f = [p[1]./((x.-p[2]).^2 .+ p[3]), p[1]./((x.-p[2]).^2 .+ p[3])]
p = scatter(x, y)
scatter!(p, x, f)


# ----- And another one -----

# a two-parameter exponential model
# x: array of independent variables
# p: array of model parameters
function model(x, p)
    res = []
    for i in x
        push!(res, p[1]*exp(-i.*p[2]))
    end
    return res
end

# some example data
# xdata: independent variables
# ydata: dependent variable
xdata = range(0, stop=10, length=20)
#ydata = model(xdata, [1.0 2.0]) + 0.01*randn(length(xdata))
ydata = model(xdata, [1.0 2.0])
plot(ydata)
p0 = [0.5, 0.5]
fit = curve_fit(model, xdata, ydata, p0; lower = zeros(2), upper = ones(2))
print(fit.param)
# fit is a composite type (LsqFitResult), with some interesting values:
#   fit.dof: degrees of freedom
#   fit.param: best fit parameters
#   fit.resid: residuals = vector of residuals
#   fit.jacobian: estimated Jacobian at solution

# We can estimate errors on the fit parameters,
# to get 95% confidence error bars:
errors = margin_error(fit, 0.95)
print(errors)

# --- And another one ---
@. multimodel(x, p) = p[1]*exp(-x[:, 1]*p[2]+x[:, 2]*p[3])
print(multimodel([.2 .2], [.1, .2, .3]))
print(length([.2 .2]))
print(length([.2, 2.]))
a = [2, 2]
b = [2 2]
print(a)
print(b)
print(transpose(a))

# ----- Testing Interact.jl ----
using Interact

ui = button()
display(ui)
on(n -> println("Hello!"), ui)

using Blink

ui = slider(1:100, label = "To what extent?", value = 33) # value is the number selected
w = Window()
body!(w, "hello")
p = plot(rand(2,2))
body!(w, p)
body!(w, ui)

# ----- Julia Discourse Question -----
using LsqFit

function floudas_one(dz_dt, z, phi, t)
    r_1 = phi[1]*z[1]
    r_2 = phi[2]*z[2]

    dz_dt[1] = -r_1
    dz_dt[2] = r_1 - r_2
end

ode_fun = floudas_one

data = Float64[1.0 0.57353 0.328937 0.188654 0.108198 0.0620545 0.0355906 0.0204132 0.011708 0.00671499;
        0.0 0.401566 0.589647 0.659731 0.666112 0.639512 0.597179 0.54867 0.499168 0.451377]
t = Float64[0.0, 0.111111, 0.222222, 0.333333, 0.444444, 0.555556, 0.666667, 0.777778, 0.888889, 1.0]

function lsq_ss_estimator(time_array, phi)
    tspan = (t[1], t[end])
    ini_cond = data[:,1]
    oprob = ODEProblem(ode_fun, ini_cond, tspan, phi)
    osol  = solve(oprob, Tsit5(), saveat=t)
    estimated = reduce(hcat, osol.u)
    return vec(estimated)
end

p0 = [5., 5.]
fit = curve_fit(lsq_ss_estimator, t, vec(data), p0, lower=[0., 0.], upper=[10., 10.])
print(fit.param)

lsq_wraper(t,p) = soft_l1(vec(adams_moulton_estimator(p, data, t, ode_fun)))

fit = curve_fit(lsq_wraper, t, vec(data), p0, lower=[0., 0.], upper=[10., 10.])
print(fit.param)

# ----- Blackbox solver -----

using BlackBoxOptim
using Revise
includet("./problem_set.jl")
includet("./objective_function.jl")
includet("./utils.jl")
import .ProblemSet: get_problem, get_problem_key, DEProblem
import .ObjectiveFunction: data_shooting, single_shooting, soft_l1, huber
import .Utils: Trace, rand_guess, make_trace, get_range_traces, add_noise,
				filter_outlier, fill_trace, scale_eval,
				max_diff_states, diff_calc, step_success_rate, success_rate, box_data, box_scatter_plot,
				get_plot_data, oe_plots, sr_plots, error_plots
using DifferentialEquations
using LinearAlgebra


prob =  get_problem("goodwin_oscillator")
ds_fun(x) = data_shooting(x, prob.data, prob.t, prob.fun)
ss_fun(x) = single_shooting(x, prob.data, prob.t, prob.fun)

res = bboptimize(ss_fun;
        SearchRange = [(b[1],b[2]) for b in [getindex.(prob.bounds, i) for i in 1:length(prob.phi)]],
        NumDimensions = length(prob.phi),
        MaxSteps=10^7)
println(prob.phi)

function rosenbrock2d(x)
  return (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
end

res = bboptimize(rosenbrock2d; SearchRange = (-5.0, 5.0), NumDimensions = 2)
best_candidate(res)

function michaelis_menten_model(concentration, Vm, K)
    (Vm * concentration) / (K + concentration)
end

# MicMen data is taken from the README of the NLReg.jl package: https://github.com/dmbates/NLreg.jl
# 1st column is concentration and 2nd column is the Rate
MicMenData = [
        0.02 76;
        0.02 47;
        0.06 97;
        0.06 107;
        0.11 123;
        0.11 139;
        0.22 159;
        0.22 152;
        0.56 191;
        0.56 201;
        1.1  207;
        1.1  200]

MicMenConcentration = MicMenData[:, 1];
MicMenRate = MicMenData[:, 2];

# Fitness function takes a vector of Vm and K and calculates the RSS
function mic_men_fitness(params)
    Vm, K = params
    yhat = Float64[michaelis_menten_model(c, Vm, K) for c in MicMenConcentration]
    sum(abs2.(MicMenRate .- yhat))
end

mic_men_fitness([1.1,2.2])

result = bboptimize(mic_men_fitness;
                SearchRange = (-1000.0, 1000.0),
                NumDimensions = 2, MaxSteps = 1e4)
Vm, K = best_candidate(result)
RSS = best_fitness(result)

println("NLReg.jl uses specific MicMen object and fit method to find:")
println("Vm = 212.684, K = 0.0641212, RSS = 1195.45")
println("\nUsing BlackBoxOptim.jl we find:")
println("Vm = $(Vm), K = $(K), RSS = $(RSS)")# ----- Testando problema Floudas -----


println("\n----- Solving problem $i with $sample_size samples -----\n")
i = 1
p_solve = problem_set[i]
ode_fun = p_solve.fun
t = p_solve.t
phi = p_solve.phi
ini_cond = p_solve.data[:,1]
bounds = p_solve.bounds
true_data = p_solve.data

#sample_t = range(t[1], stop=t[end], length=sample_size)
tspan = (t[1], t[end])
oprob = ODEProblem(ode_fun, ini_cond, tspan, phi)
osol  = solve(oprob, Tsit5(), saveat=t)
osol_plot = scatter(osol)
display(osol_plot)
data = reduce(hcat, osol.u)
lb = vec([bounds[1] for i in 1:length(phi)])
ub = vec([bounds[end] for i in 1:length(phi)])


initial_guess = desired_precision[]
for i in range(1, stop=length(phi))
    rand_num = rand(Uniform(bounds[1], bounds[end]))
    push!(initial_guess, rand_num)
end
print("Initial guess:\n$initial_guess\n")

lower = desired_precision[]
upper = desired_precision[]

for i in 1:length(phi)
    push!(lower, bounds[1])
    push!(upper, bounds[2])
end

initial_guess = initial_guess

oprob = ODEProblem(ode_fun, ini_cond, tspan, [1 10])

cost_function = build_loss_objective(oprob,Tsit5(),L2Loss(t,true_data),
                             maxiters=10000,verbose=false)

cost_function = multiple_shooting_objective(oprob,lsoda(),L2Loss(t,true_data))
cost_functio([5.0 1.0])

using PyCall

py_opt = pyimport("scipy.optimize")
py_int = pyimport("scipy.integrate")

i = 1
data = floudas_samples_array[i]
t = floudas_samples_times_array[i]
ode_fun = ode_fun_array[i]
phi = phi_array[i]
ini_cond = ini_cond_array[i]

tspan = (t[1], t[end])
oprob = ODEProblem(ode_fun, ini_cond, tspan, phi)
osol  = solve(oprob, lsoda(), saveat=reduce(vcat, t))
osol_plot = scatter(osol)
display(osol_plot)

function wrapper(x)
    res = single_shooting_estimator_residuals(x, data, t, ode_fun)
    return res
end

function single_shooting_estimator_residuals(phi, data, t, ode_fun)
    function wrapped_ode_fun(z, t, phi)
        dz_dt = zeros(length(t)*length(z))
        ode_fun(dz_dt, z, phi, t)
        return dz_dt
    end
    estimated_values = py_int.odeint(wrapped_ode_fun, data[:,1], t)

    if plot_estimated
        p = scatter(transpose(estimated), title="Plot of $phi")
        display(p)
    end
    residuals = (data-estimated)
    return reduce(vcat, residuals)
end

function wrapped_ode_fun(z, t, phi)
    dz_dt = zeros(length(t)*length(z))
    ode_fun(dz_dt, z, phi, t)
    return dz_dt
end

function py_ode_fun_test_p_one(Z, t, Phi)
    #a, b, x = x
    r_1 = Phi[1]*Z[1]
    r_2 = Phi[2]*Z[2]

    dZ1_dt = -r_1
    dZ2_dt = r_1 - r_2

    res = [dZ1_dt, dZ2_dt]
    return res
end

a = py_ode_fun_test_p_one([1., 1], 0, phi)
b = wrapped_ode_fun([1., 1.], 0, phi)
print(a == b)

estimated_values = py_int.odeint(py_ode_fun_test_p_one, [0., 1.], [1, 2, 3])

print(single_shooting_estimator_residuals(phi, data, t, ode_fun))


single_shooting_estimator_residuals(phi, data[1,:], t, ode_fun)

print(wrapper(phi))
print(single_shooting_estimator_residuals(phi, data, t, ode_fun))
res = py_opt.least_squares(wrapper, phi, loss="linear")
print(res)


function F(x)
 f1 = 4 - x[1]^2
 return f1
end

x0 = 2.
x = py_opt.fsolve(F, [x0])
print(x)

x = py_opt.minimize(F, x0)
print(x)

[4.00273, 2.01261, 39.9663, 19.9744]

# ----- Testando JuMP

using Ipopt
using JuMP
model = Model(with_optimizer(Ipopt.Optimizer))
@variable(model, x, start = BigFloat(0.0))
@variable(model, y, start = BigFloat(0.0))

@NLobjective(model, Min, (1 - x)^2 + 100 * (y - x^2)^2)

optimize!(model)
println("x = ", value(x), " y = ", value(y))

# adding a (linear) constraint
@constraint(model, x + y == BigFloat(10))
optimize!(model)
println("x = ", value(x), " y = ", value(y))

# ----- Checking lower and upper bounds -----

problem_set[2].bounds[1]

f(x) = (x[1] - x[2])^2
f([10., 9.])
res_lso_am = lso.optimize(f, [.5, 1.5], lso.Dogleg(), lower=[0., 1.], upper=[1., 2.])

function rosenbrock(x)
	[1 - x[1] 100 * (x[2]-x[1]^2)]
end
rosenbrock([2, 2])
rosenbrock([2 2])
rosenbrock([2; 2])

# Works
[1.9, 1.4]
reshape([1.9 1.4], 2)

# Does not
transpose([1.9, 1.4])
[1.9 1.4]

lso.optimize(rosenbrock, reshape([1.9 1.4], 2), lso.Dogleg(), lower=[-.1, -.5], upper=[2., 1.5])
lso.optimize(rosenbrock, [1.9, 1.4], lso.LevenbergMarquardt(), lower=[-.1, -.5], upper=[2., 1.5])

reshape(convert(Array{Float64}, [2 2]), 2)


reshape([i for i in 1:10], 10)

# ----- Problem B4 CHO (8) causes errors -----

using LSODA
i = 7
p_solve = problem_set[i]
t = p_solve.t
tspan = (t[1], t[end])
ini_cond = p_solve.data[:,1]
ode_prob = ODEProblem(p_solve.fun, ini_cond, tspan, p_solve.phi)
# Works with lsoda but not with others
ode_sol  = solve(ode_prob, Tsit5())
plot(ode_sol)

x0 = zeros(Float64, length(p_solve.data))
p_solve.fun(x0, p_solve.data, p_solve.phi, t[2])
x0

# ----- Avoid domain error -----
# DomainError(-6.516112690059482, "log will only return a complex result if called with a complex argument. Try log(Complex(x)).")
f(x) = x[2] > 5.0 ? Inf : abs2(x[1]^2-x[2])
f(x) = abs2(x[1]^2-x[2])
f([1., 6.])

res = opt.optimize(f, [0.0, 0.0], [10.0, 10.0], [7., 0.])
f(res.minimizer)

b = zeros(10)
function fun_bb(dx_dt, x, par, t)
    if x == 0
        for i in 1:length(dx_dt)
            dx_dt[i] = Inf
        end
        return
    end
    for i in 1:length(dx_dt)
        dx_dt[i] = 1
    end
end
c = fun_bb(b, 0, 0, 0)
c
b

# ----- Testing BBox -----
import BlackBoxOptim
const bbo = BlackBoxOptim

function rosenbrock2d(x)
  return (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
end

res = bbo.bboptimize(rosenbrock2d; SearchRange = (-5.0, 5.0), NumDimensions = 2)
bbo.best_candidate(res)


# ----- Testing Stochastic Search -----
using NODAL
t = range(t[1], stop=t[end], length=10)
tspan = (t[1], t[end])
ini_cond = p_solve.data[:,1]
ode_prob = ODEProblem(p_solve.fun, ini_cond, tspan, p_solve.phi)
ode_sol  = solve(ode_prob, lsoda(), saveat=reduce(vcat, t))
data_original = reduce(hcat, ode_sol.u)
ode_data = copy(data_original)
res_lso_am = two_stage_method(ode_prob,t,ode_data; mpg_autodiff=true)

test_fun_ss = single_shooting_estimator(phi, ode_data, t, ode_fun)
sum(abs2.(test_fun_ss))

test_fun_adapt = sm_adaptative_shooting(phi, ode_data, t, ode_fun)
sum(abs2.(test_fun))

test_fun_sma = single_multiple_adams_shooting(phi, ode_data, t, ode_fun)
sum(abs2.(test_fun_sma))

test_fun_mean = sm_mean_shooting(phi, ode_data, t, ode_fun)
sum(abs2.(test_fun_mean))

test_fun_am = adams_moulton_estimator(phi, ode_data, t, ode_fun)
sum(abs2.(test_fun_am))

#function noldal_solve(p_solve, fun)

fun = single_shooting_estimator
i = 1
p_solve = problem_set[i]
ode_fun = p_solve.fun
t = p_solve.t
phi = p_solve.phi
bounds = p_solve.bounds
ode_data = p_solve.data

nodal_parameters = Dict{Symbol,Any}([(Symbol("phi_$i"), rand(0.01:100.0)) for i in 1:length(phi)])
nodal_parameters

function nodal_f(x::Configuration, parameters::Dict{Symbol,Any})
    nodal_phi = Float64[]
    for i in 1:length(phi)
        key = String(collect(keys(parameters))[i])
        push!(nodal_phi, x[key].value)
    end
    res = fun(nodal_phi,ode_data,t,ode_fun)
    res = sum(abs2.(res))
    return res
end

conf = Configuration([FloatParameter(0.0001, 100.0, rand(0.01:0.01:100),
                                    String(collect(keys(nodal_parameters))[i])) for i in 1:length(phi)],
                               "nodal_f")

tuning_run = Run(cost                = nodal_f,
                 cost_arguments = nodal_parameters,
                 starting_point      = conf,
                 stopping_criterion  = elapsed_time_criterion,
                 report_after        = 10,
                 reporting_criterion = elapsed_time_reporting_criterion,
                 duration            = 30,
                 methods             = [:simulated_annealing 1])
optimize(tuning_run)

result = take!(tuning_run.channel)
result.is_final
print(result)

print("End")

while !result.is_final
    result = take!(tuning_run_other.channel)
    println(result)
end
#close(tuning_run.channel)
result

print("h")

# ----- NODAL Example -----

function rosenbrock(x::Configuration, parameters::Dict{Symbol, Any})
    return (1.0 - x["i0"].value)^2 + 100.0 * (x["i1"].value - x["i0"].value^2)^2
end

configuration = Configuration([FloatParameter(-2.0, 2.0, 0.0, "i0"),
                               FloatParameter(-2.0, 2.0, 0.0, "i1")],
                               "rosenbrock_config")

tuning_run = Run(cost                = rosenbrock,
                starting_point      = configuration,
                stopping_criterion  = elapsed_time_criterion,
                report_after        = 10,
                reporting_criterion = elapsed_time_reporting_criterion,
                duration            = 60,
                methods             = [:simulated_annealing 1])

optimize(tuning_run)
result = take!(tuning_run.channel)
print(result)

# ----- Plots -----
y = rand(10)
plot(y, annotations = (3,y[3],text("this is #3",:left)), leg=false)
annotate!([(5, y[5], text("this is #5",16,:red,:center)),
          (10, y[10], text("this is #10",:right,20,"courier"))])
scatter!(range(2, stop=8, length=6), rand(6), marker=(50,0.2,:orange),
         series_annotations = ["series","annotations","map","to","series",
                               text("data",:green)])


# ----- Testing NLOpt -----
import NLopt
const nlo = NLopt

funcs = [adams_moulton_estimator, single_shooting_estimator, sm_adaptative_shooting, sm_adaptative_hard_shooting]
fnames = ["adams_moulton_estimator", "single_shooting_estimator", "sm_adaptative_shooting", "sm_adaptative_hard_shooting"]

function nlopt_solve(p_solve, fun, rand_phi)
    ode_fun = p_solve.fun
    ode_data = p_solve.data
    t = p_solve.t
    bounds = p_solve.bounds
    phi = p_solve.phi

    nlop_opt = nlo.Opt(nlo.:GN_CRS2_LM, length(phi))
    nlo.upper_bounds!(nlop_opt, bounds[end])
    nlo.lower_bounds!(nlop_opt, bounds[1])
    nlo.xtol_rel!(nlop_opt,1e-9)
    #nlo.local_optimizer!()
    nlo.maxtime!(nlop_opt, 60)

    myfunc(x,y) = sum(abs2.(fun(x,ode_data,t,ode_fun)))
    nlo.min_objective!(nlop_opt, myfunc)

    (minf,minx,ret) = nlo.optimize(nlop_opt, rand_phi)
    numevals = nlop_opt.numevals # the number of function evaluations
    println("Got\t$minf\nAt\t$minx\nAfter\t$numevals iterations (returned $ret)\n")
end

for j in 1:6
    println("\n----- Solving problem $j -----\n")
    p_solve = problem_set[j]
    p_rand = rand_guess(p_solve.bounds)
    println("Initial guess:\n$p_rand\n")
    for i in 1:length(funcs)
        println("--- Solving for $(fnames[i]) ---")
        nlopt_solve(p_solve, funcs[i], p_rand)
    end
end


# --- Testing ForwardDiff AD -----
import ForwardDiff
const fdf = ForwardDiff

f(x) = x.^2 .- 1
function f(x)
    return (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
end
#od = opt.OnceDifferentiable(f, [0.0]; autodiff = :forward)
inner_optimizer = opt.LBFGS()

res_obj = 0
timed = @elapsed res_obj = opt.optimize(f, [10.0, 10.0], opt.NewtonTrustRegion(); autodiff=:forward)
res_obj
timed

timed = @elapsed res_obj = opt.optimize(f, [10.0, 10.0], opt.LBFGS(); autodiff=:forward)
timed
res_obj

timed = @elapsed res_obj = opt.optimize(f, BigFloat[10.0, 10.0], opt.NewtonTrustRegion(); autodiff=:forward)
timed
res_obj

timed = @elapsed res_obj = opt.optimize(f, [-1.0, -1.0],
	[40.0, 40.0], [10.0, 10.0], opt.Fminbox(opt.LBFGS()); autodiff=:forward)
res_obj
timed

g = x->fdf.gradient!(r, f, x)
g = x->fdf.gradient(f, x)
timed = @elapsed res_obj = opt.optimize(f, g, [-1.0, -1.0],
	[40.0, 40.0], [10.0, 10.0], opt.Fminbox(opt.LBFGS()))
res_obj
timed

timed = @elapsed res_obj = opt.optimize(f, [-1.0, -1.0],
	[40.0, 40.0], [10.0, 10.0], opt.Fminbox(opt.LBFGS()))
timed
res_obj

# --- Testing missing states optimization ---
include("./utils.jl")
using .Utils
include("./floudas_problems.jl")
using .ODEProblems

i_prob = "floudas_1"
p_solve = get_problem(i_prob)
fun = p_solve.fun
phi = p_solve.phi
bounds = p_solve.bounds
ini_cond = p_solve.data[:,1]

t = p_solve.t
t = range(t[1], stop=t[end], length=length(t))
tspan = (t[1], t[end])
ode_prob = ODEProblem(p_solve.fun, ini_cond, tspan, phi)
ode_sol  = solve(ode_prob, AutoVern9(Rodas5()), saveat=reduce(vcat, t))
data_original = reduce(hcat, ode_sol.u)
data = copy(data_original)

# -- Estimation using only partial data?? --
plot_data = plot(t,data')

states = [1,2]
unknown_states = []
known_states = setdiff(1:length(states),unknown_states)
data = data[filter(x -> x in known_states, states),:]

plot_data = plot(t,data')

var = 0.05
add_noise!(data, var)
plot_data = plot(t,data')
display(plot_data)

linear(x) = x
#loss = soft_l1
loss = linear

lb = []
append!(lb,bounds[1])
append!(lb,0.0)
lb = convert(Array{Float64,1},lb)

ub = []
append!(ub,bounds[end])
append!(ub,100.0)
ub = convert(Array{Float64,1},ub)

# -- Why is p0 [4]???
p0 = rand_guess(bounds)
push!(p0, 0.001)

p0 = [5.0,1.0,1.0]

# Add guess for initial value of state

SAMIN_options = opt.Options(x_tol=10^-6, f_tol=10^-12, iterations=10^4, store_trace=true, show_trace=true)
Grad_options = opt.Options(x_tol=10^-6, f_tol=10^-12, iterations=10^4, store_trace=true, show_trace=true)
inner_optimizer = opt.LBFGS()
function time_res!(f, obj, opt_type)
	if opt_type == "G"
		timed = @elapsed res_obj = opt.optimize(f, lb, ub, p0, opt.SAMIN(verbosity=0), SAMIN_options)
	else
    	#od = opt.OnceDifferentiable(adams_moulton_error, p0; autodiff = :forward)
    	timed = @elapsed res_obj = opt.optimize(f, lb, ub, p0, opt.Fminbox(inner_optimizer), Grad_options)
    	#timed = @elapsed res_obj = opt.optimize(f, p0, store_trace=true)
	end
    println("Trace")
    println(opt.trace(res_obj))
    println("F Trace")
    println(opt.f_trace(res_obj))
    #dist = diff_states(fun, ini_cond, (t[1],t[end]), phi, res_obj.minimizer[1:length(phi)])
    opt.trace(res_obj)
end

res_am = Dict("time" => [], "error" => [])
adams_moulton_error(p) = sum(
                            abs2.(
                                loss.(data_shooting(p, data,
                                                                t, p_solve.fun;
                                                                unknown_vars=unknown_states,
                                                                plot_estimated=false)
                                                                )
                                    )
                            )

single_shooting_error(p) = sum(
                            abs2.(
                                loss.(single_shooting(p, data,
                                                                t, p_solve.fun;
                                                                unknown_vars=unknown_states,
                                                                plot_estimated=false)
                                                                )
                                    )
                            )


trace = time_res!(adams_moulton_error, res_am, "L")
trace
plot(log10.(opt.f_trace(res)))

trace_err = []
trace_time = []
for i in 1:length(trace)
    append!(trace_err, parse(Float64, split(string(trace[i]))[2]))
    append!(trace_time, parse(Float64, split(string(trace[i]))[end]))
end

plot(log10.(cumsum(trace_time)), log10.(trace_err/trace_err[end]))

trace = time_res!(single_shooting_error, res_am, "L")
trace
print(trace)
plot(log10.(opt.f_trace(res)))
opt.f_trace(res)
opt.minimum(res)

trace_err = []
trace_time = []
for i in 1:length(trace)
    append!(trace_err, parse(Float64, split(string(trace[i]))[2]))
    append!(trace_time, parse(Float64, split(string(trace[i]))[2]))
end

plot(log10.(cumsum(trace_time)), log10.(trace_err/trace_err[end]))

transpose(data[:,1])
data[4:end]

a = [
    1 2 3;
    1 2 3
    ]
typeof(a)
b = [4,5,6]
bb = b[:,:]
typeof(bb)

c = zeros((5,3))
c[:,1] = vcat(a[:,1], b)
c
vcat(a[:,1],b[0])
b[end:end]
d = zeros(size(p_solve.data))
d[:,1] = p_solve.data[:,1]
d


# --- Testing DAE ---
using DifferentialEquations
using Sundials
using Plots
plotlyjs()
gr()

phi_stiff = [0.04, 3e7,1e4]
phi = [4.0,3.0,1.0]

function f(out,du,u,p,t)
  out[1] = - 4u[1]              + u[2]*u[3] - du[1]
  out[2] = + 4u[1] - 3*u[2]^2 - u[2]*u[3] - du[2]
  out[3] = u[1] + u[2] + u[3] - 1.0
end

u₀ = [1.0, 0, 0]
du₀ = [-0.04, 0.04, 0.0]
tspan = (0.0,2.5)
tspan_stiff = (0.0,1e6)

differential_vars = [true,true,false]
prob = DAEProblem(f,du₀,u₀,tspan,p=phi_stiff,differential_vars=differential_vars)
prob_stiff = DAEProblem(f,du₀,u₀,tspan_stiff,p=phi_stiff,differential_vars=differential_vars)

sol = solve(prob,IDA())
sol_stiff = solve(prob_stiff,IDA())
# Normal
plot(sol, tspan=(tspan[1], tspan[end]))
# Stiff
plot(sol_stiff, xscale=:log10, tspan=(1e-6, 1e5), layout=(3,1))


function f(du,u,p,t)
  # Algebrical constraints
  du[3] = - u[1] - u[2] + 1.0
  u[3] = du[3]
  # Differential equations
  du[1] = - 4u[1]              + u[2]*u[3] - du[1]
  du[2] = + 4u[1] - 3*u[2]^2 - u[2]*u[3] - du[2]
end

u0 = [1.0, 0, 0]
tspan = (0.0,2.5)
delta_t = 0.005
num_samples = trunc(Int, tspan[end]/delta_t)
delta_t_stiff = 0.5
num_samples_stiff = trunc(Int, tspan_stiff[end]/delta_t_stiff)

differential_vars = [true,true,false]
ode_fun = f
num_states = 3

estimated = zeros(num_states,num_samples)
estimated[:, 1] = u0

for i in 1:num_samples-1
    x_k_0 = estimated[:, i]

    f_eval_0 = zeros(num_states)
    ode_fun(f_eval_0, x_k_0, phi, 0)

    x_k_1_est = zeros(num_states)
    x_k_1_est = x_k_0 + delta_t*f_eval_0

    estimated[:, i+1] = x_k_1_est
end

plot(estimated')
plot!(sol, tspan=(tspan[1], tspan[end]))

plot(range(1e-5,stop=tspan_stiff[end],length=Integer(num_samples_stiff)),estimated',xscale=:log10,layout=(3,1))
plot!(sol_stiff, tspan=(1e-5, tspan_stiff[end]), xscale=:log10, layout=(3,1))


function f_enzyme(du,u,p,t)
  # Algebrical constraints
  du[1] = 1.0 - u[3]
  u[1] = du[1]
  du[2] = 1.0 - (u[3] + u[4])
  u[2] = du[2]
  # Differential equations
  du[3] = p[1]*u[1]*u[2] - (p[2]+p[3])*u[3]
  du[4] = p[3]*u[3]
end

u0 = [1.0, 1.0, 0.0, 0.0]
tspan = (0.0,10.0)
delta_t = 0.01
num_samples = trunc(Int, tspan[end]/delta_t)

differential_vars = [false,false,true,true]
ode_fun = f_enzyme
num_states = 4

phi = [1.0,0.5,0.5,0.5]

estimated = zeros(num_states,num_samples)
estimated[:, 1] = u0


for i in 1:num_samples-1
    x_k_0 = estimated[:, i]

    f_eval_0 = zeros(num_states)
    ode_fun(f_eval_0, x_k_0, phi, 0)

    x_k_1_est = zeros(num_states)
    x_k_1_est[differential_vars]  = x_k_0[differential_vars] + delta_t*f_eval_0[differential_vars]
    x_k_1_est[.~differential_vars] .= f_eval_0[.~differential_vars]

    estimated[:, i+1] = x_k_1_est
end


plot(range(tspan[1], stop=tspan[end], length=num_samples),
    estimated')

function f_enzyme_dae(out,du,u,p,t)
  # Algebrical constraints
  out[1] = 1.0 - u[3] - u[1]
  out[2] = 1.0 - (u[3] + u[4]) - u[2]
  # Differential equations
  out[3] = p[1]*u[1]*u[2] - (p[2]+p[3])*u[3] - du[3]
  out[4] = p[3]*u[3] - du[4]
end

du0 = [0.0, 0.0, 0.0, 0.0]
tspan = (0.0,10.0)
saveat_t = range(tspan[1], stop=tspan[end], length=15)
prob = DAEProblem(f_enzyme_dae,du0,u0,tspan,
    p=phi,differential_vars=differential_vars,saveat=saveat_t)

sol = solve(prob,IDA())
data = reduce(hcat, sol.u)
plot(sol)


using Revise
includet("./objective_functions.jl")
import .ObjectiveFunctions: data_shooting

data_shooting(phi,data,saveat_t,f_enzyme,
    plot_estimated=true,differential_vars=differential_vars)

includet("./floudas_problems.jl")
import .ODEProblems: get_ode_problem

prob = get_problem("floudas_1")
phi = prob.phi
data = prob.data[:,1:1:size(prob.data)[end]]
saveat_t = prob.t[1:1:length(prob.t)]
fun = prob.fun

data_shooting(phi,data,saveat_t,fun,
    plot_estimated=true)


using Plots

xs = 1:100
μs = log.(xs)
σs = rand(length(xs))

plot(xs,μs,grid=false,ribbon=σs,fillalpha=.5)

a = [1,2,3,4,5]
dp = Float64
a = dp[x for x in a]

a = Dict()
a["um"] = Dict("dois" => [1,2])
a["um"]["dois"]
for k in keys(a)
    print(a[k])
end


xs = 1:100
μs = log.(xs)
σs = rand(length(xs))

plot(μs,grid=false,ribbon=sum(σs)/length(σs),fillalpha=.5)
plot!(xs,μs,grid=false,ribbon=σs,fillalpha=.5)

plot(xs,μs, ribbon = 5.0)  # both sides are 5
plot(xs,μs, ribbon = 1:3) # both sides cycle through 1:3
plot(xs,μs, ribbon = (5, 1:3))

# --- Fun with Trace ---
import Optim
const opt = Optim

trace_errs = []
trace_times = []
f = (x) -> sum(x)^4
res = opt.optimize(f, [rand(-100.0:100.0),rand(-100.0:100.0)], store_trace=true)

for i in 1:10
    res = opt.optimize(f, [rand(-100.0:100.0),rand(-100.0:100.0)], store_trace=true)
    trace = opt.trace(res)

    trace_err = []
    trace_time = []
    for i in 1:length(trace)
        append!(trace_err, parse(Float64, split(string(trace[i]))[2]))
        append!(trace_time, parse(Float64, split(string(trace[i]))[end]))
    end
    append!(trace_errs, [trace_err])
    append!(trace_times, [trace_time])
end

#plot(log10.(trace_time), log10.(trace_err/trace_err[end]))
#plot(trace_time, trace_err/trace_err[end])
#plot(trace_time, trace_err)
trace_times
""" Best One: """
p = plot(mean(trace_times), log10.(mean(trace_errs/trace_errs[end][end])))

for i in 1:length(trace_errs)
    print(i)
    plot!(p, trace_times[i], log10.(trace_errs[i]/trace_errs[i][end]))
end
display(p)

""" Now comparing """


trace_errs = []
trace_times = []
for i in 1:10
    res = opt.optimize((x-> sum(x.^4)), [100.0, 200.0], store_trace=true)
    trace = opt.trace(res)

    trace_err = []
    trace_time = []
    for i in 1:length(trace)
        append!(trace_err, parse(Float64, split(string(trace[i]))[2]))
        append!(trace_time, parse(Float64, split(string(trace[i]))[end]))
    end
    append!(trace_errs, [trace_err])
    append!(trace_times, [trace_time])
end

plot!(p, mean(trace_times), log10.(mean(trace_errs/trace_errs[end][end])))

for i in 1:length(trace_errs)
    print(i)
    plot!(p, trace_times[i], log10.(trace_errs[i]/trace_errs[i][end]))
end
display(p)

# --- Util for resizing array of arryas ---

a = [1.0, 2.0]
b = [1.5, 1.5, 0.1]
mean([a,b])

function fill_to_max_size(array:: AbstractArray)
    max_array_len = maximum(map(x -> length(x), array))
    resized = map( x -> append!(x, zeros(max_array_len - length(x))), array)
    return resized
end

new_ab = fill_resize([a,b])
mean(new_ab)

# --- Max and maximum ---

struct Trace
    time::AbstractArray
    eval::AbstractArray
end

a = [1.0, 2.0]
b = [0.1, 0.2]
c = [2.0, 3.0]
d = [0.2, 0.3]
traces = [Trace(a,b), Trace(c,d)]

maximum(a)
minimum(a)
(_,min_idx) = findmin(map(t -> sum(t.time), traces))
traces[min_idx]

# --- Testing Box Plots ---
using Random
using Revise
includet("./utils.jl")
import Utils

b = randn(10000)
b = randexp(50)
med_b = median(b)
max_b = maximum(b)
min_b = minimum(b)
scatter((0.0, med_b),
		yerror=[(med_b-min_b,
				max_b-med_b)])

c = quantile(b)

scatter!((1.0, c[3]),
		yerror=[(c[3]-c[1],
				c[5]-c[3])])

min_val,med_val,max_val = box_scatter_plot(b)

scatter!((2.0, med_val),
		yerror=[(min_val,
				max_val)])

scatter!((3.0, mean(b)),
		yerror=std(d))

boxplot!(["4.0"],b,leg=false)

violin!(["5.0"],b,leg=false)


# --- Testing scaling functions ---
using Plots

pp = 0:0.1:10
f1 = [x -> (i+1)/(i+exp2(x)) for i in [1,5,10]]
f2 = [x -> (i+1)/(i+exp(x)) for i in [1,5,10]]
f3 = [x -> (i+1)/(i+exp10(x)) for i in [1,5,10]]
p = plot()
plot!(p,pp,map(f -> f.(pp),f1))
plot!(p,pp,map(f -> f.(pp),f2), linestyle=:dash)
plot!(p,pp,map(f -> f.(pp),f3), linestyle=:dot)
vline!(p,[1.0])
success_rate([1.5,0.1,0.1])

# --- Testing error metrics ---

a = [100.0,200.0,400.0]
b = [150.0,205.0,4050.0]

mae = mean(abs.(a-b))
smaea = mean(abs.(a-b)./(b))
smaeb = mean(abs.(a-b)./(a))
smaec = mean(abs.(a-b)./(maximum(a)-minimum(a)))
smaec = mean(abs.(a-b)./(maximum(b)-minimum(b)))
ssmae = mean(abs.(a-b)./(a+b))

mae = mean(abs.(10*a-10*b))
smaea = mean(abs.(10*a-10*b)./(10*b))
smaeb = mean(abs.(10*a-10*b)./(10*a))
ssmae = mean(abs.(10*a-10*b)./(10*a+10*b))

println("$(round(ssmae,digits=3))")

# --- Transducers ---
using Transducers
using BenchmarkTools

maximum(Map(x -> x^2),[1,2,3])
var = 0.1
runs = 20
f = (x,y) -> x+y
g(x,y) = begin
    sleep(1)
    x+y
end
h(x) = begin
    a,b = x
    return a+b
end


u0 = [1.1,2.2]
arr = eduction(u0 for _ in 1:runs)
arr2 = eduction(var for _ in 1:runs)
u0_arr = collect(Map(x -> f.(x,var)),arr)

collect(Map(x -> f.(x,var)), arr)
arr = tcollect(eduction(u0 for i in 1:3))
@time u0_arr = collect(Map(x -> abs.(f.(x,var))),eduction(u0 for _ in 1:runs))
@time tcollect(Map(x -> 2x), 1:1_000_000)
@time collect(Map(x -> 2x), 1:1_000_000)
@time eduction(abs.(f.(x,var)) for x in eduction(u0 for _ in 1:runs))

@btime collect(eduction(abs.(f.(x,var)) for x in eduction(u0 for _ in 1:runs)))
@btime collect(Map(x -> abs.(f.(x,var))), eduction(u0 for _ in 1:runs))
@btime tcollect(Map(x -> abs.(f.(x,var))), [u0 for _ in 1:runs])
@btime collect(map(x -> abs.(f.(x,var)), [u0 for _ in 1:runs]))
@btime [abs.(f.(x,var)) for x in [u0 for _ in 1:runs]]

@time collect(eduction(abs.(g.(u0,var)) for _ in 1:runs))
@time collect(Map(x->abs.(g.(u0,var))), 1:runs)
@time tcollect(Map(x->abs.(g.(u0,var))), 1:runs)
@time collect(map(x->abs.(g.(u0,var)), 1:runs))
@time [abs.(g.(u0,var)) for _ in 1:runs]


collect(Map(x -> abs.(f.(x,var))), eduction(u0 for _ in 1:runs))
collect(Map((x,y) -> abs.(f.(x,y))), [u0 for _ in 1:runs], [u0 for _ in 1:runs])
tcollect(Map(x -> abs.(h(x))),
    collect(zip([u0 for _ in 1:runs], [u0 for _ in 1:runs])))
collect(zip([u0 for _ in 1:runs], [u0 for _ in 1:runs], [u0 for _ in 1:runs]))

collect(map((x,y) -> abs.(f.(x,y)), [u0 for _ in 1:runs], [u0 for _ in 1:runs]))

# --- DUMP Plot Trace ---

"""
Plots
Trace
"""

p = plot(xlabel="Time", ylabel="Function Evaluation")
for m in methods
	p2 = plot(xlabel="Time", ylabel="Function Evaluation")
	trace = res[m]["trace"]
	if length(trace.eval) > 0
		trace = scale_eval(fill_trace(trace))
		#(best,med,worst) = get_range_traces(trace)

		plot!(p, mean(trace.time), log10.(mean(trace.eval)),
				label="Mean "*m, color=method_color[m])

		plot!(p2, mean(trace.time), log10.(mean(trace.eval)),
				label="Mean "*m, color=method_color[m])

		#=>
		plot!(p, mean(trace.time), log10.(mean(trace.eval)), label="Mean "*m,
				grid=true,
				ribbon=log10.(mean(std(trace.eval))),
				fillalpha=.4)

		plot!(p2, mean(trace.time), log10.(mean(trace.eval)), label="Mean "*m,
				grid=true,
				ribbon=log10.(mean(std(trace.eval))),
				fillalpha=.4)
		<=#

		#=>
		plot!(p, med.time, log10.(med.eval), label="Mean "*m,
				grid=true,
				ribbon=(log10.(abs.(med.eval-worst.eval)),
						log10.(abs.(best.eval-med.eval))),
				fillalpha=.4)

		plot!(p2, med.time, log10.(med.eval), label="Mean "*m,
				grid=true,
				ribbon=(log10.(abs.(med.eval-worst.eval)),
						log10.(abs.(best.eval-med.eval))),
				fillalpha=.4)
		<=#

		#=>
		plot!(p, best.time, log10.(best.eval), label="Best "*m)
		plot!(p, med.time, log10.(med.eval), label="Median "*m)
		plot!(p, mean(trace.time), log10.(mean(trace.eval)), label="Mean "*m)
		plot!(p, worst.time, log10.(worst.eval), label="Worst "*m)

		plot!(p2, best.time, log10.(best.eval), label="Best "*m)
		plot!(p2, med.time, log10.(med.eval), label="Median "*m)
		plot!(p2, mean(trace.time), log10.(mean(trace.eval)), label="Mean "*m)
		plot!(p2, worst.time, log10.(worst.eval), label="Worst "*m)
		<=#

		display(p2)
		savefig(p2,"./trace_$(m)_$(sam)_$(replace(string(var),"."=>"")).svg")
	end
end
savefig(p,"./trace_all_$(sam)_$(replace(string(var),"."=>"")).svg")

# --- AD Testing ---
using DifferentialEquations, Optim, LinearAlgebra

function f(dz_dt, z, phi, t)
    r_1 = phi[1]*z[1]
    r_2 = phi[2]*z[2]

    dz_dt[1] = -r_1
    dz_dt[2] = r_1 - r_2
end

function prob_sol()::Tuple
    tspan::T where T<:Tuple = (0.,8.)
    t::T where T<:AbstractArray = range(0.,stop=8.,length=10)
    p0::T where T<:AbstractArray{<:AbstractFloat} = [1.,2.]
    x0::T where T<:AbstractArray{<:AbstractFloat} = [10.,0.]

    prob = ODEProblem(f,x0,tspan,p0)

    sol = DifferentialEquations.solve(prob, Rodas5(); saveat=t)

    sol_u::AbstractArray{<:AbstractFloat} = reduce(hcat,sol.u)

    return (prob,sol_u)
end

function g(data::Array,t::AbstractArray,prob::DifferentialEquations.ODEProblem)::Function
    function loss_function(x::Array)
        _prob = remake(prob;u0=convert.(eltype(x),prob.u0),p=x)
        sol = DifferentialEquations.solve(_prob; saveat=t)
        sol_u = reduce(hcat,sol.u)
        sum(abs2.(sol_u - data))
    end
    return loss_function
end

function get_opt_results(h::Function)::Array
    res = optimize(h, [2.0,1.0], LBFGS(),autodiff=:forward)
    res.minimizer
end

prob, sol_u = prob_sol()
xdata_arr = [sol_u + randn(size(sol_u)) for _ in 1:10]

g_args = zip(xdata_arr,
            [t for _ in 1:length(xdata_arr)],
            [prob for _ in 1:length(xdata_arr)])

gs = [g(arg...) for arg in g_args]

[get_opt_results(f) for f in gs]

minimum([a... for a in [(-1,2),(-10,11)]])

# --- Ploting Problems ---

using Revise
includet("./problem_set.jl")
includet("./utils.jl")
import .ProblemSet: get_problem, get_problem_key,
        problem_plot, DEProblem
import .Utils: add_noise
using DifferentialEquations
using Plots
using PlotThemes

gr()
theme(:vibrant)

function save_regular_plots(dir::String,name::String,p::DEProblem)::Nothing
    # Continuous plot
    len = round(Int64,1e4*(p.t[2]-p.t[1]))
    t = range(p.t[1], stop=p.t[end], length=len)
    prob = ODEProblem(p.fun, p.data[1], (t[1],t[end]), p.phi)
    sol = solve(prob, OwrenZen3(), saveat=t)
    data = sol.u
    p = DEProblem(p.fun, p.phi,
                p.bounds, data, t)
    d_plot = problem_plot(p,"line")
    #display(d_plot)
    savefig(d_plot,joinpath(dir,"line_$name.pdf"))

    # Sample plot
    len = maximum([10,round(Int64, 2*(p.t[2]-p.t[1]))])
    t = range(p.t[1], stop=p.t[end], length=len)
    prob = ODEProblem(p.fun, p.data[1], (t[1],t[end]), p.phi)
    sol = solve(prob, Tsit5(), saveat=t)
    data = sol.u
    p = DEProblem(p.fun, p.phi,
                p.bounds, data, t)
    d_plot = problem_plot(p,"scatter")
    display(d_plot)
    savefig(d_plot,joinpath(dir,"scatter_$name.pdf"))

    nothing
end

function save_noisy_plots(dir::String,name::String,p::DEProblem,var::Float64)::Nothing
    # Sample plot
    len = maximum([10,round(Int64, 0.25*(p.t[2]-p.t[1]))])
    t = range(p.t[1], stop=p.t[end], length=len)
    prob = ODEProblem(p.fun, p.data[1], (t[1],t[end]), p.phi)
    sol = solve(prob, OwrenZen3(), saveat=t)
    data = add_noise(sol.u,var)
    p = DEProblem(p.fun, p.phi,
                p.bounds, data, t)
    d_plot = problem_plot(p,"scatter_line")
    display(d_plot)
    savefig(d_plot,joinpath(dir,"noise_$(name)_$(round(var; digits=3)
                                        |> x -> string(x)
                                        |> y -> replace(y, "." => "")).pdf"))

    nothing
end


dir = "/home/andrew/git/ChemParamEst/plots/problems/scatter_line/"
map(n -> save_regular_plots(dir,n,get_problem(n)),
    [get_problem_key(p) for p in 1:10])

dir = "/home/andrew/git/ChemParamEst/plots/problems/noise/"
map(x -> save_noisy_plots(dir,x[1],get_problem(x[1]),x[2]),
    Iterators.product([get_problem_key(p) for p in 1:10],[0.0, 0.05, 0.1, 0.15])
    |> q -> collect(q) |> r -> reduce(vcat,r)
    )

# --- Reduce test ---

using Statistics
using Distributions

a = [[1,2,3],[1,2,3]]
mean(mean.(a))
d = Normal(0,10)
Base.rand(d)
mean_arr = [mean(getindex.(a,i)) for i in 1:length(a[1])]

# --- Slice and set op ---

a = rand(5)

a[filter(x -> x != 2, 1:length(a))]

a = [1.1,2.2,NaN64]
mean(a)
mean(a[filter(x -> !isnan(a[x]),1:length(a))])

# --- ASYNC ---

function return_3()
	x = 1
	y = 1
	x = 2
	sleep(1)
	return x+y
end

function test()
	reps = 5
	n = 2
	res_partial = Vector{Int64}(undef, 5)
	res = fill(res_partial,reps)
	for i in 1:reps
		res[i] = [return_3() for _ in 1:3]
	end
	sleep(1)
	return res
end

function test_1()
	reps = 5
	n = 2
	res_partial = Vector{Int64}(undef, 5)
	res = fill(res_partial,reps)
	@sync for i in 1:reps
		@async res[i] = [return_3() for _ in 1:3]
	end
	sleep(1)
	return res
end

@time test_1()
# Speed up!

function test_2()
	res = 0
	@sync begin
		@async res = return_3()
	end
	return res
end

@time test_2()
# Returns correct value!

@time begin
    @sync begin
        @async sleep(1)
        @async sleep(1)
    end
    sleep(0.5)
end
# Speed up!

@time begin
    a = fill(nothing,3)
	@sync for j in 1:3
        @async for i in 1:3
            a[i] = sleep(1)
        end
	end
    sleep(1)
end

function test(x,y)
	sleep(0.1(x*y))
	return x*y
end


@time begin
	res = [0.0]
	for i in 1:3
		for j in 1:4
			Threads.@spawn res .= test(i,j)
		end
	end
	println(res)
	sleep(1)
end
# 1

@time begin
	res = [Vector{Float64}(undef,4) for _ in 1:3]
	@sync for i in 1:3
		for j in 1:4
			Threads.@spawn res[i][j] = test(i,j)
		end
	end
	println(res)
	sleep(1)
end
# 2.2 ok

@time begin
	res = [Vector{Float64}(undef,4) for _ in 1:3]
	@sync for i in 1:3
		Threads.@spawn res[i] = test.([i for _ in 1:4],[j for j in 1:4])
	end
	println(res)
	sleep(1)
end
# 4.1 ok

@time begin
	res = [0.0]
	@sync for i in 1:3
		for j in 1:4
			Threads.@spawn res .= test(i,j)
		end
	end
	println(res)
	sleep(1)
end

@time begin
	res = [0.0]
	@sync for i in 1:3
		Threads.@threads for j in 1:4
			res .= test(i,j)
		end
	end
	sleep(1)
	println(res)
end
# 5.2 ok

@time begin
	res = [0.0]
	for i in 1:3
		Threads.@threads for j in 1:4
			res .= test(i,j)
		end
	end
	sleep(1)
	println(res)
end
# 5.2 ok

@time begin
	res = [0.0]
	Threads.@threads for i in 1:3
		Threads.@threads for j in 1:4
			res .= test(i,j)
		end
	end
	sleep(1)
	println(res)
end
# 4 ok

# --- Tikhonov ---
using LinearAlgebra
using BenchmarkTools

function tikhonov(lambda::T, phi::Vector{T}, phi_ref::Vector{T},
					w::Array{T,N} where N)::T where T
	return lambda*(phi-phi_ref)'*w'*w*(phi-phi_ref)
end

function tikhonov2(lambda::T, phi::Vector{T}, phi_ref::Vector{T},
					w::Array{T,N} where N)::T where T
	return lambda*sum(abs2.((phi-phi_ref).*w))
end

function tikhonov3(lambda::T, phi::Vector{T}, phi_ref::Vector{T},
					w::Vector{T})::T where T
    res = zero(T)
    @simd for i in 1:length(phi)
        @inbounds res += abs2((phi[i]-phi_ref[i])*w[i])
    end
    res *= lambda
	return res
end

function tikhonov4(lambda::T, phi::Vector{T}, phi_ref::Vector{T},
					w::Vector{T})::T where T
    res = zero(T)
    @simd for i in 1:length(phi)
        @inbounds res += ((phi[i]-phi_ref[i])*w[i])^2
    end
    res *= lambda
	return res
end

phi_ref = randn(10)
phi = randn(10)
m = length(phi)
w = Matrix{Float64}(I,m,m)./phi_ref
w2 = ones(m)./phi_ref
alpha = randn(1)[1]

tikhonov4(alpha,phi,phi_ref,w2)
tikhonov3(alpha,phi,phi_ref,w2)
tikhonov2(alpha,phi,phi_ref,w2)
tikhonov(alpha,phi,phi_ref,w)

@btime tikhonov(1.0,phi,phi_ref,w)
@btime tikhonov2(1.0,phi,phi_ref,w2)
@btime tikhonov3(1.0,phi,phi_ref,w2)
@btime tikhonov4(1.0,phi,phi_ref,w2)
clearconsole()

# --- Floudas 6 Plot Compare ---
using DifferentialEquations
using Optim
using Plots
using Revise
includet("./problem_set.jl")
includet("./objective_function.jl")
includet("./utils.jl")
import .ObjectiveFunction: data_shooting, single_shooting, tikhonov
import .ProblemSet: get_problem, get_problem_key, DEProblem
import .Utils: Trace, rand_guess, make_trace, get_range_traces, add_noise,
				filter_outlier, fill_trace, scale_eval,
				max_diff_states, diff_calc, step_success_rate, success_rate, box_data, box_scatter_plot,
				get_plot_data, oe_plots, sr_plots, error_plots, nmse, plot_compare


p = get_problem("floudas_1")

lb = p.bounds[1]
ub = p.bounds[2]
data = add_noise(p.data,0.01)

noisy_ini = add_noise(data[1], 5.0, true)
noisy_ini = [0.6, 1.2]
tspan = (p.t[1], p.t[end])
ode_prob = ODEProblem(p.fun, noisy_ini, tspan, p.phi)
ode_sol  = solve(ode_prob, Tsit5(), saveat=p.t)
data = ode_sol.u
data = add_noise(data,0.015)

obj_fun = function (x)
			total_error = data_shooting(x,
						data,
						p.t,
						p.fun)
			total_error += tikhonov(0.0, x, zeros(length(p.phi)), ones(length(p.phi)))
			return total_error
		end

elapsed_time = @elapsed res_obj = Optim.optimize(obj_fun,
									lb,ub,
									rand_guess(p.bounds),
                                    Optim.SAMIN(verbosity=0, rt=0.5))

elapsed_time = @elapsed res_obj = Optim.optimize(obj_fun,
									lb,ub,
									res_obj.minimizer,
									Fminbox(Optim.LBFGS()),
									autodiff = :forward)

phi_est = res_obj.minimizer
tspan = (p.t[1], p.t[end])
ode_prob = ODEProblem(p.fun, noisy_ini, tspan, phi_est)
ode_sol  = solve(ode_prob, Tsit5(), saveat=p.t)
p = plot_compare(data, ode_sol.u)
savefig(p,"./error_out.pdf")
phi_est

p02 = rand_guess(p.bounds)

obj_fun = function (x)
			total_error = single_shooting(x,
						data,
						p.t,
						p.fun)
			total_error += tikhonov(0.001, x, phi_est, ones(length(p.phi))./phi_est)
			return total_error
		end

elapsed_time = @elapsed res_obj = Optim.optimize(obj_fun,
									lb,ub,
									p02,
                                    Optim.SAMIN(verbosity=0, rt=0.9))

elapsed_time = @elapsed res_obj = Optim.optimize(obj_fun,
									lb,ub,
									res_obj.minimizer,
									Fminbox(Optim.LBFGS()),
									autodiff = :forward)

phi_est2 = res_obj.minimizer
tspan = (p.t[1], p.t[end])
ode_prob = ODEProblem(p.fun, p.data[1], tspan, phi_est2)
ode_sol  = solve(ode_prob, Tsit5(), saveat=p.t)
plot_compare(data, ode_sol.u)
phi_est2

p.phi

# --- Plotting formats ---
using StatsPlots
import Distributions: Uniform

gr()
theme(:vibrant)

cur_colors = get_color_palette(:lighttest, plot_color(:white), 10)

all_method_arr = ["DS","SS","DDS"]

method_color = Dict()
for (m,c) in zip(all_method_arr,cur_colors)
	method_color[m] = c
end

method_label = Dict()
for m in all_method_arr
	method_label[m] = m
end

plot_data = Dict()
for m in all_method_arr
    plot_data[m] = Dict()
    plot_data[m]["error"] = randn(10)
    plot_data[m]["time"] = randn(10)
    plot_data[m]["est"] = [rand(Uniform(0,10),5)
							for _ in 1:10]
end

function parameter_plots(plot_data::Dict)::Nothing
    num_pars = length(plot_data[all_method_arr[1]]["est"][1])
    p_arr = []
    for m in all_method_arr
        p = plot(legend=:outertopright, ylabel="Value", xlabel="Parameter")
    	for i in 1:num_pars
    		if i == 1
    			boxplot!(p, [string(i)], log10.(getindex.(plot_data[m]["est"],i)), color=method_color[m], label=m)
    		else
    			boxplot!(p, [string(i)], log10.(getindex.(plot_data[m]["est"],i)), color=method_color[m], label="")
    		end
    	end
    	push!(p_arr,p)
    end

    p = plot(p_arr...,layout=(length(all_method_arr),1))
	display(p)
	nothing
end

parameter_plots(plot_data)


p = scatter((rand(5),rand(5)),
            xerror=(rand(5),rand(5)),
            yerror=(rand(5),rand(5)))
savefig(p,"/home/andrew/git/ChemParamEst/test2.pdf")

all_method_arr = ["DS","SS","DSS"]
cur_colors = get_color_palette(:lighttest, plot_color(:white), 10)
method_color = Dict()
for (m,c) in zip(all_method_arr,cur_colors)
	method_color[m] = c
end

method_label = Dict()
for m in all_method_arr
	method_label[m] = m
end

method_arr = ["DS","DSS"]

for sam in [2]
	results = res[sam]
	for (v,_) in results
		plot_data = get_plot_data(results, [v], ["DS","DSS"])

		#box_error_plots(plot_data,v,method_arr,method_label,method_color,sam,full_path)

		#parameter_plots(plot_data,v,method_arr,method_label,method_color,sam,full_path)

		sr_plots(plot_data,v,method_arr,method_label,method_color,sam,"")
	end
end

plot_data = Dict()
for m in ["DS","SS","DSS"]
    plot_data[m] = Dict()
    plot_data[m]["error"] = []
    push!(plot_data[m]["error"], randn(10))
    append!(plot_data[m]["error"][1], Inf64)
    plot_data[m]["time"] = [randn(10)]
    plot_data[m]["est"] = [[rand(Uniform(0,10),5)
							for _ in 1:10]]
end
plot_data
data = [[x for x in plot_data[m]["error"][1]]
        for m in method_arr]
data_ = []
for d in data
    push!(data_,collect(skipmissing([ifelse(isinf(x),missing,x) for x in d])))
end
data_

method_arr = ["DS","SS","DSS"]
method_arr_ = reduce(hcat,method_arr)
p = plot(legend=false, ylabel="Error", xlabel="Method")
boxplot!(p, method_arr_, data_,
            color=[method_color[m] for m in method_arr_])


y = rand(100,4) # Four series of 100 points each
boxplot(["Series 1" "Series 2" "Series 3" "Series 4"],y,leg=false)

import Statistics: quantile
pyplot()
gr()

using Pkg
ENV["GRDIR"]=""; Pkg.build("GR")
plot([1,2,3,4])

# -- Contour Plot ---

using Plots
using PlotThemes
theme(:default)
gr()
x = 0.5:0.01:1.50
y = 0.5:0.01:1.50
f(x,y) = abs2(x-y)
p = plot(x,y,f,st=:wireframe)
plot!(p,xrotation=45,yrotation=45)

savefig(p,"./fig3.png")

# --- Latex Tables ---
using Revise
includet("./problem_set.jl")
import .ProblemSet: get_problem, get_problem_key, DEProblem
using Printf

clearconsole()

function print_table(r)
    begin_table = "\\begin{table}\n\\begin{tabularx}{\\linewidth}{ c"
    for i in r
        begin_table *= "|c"
    end
    begin_table *= " }\n\\centering"
    end_table = "\\end{tabularx}\n\\end{table}"
    short_name = "Short Name & "
    description = "Description & "
    reference = "Reference & "
    params = "Estimated parameters & "
    nominal = "Nominal values & "
    states = "Observed states & "

    for i in r
        name = get_problem_key(i)
        p = get_problem(name)
        name = replace(length(name)>3 ? titlecase(name) : uppercase(name),("_"=>" "))
        if i != r[end]
            short_name *= string(name)*" & "
            description *= "desc & "
            reference *= "ref & "
            params *= string(length(p.phi))*" & "
            nominal *= "["
            counter = 1
            for k in p.phi
                k = k < 1e-3 ? @sprintf("%.4E, ",k) : string(k)*", "
                nominal *= k
                counter > 2 ? begin counter = 0; nominal *= " \\newline " end : counter += 1
            end
            nominal = nominal[1:end-2]
            nominal *= "] & "
            states *= string(length(p.data[1]))*" & "
        else
            short_name *= string(name)*" \\\\ \\hline"
            description *= "desc \\\\ \\hline"
            reference *= "ref \\\\ \\hline"
            params *= string(length(p.phi))*" \\\\ \\hline"
            nominal *= "["
            counter = 0
            for k in p.phi
                k = k < 1e-3 ? @sprintf("%.4E, ",k) : string(k)*", "
                nominal *= k
                counter > 2 ? begin counter = 0; nominal *= "\\newline " end : counter += 1
            end
            nominal = nominal[1:end-2]
            nominal *= "] \\\\"
            states *= string(length(p.data[1]))*" \\\\ "
        end
    end

    println(begin_table)
    println(short_name)
    println(description)
    println(reference)
    println(params)
    #println(nominal)
    println(states)
    println(end_table)
end

print_table(1:3)
print_table(4:6)
print_table(7:10)

clearconsole()

# --- Serialization ---
using JLSO

data = []
str_a = "a"
a = rand(100)
push!(data, Pair(Symbol(str_a),a))
b = rand(100)
str_b = "b"
push!(data, Pair(Symbol(str_b),b))
sum(a)
sum(b)
data
JLSO.save("test.jlso", data...)
x = JLSO.load("test.jlso")
sum(x[:a])
sum(x[:b])
x[:b] == b
x[:a] == a
