using Plots
using DifferentialEquations
plotly()

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

ode_fun = floudas_one

data = Float64[1.0 0.57353 0.328937 0.188654 0.108198 0.0620545 0.0355906 0.0204132 0.011708 0.00671499;
        0.0 0.401566 0.589647 0.659731 0.666112 0.639512 0.597179 0.54867 0.499168 0.451377]
t = Float64[0.0, 0.111111, 0.222222, 0.333333, 0.444444, 0.555556, 0.666667, 0.777778, 0.888889, 1.0]


function lsq_ss_estimator(phi)
    tspan = (t[1], t[end])
    ini_cond = data[:,1]
    oprob = ODEProblem(ode_fun, ini_cond, tspan, phi)
    osol  = solve(oprob, Tsit5(), saveat=t)
    estimated = reduce(hcat, osol.u)
    p = plot(transpose(data))
    plot!(p, transpose(estimated))
    display(p)
    return sum((estimated-data).^2)
end

p0 = [5., 5.]
res = bboptimize(lsq_ss_estimator; SearchRange = (0., 5.), NumDimensions = 2)

print(lsq_ss_estimator([5.0035, 1]))

print(phi_array[1])

print("Testing commit from Juno.")

# ----- Testando problema Floudas -----


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
i_prob = "floudas_1"
p_solve = get_ode_problem(i_prob)
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

SAMIN_options = opt.Options(x_tol=10^-12, f_tol=10^-24, iterations=10^6)
Grad_options = opt.Options(x_tol=10^-12, f_tol=10^-24, iterations=10^6)
inner_optimizer = opt.LBFGS()
function time_res!(f, obj, opt_type)
	if opt_type == "G"
		timed = @elapsed res_obj = opt.optimize(f, lb, ub, p0, opt.SAMIN(verbosity=0), SAMIN_options)
	else
    	#od = opt.OnceDifferentiable(adams_moulton_error, p0; autodiff = :forward)
    	#timed = @elapsed res_obj = opt.optimize(od, lb, ub, p0, opt.Fminbox(inner_optimizer), Grad_options)
    	timed = @elapsed res_obj = opt.optimize(f, lb, ub, p0, opt.Fminbox(inner_optimizer), Grad_options)
	end
    dist = diff_states(fun, ini_cond, (t[1],t[end]), phi, res_obj.minimizer[1:length(phi)])
    print(dist)
    print(timed)
end

res_am = ErrorTimeData([], [])
adams_moulton_error(p) = sum(
                            abs2.(
                                loss.(adams_moulton_estimator_x(p, data,
                                                                t, p_solve.fun;
                                                                unknown_states=unknown_states,
                                                                plot_estimated=true)
                                                                )
                                    )
                            )

single_shooting_error(p) = sum(
                            abs2.(
                                loss.(single_shooting_estimator(p, data,
                                                                t, p_solve.fun;
                                                                unknown_states=unknown_states,
                                                                plot_estimated=true)
                                                                )
                                    )
                            )


time_res!(adams_moulton_error, res_am, "L")

time_res!(single_shooting_error, res_am, "L")

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

function f(out,du,u,p,t)
  out[1] = - 0.04u[1]              + 1e4*u[2]*u[3] - du[1]
  out[2] = + 0.04u[1] - 3e7*u[2]^2 - 1e4*u[2]*u[3] - du[2]
  out[3] = u[1] + u[2] + u[3] - 1.0
end

u₀ = [1.0, 0, 0]
du₀ = [-0.04, 0.04, 0.0]
tspan = (0.0,100000.0)

differential_vars = [true,true,false]
prob = DAEProblem(f,du₀,u₀,tspan,differential_vars=differential_vars)

using Sundials

sol = solve(prob,IDA())
plot(sol, xscale=:log10, tspan=(1e-6, 1e5), layout=(3,1))

delta_t=tspan[2]-tspan[1]

a = [1,2]
b = copy(a)
b[1] = 2
a
