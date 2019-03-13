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
ode_fun = ode_fun_array[i]
t = t_array[i]
phi = phi_array[i]
ini_cond = ini_cond_array[i]

#sample_t = range(t[1], stop=t[end], length=sample_size)
tspan = (t[1], t[end])
oprob = ODEProblem(ode_fun, ini_cond, tspan, phi)
osol  = solve(oprob, Tsit5(), saveat=sample_t)
osol_plot = scatter(osol)
display(osol_plot)
data = reduce(hcat, osol.u)
bounds = bounds_array[i]
lb = vec([bounds[1] for i in 1:length(phi)])
ub = vec([bounds[end] for i in 1:length(phi)])

data = floudas_samples_array[i]
sample_t = floudas_samples_times_array[i]
rand_range = rand_range_array[i]

initial_guess = desired_precision[]
for i in range(1, stop=length(phi))
    rand_num = rand(Uniform(rand_range[1], rand_range[end]))
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

cost_function = build_loss_objective(oprob,Tsit5(),L2Loss(sample_t,data),
                             maxiters=10000,verbose=false)

cost_function([.1 .1])

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
