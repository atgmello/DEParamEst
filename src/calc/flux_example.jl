using Flux, DiffEqFlux, DifferentialEquations, Plots

## Setup ODE to optimize
function lotka_volterra(du,u,p,t)
  x, y = u
  α, β, δ, γ = p
  du[1] = dx = α*x - β*x*y
  du[2] = dy = -δ*y + γ*x*y
end
u0 = [1.0,1.0]
tspan = (0.0,10.0)
p = [1.5,1.0,3.0,1.0] #True Params
prob = ODEProblem(lotka_volterra,u0,tspan,p)

# Verify ODE solution
sol = solve(prob,Tsit5())
plot(sol)

# Generate data from the ODE
data_sol = solve(prob,Tsit5(),saveat=0.1)
A1 = data_sol[1,:] # length 101 vector
A2 = data_sol[2,:] # length 101 vector
t = 0:0.1:10.0
scatter!(t,A1,color=[1],label = "rabbits")
scatter!(t,A2,color=[2],label = "wolves")

# Build a neural network that sets the cost as the difference from the
# generated data and true data

p = param([4., 1.0, 2.0, 0.4]) # Initial Parameter Vector
function predict_rd() # Our 1-layer neural network
  diffeq_rd(p,prob,Tsit5(),saveat=0.1)
end
loss_rd() = sum(abs2,predict_rd()-data_sol) # loss function

# Optimize the parameters so the ODE's solution stays near 1

data = Iterators.repeated((), 100)
opt = ADAM(0.1)
cb = function () #callback function to observe training
  #= display(loss_rd()) =#
  # using `remake` to re-create our `prob` with current parameters `p`
  scatter(t,A1,color=[1],label = "rabbit data")
  scatter!(t,A2,color=[2],label = "wolves data")
  display(plot!(solve(remake(prob,p=Flux.data(p)),Tsit5(),saveat=0.1),ylim=(0,6),labels = ["rabbit model","wolf model"],color=[1 2]))
end
# Display the ODE with the initial parameter values.
cb()
Flux.train!(loss_rd, [p], data, opt, cb = cb)
p

# ----- Testing with proposed objective function -----

p = param([4., 1.0, 2.0, 0.4]) # Initial Parameter Vector
data_sim = reduce(hcat, data_sol.u)
function loss_rd() # Our 1-layer neural network
  sum(abs2.(adams_moulton_estimator(p, data_sim, t, lotka_volterra)))
end

# Optimize the parameters so the ODE's solution stays near 1

data = Iterators.repeated((), 100)
opt = ADAM(0.1)
cb = function () #callback function to observe training
  #= display(loss_rd()) =#
  # using `remake` to re-create our `prob` with current parameters `p`
  scatter(t,A1,color=[1],label = "rabbit data")
  scatter!(t,A2,color=[2],label = "wolves data")
  display(plot!(solve(remake(prob,p=Flux.data(p)),Tsit5(),saveat=0.1),ylim=(0,6),labels = ["rabbit model","wolf model"],color=[1 2]))
end
# Display the ODE with the initial parameter values.
cb()
Flux.train!(loss_rd, [p], data, opt, cb = cb)

# ----- Testing with Floudas' dataset -----

i = 1
prob = problem_set[i]
f = prob.fun
exp_data = prob.data
lb = prob.bounds[1]
ub = prob.bounds[end]
true_par = prob.phi
t = prob.t
shared_rand = rand(1:1234)
rng = MersenneTwister(shared_rand)
p_am = param([rand(rng, Uniform(lb[i], ub[i])) for i in 1:length(true_par)])
rng = MersenneTwister(shared_rand)
p_rd = param([rand(rng, Uniform(lb[i], ub[i])) for i in 1:length(true_par)])
println(p_am)
println(p_rd)

function loss_am() # Our 1-layer neural network
  sum(abs2.(adams_moulton_estimator(p_am, exp_data, t, f)))
end


# Optimize the parameters so the ODE's solution stays near 1

data = Iterators.repeated((), 1000)
opt_method = ADAM(2.0)
u0 = exp_data[:,1]
tspan = (t[1], t[end])
prob_instance = ODEProblem(f, u0, tspan)

function predict_rd() # Our 1-layer neural network
  diffeq_rd(p_rd,prob_instance,Tsit5(),saveat=t)
end
loss_rd() = sum(soft_l1.(abs2.((predict_rd()-exp_data)))) # loss function

cb_am = function () #callback function to observe training
  # @show(loss_am())
  # using `remake` to re-create our `prob` with current parameters `p`
  plot_canvas = plot()
  for i in 1:length(u0)
    scatter!(plot_canvas,t,transpose(exp_data[i,:]), color=[i], label="[A_$i]")
  end
  display(plot!(plot_canvas,solve(remake(prob_instance,p=Flux.data(p_am)),Tsit5(),saveat=t)))
end
# Display the ODE with the initial parameter values.
cb_am()
Flux.train!(loss_am, [p_am], data, opt_method, cb = cb_am)
println("True:\n$true_par\nEst:\n$p_am\nDistance:\n$(euclidean(true_par,p_am))")

cb_rd = function () #callback function to observe training
  # @show(loss_rd())
  # using `remake` to re-create our `prob` with current parameters `p`
  plot_canvas = plot()
  for i in 1:length(u0)
    scatter!(plot_canvas,t,transpose(exp_data[i,:]), color=[i], label="[A_$i]")
  end
  display(plot!(plot_canvas,solve(remake(prob_instance,p=Flux.data(p_rd)),Tsit5(),saveat=t)))
end
# Display the ODE with the initial parameter values.
cb_rd()
Flux.train!(loss_rd, [p_rd], data, opt_method, cb = cb_rd)

println("True:\n$true_par\nEst:\n$p_rd\nDistance:\n$(euclidean(true_par,p_rd))")
m = Chain(
          p_rd -> diffeq_rd(p_rd, prob_instance, Tsit5(), saveat=t),
          )
ps = Flux.params(m)
Flux.train!(loss_rd, ps, data, opt_method)

true_par
p_rd

# ----- Second example -----
u0 = Float32[2.; 0.]
datasize = 30
tspan = (0.0f0,1.5f0)

function trueODEfunc(du,u,p,t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u.^3)'true_A)'
end

function ODEfunc(du,u,p,t)
    du .= ((u.^3)'*p)'
end

t = range(tspan[1],tspan[2],length=datasize)
prob = ODEProblem(trueODEfunc,u0,tspan)
ode_data = Array(solve(prob,Tsit5(),saveat=t))
plot(transpose(ode_data))

par = [param(1.0) param(1.0); param(1.0) param(1.0)]
#par = param([1. 1.; 1. 1.])
function f1(x)
  data_shooting_estimator_node(par, x, t, ODEfunc; steps=1)
end
function f2(x)
  data_shooting_estimator_node(par, x, t, ODEfunc; steps=2)
end

function fe(x)
    sum(abs2.(data_shooting_estimator_endnode(par, x, t, ODEfunc; steps=3)))
end

dudt = Chain(x -> f1(x), x -> f2(x))
ps = Flux.params(dudt)

pred = n_ode(ode_data) # Get the prediction using the correct initial condition
scatter(t,ode_data[1,:],label="data")
scatter!(t,Flux.data(pred[1,:]),label="prediction")
function predict_n_ode()
  n_ode(u0)
end
loss_n_ode() = sum(abs2,ode_data .- predict_n_ode())

data = Iterators.repeated((), 100)
opt_method = ADAM(0.1)
cb = function () #callback function to observe training
  display(loss_n_ode())
  # plot current prediction against data
  cur_pred = Flux.data(predict_n_ode())
  pl = scatter(t,ode_data[1,:],label="data")
  scatter!(pl,t,cur_pred[1,:],label="prediction")
  display(plot(pl))
end

# Display the ODE with the initial parameter values.
cb()


dudt(ode_data)
loss_testing() = sum(abs2, dudt(ode_data).-ode_data)
Flux.train!(loss_testing, par, data, opt_method)
params(dudt)
par


# ----- Try simples example -----

# Parameter that I want to learn
learn_p = param([4.01;1.1])
# Function used for generating output using learn_p
function f_square_cube(x)
    x.^learn_p
end
# Single layer neural network
model_chain = Chain(x -> f_square_cube(x),
                    x -> f_square_cube(x)
                    )
tracking = params(learn_p)
loss(x,y) = sum(abs2, model_chain(x) - y)
rand_data = rand((1.0:0.1:100.0),2,300)
square_cube_rand = rand_data.^[4;9]
data = Iterators.repeated((rand_data, square_cube_rand), 2000)
Flux.train!(loss, tracking, data, ADAM(0.1))
tracking

#----- Now to ODEs -----
i = 1
p_solve = problem_set[i]
ode_fun = p_solve.fun
phi = p_solve.phi
t = p_solve.t
ode_data = p_solve.data
learn_p = param([rand(1:100) for i in 1:length(phi)])
learn_p
t = range(t[1], stop=t[end], length=10)
tspan = (t[1], t[end])
ini_cond = p_solve.data[:,1]
ode_prob = ODEProblem(p_solve.fun, ini_cond, tspan, p_solve.phi)
ode_sol  = solve(ode_prob, lsoda(), saveat=reduce(vcat, t))
data_original = reduce(hcat, ode_sol.u)
ode_data = copy(data_original)

function run_for_steps(steps, learn_p)
    function f_learn(x)
        data_shooting_estimator_node(learn_p,x,t,ode_fun; steps=steps)
    end
    # Single layer neural network
    model_chain = Chain(x -> f_learn(x))

    tracking = params(learn_p)
    function loss(x,y)
        res = Flux.mse(model_chain(x), y)
        if res == NaN
            res = 10^5
        end
        return res
    end
    data = Iterators.repeated((ode_data, ode_data), 500)
    Flux.train!(loss, tracking, data, ADAM(0.1))
    tracking
end

@time run_for_steps(1, learn_p)
run_for_steps(2, learn_p)
run_for_steps(3, learn_p)
run_for_steps(40, learn_p)

run_for_steps(40, learn_p)

# One step only
# Function used for generating output using learn_p
learn_p = param([7.71, 8.81])
function f_learn(x)
    data_shooting_estimator_node(learn_p,x,t,ode_fun; steps=1)
end
# Single layer neural network
model_chain = Chain(x -> f_learn(x))

tracking = params(learn_p)
loss(x,y) = sum(abs2, model_chain(x) - y)
data = Iterators.repeated((ode_data, ode_data), 10000)
Flux.train!(loss, tracking, data, ADAM(0.1))
tracking


# ----- Testing Adams-Moulton -----
i = 6
p_solve = problem_set[i]
ode_fun = p_solve.fun
phi = p_solve.phi
t = p_solve.t
ode_data = p_solve.data
t = p_solve.t
t = range(t[1], stop=t[end], length=50)
tspan = (t[1], t[end])
ini_cond = p_solve.data[:,1]
ode_prob = ODEProblem(p_solve.fun, ini_cond, tspan, p_solve.phi)
ode_sol  = solve(ode_prob, lsoda(), saveat=reduce(vcat, t))
data_original = reduce(hcat, ode_sol.u)
ode_data = copy(data_original)
plot(transpose(ode_data))
learn_p = param([rand(0.0001:0.0001:100) for i in 1:length(phi)])
tracking = params(learn_p)
function f_learn(x)
    adams_moulton_fourth_estimator(learn_p,x,t,ode_fun; return_estimated=true)
end
# Single layer neural network
model_chain = Chain(x -> f_learn(x))
function loss_mse(x,y)
    res = sum(abs2, model_chain(x) - y) + 10^10
    return res
end
data = Iterators.repeated((ode_data, ode_data), 1000)
function_error = []
function cb()
    err = Tracker.data(loss_mse(ode_data, ode_data))
    if err < 100
        push!(function_error, err)
    end
end
@time Flux.train!(loss_mse, tracking, data, ADAM(0.1), cb = cb)
plot(function_error[2:end])
tracking
euclidean(learn_p, phi)

# ----- Adams Moulton One on Each Neuron -----
i = 7
p_solve = problem_set[i]
ode_fun = p_solve.fun
phi = p_solve.phi
t = p_solve.t
ode_data = p_solve.data
t = p_solve.t
t = range(t[1], stop=t[end], length=50)
tspan = (t[1], t[end])
ini_cond = p_solve.data[:,1]
ode_prob = ODEProblem(p_solve.fun, ini_cond, tspan, p_solve.phi)
ode_sol  = solve(ode_prob, lsoda(), saveat=reduce(vcat, t))
data_original = reduce(hcat, ode_sol.u)
ode_data = copy(data_original)
plot(transpose(ode_data))
learn_p = param([rand(0.0001:0.0001:100) for i in 1:length(phi)])
tracking = params(learn_p)
function f_learn(x)
    adams_moulton_estimator(learn_p,x,t,ode_fun; return_estimated=true)
end
# Single layer neural network
function concat_neurons(args...)
    reestructure = zeros(eltype(args[1]), length(args[1][:,end]), length(args))
    for i in 1:length(args)
        reestructure[:,i] = args[i][:,end]
    end
    reestructure
end
a1 = adams_moulton_estimator(phi, ode_data[:,1:2], t[1:2], ode_fun; return_estimated=true)
a2 = adams_moulton_estimator(phi, ode_data[:,2:3], t[2:3], ode_fun; return_estimated=true)
a3 = adams_moulton_estimator(phi, ode_data[:,3:4], t[3:4], ode_fun; return_estimated=true)
con = concat_neurons(a1, a2, a3)
ode_data[:,2:4]
con - ode_data[:,2:4]

model_chain = Chain(x -> f_learn(x), x -> concat_neurons(x))
function loss_mse(x,y)
    res = Flux.mse(model_chain(x), y)
end

function loss_mse()
    proc_data = [ode_data[:,i:i+1] for i in 1:size(ode_data)[2]-1]
    segment = [f_learn(data_piece) for data_piece in proc_data]
    estimated_data = concat_neurons(segment...)
    res = Flux.mse(estimated_data, ode_data[:,2:end])
end

#data = Iterators.repeated([(ode_data[:,i:i+1], ode_data[:,2:end]) for i in 1:size(ode_data)[2]-1], 10000)
#data = [(ode_data[:,i:i+1], ode_data[:,2:end]) for i in 1:size(ode_data)[2]-1]
data = Iterators.repeated((), 1000)
function_error = []
function cb()
    err = Tracker.data(loss(ode_data, ode_data))
    if err < 100
        push!(function_error, err)
    end
end
@time Flux.train!(loss_mse, tracking, data, ADAM(0.1))
tracking
euclidean(learn_p, phi)
phi

# ----- Another test, adding layers  -----
i = 7
p_solve = problem_set[i]
ode_fun = p_solve.fun
phi = p_solve.phi
t = p_solve.t
ode_data = p_solve.data
t = p_solve.t
t = range(t[1], stop=t[end], length=10)
tspan = (t[1], t[end])
ini_cond = p_solve.data[:,1]
ode_prob = ODEProblem(p_solve.fun, ini_cond, tspan, p_solve.phi)
ode_sol  = solve(ode_prob, lsoda(), saveat=reduce(vcat, t))
data_original = reduce(hcat, ode_sol.u)
ode_data = copy(data_original)
plot(transpose(ode_data))
rand_ini = [rand(0.0001:0.0001:100) for i in 1:length(phi)]
learn_p = param(rand_ini)
#learn_p = param(phi)
#tracking = params(learn_p)
function f_learn(p)
    vec(adams_moulton_estimator(p, ode_data, t, ode_fun; return_estimated=true))
end
# 10 layer neural network
size(ode_data)
model_chain = Chain(Dense(length(ode_data),length(phi), σ),
                    Dense(length(phi),100, σ),
                    Dense(100,100, σ),
                    Dense(100,100, σ),
                    Dense(100,100, σ),
                    Dense(100,100, σ),
                    Dense(100,100, σ),
                    Dense(100,length(phi), abs),
                    x -> f_learn(x))
tracking = params(model_chain)
function loss(x,y)
    res = Flux.mse(model_chain(x), y)
    if res == NaN
        res = 10^10
    end
    return res
end
data = Iterators.repeated((vec(ode_data), vec(ode_data)), 1000000)
function_error = []
function cb()
    println(Flux.data(loss(vec(ode_data), vec(ode_data))))
end
@time Flux.train!(loss, tracking, data, ADAM(0.1))
print(params(model_chain))
# ----- Simpler model -----
learn_p = param([7.71]) |> gpu
function f(x)
    x.*learn_p
end
# Single layer neural network
model_chain = Chain(x -> f(x)) |> gpu
tracking = params(learn_p)
loss(x,y) = sum(abs2, model_chain(x) - y)
rand_data = rand((1.0:0.1:100.0),500,500) |> gpu
label = rand_data.*2 |> gpu
data = Iterators.repeated((rand_data, label), 5000)
@time Flux.train!(loss, tracking, data, ADAM(0.1))
tracking

# ----- Other one -----
model_chain = Chain(Dense(100, 50, σ),
                    Dense(50,100), softmax) |> gpu
tracking = params(model_chain) |> gpu
loss(x,y) = sum(abs2, model_chain(x) - y)
rand_data = rand((1.0:0.1:100.0),100,100)
label = rand_data.*[2]
data = Iterators.repeated((rand_data, label), 500)
@time Flux.train!(loss, tracking, data, ADAM(0.1))
loss(rand_data, label)

# ----- CuArrays are fast as they get big -----
a = rand((1:0.001:1000),10000,10000)
b = cu(a)
@time a*a
@time b*b
