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

i = 5
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

data = Iterators.repeated((), 100)
opt_method = ADAM(0.4)
u0 = exp_data[:,1]
tspan = (t[1], t[end])
prob_instance = ODEProblem(f, u0, tspan)

function predict_rd() # Our 1-layer neural network
  diffeq_rd(p_rd,prob_instance,Tsit5(),saveat=t)
end
loss_rd() = sum(abs2,predict_rd()-exp_data) # loss function

cb_am = function () #callback function to observe training
  #= display(loss_rd()) =#
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
  #= display(loss_rd()) =#
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
