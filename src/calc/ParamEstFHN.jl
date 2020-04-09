using ParameterizedFunctions, OrdinaryDiffEq, DiffEqParamEstim
using BlackBoxOptim, NLopt, Plots
plotlyjs()

loc_bounds = Tuple{Float64,Float64}[(0, 1), (0, 1), (0, 1), (0, 1)]
glo_bounds = Tuple{Float64,Float64}[(0, 5), (0, 5), (0, 5), (0, 5)]
loc_init = [0.5,0.5,0.5,0.5]
glo_init = [2.5,2.5,2.5,2.5]

fitz = @ode_def FitzhughNagumo begin
  dv = v - v^3/3 -w + l
  dw = τinv*(v +  a - b*w)
end a b τinv l

p = [0.7,0.8,0.08,0.5]              # Parameters used to construct the dataset
r0 = [1.0; 1.0]                     # initial value
tspan = (0.0, 30.0)                 # sample of 3000 observations over the (0,30) timespan
prob = ODEProblem(fitz, r0, tspan,p)
tspan2 = (0.0, 3.0)                 # sample of 300 observations with a timestep of 0.01
prob_short = ODEProblem(fitz, r0, tspan2,p)

dt = 30.0/3
tf = 30.0
tinterval = 0:dt:tf
t  = collect(tinterval)

h = 1
M = 3
tstart = 0.0
tstop = tstart + M * h
tinterval_short = 0:h:tstop
t_short = collect(tinterval_short)

#Generate Data
data_sol_short = solve(prob_short,Vern9(),saveat=t_short,reltol=1e-9,abstol=1e-9)
data_short = convert(Array, data_sol_short) # This operation produces column major dataset obs as columns, equations as rows
data_sol = solve(prob,Vern9(),saveat=t,reltol=1e-9,abstol=1e-9)
data = convert(Array, data_sol)

plot(data_sol_short)

plot(data_sol)

obj_short = build_loss_objective(prob_short,Tsit5(),L2Loss(t_short,data_short),tstops=t_short)
res1 = bboptimize(obj_short;SearchRange = glo_bounds, MaxSteps = 7e3)
println("Real params are:\n$p")
# Lower tolerance could lead to smaller fitness (more accuracy)

obj_short_am(x) = sum(abs2.(adams_moulton_estimator(x, data_short, t_short, fitz)))
res1 = bboptimize(obj_short_am;SearchRange = glo_bounds, MaxSteps = 7e3)
println("Real params are:\n$p")
obj_short_ds(x) = sum(abs2.(data_shooting_estimator(x, data_short, t_short, fitz, steps=1, euler=true)))
res1 = bboptimize(obj_short_ds;SearchRange = glo_bounds, MaxSteps = 7e3)
println("Real params are:\n$p")
obj_short_ds(x) = sum(abs2.(data_shooting_estimator(x, data_short, t_short, fitz, steps=10, euler=true)))
res1 = bboptimize(obj_short_ds;SearchRange = glo_bounds, MaxSteps = 7e3)
println("Real params are:\n$p")
# Testing same conditions as above but with a different objective functions. First using Adams-Moulton. Then, Euler, with 1 and 10 steps. The more steps the worse.

obj_short = build_loss_objective(prob_short,Tsit5(),L2Loss(t_short,data_short),tstops=t_short,reltol=1e-9)
res1 = bboptimize(obj_short;SearchRange = glo_bounds, MaxSteps = 7e3)
println("Real params are:\n$p")
# Change in tolerance makes it worse

obj_short = build_loss_objective(prob_short,Vern9(),L2Loss(t_short,data_short),tstops=t_short,reltol=1e-9,abstol=1e-9)
res1 = bboptimize(obj_short;SearchRange = glo_bounds, MaxSteps = 7e3)
println("Real params are:\n$p")
# using the moe accurate Vern9() reduces the fitness marginally and leads to some increase in time taken

obj_short = build_loss_objective(prob_short,Vern9(),L2Loss(t_short,data_short),tstops=t_short,reltol=1e-9,abstol=1e-9)

opt = Opt(:GN_ORIG_DIRECT_L, 4)
lower_bounds!(opt,[0.0,0.0,0.0,0.0])
upper_bounds!(opt,[5.0,5.0,5.0,5.0])
min_objective!(opt, obj_short.cost_function2)
xtol_rel!(opt,1e-12)
maxeval!(opt, 10000)
@time (minf,minx,ret) = NLopt.optimize(opt,glo_init)
println("\nObjective function:\t$minf\nFor parameters:\n$minx\nReal parameters are:\n$p\nMessage:\t$ret")

obj_short_am(x,y) = sum(abs2.(adams_moulton_estimator(x, data_short, t_short, fitz)))

opt = Opt(:GN_ORIG_DIRECT_L, 4)
lower_bounds!(opt,[0.0,0.0,0.0,0.0])
upper_bounds!(opt,[5.0,5.0,5.0,5.0])
min_objective!(opt, obj_short_am)
xtol_rel!(opt,1e-12)
maxeval!(opt, 10000)
@time (minf,minx,ret) = NLopt.optimize(opt,glo_init)
println("\nObjective function:\t$minf\nFor parameters:\n$minx\nReal parameters are:\n$p\nMessage:\t$ret")
# Very Bad

opt = Opt(:GN_CRS2_LM, 4)
lower_bounds!(opt,[0.0,0.0,0.0,0.0])
upper_bounds!(opt,[5.0,5.0,5.0,5.0])
min_objective!(opt, obj_short_am)
xtol_rel!(opt,1e-12)
maxeval!(opt, 10000)
@time (minf,minx,ret) = NLopt.optimize(opt,glo_init)
println("\nObjective function:\t$minf\nFor parameters:\n$minx\nReal parameters are:\n$p\nMessage:\t$ret")
# Very Good

opt = Opt(:GN_ISRES, 4)
lower_bounds!(opt,[0.0,0.0,0.0,0.0])
upper_bounds!(opt,[5.0,5.0,5.0,5.0])
min_objective!(opt,obj_short_am)
xtol_rel!(opt,1e-12)
maxeval!(opt, 10000)
@time (minf,minx,ret) = NLopt.optimize(opt,glo_init)
println("\nObjective function:\t$minf\nFor parameters:\n$minx\nReal parameters are:\n$p\nMessage:\t$ret")

opt = Opt(:GN_ESCH, 4)
lower_bounds!(opt,[0.0,0.0,0.0,0.0])
upper_bounds!(opt,[5.0,5.0,5.0,5.0])
min_objective!(opt, obj_short_am)
xtol_rel!(opt,1e-12)
maxeval!(opt, 10000)
@time (minf,minx,ret) = NLopt.optimize(opt,glo_init)
println("\nObjective function:\t$minf\nFor parameters:\n$minx\nReal parameters are:\n$p\nMessage:\t$ret")

opt = Opt(:LN_BOBYQA, 4)
lower_bounds!(opt,[0.0,0.0,0.0,0.0])
upper_bounds!(opt,[1.0,1.0,1.0,1.0])
min_objective!(opt, obj_short_am)
xtol_rel!(opt,1e-12)
maxeval!(opt, 10000)
@time (minf,minx,ret) = NLopt.optimize(opt,loc_init)
println("\nObjective function:\t$minf\nFor parameters:\n$minx\nReal parameters are:\n$p\nMessage:\t$ret")
# Very Good

opt = Opt(:LN_NELDERMEAD, 4)
lower_bounds!(opt,[0.0,0.0,0.0,0.0])
upper_bounds!(opt,[1.0,1.0,1.0,1.0])
min_objective!(opt, obj_short_am)
xtol_rel!(opt,1e-12)
maxeval!(opt, 10000)
@time (minf,minx,ret) = NLopt.optimize(opt,loc_init)
println("\nObjective function:\t$minf\nFor parameters:\n$minx\nReal parameters are:\n$p\nMessage:\t$ret")

opt = Opt(:LD_SLSQP, 4)
lower_bounds!(opt,[0.0,0.0,0.0,0.0])
upper_bounds!(opt,[1.0,1.0,1.0,1.0])
min_objective!(opt, obj_short.cost_function2)
xtol_rel!(opt,1e-12)
maxeval!(opt, 10000)
@time (minf,minx,ret) = NLopt.optimize(opt,loc_init)
println("\nObjective function:\t$minf\nFor parameters:\n$minx\nReal parameters are:\n$p\nMessage:\t$ret")

opt = Opt(:LN_COBYLA, 4)
lower_bounds!(opt,[0.0,0.0,0.0,0.0])
upper_bounds!(opt,[1.0,1.0,1.0,1.0])
min_objective!(opt, obj_short_am)
xtol_rel!(opt,1e-12)
maxeval!(opt, 10000)
@time (minf,minx,ret) = NLopt.optimize(opt,loc_init)
println("\nObjective function:\t$minf\nFor parameters:\n$minx\nReal parameters are:\n$p\nMessage:\t$ret")

opt = Opt(:LN_NEWUOA_BOUND, 4)
lower_bounds!(opt,[0.0,0.0,0.0,0.0])
upper_bounds!(opt,[1.0,1.0,1.0,1.0])
min_objective!(opt, obj_short_am)
xtol_rel!(opt,1e-12)
maxeval!(opt, 10000)
@time (minf,minx,ret) = NLopt.optimize(opt,loc_init)
println("\nObjective function:\t$minf\nFor parameters:\n$minx\nReal parameters are:\n$p\nMessage:\t$ret")

opt = Opt(:LN_PRAXIS, 4)
lower_bounds!(opt,[0.0,0.0,0.0,0.0])
upper_bounds!(opt,[1.0,1.0,1.0,1.0])
min_objective!(opt, obj_short_am)
xtol_rel!(opt,1e-12)
maxeval!(opt, 10000)
@time (minf,minx,ret) = NLopt.optimize(opt,loc_init)
println("\nObjective function:\t$minf\nFor parameters:\n$minx\nReal parameters are:\n$p\nMessage:\t$ret")
# Very Good - Improved Speed Tenfold

opt = Opt(:LN_SBPLX, 4)
lower_bounds!(opt,[0.0,0.0,0.0,0.0])
upper_bounds!(opt,[1.0,1.0,1.0,1.0])
min_objective!(opt, obj_short_am)
xtol_rel!(opt,1e-12)
maxeval!(opt, 10000)
@time (minf,minx,ret) = NLopt.optimize(opt,loc_init)
println("\nObjective function:\t$minf\nFor parameters:\n$minx\nReal parameters are:\n$p\nMessage:\t$ret")
# Very Good - Improved Speed Tenfold

opt = Opt(:LD_MMA, 4)
lower_bounds!(opt,[0.0,0.0,0.0,0.0])
upper_bounds!(opt,[1.0,1.0,1.0,1.0])
min_objective!(opt, obj_short.cost_function2)
xtol_rel!(opt,1e-12)
maxeval!(opt, 10000)
@time (minf,minx,ret) = NLopt.optimize(opt,loc_init)
println("\nObjective function:\t$minf\nFor parameters:\n$minx\nReal parameters are:\n$p\nMessage:\t$ret")

obj = build_loss_objective(prob,Vern9(),L2Loss(t,data),tstops=t,reltol=1e-9,abstol=1e-9)
res1 = bboptimize(obj;SearchRange = glo_bounds, MaxSteps = 4e3)

obj_short_am(x) = sum(abs2.(adams_moulton_estimator(x, data, t, fitz)))
res1 = bboptimize(obj_short_am;SearchRange = glo_bounds, MaxSteps = 4e3)
println("Real params are:\n$p")
obj_short_ds(x) = sum(abs2.(data_shooting_estimator(x, data, t, fitz, steps=1, euler=true)))
res1 = bboptimize(obj_short_ds;SearchRange = glo_bounds, MaxSteps = 4e3)
println("Real params are:\n$p")
obj_short_ds(x) = sum(abs2.(data_shooting_estimator(x, data, t, fitz, steps=10, euler=true)))
res1 = bboptimize(obj_short_ds;SearchRange = glo_bounds, MaxSteps = 4e3)
println("Real params are:\n$p")

obj_am(x,y) = sum(abs2.(adams_moulton_estimator(x, data, t, fitz)))

opt = Opt(:GN_ORIG_DIRECT_L, 4)
lower_bounds!(opt,[0.0,0.0,0.0,0.0])
upper_bounds!(opt,[5.0,5.0,5.0,5.0])
min_objective!(opt, obj_am)
xtol_rel!(opt,1e-12)
maxeval!(opt, 10000)
@time (minf,minx,ret) = NLopt.optimize(opt,glo_init)
println("\nObjective function:\t$minf\nFor parameters:\n$minx\nReal parameters are:\n$p\nMessage:\t$ret")
# Very Good!! - Improved Speed and Estimation CAVEAT: Requires somewhat tight bounds.

opt = Opt(:GN_CRS2_LM, 4)
lower_bounds!(opt,[0.0,0.0,0.0,0.0])
upper_bounds!(opt,[5.0,5.0,5.0,5.0])
min_objective!(opt, obj_am)
xtol_rel!(opt,1e-12)
maxeval!(opt, 20000)
@time (minf,minx,ret) = NLopt.optimize(opt,glo_init)
println("\nObjective function:\t$minf\nFor parameters:\n$minx\nReal parameters are:\n$p\nMessage:\t$ret")
# Very Good

opt = Opt(:GN_ISRES, 4)
lower_bounds!(opt,[0.0,0.0,0.0,0.0])
upper_bounds!(opt,[5.0,5.0,5.0,5.0])
min_objective!(opt, obj_am)
xtol_rel!(opt,1e-12)
maxeval!(opt, 50000)
maxtime!(opt, 60)
@time (minf,minx,ret) = NLopt.optimize(opt,glo_init)
println("\nObjective function:\t$minf\nFor parameters:\n$minx\nReal parameters are:\n$p\nMessage:\t$ret")
# Very Good - Improved convergence: AM is better than classic using the same max_time

opt = Opt(:GN_ESCH, 4)
lower_bounds!(opt,[0.0,0.0,0.0,0.0])
upper_bounds!(opt,[5.0,5.0,5.0,5.0])
min_objective!(opt, obj_am)
xtol_rel!(opt,1e-12)
maxeval!(opt, 20000)
maxtime!(opt, 30)
@time (minf,minx,ret) = NLopt.optimize(opt,glo_init)
println("\nObjective function:\t$minf\nFor parameters:\n$minx\nReal parameters are:\n$p\nMessage:\t$ret")
# Very Good - Improved convergence: AM is better than classic using the same max_time

opt = Opt(:LN_BOBYQA, 4)
lower_bounds!(opt,[0.0,0.0,0.0,0.0])
upper_bounds!(opt,[10.0,10.0,10.0,10.0])
min_objective!(opt, obj)
xtol_rel!(opt,1e-12)
maxeval!(opt, 10000)
@time (minf,minx,ret) = NLopt.optimize(opt,glo_init)
println("\nObjective function:\t$minf\nFor parameters:\n$minx\nReal parameters are:\n$p\nMessage:\t$ret")
# Very Good - Improved convergence: AM gets the same results much faster 1.3 seconds vs 0.3 seconds
# Converges even from glo_init
# Both obj and obj_am converge from global_init

opt = Opt(:LN_NELDERMEAD, 4)
lower_bounds!(opt,[0.0,0.0,0.0,0.0])
upper_bounds!(opt,[1.0,1.0,1.0,1.0])
min_objective!(opt, obj_am)
xtol_rel!(opt,1e-9)
maxeval!(opt, 10000)
@time (minf,minx,ret) = NLopt.optimize(opt,loc_init)
println("\nObjective function:\t$minf\nFor parameters:\n$minx\nReal parameters are:\n$p\nMessage:\t$ret")
# Very Good - Improved: Converges

using ForwardDiff

obj_am(x,y) = sum(abs2.(adams_moulton_estimator(x, data, t, fitz)))
obj_am_g(x) = obj_am(x, 0)

function obj_am_grad(x, y)
    g = z -> ForwardDiff.gradient(obj_am_g,z)
    aux = g(y)
    y[:] = aux
    return obj_am_g(x)
end

opt = Opt(:LD_SLSQP, 4)
lower_bounds!(opt,[0.0,0.0,0.0,0.0])
upper_bounds!(opt,[1.0,1.0,1.0,1.0])
min_objective!(opt, obj.cost_function2)
xtol_rel!(opt,1e-12)
maxeval!(opt, 10000)
@time (minf,minx,ret) = NLopt.optimize(opt,loc_init)
println("\nObjective function:\t$minf\nFor parameters:\n$minx\nReal parameters are:\n$p\nMessage:\t$ret")

obj_short = build_loss_objective(prob_short,Tsit5(),L2Loss(t_short,data_short),tstops=t_short)
lower = [0,0,0,0]
upper = [1,1,1,1]
splits = ([0,0.3,0.7],[0,0.3,0.7],[0,0.3,0.7],[0,0.3,0.7])
@time root, x0 = analyze(obj_short,splits,lower,upper)

minimum(root)

obj = build_loss_objective(prob,Vern9(),L2Loss(t,data),tstops=t,reltol=1e-9,abstol=1e-9)
lower = [0,0,0,0]
upper = [5,5,5,5]
splits = ([0,0.5,1],[0,0.5,1],[0,0.5,1],[0,0.5,1])
@time root, x0 = analyze(obj_short,splits,lower,upper)

minimum(root)
