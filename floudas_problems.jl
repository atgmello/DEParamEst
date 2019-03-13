#desired_precision = Float64
desired_precision = BigFloat

struct ChemKinProblem
    fun
    phi
    bounds
    data
    t
end

ode_fun_array = []
bounds_array = Vector{desired_precision}[]
rand_range_array = Vector{desired_precision}[]
phi_array = Vector{desired_precision}[]
ini_cond_array = Vector{desired_precision}[]
#x_t_array = Vector{desired_precision}[]
#delta_t_array = Vector{desired_precision}[]
t_array = Vector{desired_precision}[]
floudas_samples_array = Array{desired_precision}[]
floudas_samples_times_array = Array{desired_precision}[]
floudas_plots = []

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
bounds = [0., 10.]
push!(bounds_array, [0.,10.])
push!(rand_range_array, [0.,10.])
ini_cond = [1., 0.]
push!(ini_cond_array, ini_cond)
t = range(0, 1, length=10)
push!(t_array, t)
floudas_samples = [
                    1. 0.606 0.368 0.223 0.135 0.082 0.050 0.030 0.018 0.011 0.007;
                    0. 0.373 0.564 0.647 0.669 0.656 0.642 0.583 0.539 0.494 0.451
                    ]
push!(floudas_samples_array, floudas_samples)
floudas_samples_times = [
                        0. 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1
                        ]
push!(floudas_samples_times_array, floudas_samples_times)
p = scatter(transpose(floudas_samples_times), transpose(floudas_samples))
push!(floudas_plots, p)

p_one = ChemKinProblem(floudas_one, phi, bounds, floudas_samples, floudas_samples_times)

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
floudas_samples = [
                    1. 0.8241 0.6852 0.5747 0.4867 0.4166 0.3608 0.3164 0.2810 0.2529 0.2304 0.2126 0.1984 0.1870 0.1870 0.1709 0.1651 0.1606 0.1570 0.1541 0.1518;
                    0. 0.0937 0.1345 0.1654 0.1899 0.2094 0.2249 0.2373 0.2472 0.2550 0.2613 0.2662 0.2702 0.2733 0.2759 0.2779 0.2794 0.2807 0.2817 0.2825 0.2832;
                    0. 0.0821 0.1802 0.2598 0.3233 0.3738 0.4141 0.4461 0.4717 0.4920 0.5082 0.5210 0.5313 0.5395 0.5460 0.5511 0.5553 0.5585 0.5612 0.5632 0.5649
                   ]
push!(floudas_samples_array, floudas_samples)
floudas_samples_times = [
                        0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0
                        ]
push!(floudas_samples_times_array, floudas_samples_times)
p = scatter(transpose(floudas_samples_times), transpose(floudas_samples))
push!(floudas_plots, p)

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
floudas_samples = [
                    1. 0.7307 0.5982 0.4678 0.4267 0.3436 0.3126 0.2808 0.2692 0.2210 0.2122 0.1903 0.1735 0.1615 0.1240 0.1190 0.1109 0.0890 0.0820 0.0745 0.0639;
                    0. 0.1954 0.2808 0.3175 0.3047 0.2991 0.2619 0.2391 0.2210 0.1898 0.1801 0.1503 0.1030 0.0964 0.0581 0.0471 0.0413 0.0367 0.0219 0.0124 0.0089
                    ]
push!(floudas_samples_array, floudas_samples)
floudas_samples_times = [
                        0. .025 .05 .075 .1 .125 .150 .175 .2 .225 .25 .3 .35 .4 .45 .5 .55 .65 .75 .85 .95
                        ]
push!(floudas_samples_times_array, floudas_samples_times)
p = scatter(transpose(floudas_samples_times), transpose(floudas_samples))
push!(floudas_plots, p)

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
floudas_samples = [
                    0. 1.4 6.3 10.4 14.2 17.6 21.4 23.0 27.0 30.4 34.4 38.8 41.6 43.5 45.3
                    ]
push!(floudas_samples_array, floudas_samples)
floudas_samples_times = [
                        0. 1.0 2.0 3.0 4.0 5.0 6.0 7.0 9.0 11.0 14.0 19.0 24.0 29.0 39.0
                        ]
push!(floudas_samples_times_array, floudas_samples_times)
p = scatter(transpose(floudas_samples_times), transpose(floudas_samples))
push!(floudas_plots, p)

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
floudas_samples = [
                    1. 0.461 0.426 0.383 0.305 0.195 0.170 0.139 0.112  0.112 0.090 0.082 0.066 0.053 0.043 0.041 0.029;
                    0. 0.114 0.135 0.157 0.194 0.231 0.234 0.228 0.228  0.226 0.220 0.214 0.178 0.188 0.183 0.184 0.166;
                    0. 0.018 0.035 0.045 0.047 0.084 0.095 0.111 0.134 0.168 0.148 0.157 0.206 0.206 0.214 0.213 0.230
                    ]
push!(floudas_samples_array, floudas_samples)
floudas_samples_times = [
                        0. 0.050 0.065 0.080 0.123 0.233 0.273 0.354 0.397 0.418 0.502 0.553 0.681 0.750 0.916 0.937 1.122
                        ]
push!(floudas_samples_times_array, floudas_samples_times)
p = scatter(transpose(floudas_samples_times), transpose(floudas_samples))
push!(floudas_plots, p)

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
floudas_samples = [
                    1.2 0.7990 0.8731 1.2487 1.0362 0.7483 1.0024 1.2816 0.8944 0.7852 1.1527;
                    1.1 1.0758 0.8711 0.9393 1.1468 1.0027 0.8577 1.0274 1.1369 0.9325 0.9074
                    ]
push!(floudas_samples_array, floudas_samples)
floudas_samples_times = [
                        0. 1. 2. 3. 4. 5. 6. 7. 8. 9. 10.
                        ]
push!(floudas_samples_times_array, floudas_samples_times)
p = scatter(transpose(floudas_samples_times), transpose(floudas_samples))
push!(floudas_plots, p)

# ----- Plotting -----
for p in 1:length(floudas_plots)
    print("\n----- Plot for Floudas Problem $p-----\n")
    display(floudas_plots[p])
end
