module ProblemSet

using DifferentialEquations
using ParameterizedFunctions
export DEProblem, get_ode_problem,
        get_problem_key, problem_info, problem_plot

desired_precision = Float64

struct DEProblem
    fun::Function
    phi::Vector
    bounds::Vector
    data::Vector
    t::AbstractArray
end

Base.copy(p::DEProblem) = DEProblem(p.fun, p.phi, p.bounds,
                                    p.data, p.t)

function problem_info(p::DEProblem)::String
    num_states = length(p.data[1])
    num_params = length(p.phi)
    sep = "\n--- // ---\n"

    rep = sep
    res *= "Number of State Variables:\n\t $(num_states)"
    res *= "\n\t Initial Condition: \n\t $(p.data[1])"
    res *= "\n\t Number of Parameters: \n\t $(num_params) "
    res *= "\n\t Parameter Values: \n\t $(p.phi)"
    res *= "\n\t Bounds: \n\t $(p.bounds)"
    res *= sep
    return res
end

function get_problem_key(i::Int=0)::String
    keys = ["floudas_1","floudas_2",
            "floudas_3","floudas_4",
            "floudas_5","floudas_6",
            "bbg","fhn",
            "mapk",
            "gosc",
            "tsmp","cho","exponential"]
    if i > 0
        return keys[i]
    else
        return keys
    end
end

function get_problem(p::String)::DEProblem
    if p == "exponential"
        # ----- Problem 0 -----

        function f_exp(dz_dt, z, phi, t)
            dz_dt[1] = z[1]*phi[1]
        end

        k1 = 1.0
        phi = [k1]
        bounds = [0., 100.]
        ini_cond = [1.]

        t = range(0.0, stop=10.0, length=10)
        de_prob = ODEProblem(f_exp, ini_cond, (t[1],t[end]), phi)
        de_sol = solve(de_prob, saveat=t)
        ode_data = de_sol.u

        return DEProblem(f_exp, phi, bounds, ode_data, t)

    elseif p == "floudas_1"
        # ----- Problem 1 -----

        function floudas_one(dz_dt, z, phi, t)
            r_1 = phi[1]*z[1]
            r_2 = phi[2]*z[2]

            dz_dt[1] = -r_1
            dz_dt[2] = r_1 - r_2
        end

        k1 = 5.0035
        k2 = 1.
        phi = [k1, k2]
        ini_cond = [1., 0.]
        bounds = [[0. for i in 1:length(phi)],
                [10. for i in 1:length(phi)]]
        #=>
        floudas_samples = [
                            1. 0.606 0.368 0.223 0.135 0.082 0.050 0.030 0.018 0.011 0.007;
                            0. 0.373 0.564 0.647 0.669 0.656 0.642 0.583 0.539 0.494 0.451
                            ]
        <=#
        t = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
        de_prob = ODEProblem(floudas_one, ini_cond, (t[1],t[end]), phi)
        de_sol = solve(de_prob, saveat=t)
        ode_data = de_sol.u

        return DEProblem(floudas_one, phi, bounds, ode_data, t)

    elseif p == "floudas_2"
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

        k1 = 4.
        k2 = 2.
        k3 = 40.013
        k4 = 20.007
        phi = [k1, k2, k3, k4]
        ini_cond = [1., 0., 0.]
        bounds = [[0., 0., 10., 10.],
                [10., 10., 50., 50.]]
        #=>
        floudas_samples = [
                            1. 0.8241 0.6852 0.5747 0.4867 0.4166 0.3608 0.3164 0.2810 0.2529 0.2304 0.2126 0.1984 0.1870 0.1870 0.1709 0.1651 0.1606 0.1570 0.1541 0.1518;
                            0. 0.0937 0.1345 0.1654 0.1899 0.2094 0.2249 0.2373 0.2472 0.2550 0.2613 0.2662 0.2702 0.2733 0.2759 0.2779 0.2794 0.2807 0.2817 0.2825 0.2832;
                            0. 0.0821 0.1802 0.2598 0.3233 0.3738 0.4141 0.4461 0.4717 0.4920 0.5082 0.5210 0.5313 0.5395 0.5460 0.5511 0.5553 0.5585 0.5612 0.5632 0.5649
                           ]
        <=#

        t = [0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0]
        de_prob = ODEProblem(floudas_two, ini_cond, (t[1],t[end]), phi)
        de_sol = solve(de_prob, saveat=t)
        ode_data = de_sol.u
        return DEProblem(floudas_two, phi, bounds, ode_data, t)

    elseif p == "floudas_3"
        # ----- Problem 3 -----

        function floudas_three(dz_dt, z, phi, t)
            r_1 = phi[1]*z[1]^2
            r_2 = phi[2]*z[2]
            r_3 = phi[3]*z[1]^2

            dz_dt[1] = - r_1 - r_3
            dz_dt[2] = r_1 - r_2
        end

        k1 = 12.214
        k2 = 7.9798
        k3 = 2.2216
        phi = [k1, k2, k3]
        ini_cond = [1., 0.]
        bounds = [[0. for i in 1:length(phi)],
                [20. for i in 1:length(phi)]]
        #=>
        floudas_samples = [
                            1. 0.7307 0.5982 0.4678 0.4267 0.3436 0.3126 0.2808 0.2692 0.2210 0.2122 0.1903 0.1735 0.1615 0.1240 0.1190 0.1109 0.0890 0.0820 0.0745 0.0639;
                            0. 0.1954 0.2808 0.3175 0.3047 0.2991 0.2619 0.2391 0.2210 0.1898 0.1801 0.1503 0.1030 0.0964 0.0581 0.0471 0.0413 0.0367 0.0219 0.0124 0.0089
                            ]
        <=#
        t =[0.0,.025,.05,.075,.1,.125,.150,.175,.2,.225,.25,.3,.35,.4,.45,.5,.55,.65,.75,.85,.95]
        de_prob = ODEProblem(floudas_three, ini_cond, (t[1],t[end]), phi)
        de_sol = solve(de_prob, saveat=t)
        ode_data = de_sol.u
        return DEProblem(floudas_three, phi, bounds, ode_data, t)

    elseif p == "floudas_4"
        # ----- Problem 4 -----

        function floudas_four(dz_dt, z, phi, t)
            dz_dt[1] = phi[1]*(126.2 - z[1])*(91.9 - z[1])^2 - phi[2]*z[1]^2
        end

        k1 = 4.5704*10^(-6)
        k2 = 2.7845*10^(-4)
        phi = [k1, k2]
        ini_cond = [0.]
        bounds = [[0. for i in 1:length(phi)],
                [.1 for i in 1:length(phi)]]
        #=>
        floudas_samples = [
                            0. 1.4 6.3 10.4 14.2 17.6 21.4 23.0 27.0 30.4 34.4 38.8 41.6 43.5 45.3
                            ]
        <=#
        t = [0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,9.0,11.0,14.0,19.0,24.0,29.0,39.0]
        de_prob = ODEProblem(floudas_four, ini_cond, (t[1],t[end]), phi)
        de_sol = solve(de_prob, saveat=t)
        ode_data = de_sol.u
        return DEProblem(floudas_four, phi, bounds, ode_data, t)

    elseif p == "floudas_5"
        # ----- Problem 5 -----

        function floudas_five(dz_dt, z, phi, t)
            dz_dt[1] = - (2*phi[1] - ((phi[1]*z[2])/((phi[2] + phi[5])*z[1] + z[2])) + phi[3] + phi[4])*z[1]
            dz_dt[2] = ((phi[1]*z[1])*(phi[2]*z[1] - z[2]))/((phi[2] + phi[5])*z[1] + z[2]) + phi[3]*z[1]
            dz_dt[3] = ((phi[1]*z[1])*(z[2] + phi[5]*z[1]))/((phi[2] + phi[5])*z[1] + z[2]) + phi[4]*z[1]
        end

        k1 = 5.2407
        k2 = 1.2176
        k3 = 0.
        k4 = 0.
        k5 = 0.
        phi = [k1, k2, k3, k4, k5]
        ini_cond = [1., 0., 0.]
        bounds = [[0. for i in 1:length(phi)],
                [100. for i in 1:length(phi)]]
        #=>
        floudas_samples = [
                            1. 0.461 0.426 0.383 0.305 0.195 0.170 0.139 0.112  0.112 0.090 0.082 0.066 0.053 0.043 0.041 0.029;
                            0. 0.114 0.135 0.157 0.194 0.231 0.234 0.228 0.228  0.226 0.220 0.214 0.178 0.188 0.183 0.184 0.166;
                            0. 0.018 0.035 0.045 0.047 0.084 0.095 0.111 0.134 0.168 0.148 0.157 0.206 0.206 0.214 0.213 0.230
                            ]
        <=#
        t = [0.,0.050,0.065,0.080,0.123,0.233,0.273,0.354,0.397,0.418,0.502,0.553,0.681,0.750,0.916,0.937,1.122]
        de_prob = ODEProblem(floudas_five, ini_cond, (t[1],t[end]), phi)
        de_sol = solve(de_prob, saveat=t)
        ode_data = de_sol.u
        return DEProblem(floudas_five, phi, bounds, ode_data, t)

    elseif p == "floudas_6"
        # ----- Problem 6 -----

        function floudas_six(dz_dt, z, phi, t)
            dz_dt[1] = phi[1]*z[1]*(1-z[2])
            dz_dt[2] = phi[2]*z[2]*(z[1]-1)
        end

        k1 = 3.2434
        k2 = 0.9209
        phi = [k1, k2]
        ini_cond = [1.2, 1.1]
        bounds = [[0. for i in 1:length(phi)],
                [10. for i in 1:length(phi)]]
        #=>
        floudas_samples = [
                            1.2 0.7990 0.8731 1.2487 1.0362 0.7483 1.0024 1.2816 0.8944 0.7852 1.1527;
                            1.1 1.0758 0.8711 0.9393 1.1468 1.0027 0.8577 1.0274 1.1369 0.9325 0.9074
                            ]
        <=#
        t = [0.,1.,2.,3.,4.,5.,6.,7.,8.,9.,10.]
        de_prob = ODEProblem(floudas_six, ini_cond, (t[1],t[end]), phi)
        de_sol = solve(de_prob, saveat=t)
        ode_data = de_sol.u
        return DEProblem(floudas_six, phi, bounds, ode_data, t)

    elseif p == "bbg"
        # ----- BBG -----
        """
        Biomass Batch Growth
        Robust and Efficient Parameter Estimation in Dynamic
        Models of Biological Systems
        [ROEF]
        """

        phi = [0.4, 5, 0.05, 0.5]

        bounds = [[10^(-5) for i in 1:length(phi)],
                [10^(2) for i in 1:length(phi)]]

        ini_cond = [2.0, 30.0]
        t = range(0.0, stop=12.0, length=7)

        function f_bbg(dz_dt, z, phi, t)
            mi, Ks, Kd, yield = phi
            Cb, Cs = z
            dz_dt[1] = mi*(Cs*Cb/(Ks+Cs))-Kd*Cb
            dz_dt[2] = -(mi/yield)*(Cs*Cb/(Ks+Cs))
        end

        de_prob = ODEProblem(f_bbg, ini_cond, (t[1],t[end]), phi)
        de_sol = solve(de_prob, saveat=t)
        ode_data = de_sol.u
        return DEProblem(f_bbg, phi, bounds, ode_data, t)

    elseif p == "fhn"
        # ----- FHN -----
        """
        FitzHugh-Nagumo neural spike
        Robust and Efficient Parameter Estimation in Dynamic
        Models of Biological Systems
        [ROEF]
        """

        phi = [0.2, 0.2, 3.0]

        bounds = [[10^(-5) for i in 1:length(phi)],
                [10^(5) for i in 1:length(phi)]]

        ini_cond = [-1.0, 1.0]
        t = range(0.0, stop=20.0, length=7)

        function f_fhn(dz_dt, z, phi, t)
            a, b, c = phi
            V, R = z
            u = 0
            dz_dt[1] = c*(V - (V^3)/3 + R) + u
            dz_dt[2] = -(1/c)*(V - a + b*R)
        end

        de_prob = ODEProblem(f_fhn, ini_cond, (t[1],t[end]), phi)
        de_sol = solve(de_prob, saveat=t)
        ode_data = de_sol.u
        return DEProblem(f_fhn, phi, bounds, ode_data, t)

    elseif p == "mapk"
        # ----- MPK -----
        """
        Kholodenko MAPK signalling pathway (MAPK)
        ROEF
        """
        phi = [2.5, 0.25, 0.75, 0.75, 0.5, 0.5]

        bounds = [[0.01 for i in 1:length(phi)],
                [50.0 for i in 1:length(phi)]]

        ini_cond = [90.0, 10.0, 280.0, 10.0, 10.0, 280.0, 10.0, 10.0]
        #t = range(0.0, stop=20.0, length=7)
        t = [50.0, 100.0, 150.0, 200.0, 300.0,
            400.0, 500.0, 600.0, 800.0, 1000.0]

        function f_mapk(dz_dt, z, phi, t)
            J0V1, J1V2, J4V5, J5V6, J8V9, J9V10 = phi
            J0Ki, J0n, J0k1, J1KK2, J2k3, J2KK3, J3k4,
            J3KK4, J4KK5, J5KK6, J6k7, J6KK7, J7k8, J7KK8,
            J8KK9, J9KK10 = convert(Array{eltype(phi)},
                [9.0, 1.0, 10.0, 8.0, 0.025, 15.0, 0.025, 15.0, 15.0,
                15.0, 0.025, 15.0, 0.025, 15.0, 15.0, 15.0])

            RJ0 = J0V1*(z[1]/(1+((z[8]/J0Ki)^J0n)*(J0k1+z[1])))
            RJ1 = J1V2*(z[2]/(J1KK2+z[2]))
            RJ2 = J2k3*(z[2]*z[3]/(J2KK3+z[3]))
            RJ3 = J3k4*(z[2]*z[4]/(J3KK4+z[4]))
            RJ4 = J4V5*(z[5]/(J4KK5+z[5]))
            RJ5 = J5V6*(z[4]/(J5KK6+z[4]))
            RJ6 = J6k7*(z[5]*z[6]/(J6KK7+z[6]))
            RJ7 = J7k8*(z[5]*z[7]/(J7KK8+z[7]))
            RJ8 = J8V9*(z[8]/(J8KK9+z[8]))
            RJ9 = J9V10*(z[7]/(J9KK10+z[7]))

            dz_dt[1] = - RJ0 + RJ1
            dz_dt[2] = RJ0 - RJ1
            dz_dt[3] = - RJ2 + RJ5
            dz_dt[4] = RJ2 - RJ3 + RJ4 - RJ5
            dz_dt[5] = RJ3 - RJ4
            dz_dt[6] = - RJ6 + RJ9
            dz_dt[7] = RJ6 - RJ7 + RJ8 - RJ9
            dz_dt[8] = RJ7 - RJ8
        end

        de_prob = ODEProblem(f_mapk, ini_cond, (t[1],t[end]), phi)
        de_sol = solve(de_prob, saveat=t)
        ode_data = de_sol.u
        return DEProblem(f_mapk, phi, bounds, ode_data, t)

    elseif p == "gosc"
        # ----- GOsc -----
        """
        Goodwin Oscillator
        Robust and Efficient Parameter Estimation in Dynamic
        Models of Biological Systems
        [ROEF]
        """
        phi = [1.0, 0.1, 1.0, 0.1, 1.0, 0.1, 1.0, 10.0]

        bounds = [[10^(-2) for i in 1:length(phi)],
                [10^(2) for i in 1:length(phi)]]
        bounds[1][end] = 1.0
        bounds[2][end] = 12.0

        ini_cond = [0.1, 0.2, 2.5]
        t = range(0.0, stop=240.0, length=10)

        function f_gosc(dz_dt, z, phi, t)
            k1, k2, k3, k4, k5, k6, Ki, n = phi

            dz_dt[1] = k1*Ki^n/(Ki^n + z[3]^n) - k2*z[1]
            dz_dt[2] = k3*z[1] - k4*z[2]
            dz_dt[3] = k5*z[2] - k6*z[3]
        end

        de_prob = ODEProblem(f_gosc, ini_cond, (t[1],t[end]), phi)
        de_sol = solve(de_prob, saveat=t)
        ode_data = de_sol.u
        new_prob = DEProblem(f_gosc, phi, bounds, ode_data, t)

        #=>
        # ----- TGFB -----
        """
        TGF - β signalling pathway model
        Robust and Efficient Parameter Estimation in Dynamic
        Models of Biological Systems
        [ROEF]
        """
        phi = [0.00015, 0.023, 0.01, 0.01, 0.01, 0.1, 0.000403, 0.0026, 0.0056, 0.002, 0.016, 5.7, 0.00657, 0.0017, 1.0, 0.0008, 0.0001, 0.0021, 0.001, 9000.0, 1800.0]

        bounds = [[10^(-3) for i in 1:length(phi)],
                [10^(3) for i in 1:length(phi)]]
        bounds[:,end] = [1, 12]

        ini_cond = [0.1, 0.2, 2.5]
        t = range(0.0, stop=240.0, length=10)

        function f_gosc(dz_dt, z, phi, t)
            CTGFb, CTGFbR, CTGFbTGFbR, CTGFbTGFbRP, CISmadTGFbTGFbRP, CSmad, CSmadP, CCoSmad, CSmadPSmaP, CSmadPCoSmad, CSmadN, CSmadPSmadPN, CSmadPN, CSmadPCoSmadN, CCoSmadN, CISmadmRNA1, CISmadmRNA2, CISmad = z
            k1, k2, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19 = phi
            k3 = 0.01
            k20 = 9000.0
            k21 = 1800.0
            r1 = k1*CTGFbTGFbR
            r2 = k2*CTGFbR*CTGFb
            r3 = k3*CTGFbTGFbR*(1-e^(-((t-k20)/(k21)))^10)
            r4 = k4*CTGFbTGFbRP
            r5 = k5*CTGFbTGFbRP*CISmad
            r6 = k6*CISmadTGFbTGFbRP
            r7 = k7*C Smad*CTGFbTGFbRP
            r8 = k8*CSmad
            r9 = k9*CSmadN
            r10 = k10*2*CSmadP*CSmadP
            r11 = k11*CSmadPCoSmad
            r12 = k12*CSmadP*CCoSmad
            r13 = k11*CSmadP*CoSmad
            r14 = k8*CCoSmad
            r15 = k9*CCoSmadN
            r16 = k12*k8*CSmadPSmaP
            r17 = k8*CSmadP
            r18 = k9*CSmadPN
            r19 = k12*k8*CSmadPCoSmad
            r20 = k13*CSmadPN
            r21 = k10*2*CSmadPNCSmadPN
            r22 = k11*CSmadPSmadPN
            r23 = k10*CSmadPN*CCoSmadN
            r24 = k11*CSmadPCoSmadN
            r25 = k14*((CSmadPCoSmadN^2)\(CSmadPCoSmadN^2+k15^2))
            r26 = k16*CISmadmRNA1
            r27 = k17*CISmadmRNA2
            r28 = k18*CISmadmRNA2
            r29 = k19*CISmad

            dz_dt[1] = r1 - r2
            dz_dt[1] = r1 - r2
            dz_dt[1] = - r1 + r2 - r3 + r4 + r6
            dz_dt[1] = r3 - r4 - r5
            dz_dt[1] = r5 - r6
            dz_dt[1] = - r7 - r8 + r9
            dz_dt[1] = r7 - r10 + r11 - r12 + r13 - r17 + r18
            dz_dt[1] = - r12 + r13 - r14 + r15
            dz_dt[1] = r10 - r11 - r16
            dz_dt[1] = r12 - r13 - r19
            dz_dt[1] = r8 - r9 + r20
            dz_dt[1] = r16 + r21 - r22
            dz_dt[1] = r17 - r18 - r20 - r21 + r22 - r23 + r24
            dz_dt[1] = r19 + r23 - r24
            dz_dt[1] = r14 - r15 - r23 + r24
            dz_dt[1] = r25 - r26
            dz_dt[1] = r26 - r27
            dz_dt[1] = r28 - r29 - r5 + r6
        end

        de_prob = ODEProblem(f_fhn, ini_cond, (t[1],t[end]), phi)
        de_sol = solve(de_prob, OwrenZen3(), saveat=t)
        ode_data = de_sol.u

        new_prob = DEProblem(f_gosc, phi, bounds, ode_data, t)
        push!(problem_set, new_prob)
        <=#

    elseif p == "tsmp"
        # ----- TSP -----
        """
        Three Step Metabolic Pathway from:
        Parameter Estimation in Biochemical Pathways:
        A Comparison of Global Optimization Methods
        [BioParamEst-GO]
        [ROEF]
        """

        s = [.1, .46416, 2.1544, 10]
        p = [.05, .13572, .36840, .1]

        phi = [1.0, 1.0, 2.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0,
                1.0, 2.0, 1.0, 1.0, 1.0, 2.0, 1.0, 2.0, 1.0,
                .1, 1.0, .1, .1, 1.0, .1, .1, 1.0, .1, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        bounds = [[10^(-5) for i in 1:length(phi)],
                [10^(5) for i in 1:length(phi)]]
        for i in [2,4,8,10,14,16]
            bounds[1][i] = 0.1
            bounds[2][i] = 10
        end

        ini_cond = [.6667, .5725, .4176, .4, .3641, .2946, 1.419, .9346]
        t = range(0, stop=120, length=21)

        function f_tsp(dz_dt, z, phi, t)
            G1 = z[1]
            G2 = z[2]
            G3 = z[3]

            E1 = z[4]
            E2 = z[5]
            E3 = z[6]

            M1 = z[7]
            M2 = z[8]

            S = s[3]
            P = p[4]

            V1 = phi[1]
            Ki1 = phi[2]
            ni1 = phi[3]
            Ka1 = phi[4]
            na1 = phi[5]
            k1 = phi[6]
            V2 = phi[7]
            Ki2 = phi[8]
            ni2 = phi[9]
            Ka2 = phi[10]
            na2 = phi[11]
            k2 = phi[12]
            V3 = phi[13]
            Ki3 = phi[14]
            ni3 = phi[15]
            Ka3 = phi[16]
            na3 = phi[17]
            k3 = phi[18]
            V4 = phi[19]
            K4 = phi[20]
            k4 = phi[21]
            V5 = phi[22]
            K5 = phi[23]
            k5 = phi[24]
            V6 = phi[25]
            K6 = phi[26]
            k6 = phi[27]
            kcat1 = phi[28]
            Km1 = phi[29]
            Km2 = phi[30]
            kcat2 = phi[31]
            Km3 = phi[32]
            Km4 = phi[33]
            kcat3 = phi[34]
            Km5 = phi[35]
            Km6 = phi[36]

            dz_dt[1] = (V1/(1+(P/Ki1)^ni1+(Ka1/S)^na1))-k1*G1
            dz_dt[2] = (V2/(1+(P/Ki2)^ni2+(Ka2/M1)^na2))-k2*G2
            dz_dt[3] = (V3/(1+(P/Ki3)^ni3+(Ka3/M2)^na3))-k3*G3

            dz_dt[4] = (V4*G1/(K4+G1))-k4*E1
            dz_dt[5] = (V5*G2/(K5+G2))-k5*E2
            dz_dt[6] = (V6*G3/(K6+G3))-k6*E3

            dz_dt[7] = ((kcat1*E1*(1/Km1)*(S-M1)))/(1+(S/Km1)+(M1/Km2)) - ((kcat2*E2*(1/Km3)*(M1-M2)))/(1+(M1/Km3)+(M2/Km4))
            dz_dt[8] = ((kcat2*E2*(1/Km3)*(M1-M2)))/(1+(M1/Km3)+(M2/Km4)) - ((kcat3*E3*(1/Km5)*(M2-P)))/(1+(M2/Km5)+(P/Km6))
        end

        de_prob = ODEProblem(f_tsp, ini_cond, (t[1],t[end]), phi)
        de_sol = solve(de_prob, saveat=t)
        ode_data = de_sol.u
        return DEProblem(f_tsp, phi, bounds, ode_data, t)

    elseif p == "cho"
    # ----- CHO -----
	# [BioPreDyn]

        x0 = zeros(Float64, 35)
        x0[1] = 5
        x0[2] = 1
        x0[3] = 5
        x0[4] = 5
        x0[5] = 1
        x0[6] = 1
        x0[7] = 1
        x0[8] = 1
        x0[9] = 1
        x0[10] = 1
        x0[11] = 1
        x0[12] = 1
        x0[13] = 1
        x0[14] = 1
        x0[15] = 1
        x0[16] = 1
        x0[17] = 1
        x0[18] = 1
        x0[19] = 1
        x0[20] = 1
        x0[21] = 1
        x0[22] = 1
        x0[23] = 1
        x0[24] = 1
        x0[25] = 1
        x0[26] = 1
        x0[27] = 1
        x0[28] = 1
        x0[29] = 1
        x0[30] = 1
        x0[31] = 1
        x0[32] = 1
        x0[33] = 1
        x0[34] = 1
        x0[35] = 6

        p = zeros(Float64, 117)
        p[1] = 1000
        p[2] = 1000
        p[3] = 1000
        p[4] = 1000
        p[5] = 1000
        p[6] = 1000
        p[7] = 1000
        p[8] = 1000
        p[9] = 1
        p[10] = 1
        p[11] = 0.7
        p[12] = 0.7
        p[13] = 0.7
        p[14] = 0.7
        p[15] = 0.2
        p[16] = 0.2
        p[17] = 1
        p[18] = 1
        p[19] = 0.7
        p[20] = 0.7
        p[21] = 0.05
        p[22] = 1
        p[23] = 0.5
        p[24] = 0.7
        p[25] = 0.5
        p[26] = 2
        p[27] = 0.7
        p[28] = 0.7
        p[29] = 2
        p[30] = 0.2
        p[31] = 0.2
        p[32] = 1
        p[33] = 0.7
        p[34] = 0.2
        p[35] = 0.21
        p[36] = 1
        p[37] = 0.7
        p[38] = 1
        p[39] = 0.2
        p[40] = 0.2
        p[41] = 0.05
        p[42] = 0.01
        p[43] = 0.7
        p[44] = 1
        p[45] = 0.5
        p[46] = 1
        p[47] = 0.04
        p[48] = 0.01
        p[49] = 0.01
        p[50] = 0.01
        p[51] = 1
        p[52] = 1
        p[53] = 0.5
        p[54] = 1
        p[55] = 1
        p[56] = 0.5
        p[57] = 0.1
        p[58] = 0.1
        p[59] = 0.5
        p[60] = 1
        p[61] = 1
        p[62] = 1
        p[63] = 0.3
        p[64] = 1
        p[65] = 1
        p[66] = 0.7
        p[67] = 0.4
        p[68] = 0.7
        p[69] = 1
        p[70] = 0.7
        p[71] = 0.7
        p[72] = 2
        p[73] = 0.2
        p[74] = 2
        p[75] = 1
        p[76] = 0.7
        p[77] = 0.7
        p[78] = 0.2
        p[79] = 0.7
        p[80] = 0.5
        p[81] = 0.21
        p[82] = 0.2
        p[83] = 0.2
        p[84] = 0.01
        p[85] = 2
        p[86] = 2
        p[87] = 2
        p[88] = 2
        p[89] = 2
        p[90] = 2
        p[91] = 2
        p[92] = 2
        p[93] = 1
        p[94] = 1
        p[95] = 1
        p[96] = 1
        p[97] = 1
        p[98] = 0.5
        p[99] = 0.7
        p[100] = 0.7
        p[101] = 2
        p[102] = 0.2
        p[103] = 2
        p[104] = 1
        p[105] = 1
        p[106] = 0.7
        p[107] = 1
        p[108] = 0.7
        p[109] = 1
        p[110] = 0.7
        p[111] = 1
        p[112] = 0.7
        p[113] = 2
        p[114] = 2
        p[115] = 0.7
        p[116] = 0.2
        p[117] = 0.2

        t = collect(1.0:20.0)
        bounds = [[10^-5 for i in 1:length(p)],
                [10^5 for i in 1:length(p)]]

        function f_cho(dx_dt, x, par, t)
            p = zeros(347)
            p[118] = 1
            p[119] = 1
            p[120] = 1
            p[121] = 1
            p[122] = 1
            p[123] = 1
            p[124] = 1
            p[125] = 1
            p[126] = 1
            p[127] = 1
            p[128] = 1
            p[129] = 1
            p[130] = 1
            p[131] = 1
            p[132] = 1
            p[133] = 1
            p[134] = 1
            p[135] = 1
            p[136] = 1
            p[137] = 1
            p[138] = 1
            p[139] = 1
            p[140] = 1
            p[141] = 1
            p[142] = 1
            p[143] = 1
            p[144] = 1
            p[145] = 1
            p[146] = 1
            p[147] = 1
            p[148] = 1
            p[149] = 1
            p[150] = 0
            p[151] = 141.471
            p[152] = 0.9
            p[153] = 0.1
            p[154] = 83402
            p[155] = 127877
            p[156] = 603.414
            p[157] = 603.414
            p[158] = 5.02845
            p[159] = 21722.9
            p[160] = 603.414
            p[161] = 361200
            p[162] = 673524
            p[163] = 1.62679e+06
            p[164] = 164390
            p[165] = 36512.9
            p[166] = 166804
            p[167] = 39530
            p[168] = 1810.24
            p[169] = 127877
            p[170] = 166201
            p[171] = 1.94454e+06
            p[172] = 5.02845
            p[173] = 317755
            p[174] = 0
            p[175] = 532531
            p[176] = 38926.6
            p[177] = 40133.4
            p[178] = 689982
            p[179] = 4.45873e+06
            p[180] = 2413.66
            p[181] = 83402
            p[182] = 603.414
            p[183] = 603.414
            p[184] = 2413.66
            p[185] = 533135
            p[186] = 100000
            p[187] = 1
            p[188] = 1000
            p[189] = 1000
            p[190] = 30
            p[191] = 1000
            p[192] = 1000
            p[193] = 1000
            p[194] = 1000
            p[195] = 1000
            p[196] = 1000
            p[197] = 1000
            p[198] = 3000
            p[199] = 1000
            p[200] = 1000
            p[201] = 1000
            p[202] = 100
            p[203] = 100
            p[204] = 100
            p[205] = 100
            p[206] = 1000
            p[207] = 1000
            p[208] = 1000
            p[209] = 1000
            p[210] = 1000
            p[211] = 1000
            p[212] = 1000
            p[213] = 1000
            p[214] = 1000
            p[215] = 3000
            p[216] = 100000
            p[217] = 1000
            p[218] = 1000
            p[219] = 1000
            p[220] = 1000
            p[221] = 1000
            p[222] = 1000
            p[223] = 1000
            p[224] = 1000
            p[225] = 1000
            p[226] = 1000
            p[227] = 1000
            p[228] = 0
            p[229] = 1
            p[230] = 0
            p[231] = 0
            p[232] = 1
            p[233] = 1
            p[234] = 1
            p[235] = -0.7
            p[236] = -0.7
            p[237] = 0.7
            p[238] = 0.7
            p[239] = -0.2
            p[240] = -0.2
            p[241] = 1
            p[242] = 1
            p[243] = -0.7
            p[244] = -0.7
            p[245] = 0.05
            p[246] = 1
            p[247] = 0.5
            p[248] = -0.7
            p[249] = -0.5
            p[250] = 2
            p[251] = 0.7
            p[252] = 0.7
            p[253] = -2
            p[254] = -0.2
            p[255] = -0.2
            p[256] = 1
            p[257] = 0.7
            p[258] = -0.2
            p[259] = -0.21
            p[260] = 0
            p[261] = 1
            p[262] = 0.7
            p[263] = 1
            p[264] = -0.2
            p[265] = -0.2
            p[266] = 0.05
            p[267] = 0
            p[268] = -0.01
            p[269] = 0.7
            p[270] = 1
            p[271] = -0.5
            p[272] = -1
            p[273] = 0.04
            p[274] = -0.01
            p[275] = -0.01
            p[276] = -0.01
            p[277] = 1
            p[278] = 1
            p[279] = 0.5
            p[280] = -1
            p[281] = -1
            p[282] = -0.5
            p[283] = 0.1
            p[284] = -0.1
            p[285] = 0.5
            p[286] = 1
            p[287] = 1
            p[288] = -1
            p[289] = -0.3
            p[290] = -1
            p[291] = 1
            p[292] = 0.7
            p[293] = -0.4
            p[294] = -0.7
            p[295] = 1
            p[296] = -0.7
            p[297] = 0.7
            p[298] = 2
            p[299] = -0.2
            p[300] = -2
            p[301] = 1
            p[302] = 0.7
            p[303] = 0.7
            p[304] = 0.2
            p[305] = 0.7
            p[306] = -0.5
            p[307] = -0.21
            p[308] = -0.2
            p[309] = -0.2
            p[310] = 0
            p[311] = 0
            p[312] = -0.01
            p[313] = 0.7
            p[314] = -0.2
            p[315] = 2
            p[316] = 2
            p[317] = -2
            p[318] = -2
            p[319] = 2
            p[320] = 2
            p[321] = -2
            p[322] = -2
            p[323] = 1
            p[324] = 1
            p[325] = -1
            p[326] = -1
            p[327] = 1
            p[328] = -0.5
            p[329] = 0.7
            p[330] = 0.7
            p[331] = 2
            p[332] = -0.2
            p[333] = -2
            p[334] = 1
            p[335] = 1
            p[336] = -0.7
            p[337] = 1
            p[338] = -0.7
            p[339] = 1
            p[340] = -0.7
            p[341] = 1
            p[342] = -0.7
            p[343] = 2
            p[344] = -2
            p[345] = 0.7
            p[346] = -0.2
            p[347] = -0.2

            for i in 1:length(par)
                p[i] = par[i]
            end
            #endif /* FIXED */

            #ifdef ASSIGNMENT
            y = zeros(promote_type(eltype(x),eltype(t)),102)
            y[1] = log(x[1])
            y[2] = log(x[2])
            y[3] = log(x[3])
            y[4] = log(x[4])
            y[5] = log(x[5])
            y[6] = log(x[6])
            y[7] = log(x[7])
            y[8] = log(x[8])
            y[9] = log(x[9])
            y[10] = log(x[10])
            y[11] = log(x[11])
            y[12] = log(x[12])
            y[13] = log(x[13])
            y[14] = log(x[14])
            y[15] = log(x[15])
            y[16] = log(x[16])
            y[17] = log(x[17])
            y[18] = log(x[18])
            y[19] = log(x[19])
            y[20] = log(x[20])
            y[21] = log(x[21])
            y[22] = log(x[22])
            y[23] = log(x[23])
            y[24] = log(x[24])
            y[25] = log(x[25])
            y[26] = log(x[26])
            y[27] = log(x[27])
            y[28] = log(x[28])
            y[29] = log(x[29])
            y[30] = log(x[30])
            y[31] = log(x[31])
            y[32] = log(x[32])
            y[33] = log(x[33])
            y[34] = log(x[34])
            y[35] = log(x[35])
            y[36] = x[1]*p[186]
            y[37] = x[2]*p[187]
            y[38] = x[3]*p[188]
            y[39] = x[4]*p[189]
            y[40] = x[5]*p[190]
            y[41] = x[6]*p[191]
            y[42] = x[7]*p[192]
            y[43] = x[8]*p[193]
            y[44] = x[9]*p[194]
            y[45] = x[10]*p[195]
            y[46] = x[11]*p[196]
            y[47] = x[12]*p[197]
            y[48] = x[13]*p[198]
            y[49] = x[14]*p[199]
            y[50] = x[15]*p[200]
            y[51] = x[16]*p[201]
            y[52] = x[17]*p[202]
            y[53] = x[18]*p[203]
            y[54] = x[19]*p[204]
            y[55] = x[20]*p[205]
            y[56] = x[21]*p[206]
            y[57] = x[22]*p[207]
            y[58] = x[23]*p[208]
            y[59] = x[24]*p[209]
            y[60] = x[25]*p[210]
            y[61] = x[26]*p[211]
            y[62] = x[27]*p[212]
            y[63] = x[28]*p[213]
            y[64] = x[29]*p[214]
            y[65] = x[30]*p[215]
            y[66] = x[31]*p[216]
            y[67] = x[32]*p[217]
            y[68] = x[33]*p[218]
            y[69] = x[34]*p[219]
            if t>5 && t<10
                y[70] = p[150]
            else
                y[70] = 0.00000000000000000
            end
            y[71] = 0.00000000000000000
            y[72] = 0.00000000000000000
            y[73] = 0.00000000000000000
            y[74] = 0.00000000000000000
            y[75] = 0.00000000000000000
            y[76] = p[123]*p[159]*(1.00000000000000000+p[233]*y[8]+p[234]*y[9]+p[235]*y[6]+p[236]*y[7])
            y[77] = p[124]*p[160]*(1.00000000000000000+p[237]*y[12]+p[238]*y[13]+p[239]*y[10]+p[240]*y[11])
            y[78] = p[125]*p[161]*(1.00000000000000000+p[241]*y[15]+p[242]*y[6]+p[243]*y[8]+p[244]*y[14]+p[245]*y[16])
            y[79] = p[126]*p[162]*(1.00000000000000000+p[246]*y[16]+p[247]*y[7]+p[248]*y[15]+p[249]*y[9])
            y[80] = p[127]*p[163]*(1.00000000000000000+p[250]*y[9]+p[251]*y[19]+p[252]*y[20]+p[253]*y[7]+p[254]*y[17]+p[255]*y[18])
            y[81] = p[128]*p[164]*(1.00000000000000000+p[256]*y[22]+p[257]*y[11]+p[258]*y[21]+p[259]*y[13]+p[260]*y[13])
            y[82] = p[129]*p[165]*(1.00000000000000000+p[261]*y[15]+p[262]*y[7]+p[263]*y[21]+p[264]*y[8]+p[265]*y[9]+p[266]*y[32]+p[267]*y[9]+p[268]*y[30])
            y[83] = p[130]*p[166]*(1.00000000000000000+p[269]*y[25]+p[270]*y[26]+p[271]*y[23]+p[272]*y[24]+p[273]*y[11]+p[274]*y[11]+p[275]*y[13]+p[276]*y[22])
            y[84] = p[131]*p[167]*(1.00000000000000000+p[277]*y[28]+p[278]*y[29]+p[279]*y[23]+p[280]*y[12]+p[281]*y[27]+p[282]*y[25]+p[283]*y[27]+p[284]*y[10])
            y[85] = p[132]*p[168]*(1.00000000000000000+p[285]*y[32]+p[286]*y[22]+p[287]*y[16]+p[288]*y[15]+p[289]*y[30]+p[290]*y[27])
            y[86] = p[133]*p[169]*(1.00000000000000000+p[291]*y[21]+p[292]*y[23]+p[293]*y[2]+p[294]*y[25])
            y[87] = p[134]*p[170]*(1.00000000000000000+p[295]*y[24]+p[296]*y[22])
            y[88] = p[135]*p[171]*(1.00000000000000000+p[297]*y[20]+p[298]*y[17]+p[299]*y[18]+p[300]*y[19])
            y[89] = p[136]*(p[172]/(p[209]/p[220]*(p[210]/p[221])*(p[197]/p[222])*(p[218]/p[223])*(p[219]/p[224])*(p[214]/p[225])*(p[195]/p[226])*(p[198]/p[227])/((1.00000000000000000+p[209]/p[220])*(1.00000000000000000+p[210]/p[221])*(1.00000000000000000+p[197]/p[222])*(1.00000000000000000+p[218]/p[223])*(1.00000000000000000+p[219]/p[224])*(1.00000000000000000+p[214]/p[225])*(1.00000000000000000+p[195]/p[226])*(1.00000000000000000+p[198]/p[227]))))*(x[24]*p[209]/p[220]*(x[25]*p[210]/p[221])*(x[12]*p[197]/p[222])*(x[33]*p[218]/p[223])*(x[34]*p[219]/p[224])*(x[29]*p[214]/p[225])*(x[10]*p[195]/p[226])*(x[13]*p[198]/p[227])/((1.00000000000000000+x[24]*p[209]/p[220])*(1.00000000000000000+x[25]*p[210]/p[221])*(1.00000000000000000+x[12]*p[197]/p[222])*(1.00000000000000000+x[33]*p[218]/p[223])*(1.00000000000000000+x[34]*p[219]/p[224])*(1.00000000000000000+x[29]*p[214]/p[225])*(1.00000000000000000+x[10]*p[195]/p[226])*(1.00000000000000000+x[13]*p[198]/p[227])))
            y[90] = p[137]*p[173]*(1.00000000000000000+p[301]*y[8]+p[302]*y[7]+p[303]*y[19]+p[304]*y[31]+p[305]*y[32]+p[306]*y[16]+p[307]*y[9]+p[308]*y[17]+p[309]*y[30]+p[310]*y[31]+p[311]*y[9]+p[312]*y[15])
            y[91] = p[138]*p[174]*(1.00000000000000000+p[313]*y[18]+p[314]*y[20])
            y[92] = p[139]*p[175]*(1.00000000000000000+p[315]*y[11]+p[316]*y[30]+p[317]*y[32]+p[318]*y[13])
            y[93] = p[140]*p[176]*(1.00000000000000000+p[319]*y[8]+p[320]*y[27]+p[321]*y[28]+p[322]*y[16])
            y[94] = p[141]*p[177]*(1.00000000000000000+p[323]*y[14]+p[324]*y[12]+p[325]*y[29]+p[326]*y[6])
            y[95] = p[142]*p[178]*(1.00000000000000000+p[327]*y[13]+p[328]*y[11])
            y[96] = p[143]*p[179]*(1.00000000000000000+p[329]*y[32]+p[330]*y[31]+p[331]*y[18]+p[332]*y[30]+p[333]*y[20])
            y[97] = p[144]*p[180]*(1.00000000000000000+p[334]*y[27]+p[335]*y[31]+p[336]*y[16])
            y[98] = p[145]*p[181]*(1.00000000000000000+p[337]*y[1]+p[338]*y[26])
            y[99] = p[146]*p[182]*(1.00000000000000000+p[339]*y[3]+p[340]*y[33])
            y[100] = p[147]*p[183]*(1.00000000000000000+p[341]*y[4]+p[342]*y[34])
            y[101] = p[148]*p[184]*(1.00000000000000000+p[343]*y[6]+p[344]*y[12])
            y[102] = p[149]*p[185]*(1.00000000000000000+p[345]*y[18]+p[346]*y[31]+p[347]*y[20])
            #endif /* ASSIGNMENT */

            #ifdef ODEs
            dx_dt[1] = (y[71]-y[98]*1.00000000000000000/p[151]-x[1]*p[186]/x[35]*y[70])/p[186]
            dx_dt[2] = ((-y[72])+y[86]*1.00000000000000000/p[151]-x[2]*p[187]/x[35]*y[70])/p[187]
            dx_dt[3] = (y[73]-y[99]*1.00000000000000000/p[151]-x[3]*p[188]/x[35]*y[70])/p[188]
            dx_dt[4] = (y[74]-y[100]*1.00000000000000000/p[151]-x[4]*p[189]/x[35]*y[70])/p[189]
            dx_dt[5] = ((-y[75])+y[89]*1.00000000000000000/p[151]-x[5]*p[190]/x[35]*y[70])/p[190]
            dx_dt[6] = (y[76]-y[78]+y[94]*p[152]/p[153]-y[101]*p[152]/p[153])/p[191]
            dx_dt[7] = (y[76]-y[79]+y[80]-2.00000000000000000*y[82]*p[152]/p[153]-y[90])/p[192]
            dx_dt[8] = ((-y[76])+y[78]+y[82]*p[152]/p[153]-y[90]-y[93]*p[152]/p[153])/p[193]
            dx_dt[9] = ((-y[76])+y[79]-y[80]+2.00000000000000000*y[82]*p[152]/p[153]+y[90])/p[194]
            dx_dt[10] = (y[77]-120.00000000000000000*y[89])/p[195]
            dx_dt[11] = (y[77]-y[81]+1260.00000000000000000*y[89]-y[92]+y[95])/p[196]
            dx_dt[12] = ((-y[77])+y[84]-240.00000000000000000*y[89]-y[94]+y[101])/p[197]
            dx_dt[13] = ((-y[77])+y[81]-1260.00000000000000000*y[89]+y[92]-y[95])/p[198]
            dx_dt[14] = (y[78]-y[94]*p[152]/p[153])/p[199]
            dx_dt[15] = ((-y[78])+y[79]-y[82]*p[152]/p[153]+y[85]*p[152]/p[153])/p[200]
            dx_dt[16] = ((-y[79])-y[85]*p[152]/p[153]+y[90]+y[93]*p[152]/p[153]+y[97]*p[152]/p[153])/p[201]
            dx_dt[17] = (2.00000000000000000*y[80]-2.00000000000000000*y[88]+2.00000000000000000*y[90])/p[202]
            dx_dt[18] = (4.00000000000000000*y[80]+6.00000000000000000*y[88]-y[91]-3.00000000000000000*y[96]-y[102]*p[152]/p[153])/p[203]
            dx_dt[19] = ((-2.00000000000000000)*y[80]+2.00000000000000000*y[88]-2.00000000000000000*y[90])/p[204]
            dx_dt[20] = ((-4.00000000000000000)*y[80]-6.00000000000000000*y[88]+y[91]+3.00000000000000000*y[96]+y[102]*p[152]/p[153])/p[205]
            dx_dt[21] = (y[81]-y[82]-y[86])/p[206]
            dx_dt[22] = ((-y[81])-y[85]+y[87])/p[207]
            dx_dt[23] = (y[83]-y[84]-y[86]+120.00000000000000000*y[89])/p[208]
            dx_dt[24] = (y[83]-y[87]-120.00000000000000000*y[89])/p[209]
            dx_dt[25] = ((-y[83])+y[84]+y[86]-120.00000000000000000*y[89])/p[210]
            dx_dt[26] = ((-0.50000000000000000)*y[83]+y[98])/p[211]
            dx_dt[27] = (y[84]+y[85]-y[93]-y[97])/p[212]
            dx_dt[28] = ((-y[84])+120.00000000000000000*y[89]+y[93])/p[213]
            dx_dt[29] = ((-y[84])-120.00000000000000000*y[89]+y[94])/p[214]
            dx_dt[30] = (y[85]*p[152]/p[153]+y[90]-y[92]*p[152]/p[153]+y[96])/p[215]
            dx_dt[31] = ((-y[90])-y[96]-y[97]*p[152]/p[153]+y[102]*p[152]/p[153])/p[216]
            dx_dt[32] = ((-y[85])*p[152]/p[153]-y[90]+y[92]*p[152]/p[153]-y[96])/p[217]
            dx_dt[33] = ((-120.00000000000000000)*y[89]+y[99])/p[218]
            dx_dt[34] = ((-120.00000000000000000)*y[89]+y[100])/p[219]
            dx_dt[35] = y[70]
        end

        de_prob = ODEProblem(f_cho, x0, (t[1],t[end]), p)
        de_sol = solve(de_prob, saveat=t)
        ode_data = de_sol.u
        return DEProblem(f_cho, p, bounds, ode_data, t)

    end
end

end
