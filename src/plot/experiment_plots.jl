using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using Gadfly
using JLSO
using Revise
includet("./utils.jl")
using .PlottingUtils: get_plot_data, sr_plots,
                        box_error_plots, oe_plots


"""
Load serialized results and save the
most relevant plots
"""
function plot_main_results(results::Dict,
                           path::String,
                           methods::Vector=[:SS,:DSS])::Nothing

    first_key = collect(keys(results))[1]
    first_result = results[first_key]
    samples = collect(keys(first_result))
    noise_levels = Array{Symbol}(collect(keys(first_result[samples[1]])))
    #methods = Array{Symbol}(collect(keys(first_result[samples[1]][noise_levels[1]])))

    # cur_colors = [colorant"#e6612f",colorant"#42b3d5",colorant"seagreen"]
    # [Flame, Maximum Blue, Seagreen]
    cur_colors = [colorant"#ff595e",colorant"#ffca3a",colorant"#8ac926",colorant"#1982c4"]
    # [Sizzling Red, Sunglow, Yellow Green, Green Blue Crayola]

    method_label = Dict()
    for m in methods
        method_label[m] = String(m)
    end

    method_color = Dict()
    for (m,c) in zip([:SS,:SSR,:DS,:DSS],cur_colors)
        method_color[m] = c
    end

    for (method,result) in results
        dir_path = joinpath(path,string(method))
        mkdir(dir_path)

        for sam in samples
            res = result[sam]
            plot_data = get_plot_data(res,noise_levels,methods)

            sr_plots(plot_data,noise_levels,methods,
                    method_label,method_color,sam,dir_path)

            oe_plots(plot_data,noise_levels,methods,
                    method_label,method_color,sam,dir_path)

            if :DS in methods
                oe_plots(plot_data,noise_levels,filter(x -> x != :DS, methods),
                         method_label,method_color,sam,dir_path)
            end

            for noise in noise_levels
                plot_data = get_plot_data(res,[noise],methods)

                box_error_plots(plot_data,noise,methods,
                                  method_label,method_color,sam,dir_path)

                #parameter_plots(plot_data,noise,methods,
                #                   method_label,method_color,sam,dir_path)

                sr_plots(plot_data,noise,methods,
                        method_label,method_color,sam,dir_path)

                oe_plots(plot_data,[noise],methods,
                        method_label,method_color,sam,dir_path)

            end
        end
    end
end


function main(args::Array{<:String})::Nothing
    jlso_file_or_path = string(args[1])
    save_path = string(args[2])

    println(jlso_file_or_path)

    if isdir(jlso_file_or_path)
        jlso_files = map(file -> joinpath(jlso_file_or_path, file),
                         readdir(jlso_file_or_path))
    else
        jlso_files = [jlso_file_or_path]
    end

    # base_dir = "/home/andrew/git/DEParamEst/"
    # save_path = joinpath(base_dir,"results/remote/results_low/")
    # jlso_file = joinpath(save_path,"experiment_results.jlso")
    results = map(JLSO.load, jlso_files)

    for r in results
        plot_main_results(r, save_path, [:SS, :SSR, :DS, :DSS])
    end
end

main(ARGS)

