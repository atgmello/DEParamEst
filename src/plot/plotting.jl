module PlottingUtils

using Plots

function box_data(arr::Array{<:AbstractFloat})::Array{<:AbstractFloat}
    q = quantile(arr,[0.25,0.5,0.75])
    iqr = q[3] - q[1]
    min_val = q[1] - 1.5*iqr
    max_val = q[3] + 1.5*iqr
    return [maximum([min_val,minimum(arr)]),
            q[1],q[2],q[3],
            minimum([max_val,maximum(arr)])]
end

function box_scatter_plot(arr::Array{<:AbstractFloat})::Array{<:AbstractFloat}
    qarr = box_data(arr)
    return [qarr[3]-qarr[1],qarr[3],qarr[5]-qarr[3]]
end

function get_plot_data(results::Dict,
                        noise_level::AbstractArray{Symbol},
                        methods::Vector{Symbol})::Dict
    plot_data = Dict()
    for m in methods
        plot_data[m] = Dict()
        plot_data[m]["error"] = []
        plot_data[m]["time"] = []
        plot_data[m]["est"] = []
    end
    for v in noise_level
        for m in methods
            error = [e[1][1] for e in results[v][m]]
            if length(error) > 0
                push!(plot_data[m]["error"], error)
            else
                push!(plot_data[m]["error"], [NaN])
            end
            time = [e[2][1] for e in results[v][m]]
            if length(time) > 0
                push!(plot_data[m]["time"], time)
            else
                push!(plot_data[m]["time"], [NaN])
            end
            est = [e[3] for e in results[v][m]]
            if length(time) > 0
                push!(plot_data[m]["est"], est)
            else
                push!(plot_data[m]["est"], [NaN])
            end
        end
    end
    return plot_data
end

function error_plots(plot_data::Dict,
                    noise_level::AbstractArray{<:AbstractFloat},
                    methods::Array,
                    method_label::Dict,
                    method_color::Dict,
                    sam::Int,
                    path::String)::Nothing
    """
    Error Plots
    xaxis: Noise Percentage
    yaxis: Mean Error
    """

    p = plot(x=noise_level, y=[],Geom.line,
            Theme(background_color=colorant"white",
                panel_fill=colorant"white",
                major_label_font="Hack",
                minor_label_font="Hack"))

    p = plot(x=noise_level, xlabel="Noise Percentage", ylabel="Error", legend=:outertopright)
    ylim_arr = []
    for m in methods
        p2 = plot(x=noise_level, xlabel="Noise Percentage", ylabel="Error", legend=:outertopright)
        error = plot_data[m]["error"]
        # Proceed only if there are no NaN
        if !any(isnan.(vcat(error...)))
            qerror = hcat(box_scatter_plot.(error)...)
            # Don't append to plot p if data is too far
            # from the expected values (i.e. plot only if
            # data < 5)
            if !any(qerror[3,:] .> 5.0)

                append!(p.layers,layer(x=x1, y=y1, Theme(default_color=colorant"#e6612f"), #Vivid Vermilion
                        Geom.PointGeometry))

#                plot!(p, noise_level, qerror[2,:],
#                            grid=true,
#                            ribbon=(qerror[1,:],
#                                    qerror[3,:]),
#                            fillalpha=.5, label=method_label[m], color=method_color[m])
                push!(ylim_arr, ylims(p))
            end

            plot!(p2, noise_level, qerror[2,:],
                        grid=true,
                        ribbon=(qerror[1,:],
                                qerror[3,:]),
                        fillalpha=.5, label=method_label[m], color=method_color[m])
            #display(p2)

            savefig(p,path*"/error_inter_$(m)_$(sam).pdf")
            savefig(p2,path*"/error_$(m)_$(sam).pdf")
        end
    end
    #=>
    if length(ylim_arr) > 1
        ylims!(p, (minimum(abs.([yl[1] for yl in ylim_arr])),
                    minimum(abs.([yl[2] for yl in ylim_arr]))))
    end
    <=#
    #display(p)

    savefig(p,path*"/error_all_$(sam).pdf")
    nothing
end

"""
Box Error Plots
xaxis: Method
yaxis: Error Distribution
"""
function box_error_plots(plot_data::Dict,
                    var::Symbol,
                    methods::Vector{Symbol},
                    method_label::Dict,
                    method_color::Dict,
                    sam::Symbol,
                    path::String)::Nothing

    p = plot(legend=false, ylabel="Error", xlabel="Method")
    for m in methods
        # Substitute infinite for missing
        # so that boxplot can still work
        # with the data
        data = [ifelse(isinf(x),missing,x) for x in plot_data[m]["error"][1]]
        boxplot!(p, [method_label[m]], data, color=method_color[m])
    end
    savefig(p,joinpath(path,replace("box_$(sam)_$(var)","."=>"")*".pdf"))
end

"""
Parameter Distribution Plots
xaxis: Parameter
yaxis: Value Distribution
"""
function parameter_plots(plot_data::Dict,
                    var::Symbol,
                    methods::Array,
                    method_label::Dict,
                    method_color::Dict,
                    sam::Symbol,
                    path::String)::Nothing
    num_pars = length(plot_data[methods[1]]["est"][1][1])
    p_arr = []
    for m in methods
        p = plot(legend=:outertopright, ylabel="Value", xlabel="Parameter")
    	for i in 1:num_pars
            data = getindex.(plot_data[m]["est"][1],i)
    		if i == 1
    			boxplot!(p, [string(i)], log10.(data), color=method_color[m], label=m)
    		else
    			boxplot!(p, [string(i)], log10.(data), color=method_color[m], label="")
    		end
    	end
        savefig(p,joinpath(path,replace("par_$(m)_$(sam)_$(var)","."=>"")*".pdf"))
    	push!(p_arr,p)
    end

    p = plot(p_arr...,layout=(length(methods),1))
    savefig(p,joinpath(path,replace("par_all_$(sam)_$(var)","."=>"")*".pdf"))
end

"""
Success Rate vs Time Plots
xaxis: Mean Computation Time
yaxis: 1 / Success Rate
"""
function sr_plots(plot_data::Dict,
                    noise_level::AbstractArray{Symbol},
                    methods::Vector{Symbol},
                    method_label::Dict,
                    method_color::Dict,
                    sam::Symbol,
                    path::String)::Nothing

    p = scatter(xlabel="Time", ylabel="1 / Success Rate", legend=:outertopright)
    p2 = scatter(xlabel="Time", ylabel="1 / Success Rate", legend=:outertopright)
    ylim_arr = []
    for m in methods
        p3 = scatter(xlabel="Time", ylabel="1 / Success Rate", legend=:outertopright)

        sr = mean.([step_success_rate.(e) for e in plot_data[m]["error"]])
        isr = 1.0./sr
        timed = plot_data[m]["time"]
        if !any(isnan.(vcat(timed...)))
            qtime = hcat(box_scatter_plot.(timed)...)
            qqtime = box_scatter_plot(qtime[2,:])
            qisr = box_scatter_plot(isr)
            if qisr[2] < 10.0
                scatter!(p, (qqtime[2],qisr[2]),
                            xerror=[(qqtime[1],qqtime[3])],
                            yerror=[(qisr[1],qisr[3])],
                            label=method_label[m], color=method_color[m])
            end

            scatter!(p2, (qtime[2,:], isr),
                        label=method_label[m], color=method_color[m])

            scatter!(p3, (qtime[2,:], isr),
                        label=method_label[m], color=method_color[m],
                        series_annotations = text.(noise_level, :top, 11))

            #display(p3)

            savefig(p3,path*"/sr_$(m)_$(sam).pdf")
        end
    end
    #display(p)
    #display(p2)

    savefig(p, path*"/sr_all_medians_$(sam).pdf")
    savefig(p2, path*"/sr_all_$(sam).pdf")
    nothing
end
function sr_plots(plot_data::Dict,
                    var::Symbol,
                    methods::Vector{Symbol},
                    method_label::Dict,
                    method_color::Dict,
                    sam::Symbol,
                    path::String)::Nothing

    p = scatter(xlabel="Time", ylabel="1 / Success Rate", legend=:outertopright)
    ylim_arr = []
    for m in methods
        sr = mean([step_success_rate(e) for e in plot_data[m]["error"][1]])
        isr = [1.0./sr]
        timed = plot_data[m]["time"][1]

        if !any(isnan.(vcat(timed...)))
            qtime = box_scatter_plot(timed)
            qisr = box_scatter_plot(isr)
            if qisr[2] < 20.0 && qtime[2] < 20.0
                scatter!(p, (qtime[2],qisr[2]),
                            xerror=[(qtime[1],qtime[3])],
                            yerror=[(qisr[1],qisr[3])],
                            label=method_label[m], color=method_color[m])
            end
        end
    end

    #display(p)
    savefig(p,joinpath(path,replace("sr_all_medians_$(sam)_$(var)","."=>"")*".pdf"))
    nothing
end


function plot_compare(data::Vector, data_est::Vector)
    alphabet='A':'Z'
    label_n=reshape(["$i (Nominal)" for i in alphabet[1:length(data[1])]],(1,length(data[1])))
    label_e=reshape(["$i (Estimated)" for i in alphabet[1:length(data[1])]],(1,length(data[1])))
    err = sqrt(nmse(reduce(vcat,data), reduce(vcat,data_est)))
    p = plot(title = "RMSE = $(err)")
    plot!(p, reduce(hcat,data)', label=label_n, markershape=:circle, linestyle=:solid)
    plot!(p, reduce(hcat,data_est)', label=label_e, markershape=:cross, linestyle=:dash)
    display(p)
    return p
end

"""
Overall Efficiency (OE) Plots
xaxis: Method
yaxis: OE Score
"""
function oe_plots(plot_data::Dict,
                    noise_level::AbstractArray{<:AbstractFloat},
                    methods::Array{<:String,1},
                    method_label::Dict,
                    method_color::Dict,
                    sam::Int,
                    path::String)::Nothing

    t_succ = zeros(length(methods))
    @inbounds for i in 1:length(methods)
        m = methods[i]
        error = plot_data[m]["error"]
        sr = mean.([step_success_rate.(e) for e in error])
        time = plot_data[m]["time"]
        mean_time = mean.(time)
        t_succ_arr = mean_time./sr
        t_succ[i] = median(t_succ_arr)
    end

    oe = minimum(t_succ)./t_succ

    p = bar(xlabel="Method",
            ylabel="Overall Efficiency",
            legend=false)
    bar!(p, methods, oe,
            color=[method_color[m] for m in methods])
    #display(p)

    savefig(p, path*"/$(sam)_oe.pdf")
    nothing
end

end
