module PlottingUtils

using Gadfly
using ColorSchemes
using DataFrames
using Statistics: mean, quantile
import Cairo,
      Fontconfig

const PLOT_FONT = "Arial"
const VTR = 0.125

"""
Adds noise to a Vector of Vector
Used for adding simulated noise to
the states of a system.
Assumes that the values can be negative.
"""
function add_noise(data::Vector{Vector{T}}, percent::T,
                   positive::Bool=false)::Vector{Vector{T}} where T
    if percent == 0.0
        return data
    end

    noise_data = deepcopy(data)
    epsilon_arr = [0.01*mean(getindex.(data,i)) for i in 1:length(data[1])]
    sigma = zero(T)
    @inbounds for i in 1:length(noise_data)
        for j in 1:length(noise_data[1])
            sigma = abs(percent*noise_data[i][j] + epsilon_arr[j])
            d = Normal(0,sigma)
            noise_data[i][j] += rand(d)
            noise_data[i][j] = positive ? abs(noise_data[i][j]) : noise_data[i][j]
        end
    end
    return noise_data
end


function step_success_rate(x::T)::Int64 where T
    if x < VTR
       return 1
    else
        return 0
    end
end


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
    return [qarr[1],qarr[3],qarr[5]]
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


"""
Error Plots
xaxis: Noise Percentage
yaxis: Mean Error
"""
function error_plots(plot_data::Dict,
                    noise_level::AbstractArray{<:AbstractFloat},
                    methods::Array,
                    method_label::Dict,
                    method_color::Dict,
                    sam::Int,
                    path::String)::Nothing

    p = plot(x=noise_level, y=[],Geom.line,
            Theme(background_color=colorant"white",
                panel_fill=colorant"white",
                major_label_font=PLOT_FONT,
                minor_label_font=PLOT_FONT))

    p = plot(x=noise_level, xlabel="Noise Percentage", ylabel="NRMSE", legend=:outertopright)
    ylim_arr = []
    for m in methods
        p2 = plot(x=noise_level, xlabel="Noise Percentage", ylabel="NRMSE", legend=:outertopright)
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

    df = DataFrame()
    for m in methods
        # Substitute infinite for missing
        # so that boxplot can still work
        # with the data
        data = [ifelse(isinf(x),missing,x) for x in plot_data[m]["error"][1]]
        aux_df = DataFrame(method=method_label[m],
                    data=data, color=method_color[m])
        try
            append!(df, aux_df)
        catch e
            @show e
        end
    end
    p = plot(df, x="method", y="data", color="color", Geom.boxplot,
                Guide.xlabel("Method"), Guide.ylabel("NRMSE"),
                Theme(background_color=colorant"white",
                    panel_fill=colorant"white",
                    major_label_font=PLOT_FONT,
                    minor_label_font=PLOT_FONT))
    try
        p |> PDF(joinpath(path,replace("box_$(sam)_$(var)","."=>"")*".pdf"))
    catch e
        @show e
        println(df)
        println()
    end
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

    p = plot(x=[], y=[], Geom.point,
                Guide.xlabel("Time"), Guide.ylabel("1 / Success Rate"),
                Theme(background_color=colorant"white",
                    panel_fill=colorant"white",
                    major_label_font=PLOT_FONT,
                    minor_label_font=PLOT_FONT),
                    Guide.manual_color_key("Method",
                    [l[end] for l in method_label],
                    [c[end] for c in method_color]))

    p2 = plot(x=[], y=[], Geom.point,
                Guide.xlabel("Time"), Guide.ylabel("1 / Success Rate"),
                Theme(background_color=colorant"white",
                    panel_fill=colorant"white",
                    major_label_font=PLOT_FONT,
                    minor_label_font=PLOT_FONT),
                    Guide.manual_color_key("Method",
                    [l[end] for l in method_label],
                    [c[end] for c in method_color]))

    ylim_arr = []
    for m in methods
        p3 = plot(x=[], y=[],Geom.point,
                Guide.xlabel("Time"), Guide.ylabel("1 / Success Rate"),
                Theme(background_color=colorant"white",
                    panel_fill=colorant"white",
                    major_label_font=PLOT_FONT,
                    minor_label_font=PLOT_FONT))

        sr = mean.([step_success_rate.(e) for e in plot_data[m]["error"]])
        isr = 1.0./sr
        timed = plot_data[m]["time"]
        if !any(isnan.(vcat(timed...)))
            qtime = hcat(box_scatter_plot.(timed)...)
            qqtime = box_scatter_plot(qtime[2,:])
            qisr = box_scatter_plot(isr)
            if qisr[2] <= 5.0
                append!(p.layers,
                    layer(x=[qqtime[2]], y=[qisr[2]],
                            xmin=[qqtime[1]],
                            xmax=[qqtime[3]],
                            ymin=[qisr[1]],
                            ymax=[qisr[3]],
                            color=[method_color[m]],
                            Geom.point, Geom.errorbar))
            end

            append!(p2.layers, layer(x=qtime[2,:], y=isr,
                        color=[method_color[m]], Geom.point))

            append!(p3.layers, layer(x=qtime[2,:], y=isr,
                        color=[method_color[m]],
                        label=String.(noise_level),
                        Geom.point, Geom.label))

            try
                p3 |> PDF(joinpath(path,"sr_$(m)_$(sam).pdf"))
            catch e
                @show e
            end
        end
    end
    # display(p2)
    # display(p)

    try
        p2 |> PDF(joinpath(path,"sr_all_$(sam).pdf"))
    catch e
        @show e
    end

    try
        p |> PDF(joinpath(path,"sr_all_medians_$(sam).pdf"))
    catch e
        @show e
    end
    nothing
end


function sr_plots(plot_data::Dict,
                    var::Symbol,
                    methods::Vector{Symbol},
                    method_label::Dict,
                    method_color::Dict,
                    sam::Symbol,
                    path::String)::Nothing

    p = plot(x=[], y=[], Geom.point,
                Guide.xlabel("Time"), Guide.ylabel("1 / Success Rate"),
                Theme(background_color=colorant"white",
                    panel_fill=colorant"white",
                    major_label_font=PLOT_FONT,
                    minor_label_font=PLOT_FONT),
                    Guide.manual_color_key("Method",
                    [l[end] for l in method_label],
                    [c[end] for c in method_color]))

    ylim_arr = []
    for m in methods
        sr = mean([step_success_rate(e) for e in plot_data[m]["error"][1]])
        isr = [1.0./sr]
        timed = plot_data[m]["time"][1]

        if !any(isnan.(vcat(timed...)))
            qtime = box_scatter_plot(timed)
            qisr = box_scatter_plot(isr)
            if qisr[2] <= 5.0
                append!(p.layers,
                    layer(x=[qtime[2]], y=[qisr[2]],
                            xmin=[qtime[1]],
                            xmax=[qtime[3]],
                            ymin=[qisr[1]],
                            ymax=[qisr[3]],
                            color=[method_color[m]],
                            Geom.point, Geom.errorbar))
            end
        end
    end

    # display(p)
    try
        p |> PDF(joinpath(path,replace("sr_all_medians_$(sam)_$(var)","."=>"")*".pdf"))
    catch  e
        @show e
    end
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
                    noise_level::AbstractArray{Symbol},
                    methods::Vector{Symbol},
                    method_label::Dict,
                    method_color::Dict,
                    sam::Symbol,
                    path::String)::Nothing

    t_succ = zeros(length(methods))
    for (i,m) in enumerate(methods)
        error = plot_data[m]["error"]
        sr = mean.([step_success_rate.(e) for e in error])
        time = plot_data[m]["time"]
        mean_time = mean.(time)
        t_succ_arr = mean_time./sr
        t_succ[i] = mean(t_succ_arr)
    end

    oe = minimum(t_succ)./t_succ

    df = DataFrame()
    for (i,m) in enumerate(methods)
        aux_df = DataFrame(method=method_label[m],
                    data=oe[i], color=method_color[m])
        try
            append!(df, aux_df)
        catch e
            @show e
        end
    end

    p = plot(df, x="method", y="data", color="color", Geom.bar,
            Guide.xlabel("Method"), Guide.ylabel("Overall Efficiency"),
            Theme(background_color=colorant"white",
                panel_fill=colorant"white",
                major_label_font=PLOT_FONT,
                minor_label_font=PLOT_FONT))

    methods_str = reduce((acc,x) -> acc*"_"*string(x), methods; init="")
    try
        if length(noise_level) > 1
            p |> PDF(joinpath(path,"oe_$(sam)$(methods_str)_all.pdf"))
        else
            var = noise_level[1]
            p |> PDF(joinpath(path,replace("oe_$(sam)$(methods_str)_$(var)","."=>"")*".pdf"))
        end
    catch e
        @show e
    end
    nothing
end


function heatmap(x::AbstractArray,
                y::AbstractArray,
                z::AbstractArray;
                xlabel::String="",
                ylabel::String="",
                zlabel::String="",
                title::String="",
                fillcolor::ColorScheme=ColorSchemes.vik)::Gadfly.Plot
    is = repeat(x, length(x))
    js = vec(repeat(y', length(y)))
    values = z
    p = Gadfly.plot(x=is, y=js, color=values,
        Gadfly.Guide.xlabel(xlabel),
        Gadfly.Guide.ylabel(ylabel),
        Gadfly.Guide.colorkey(zlabel),
        Gadfly.Guide.title(title),
        Gadfly.Theme(minor_label_font=PLOT_FONT,
                    major_label_font=PLOT_FONT,
                    key_title_font=PLOT_FONT,
                    key_label_font=PLOT_FONT),
        # Gadfly.Guide.manual_color_key("Legend",
        #                                 ["True k1",
        #                                 "True k2"],
        #                                 ["magenta",
        #                                 "cyan"]),
        Gadfly.Coord.cartesian(yflip=false,
                        fixed=false,
                        xmin=minimum(x),
                        xmax=maximum(x),
                        ymin=minimum(y),
                        ymax=maximum(y)),
        Gadfly.Scale.ContinuousColorScale(p ->
                ColorSchemes.get(fillcolor, p)),
        Gadfly.Geom.rectbin,
        Gadfly.Scale.x_continuous,
        Gadfly.Scale.y_continuous)
    return p
end


function problem_plot(data::Matrix,
                        t::Vector,
                        type::String)::Gadfly.Plot

    m = size(data)[1]
    alphabet = 'A':'Z'
    label = ["$i" for i in alphabet[1:m]]

    df = DataFrame()
    for i in 1:m
        aux_df = DataFrame(x=data[i,:], State=label[i], t=t)
        append!(df, aux_df)
    end

    if type == "scatter"
        p = plot(df, x=:t, y=:x, color=:State,
                    Guide.xlabel("Time"), Guide.ylabel("State"),
                    Geom.point,
                    Gadfly.Coord.cartesian(yflip=false,
                                            fixed=false,
                                            xmin=minimum(df[:t]),
                                            xmax=maximum(df[:t]),
                                            ymin=minimum(df[:x]),
                                            ymax=maximum(df[:x])),
                    Gadfly.Theme(minor_label_font=PLOT_FONT,
                                    major_label_font=PLOT_FONT,
                                    key_title_font=PLOT_FONT,
                                    key_label_font=PLOT_FONT)
                    )

    elseif type == "line"
        p = plot(df, x=:t, y=:x, color=:State,
                    Guide.xlabel("Time"), Guide.ylabel("State"),
                    Geom.line,
                    Gadfly.Coord.cartesian(yflip=false,
                                            fixed=false,
                                            xmin=minimum(df[:t]),
                                            xmax=maximum(df[:t]),
                                            ymin=minimum(df[:x]),
                                            ymax=maximum(df[:x])),
                    Gadfly.Theme(minor_label_font=PLOT_FONT,
                                    major_label_font=PLOT_FONT,
                                    key_title_font=PLOT_FONT,
                                    key_label_font=PLOT_FONT)
                    )

    elseif type == "scatter_line"
        p = plot(df, x=:t, y=:x, color=:State,
                    Guide.xlabel("Time"), Guide.ylabel("State"),
                    Geom.point, Geom.line,
                    Gadfly.Coord.cartesian(yflip=false,
                                            fixed=false,
                                            xmin=minimum(df[:t]),
                                            xmax=maximum(df[:t]),
                                            ymin=minimum(df[:x]),
                                            ymax=maximum(df[:x])),
                    Gadfly.Theme(minor_label_font=PLOT_FONT,
                                    major_label_font=PLOT_FONT,
                                    key_title_font=PLOT_FONT,
                                    key_label_font=PLOT_FONT)
                    )
    else
        p = plot()
    end

    return p
end


end
