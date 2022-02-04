using PlotlyJS
function contour_js(x, y, z, name, method, t)
    path = "./learned_figures/vel_field_slice_t$(t)_$(method)_$(name)_Nf$(N_f).png"
    p = PlotlyJS.plot(PlotlyJS.heatmap(;z=z,x=x,y=y, zsmooth="best", connectgaps=true))
    relayout!(p, xaxis_title="x", yaxis_title="y", title="$(name)(t=$(dt*t), x, y, z_0)")
    PlotlyJS.savefig(p, path)
    return p
end

function contour_js_nopath(x, y, z, name)
    p = PlotlyJS.plot(PlotlyJS.heatmap(;z=z,x=x,y=y, zsmooth="best", connectgaps=true))
    relayout!(p, xaxis_title="X axis", yaxis_title="Y axis", title="Slice of Velocity Field: $(name)")
    # return p
end

# v = 0.02*randn(16, 16)
# x = range(0, 2*pi, length=16); y = x
# contour_js(x, y, v, "test", "js", 1)
