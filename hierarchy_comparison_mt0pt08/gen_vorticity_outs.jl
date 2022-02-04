using GLMakie
using AbstractPlotting
AbstractPlotting.inline!(true)
using Makie

function iso_surface_vorticity(vort_field, isr, isv, path)
    N = size(vort_field)[1]; l = round(Int, N^(1/3));
    x = range(0, 2*pi, length=l); y = x; z = x;
    vort_field = reshape(vort_field, (l, l, l));
    # vort_field_gt = reshape(vort_field_gt, (l, l, l));

    # alg = :mip; cmap = :coolwarm;
    # cmap = :coolwarm;
    alg = :iso; cmap = :winter;
    x = range(0, 2*pi, length=l); y = x; z = x;
    scene = Makie.volume(x,y,z, vort_field, colormap = cmap, algorithm = alg, isorange = isr, isovalue = isv)
    # Makie.save("./plot_vort_isosurface_$(isr).png", scene)
    # Makie.volume!(x,y,z, vort_field, colormap = cmap, algorithm = alg, isorange = isr/2, isovalue = isv/2)
    Makie.save(path, scene)
end
