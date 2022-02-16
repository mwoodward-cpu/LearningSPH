using NPZ


function load_data(path)
    data = npzread(path)
    println("*********** data loaded ****************")
    return data
end

#----Load datas
path_ = "./learned_data/vort_N4096_T10_ts1_h0.335_dns_θ0.0002_phys_inf.npy"
Vort = load_data(path_)
Vort_field = 0.5 * (Vort[end,:,1].^2 .+ Vort[end,:,2].^2 .+ Vort[end,:,3].^2)



#----Plotting
include("./gen_vorticity_outs.jl")

r = 0.3; v = 1.2*r
# Makie.use_plot_pane(true)
iso_surface_vorticity(Vort_field, r, v)
