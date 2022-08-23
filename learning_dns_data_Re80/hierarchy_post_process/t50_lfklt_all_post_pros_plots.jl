#in this file, we produce all the plots seen in the order of
#the paper


#set colors
include("./color_scheme_utils.jl")

#set plot dims:
include("./plot_dims.jl")

#plot loss functions
include("./plot_losses_t50_lfklt.jl")



#
#plot smoothing kernels
include("./smoothing_w_t50_lfklt.jl")


#
# #plot gen error over t for phys_inf
# #plot gen error over Mt for phys_inf
m_runs = "phys_inf"
include("./plotting_gen_errors_t50_lfklt.jl")
# include("./plotting_gen_errors_mt_t20.jl")
#
#
#


# #-------Single particle statistics
#
# #plot learned distributions for phys inf (over time)
m_kind = "phys"
T_frame = 80:100
include("./plotting_single_part_stats_nncomp_t50_lfklt.jl")


#
#
# #plot learned accl dist for phys inf over Mts
# m_kind = "phys"
# Mt = 0.16; mmt = "016"
# T_frame = 100:200
# include("./plotting_single_part_stats_nncomp_t20_Mt.jl")
# Mt = 0.04; mmt = "004"
# T_frame = 1:200;
# include("./plotting_single_part_stats_nncomp_t20_Mt.jl")
#
#
#
#
# #------ Energy Spectrum
# #plot energy spectrum for phys_inf method
#
# m_runs = "phys_inf"
# t_s = 1*70 #kolmogorov time scale
# include("./comp_energy_spectrum_t20.jl")
#
# t_s = 6*70 #eddy turn over time
# include("./comp_energy_spectrum_t20.jl")
#
#
#
#
# #---Plotting comparison over all models
#
m_runs = "comb"; tick_fs = 17 #better fit
include("./plotting_gen_errors_t50_lfklt.jl")

# include("./plotting_gen_errors_mt_t20.jl")
#
# #reset dims
include("./plot_dims.jl")
#
# #energy spectrum
# m_runs = "comb"
# t_s = 1*70 #kolmogorov time scale
# include("./comp_energy_spectrum_t20.jl")
#
# t_s = 6*70 #eddy turn over time
# include("./comp_energy_spectrum_t20.jl")
#
#
#
# #plotting Gd and Gv
# #plot learned accl dist for phys inf over Mts
m_kind = "comb"
T_frame = 120:140
include("./plotting_single_part_stats_nncomp_t50_lfklt.jl")


#
# Mt = 0.16; mmt = "016"
# T_frame = 100:200
# include("./plotting_single_part_stats_nncomp_t20_Mt.jl")
# Mt = 0.04; mmt = "004"
# T_frame = 1:200;
# include("./plotting_single_part_stats_nncomp_t20_Mt.jl")
