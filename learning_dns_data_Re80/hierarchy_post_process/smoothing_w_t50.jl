using NPZ, Plots
using LaTeXStrings, SpecialFunctions
using ForwardDiff

function sigma_f(h,a,b)
  if (D == 2)
    return  gamma(b + 2/a + 1)/(4*pi*h^2 * gamma(b+1) * gamma(2/a + 1))
  end
  if (D == 3)
    return  3*gamma(b + 3/a + 1)/(32*pi*h^3 * gamma(b+1) * gamma(3/a + 1))
  end
end

function W_ab(r,h,a,b)
  sigma = sigma_f(h,a,b);
  q = r / (2*h);
  if (q > 1.0)
    return 0.0
  end
  return sigma * (1 - q^a)^b
end

# function W_ab(r,h,a,b)
#   sigma = sigma_f(h,a,b);
#   q = r / (2*h);
#   if (q > 1.0)
#     return 0.0
#   end
#   return sigma * (1 - (q)^a)^b
# end

function ∂rW_ab_ad(r,h,a,b)
  out = ForwardDiff.derivative(x-> W_ab(x,h,a,b), r)
  return out
end

∂rW_ab_ad_r(r, h, a, b) = ∂rW_ab_ad(r,h,a,b)/(r+1e-10)

function ∂rW_ab(r,h,a,b) # H(r) = (d W / d r) / r
  sigma = sigma_f(h,a,b);
  q = r / (2*h);
  if (q > 1.0)
    return 0.0
  end
  return -sigma * r * a*b/(4*h^2)*(1-q^a)^(b-1)*q^(a-2)
end

function H_ab(r,h,a,b) # H(r) = (d W / d r) / r
  sigma = sigma_f(h,a,b);
  q = r / (2*h);
  if (q > 1.0)
    return 0.0
  end
  return -sigma * a*b/(4*h^2)*(1-q^a)^(b-1)*q^(a-2)
end



function W_cub(r, h)
  sigma = 1/(pi*h^3)  #3D normalizing factor
  q = r / h;   if (q > 2.)   return 0.;   end
  if (q > 1.)   return (sigma * (2. - q)^3 / 4.);   end
  return (sigma * (1. - 1.5 * q * q * (1. - q / 2.)));
end

function ∂rW_cub(r, h)
  sigma = 1/(pi*h^3)  #3D normalizing factor
  q = r / h;   if (q > 2.)   return 0.;   end
  if (q > 1.)   return (-3. * r * sigma * (2. - q)^2 / (4. * h * r));   end
  return (sigma * r * (-3. + 9. * q / 4.) / h^2);
end

# H(r) = (d W / d r) / r
function H_cub(r, h)
  sigma = 1/(pi*h^3)  #3D normalizing factor
  q = r / h;   if (q > 2.)   return 0.;   end
  if (q > 1.)   return (-3. * sigma * (2. - q)^2 / (4. * h * r));   end
  return (sigma * (-3. + 9. * q / 4.) / h^2);
end


function W_liu(r, h)
  sigma = 315/(208*pi*h^3);
  q = r/h;
  if q > 2.0
    out = 0.0
  else
    out = sigma * (2/3 - 9/8*q^2 + 19/24*q^3 - 5/32*q^4)
  end
  return out
end

function ∂rW_liu(r, h)
  sigma = 315/(208*pi*h^3);
  q = r/h;
  if q > 2.0
    out = 0.0
  else
    out = sigma * r * (-18/(8*h^2) + 57*r/(24*h^3) - 20*r^2/(32*h^4))
  end
  return out
end

function H_liu(r, h)
  sigma = 315/(208*pi*h^3);
  q = r/h;
  if q > 2.0
    out = 0.0
  else
    out = sigma * (-18/(8*h^2) + 57*r/(24*h^3) - 20*r^2/(32*h^4))
  end
  return out
end

using ForwardDiff
function dH_liu(r, h)
  out = ForwardDiff.derivative(x-> H_liu(x,h), r)/(r+1e-12)
  return out
end


function sigma_f2(h,a,b)
  if D==1
    out = 5*a^8*b*h*asinh(1/a)/64 - 5*a^7*b*h/(64*sqrt(1+1/(a^2))) +
        a^6*b*h*asinh(1/a)/4 - a^6*h*asinh(1/a)/8 - 53*a^5*b*h/(192*sqrt(1+1/(a^2))) +
        a^5*h/(8*sqrt(1+1/(a^2))) + a^4*b*h*asinh(1/a)/4 - a^4*h*asinh(1/a)/2 -
        31*a^3*b*h/(96*sqrt(1+1/(a^2))) + 13*a^3*h/(24*sqrt(1+1/(a^2))) -
        a^2*h*asinh(1/a) - 5*a*b*h/(24*sqrt(1+1/(a^2))) + a*h/(12*sqrt(1+1/(a^2))) +
        16*b*h*sqrt(a^2+1)/105 + 16*h*sqrt(a^2+1)/15 - b*h/(12*a*sqrt(1+1/(a^2))) -
        h/(3*a*sqrt(1+1/(a^2)));
  end
  if D==3
    out = 4*pi*((sqrt(1 + a^2)*(32*(87 + 22*b) + 21*a^2*(-48*(5 + b) +
        a^2*(-380 + 96*b + 5*a^2*(-30 + (46 + 21*a^2)*b)))))/80640 +
        (a^4*(-32 + 16*a^2*(-2 + b) + 7*a^6*b + 10*a^4*(-1 + 2*b))*
        (log(a^2) - 2*log(1 + sqrt(1 + a^2))))/512)
  end
  return out
end

function W2_ab(r,h,a2,b2)
  sigma = sigma_f2(h,a2,b2);
  q = r / (2*h);
  if (q > 1.0)
    return 0.0
  end
  if D==3
    return 1/((2*h)^D*sigma) * (sqrt(a2^2 + 1) - sqrt(a2^2 + q^2)) * (1 - q^2)^2 * (1 + b2*q^2)
  end
  if D==1
    return 1/(2*sigma) * (sqrt(a2^2 + 1) - sqrt(a2^2 + q^2)) * (1 - q^2)^2 * (1 + b2*q^2)
  end
end

function H2_ab(r,h,a2,b2) # H(r) = (d W / d r) / r
  sigma = sigma_f2(h,a2,b2);
  q = r / (2*h);
  if (q > 1.0)
    return 0.0
  end
  if D ==3
    #sigma = sigma_f2(h,a2,b2);
    tmp1 = -(1-q^2)^2 * (b2*q^2+1)/((2*h)^(D+2)*sigma*sqrt(a2^2 + q^2))
    tmp2 = 2*b2*(1-q^2)^2 * (sqrt(a2^2 + 1) - sqrt(a2^2 + q^2))/((2*h)^(D+2)*sigma)
    tmp3 = 4*(1-q^2) * ((sqrt(a2^2 +1) - sqrt(a2^2 + q^2)) * (b2*q^2 + 1))/((2*h)^(D+2)*sigma)
    return tmp1 + tmp2 - tmp3
  end
  if D == 1
    tmp1 = b2*(1 - r^2/(4*h^2))^2 * (sqrt(a2^2 + 1) - sqrt(a2^2 + r^2/(4*h^2))) / (4*h^2*sigma);
    tmp2 = (1 - r^2/(4*h^2))^2 * (b2*r^2/(4*h^2) + 1) /(8*h^2*sigma * sqrt(a2^2 + r^2/(4*h^2)));
    tmp3 = (1 - r^2/(4*h^2)) * (b2*r^2/(4*h^2) + 1) * (sqrt(a2^2+1) - sqrt(a2^2+r^2/(4*h^2))) / (2*h^2*sigma)
    return tmp1 - tmp2 - tmp3
  end
end

∂rW2_ab(r,h,a,b) = r*H2_ab(r,h,a,b);

function dH2_ab_dr_r(r, h, a, b)
  out = ForwardDiff.derivative(x-> H(x,h,a,b), r)/(r+1e-12)
  return out
end






function plot_kernels(a1, b1, a2, b2, h)
  path = "./learned_figures_t50/comp_W.png"
  l = 3.2
  plt = plot(r -> W_cub(r, h), 0, 2*h, color="indigo", label=L"W_{cubic}", linewidth = l, legendfontsize=12)
  plot!(r -> W_ab(r, h, a1, b1), 0, 2*h, color="black", linestyle=:dashdot, label=L"W_{ab}", linewidth = l, legendfontsize=12)
  plot!(r -> W_liu(r, h), 0, 2*h, color="blue", linestyle=:dashdot, label=L"W_{Liu}", linewidth = l, legendfontsize=12)
  plot!(r -> W2_ab(r, 2*h, a2, b2), 0, 2*h, color="purple", linestyle=:dot, label=L"W2_{ab}", linewidth = l, legendfontsize=12)

  title!(L"\textrm{Comparing: } W_{\theta}", titlefont=16)
  xlabel!(L"r/h", xtickfontsize=10, xguidefontsize=16)
  ylabel!(L"W(r, h)", ytickfontsize=10, yguidefontsize=14)
  # savefig(plt, path)
  # display(plt)
  return plt
end


function plot_kernels_der(a1, b1, a2, b2, h)
  path = "./learned_figures_t50/comp_W_r.png"
  l = 3.2
  plt = plot(r -> ∂rW_cub(r, h), 0, 2*h, color="indigo", label=L"\partial_r W_{cubic}", linewidth = l, legendfontsize=12)
  plot!(r -> ∂rW_ab(r, h, a1, b1), 0, 2*h, color="black", linestyle=:dashdot, label=L"\partial_r W_{ab}", linewidth = l, legendfontsize=12)
  plot!(r -> ∂rW_liu(r, h), 0, 2*h, color="blue", linestyle=:dashdot, label=L"\partial_r W_{Liu}",linewidth = l, legendfontsize=12)
  plot!(r -> ∂rW2_ab(r, 2*h, a2, b2), 0, 2*h, color="purple", linestyle=:dot, label=L"\partial_r W2_{ab}",linewidth = l, legendfontsize=12)

  title!(L"\textrm{Comparing: } \partial_r W_{\theta}", titlefont=16)
  xlabel!(L"r/h", xtickfontsize=10, xguidefontsize=16)
  ylabel!(L"\partial_r W_{\theta}(r, h)", ytickfontsize=10, yguidefontsize=14)
  # savefig(plt, path)
  # display(plt)
  return plt
end

function plot_kernels_H(a1, b1, a2, b2, h)
  path = "./learned_figures_t50/comp_H.png"
  l = 3.2
  plt = plot(r -> H_cub(r, h), 0, 2*h, color="indigo", label=L"H_{cubic}", linewidth = l, legendfontsize=12)
  plot!(r -> H_ab(r, h, a1, b1), 0, 2*h, color="forestgreen", linestyle=:dash, label=L"H_{ab}", linewidth = l, legendfontsize=12)
  plot!(r -> H_liu(r, h), 0, 2*h, color="blue", linestyle=:dashdot, label=L"H_{Liu}", linewidth = l, legendfontsize=12)
  plot!(r -> H2_ab(r, 2*h, a2, b2), 0, 2*h, color="purple", linestyle=:dot, label=L"H2_{ab}", linewidth = l, legendfontsize=12)

  title!(L"\textrm{Comparing: } H_{\theta} = \frac{\partial_r W}{r}", titlefont=16)
  xlabel!(L"r/h", xtickfontsize=10, xguidefontsize=16)
  ylabel!(L"H(r, h)", ytickfontsize=10, yguidefontsize=14)
  # savefig(plt, path)
  # display(plt)
  return plt
end




#Testing new kernel
function plot_kernels_2(a1, b1, a2, b2, h)
  gr(size=(700,600))
  path = "./learned_figures_t50/comp_W.png"
  plt = plot(r -> W_cub(r, h), 0, 2*h, color="indigo", label=L"W_{cubic}", linewidth = 3.3, legendfontsize=16)
  plot!(r -> W_liu(r, h), 0, 2*h, color="blue", markershape=:dtriangle, ms = 5.5, linestyle=:dash, label=L"W_{quart}", linewidth = 3.3, legendfontsize=16)
  plot!(r -> W_ab(r, h, a1, b1), 0, 2*h, color="black", markershape=:xcross, ms = 5.5, linestyle=:dashdot, label=L"W(a,b)", linewidth = 3.3, legendfontsize=16)
  plot!(r -> W2_ab(r, h, a2, b2), 0, 2*h, color="purple",  markershape=:auto, linestyle=:dot, label=L"W_2(a,b)", linewidth = 2.8, legendfontsize=16)

  title!(L"\textrm{Learned Smoothing Kernels } W", titlefont=20)
  xlabel!(L"r/h", xtickfontsize=12, xguidefontsize=18)
  ylabel!(L"W(r, h)", ytickfontsize=12, yguidefontsize=18)
  savefig(plt, path)
  display(plt)
end


function plot_kernels_der_2(a1, b1, a2, b2, h)
  gr(size=(700,600))
  path = "./learned_figures_t50/comp_W_r.png"
  plt = plot(r -> ∂rW_cub(r, h), 0, 2*h, color="indigo", label=L"\nabla_r W_{cubic}", linewidth = 3.3, legendfontsize=16, legend=:bottomright)
  plot!(r -> ∂rW_liu(r, h), 0, 2*h, color="blue", markershape=:dtriangle, ms = 5.5, linestyle=:dash, label=L"\nabla_r W_{quart}",linewidth = 3.3, legendfontsize=16, legend=:bottomright)
  plot!(r -> ∂rW_ab(r, h, a1, b1), 0, 2*h, color="black", markershape=:xcross, ms = 5.5, linestyle=:dashdot, label=L"\nabla_r W(a,b)", linewidth = 3.3, legendfontsize=16, legend=:bottomright)
  plot!(r -> ∂rW2_ab(r, h, a2, b2), 0, 2*h, color="purple",  markershape=:auto, linestyle=:dot, label=L"\nabla_r W_2(a,b)",linewidth = 2.8, legendfontsize=16)

  title!(L"\textrm{Learned Smoothing Gradients } \nabla_r W", titlefont=20)
  xlabel!(L"r/h", xtickfontsize=12, xguidefontsize=18)
  ylabel!(L"\nabla_r W(r, h)", ytickfontsize=12, yguidefontsize=18)
  savefig(plt, path)
  display(plt)
end

function plot_kernels_H_2(a1, b1, a2, b2, h)
  path = "./learned_figures_t50/comp_H.png"
  plt = plot(r -> H_cub(r, h), 0, 2*h, color="indigo", label=L"H_{cubic}", linewidth = 2.8, legendfontsize=16)
  plot!(r -> H_liu(r, h), 0, 2*h, color="blue", linestyle=:dashdot, label=L"H_{quart}", linewidth = 2.8, legendfontsize=16)
  plot!(r -> H_ab(r, h, a1, b1), 0, 2*h, color="black", linestyle=:dashdot, label=L"H_(a,b)", linewidth = 2.8, legendfontsize=16)
  plot!(r -> H2_ab(r, h, a2, b2), 0, 2*h, color="purple",  markershape=:auto, linestyle=:dot, label=L"H_2(a,b)", linewidth = 2.8, legendfontsize=12)

  title!(L"\textrm{Comparing } H = \frac{\partial_r W}{r}", titlefont=20)
  xlabel!(L"r/h", xtickfontsize=12, xguidefontsize=18)
  ylabel!(L"H(r, h)", ytickfontsize=12, yguidefontsize=18)
  savefig(plt, path)
  display(plt)
end



function plot_kernels_der_ad(a1, b1, a2, b2, h)
  path = "./learned_figures_t50/comp_∂rWab:AD.png"
  plt = plot(r -> ∂rW_ab(r, h, a1, b1), 0, 2*h, color="forestgreen", linestyle=:dash, label=L"\partial_r W_{ab}", linewidth = 2.8, legendfontsize=12)
  plot!(r -> ∂rW2_ab(r, 2*h, a2, b2), 0, 2*h, color="black", linestyle=:dot, label=L"\partial_r W_{2ab}",linewidth = 2.8, legendfontsize=12)
  plot!(r -> ∂rW_ab_ad(r, h, a1, b1), 0, 2*h, color="forestgreen", label=L"\partial_r W_{ab:AD}", linewidth = 2.25, legendfontsize=12)
  plot!(r -> ∂rW2_ab_ad(r, 2*h, a2, b2), 0, 2*h, color="black", label=L"\partial_r W_{2ab:AD}",linewidth = 2.25, legendfontsize=12)

  title!(L"\textrm{Comparing: } \partial_r W", titlefont=16)
  xlabel!(L"r", xtickfontsize=10, xguidefontsize=16)
  ylabel!(L"H(r, h)", ytickfontsize=10, yguidefontsize=14)
  savefig(plt, path)
  display(plt)
end


function plot_kernels_der_r_ad(a1, b1, a2, b2, h)
  path = "./learned_figures_t50/comp_H:AD.png"
  plt = plot(r -> H_ab(r, h, a1, b1), 0, 2*h, color="forestgreen", linestyle=:dash, label=L"H_{ab}", linewidth = 2.8, legendfontsize=12)
  plot!(r -> H2_ab(r, 2*h, a2, b2), 0, 2*h, color="black", linestyle=:dot, label=L"H_{2ab}", linewidth = 2.8, legendfontsize=12)
  plot!(r -> ∂rW_ab_ad_r(r, h, a1, b1), 0, 2*h, color="forestgreen", label=L"\partial_r W_{ab:AD}/r", linewidth = 2.25, legendfontsize=12)
  plot!(r -> ∂rW2_ab_ad_r(r, 2*h, a2, b2), 0, 2*h, color="black", label=L"\partial_r W_{2ab:AD}/r",linewidth = 2.25, legendfontsize=12)

  title!(L"\textrm{Comparing: } (\partial_r W)/r", titlefont=16)
  xlabel!(L"r", xtickfontsize=10, xguidefontsize=16)
  ylabel!(L"H(r, h)", ytickfontsize=10, yguidefontsize=14)
  savefig(plt, path)
  display(plt)
end

t_coarse = 1; dt = t_coarse * 0.04; l_method = "lf"

include("load_models_t50.jl")
d1, d2, d3, d4, d5, d6, d7, d8, d9 = load_dirs_names_t50()

p_fin_wcub = load_learned_model_params(d4)
p_fin_wab = load_learned_model_params(d2)
p_fin_w2ab = load_learned_model_params(d1)
p_fin_wliu = load_learned_model_params(d3)

a1 = p_fin_wab[5]; b1 = p_fin_wab[6];
a2 = p_fin_w2ab[5]; b2 = p_fin_w2ab[6];

h = 1.0; D = 3; N = 4096
plot_kernels_2(a1, b1, a2, b2, h)
plot_kernels_der_2(a1, b1, a2, b2, h)
# plot_kernels_H_2(a1, b1, a2, b2, h)
# plot_kernels_der_ad(a1, b1, a2, b2, h)
# plot_kernels_der_r_ad(a1, b1, a2, b2, h)
