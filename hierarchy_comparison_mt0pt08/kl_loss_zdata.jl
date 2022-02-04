



function trapezoid_int_KL(G, G_pred, xs, xf)
        ∫ = 0.0;
        Δx = (xf - xs)/n_int;
        x = xs:Δx:xf
        for i in 2 : n_int
                if (G(x[i - 1]) == 0.0) || (G(x[i]) == 0.0)
                        integrand = 0.0
                elseif (G_pred(x[i - 1]) == 0.0) || (G_pred(x[i]) == 0.0)
                        integrand = 1e4
                else
                        f_l = G(x[i - 1]) * log(G(x[i - 1]) / G_pred(x[i - 1]))
                        f_r = G(x[i]) * log(G(x[i]) / G_pred(x[i]))
                        integrand = (f_l + f_r)/2 * Δx
                end
                ∫ += integrand
        end
        return ∫
end


function Ikl_τ_zdat(z_data_gt, z_data_pred, t, t_s)
    #Forward KL: data is sampled from GT distritubtion
        #For each τ, this integrates over the variable z in kl(τ, z, z_data)
        L = 0.0;
        for τ in t_s :  t
                δu_gt = z_data_gt[τ, :, 1]; δu_pr = z_data_pred[τ, :, 1];
                δv_gt = z_data_gt[τ, :, 2]; δv_pr = z_data_pred[τ, :, 2];
                δw_gt = z_data_gt[τ, :, 3]; δw_pr = z_data_pred[τ, :, 3];

                Gu_pred(x) = kde(x, δu_pr); hu_pred = obtain_h_kde(δu_pr);
                Gv_pred(x) = kde(x, δv_pr); hv_pred = obtain_h_kde(δv_pr);
                Gw_pred(x) = kde(x, δw_pr); hw_pred = obtain_h_kde(δw_pr);

                Gu(x) = kde(x, δu_gt); hu_gt = obtain_h_kde(δu_gt);
                Gv(x) = kde(x, δv_gt); hv_gt = obtain_h_kde(δv_gt);
                Gw(x) = kde(x, δw_gt); hw_gt = obtain_h_kde(δw_gt);

                ηu = hu_gt; ηv = hv_gt; ηw = hw_gt
                ηpu = hu_pred; ηpv = hv_pred; ηpw = hw_pred;

                xueg = maximum(δu_gt) + r*ηu;  xusg = minimum(δu_gt) - r*ηu;
                xveg = maximum(δv_gt) + r*ηv;  xvsg = minimum(δv_gt) - r*ηv;
                xweg = maximum(δw_gt) + r*ηw;  xwsg = minimum(δw_gt) - r*ηw;

                xuep = maximum(δu_pr) + r*ηpu; xusp = minimum(δu_pr) - r*ηpu;
                xvep = maximum(δv_pr) + r*ηpv; xvsp = minimum(δv_pr) - r*ηpv;
                xwep = maximum(δw_pr) + r*ηpw; xwsp = minimum(δw_pr) - r*ηpw;

                xus = minimum([xusg, xusp]); xue = maximum([xueg, xuep]);
                xvs = minimum([xvsg, xvsp]); xve = maximum([xveg, xvep]);
                xws = minimum([xwsg, xwsp]); xwe = maximum([xweg, xwep]);

                L += dt * (trapezoid_int_KL(Gu, Gu_pred, xus, xue) + trapezoid_int_KL(Gv, Gv_pred, xvs, xve) +
                                        trapezoid_int_KL(Gw, Gw_pred, xws, xwe))
        end
        return L
end




function Ikl_τ_diff(z_data_gt, z_data_pred, t, t_s)
    #Forward KL: data is sampled from GT distritubtion
        #For each τ, this integrates over the variable z in kl(τ, z, z_data)
        L = 0.0;
        for τ in t_s :  t
                δu_gt = z_data_gt[τ, :]; δu_pr = z_data_pred[τ, :];

                Gu_pred(x) = kde(x, δu_pr); hu_pred = obtain_h_kde(δu_pr);
                Gu(x) = kde(x, δu_gt); hu_gt = obtain_h_kde(δu_gt);
                ηu = hu_gt; ηpu = hu_pred;

                xueg = maximum(δu_gt) + r*ηu;  xusg = minimum(δu_gt) - r*ηu;
                xuep = maximum(δu_pr) + r*ηpu; xusp = minimum(δu_pr) - r*ηpu;
                xus = minimum([xusg, xusp]); xue = maximum([xueg, xuep]);

                L += dt * (trapezoid_int_KL(Gu, Gu_pred, xus, xue))
        end
        return L
end
