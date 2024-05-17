# Import Libraries
import pandas as pd
import torch
import mod



def kan_loss(
        self, x:  torch.Tensor,
        lamb_l1=1.0, lamb_entropy=2.0,
        lamb_coef=0.0, lamb_coefdiff=0.0,
        small_mag_threshold=1e-16,
        small_reg_factor=1.0
):
    
    def reg(mod):

        def nonlinear(x, th=small_mag_threshold, factor=small_reg_factor):
            return (x < th) * x * factor + (x > th) * (
                x + (factor - 1) * th
            )
        
        reg = 0.0
        for i in range(len(mod.acts_scale())):
            vec = mod.acts_scale[i].reshape(
                -1,
            )

            p = vec / torch.sum(vec)
            l1 = torch.sum(nonlinear(vec))
            entropy = -torch.sum(p * torch.log2(p + 1e-4))
            reg_ += (
                lamb_l1 * l1 + lamb_entropy * entropy
            ) # both l1 and entropy

        # regularize coefficient to encourage spline to be zero
        for i in range(len(mod.act_fun())):
            coeff_l1 = torch.sum(
                torch.mean(torch.abs(mod.act_fun[i].coef), dim=1)
            )
            coeff_diff_l1 = torch.sum(
                torch.mean(
                    torch.abs(torch.diff(mod.act_fun[i].coef)), dim=1
                )
            )
            reg_ += lamb_coef * coeff_l1 + lamb_coefdiff * coeff_diff_l1

        return reg_