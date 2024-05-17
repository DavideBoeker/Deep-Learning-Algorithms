# Import Libraries
import torch
import torch.nn as nn




class KAN(nn.Module):
    def forward(self, x):
        shape_size = len(x.shape)
        x = x.view(-1, T)

        self.acts = []
        self.spline_preacts = []
        self.spline_postsplines = []
        self.spline_postacts = []
        self.acts_scale = []
        self.acts_scale_std = []

        self.acts.append(x)

        for l in range(self.depth):
            x_numerical, preacts, postacts_numerical, postspline = self.act_fun[l](x)

            if self.symbolic_enabled:
                x_symbolic, postacts_symbolic = self.symbolic_fun[l](x)
            else:
                x_symbolic = 0.0
                postacts_symbolic = 0.0

            x = x_numerical + x_symbolic
            postacts = postacts_numerical + postacts_symbolic

            grid_reshape = self.act_fun[l].grid_reshape(
                self.width[l + 1], self.width[l], -1
            )
            input_range = grid_reshape[:, :, -1] - grid_reshape[:, :, 0] + 1e-4
            output_range =torch.mean(torch.abs(postacts), dim=0)
            self.acts_scale.append(output_range / input_range)
            self.acts_scale_std.append(torch.std(postacts, dim=0))
            self.spline_preacts.append(preacts.detach())
            self.spline_postacts.append(postacts.detach())
            self.spline_postsplines.append(postspline.detach())

            x = x + self.biases[l].weight
            self.acts.append(x)

        U = x.shape[1]

        if shape_size == 3:
            x = x.view(B, C, U)
        elif shape_size == 2:
            assert x.shape == (B, U)


        return x
