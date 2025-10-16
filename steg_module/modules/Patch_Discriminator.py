import torch
import torch.nn as nn

class Patch_Discriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, blocks=3, channels=64):
        super(Patch_Discriminator, self).__init__()

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(3, channels, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, blocks):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(channels * nf_mult_prev, channels * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=False),
                nn.InstanceNorm2d(channels * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** blocks, 8)
        sequence += [
            nn.Conv2d(channels * nf_mult_prev, channels * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=False),
            nn.InstanceNorm2d(channels * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(channels * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)
        
    # @autocast()
    def forward(self, input):
        """Standard forward."""
        return self.model(input)


def compute_gradient_penalty(D, real_samples, fake_samples):
    "Calculates the gradient penalty loss for WGAN GP"
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand((real_samples.size(0), 1, 1, 1)).to(real_samples.device.index)

    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples +((1 - alpha) * fake_samples))
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
    d_interpolates = D(interpolates)
    fake = torch.autograd.Variable(torch.ones_like(d_interpolates), requires_grad=False) # Get qradient w.r.t. interpolates
    gradients = torch.autograd.grad(outputs = d_interpolates,
                                    inputs = interpolates,
                                    grad_outputs = fake,
                                    create_graph = True,
                                    retain_graph = True,
                                    only_inputs = True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2,dim=1) -1) ** 2).mean()
        
    return gradient_penalty