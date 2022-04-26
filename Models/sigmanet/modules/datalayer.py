import sys

sys.path('../../../')

import torch
import torch.nn as nn
import Datasets.medutils_torch as medutils_torch
from Datasets.medutils_torch import complex
from Datasets.medutils_torch.fft import fft2, ifft2
from Datasets.medutils_torch.mri import \
    adjointSoftSenseOpNoShift, \
    forwardSoftSenseOpNoShift


class DataIDLayer(nn.Module):
    """
    Placeholder for data layer
    """

    def __init__(self, *args, **kwargs):
        super(DataIDLayer, self).__init__()

    def forward(selfself, x, *args, **kwargs):
        return x

    def __repr__(self):
        return f'DataIDLayer()'


class DCLayer(nn.Module):
    """
    DClayer from DC-CNN, apply for single coil mainly
    """

    def __init__(self, lambda_init=0., learnable=True):
        """
        :param lambda_init -> float: Init value of data consistency block
        """
        super(DCLayer, self).__init__()
        self.lambda_ = torch.nn.Parameter(torch.Tensor(1))
        self.lambda_.data = torch.tensor(lambda_init, dtype=self.lambda_.dtype)
        self.lambda_.requires_grad = learnable
        self.set_learnable(learnable)

    def forward(self, x, y, mask):
        A_x = fft2(x)
        k_dc = (1 - mask) * A_x + mask * (self.lambda_ * A_x + (1 - self.lambda_) * y)
        x_dc = ifft2(k_dc)

        return x_dc

    def extra_repr(self):
        return f"lambda={self.lambda_.item():.4g}, learnable={self.requires_grad}"

    def set_learnable(self, flag):
        self.lambda_.requires_grad = flag


class DataGDLayer(nn.Module):
    """
    DataLayer computing the gradient on the L2 data term.
    """

    def __init__(self, lambda_init, learnable=True):
        """
        :param lambda_init (float): init value of data term weight lambda
        """

        super(DataGDLayer, self).__init__()
        self.lambda_init = lambda_init
        self.data_weight = torch.nn.Parameter(torch.Tensor(1))
        self.data_weight.data = torch.tensor(
            lambda_init,
            dtype=self.data_weight.dtype
        )
        self.set_learnable(learnable)

    def forward(self, x, y, smaps, mask):
        # use ifft2 to get zero filled image, and its residual
        A_x_y = forwardSoftSenseOpNoShift(x, smaps, mask) - y
        # use fft2 to get res-kspace
        gradD_x = adjointSoftSenseOpNoShift(A_x_y, smaps, mask)
        return x - self.data_weight * gradD_x

    def __repr__(self):
        return f'DataLayer(lambda_init={self.data_weight.item():.4g}'

    def set_learnable(self, flag):
        self.data_weight.requires_grad = flag


class DataProxCGLayer(nn.Module):
    """
    Solving the prox wrt. dataterm using Conjugate Gradient
    as proposed by Aggarwal et al.
    """

    def __init__(self, lambda_init, tol=1e-6, itera=10, learnable=True):
        super(DataProxCGLayer, self).__init__()

        self.lambda_a = torch.nn.Parameter(torch.Tensor(1))
        self.lambda_a.data = torch.tensor(lambda_init)
        self.lambda_a_init = lambda_init
        self.lambda_a.requires_grad = learnable

        self.tol = tol
        self.iter = itera

        self.op = MyCG

    def forward(self, x, y, samps, mask):
        return self.op.apply(
            x, self.lambda_a, y, samps, mask,
            self.tol, self.itera
        )

    def extra_repr(self) -> str:
        return (f"lambda_init = {self.lambdaa.item():.4g}, tol={self.tol}"
                f" iter={self.itera} learnable={self.lambda_a.requires_grad}")

    def set_learnable(self, flag):
        self.lambda_a.requires_grad = flag


class MyCG(torch.autograd.Function):
    """
    performs CG algorithm
    """
    @staticmethod
    def complexDot(data1, data2):
        nBatch = data1.shape[0]
        mult = complex.complex_mult_conj(data1, data2)
        re, im = torch.unbind(mult, dim=-1)
        return torch.stack([torch.sum(re.view(nBatch, -1), dim=-1),
                            torch.sum(im.view(nBatch, -1), dim=-1)], -1)

    @staticmethod
    def solve(x0, M, tol, max_iter):
        nBatch = x0.shape[0]
        # x0 shape tensor
        x = torch.zeros(x0.shape).to(x0.device)
        r = x0.clone()
        p = x0.clone()
        x0x0 = (x0.pow(2)).view(nBatch, -1).sum(-1)
        rr = torch.stack([
            (r.pow(2)).view(nBatch, -1).sum(-1),
            torch.zeros(nBatch).to(x0.device)
        ], dim=-1)

        it = 0

        while torch.min(rr[..., 0] / x0x0) > tol and it < max_iter:
            it += 1
            q = M(p)
            alpha = complex.complex_div(rr, MyCG.complexDot(p, q))

            x += complex.complex_mult(
                alpha.reshape(nBatch, 1, 1, 1, -1),
                p.clone()
            )

            r -= complex.complex_mult(
                alpha.reshape(nBatch, 1, 1, 1, -1),
                q.clone()
            )

            rr_new = torch.stack([
                (r.pow(2)).view(nBatch, -1).sum(-1),
                torch.zeros(nBatch).to(x0.device)
            ],
                dim=-1)

            beta = torch.stack([
                rr_new[..., 0] / rr[..., 0],
                torch.zeros(nBatch).to(x0.device)
            ],
                dim=-1)

            p = r.clone() + complex.complex_mult(
                beta.reshape(nBatch, 1, 1, 1, -1),
                p
            )

            rr = rr_new.clone()

        return x

    @staticmethod
    def forward(ctx, z, lambda_a, y, smaps, mask, tol, max_iter):
        ctx.tol = tol
        ctx.max_iter = max_iter

        def A(x):
            return forwardSoftSenseOpNoShift(x, smaps, mask)

        def AT(y):
            return adjointSoftSenseOpNoShift(y, smaps, mask)

        def M(p):
            return lambda_a * AT(A(p)) + p

        x0 = lambda_a * AT(y) + z
        ctx.save_for_backward(AT(y), x0, smaps, mask, lambda_a)

        return MyCG.solve(x0, M, ctx.tol, ctx.max_iter)

    @staticmethod
    def backward(ctx, grad_x):
        ATy, rhs, smaps, mask, lambda_a = ctx.saved_tensors

        def A(x):
            return forwardSoftSenseOpNoShift(x, smaps, mask)

        def AT(y):
            return adjointSoftSenseOpNoShift(y, smaps, mask)

        def M(p):
            return lambda_a * AT(A(p)) + p

        Qe = MyCG.solve(grad_x, M, ctx.tol, ctx.max_iter)
        QQe = MyCG.solve(Qe, M, ctx.tol, ctx.max_iter)

        grad_z = Qe

        grad_lambda_a = complex.complex_dotp(Qe, ATy).sum() \
                        - complex.complex_dotp(QQe, rhs).sum()

        return grad_z, grad_lambda_a, None, None, None, None, None


class DataVSLayer(nn.Module):
    """
    DataLayer using variable splitting formulation
    """

    def __init__(self, alpha_init, beta_init, learnable=True):
        """

        :param alpha_init -> float: Init value of data consistency block (DCB)
        :param beta_init -> float: Init value of weighted averageing blcok (WAB)
        """

        super(DataVSLayer, self).__init__()
        self.alpha = torch.nn.Parameter(torch.Tensor(1))
        self.alpha.data = torch.tensor(alpha_init, dtype=self.alpha.dtype)

        self.beta = torch.nn.Parameter(torch.Tensor(1))
        self.beta.data = torch.tensor(beta_init, dtype=self.beta.dtype)

        self.set_learnable(learnable)

    def forward(self, x, y, smaps, mask):
        A_x = forwardSoftSenseOpNoShift(x, smaps, 1.)
        k_dc = (1 - mask) * A_x + mask * (self.alpha * A_x + (1 - self.alpha) * y)
        x_dc = adjointSoftSenseOpNoShift(k_dc, smaps, 1.)
        x_wab = self.beta * x + (1 - self.beta) * x_dc

        return x_wab

    def extra_repr(self) -> str:
        return (
            f"alpha={self.alpha.item():.4g},"
            f"beta={self.beta.item():.4g},"
            f"learnable={self.learnable}"
        )

    def set_learnable(self, flag):
        self.alpha.requires_grad = flag
        self.beta.requires_grad = flag
