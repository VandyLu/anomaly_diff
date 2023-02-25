"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

import enum
import math

import numpy as np
import torch as th
import torch.nn.functional as F

from .nn import mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == 'constant':
        return np.ones(num_diffusion_timesteps, dtype=np.float64) * 1e-3
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

        self.feature_extractor = FeatureExtractor()
        # fix pretrained network
        self.feature_extractor.cuda()
        self.feature_extractor.eval()
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
    
        sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.copy()
        sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.copy()
        # def feat_cond_fn(x, x_target, feature_extractor, t):
        def feat_cond_fn(x, t, feature_extractor=None, x_target=None, diffusion_model=None):
            assert feature_extractor is not None
            assert x_target is not None

            with th.no_grad():
                # use gt image
                targets, _ = feature_extractor(x_target)

                # use feature of noised image
                # noise = th.randn_like(x_target)
                # x_t_gt = _extract_into_tensor(sqrt_alphas_cumprod, t//4, x_target.shape) * x_target
                # + _extract_into_tensor(sqrt_one_minus_alphas_cumprod, t//4, x_target.shape) * noise
                # targets, _ = feature_extractor(x_t_gt)

            with th.enable_grad():
                x_in = x.detach().requires_grad_(True)
                feats, _ = feature_extractor(x_in)
                losses = []
                for idx, (feat, target) in enumerate(zip(feats, targets)):
                    # print('feat: ', feat.shape)
                    # print('target: ', target.shape)
                    if idx <= 2:
                        # loss = mean_flat((feat - target.detach()).abs())
                        loss = mean_flat((feat - target.detach())**2)
                        losses.append(loss)
                loss = sum(losses)
                
                grad = th.autograd.grad(loss, x_in)[0] * (-50000) * 1
                # negative, minimize mse loss
                # return -10 * t.float() * grad
                grad2 = -(x-x_target) * 0.00
                print(grad.abs().mean(), grad2.abs().mean())
                return grad + grad2 

        self.feat_cond_fn = feat_cond_fn

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
            "eps": model_output,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
        # print('t: ', t, 'mean: ', p_mean_var["mean"].abs().mean())
        # print('t: ', t, 'grad: ', gradient.abs().mean())
        # print('t: ', t, 'var*grad: ', (p_mean_var["variance"] * gradient.float()).abs().mean())
        new_mean = (
            p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        )
        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(
            x, self._scale_timesteps(t), **model_kwargs
        )

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t
        )
        return out

    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"], "t": t}

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        mid = []
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample
            mid.append(sample)
        return final["sample"], mid


    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)
        
        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                img = out["sample"]

    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_next)
            + th.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        mid = []
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        ):
            final = sample
            mid.append(sample)
        return final["sample"], mid

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out
                img = out["sample"]

    def _vb_terms_bpd(
        self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        # noise = th.randn_like(x_start)
        # x_t_1 = self.q_sample(x_start=x_start, t=t-1, noise=noise)
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl_map = kl / np.log(2.0)
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll_map = decoder_nll / np.log(2.0)
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        while len(t.shape) < len(decoder_nll.shape):
            t = t[..., None]

        # output_map = th.where((t.expand_as(decoder_nll) == 0), decoder_nll_map, kl_map)
        output_map = kl_map

        return {"output": output, "pred_xstart": out["pred_xstart"], 'nll_map': output_map,
        'decoder_nll_map': decoder_nll_map, 'kl_map': kl_map}

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}
        
        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                vb_term_out = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )
                terms["vb"] = vb_term_out["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0
                pred_xstart = vb_term_out["pred_xstart"]

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape
            terms["mse"] = mean_flat((target - model_output) ** 2)
            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]

            # get prev x
            # feat_t, _ = self.feature_extractor(pred_xstart)
            # feat_start, _ = self.feature_extractor(x_start)
            # # MSE feat loss
            # for idx in range(len(feat_t)):
            #     mse_loss = mean_flat((feat_t[idx] - feat_start[idx]) ** 2)
            #     terms["feat_mse.L{}".format(idx)] = mse_loss
            #     terms["loss"] += mse_loss

        else:
            raise NotImplementedError(self.loss_type)


        return terms

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def _vb_terms_bpd_anom(
        self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        # noise = th.randn_like(x_start)
        # x_t_1 = self.q_sample(x_start=x_start, t=t-1, noise=noise)
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl_map = kl / np.log(2.0)
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll_map = decoder_nll / np.log(2.0)
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        while len(t.shape) < len(decoder_nll.shape):
            t = t[..., None]
        output_map = th.where((t.expand_as(decoder_nll) == 0), decoder_nll_map, kl_map)

        return {"output": output, "pred_xstart": out["pred_xstart"], 'nll_map': output_map,
        # 'other': [true_mean, true_log_variance_clipped, out['mean'], out['log_variance']],
        'log_variance': out['log_variance'],
        'decoder_nll_map': decoder_nll_map, 'kl_map': kl_map, "eps": out['eps']}

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None, visual=False):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.

        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        import cv2
        device = x_start.device
        batch_size = x_start.shape[0]

        # x_start = th.nn.functional.interpolate(x_start, (128, 128), mode='bilinear')
        model_kwargs['x_target'] = x_start

        def merge_list(x_list):
            x_out = dict()
            keys = x_list[0].keys()
            n = len(x_list)
            for k in keys:
                x_out[k] = sum([m[k] for m in x_list]) / n
            return x_out

        x_cur = th.randn_like(x_start)

        if not hasattr(self, 'id'):
            self.id = 0
        results = []
        vb_results = []
        with th.no_grad():
            for t in list(range(self.num_timesteps))[::-1][-100::]:
                
                t_batch = th.tensor([t] * batch_size, device=device)

                out_list = []
                if t <= 5:
                    repeat = 5
                else:
                    repeat = 1
                for i in range(repeat):
                    noise = th.randn_like(x_start)
                    x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
                    # x_t = x_start
                    with th.no_grad():
                        out = self._vb_terms_bpd_anom(
                            model,
                            x_start=x_start,
                            x_t=x_t,
                            t=t_batch,
                            clip_denoised=clip_denoised,
                            model_kwargs=model_kwargs,
                        )
                    eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
                    eps_err = (eps - noise).abs()
                    out['eps_err'] = eps_err
                    out_list.append(out)
                out = merge_list(out_list)

                vb = out['nll_map']
                vb_results.append((vb - vb.mean())/vb.std())

                diff = (out["pred_xstart"] - x_start).abs()
                eps_err = out["eps_err"]
                # diff = out["diff"] 
                # results.append(diff)
                results.append((diff - diff.mean())/diff.std())

                visual = True
                if (t <= 5 or t % 30 == 0) and visual:
                # if visual:
                    if not hasattr(self, 'id'):
                        self.id = 0

                    x0 = (th.clip((out["pred_xstart"].cpu().permute(0, 2, 3, 1)+1)*127.5, 0, 255).numpy()[0, :, :, ::-1]).astype(np.uint8)
                    xt = (th.clip((x_t.cpu().permute(0, 2, 3, 1)+1)*127.5, 0, 255).numpy()[0, :, :, ::-1]).astype(np.uint8)
                    cv2.imwrite('./img_{:04d}_{:04d}_x0_vis.jpg'.format(self.id, t), x0)
                    cv2.imwrite('./img_{:04d}_{:04d}_xt_vis.jpg'.format(self.id, t), xt)

                    eps_err_vis = (eps_err**2 *127.5* 20).cpu().permute(0, 2, 3, 1).mean(-1).clamp(0, 255).numpy()[0].astype(np.uint8)
                    cv2.imwrite('./img_{:04d}_{:04d}_epserr_vis.jpg'.format(self.id, t), eps_err_vis)


                    vb_vis = out["nll_map"]
                    vb_vis = (((vb_vis - vb_vis.mean()) / vb_vis.std()+1)*20).cpu().permute(0, 2, 3, 1).mean(-1).clamp(0, 255).numpy()[0].astype(np.uint8)
                    cv2.imwrite('./img_{:04d}_{:04d}_vb_vis.jpg'.format(self.id, t), vb_vis)
                    print(t, vb.mean(), vb.max())

                    # result_vis = ((result_vis-result_vis.min())/(result_vis.max()-result_vis.min())*255).cpu().permute(0, 2, 3, 1).mean(-1).clamp(0, 255).numpy()[0].astype(np.uint8)
                    diff_vis = diff
                    diff_vis = (((diff_vis - diff_vis.mean()) / diff_vis.std()+1)*20).cpu().permute(0, 2, 3, 1).mean(-1).clamp(0, 255).numpy()[0].astype(np.uint8)
                    # diff_vis = ((diff - diff.min())/(diff.max()-diff.min())*255).clip(0, 255).detach().cpu().numpy().astype(np.uint8)
                    cv2.imwrite('./img_{:04d}_{:04d}_diff_vis.jpg'.format(self.id, t), diff_vis)
                    

        vb_map = sum(vb_results) / len(vb_results)
        diff_map = sum(results) / len(results)
        if visual:
            x0 = (th.clip((x_start.cpu().permute(0, 2, 3, 1)+1)*127.5, 0, 255).numpy()[0, :, :, ::-1]).astype(np.uint8)
            cv2.imwrite('./img_{:04d}_origin.jpg'.format(self.id), x0)
            # print(vb_map.min(), vb_map.max(), vb_map.mean())
            # vb_map_vis = th.clip((vb_map * 5).cpu().permute(0, 2, 3, 1).sum(-1), 0, 255).numpy()[0].astype(np.uint8)
            diff_map_vis = th.clip(((diff_map+1)*20).mean(dim=1)[0].cpu(), 0, 255).numpy().astype(np.uint8)
            vb_map_vis = th.clip(((vb_map+1)*20).mean(dim=1)[0].cpu(), 0, 255).numpy().astype(np.uint8)
            # vb_map_vis = th.clip(255*(1.0-th.exp(-vb_map)).cpu().permute(0, 2, 3, 1).mean(-1), 0, 255).numpy()[0].astype(np.uint8)
            # vb_map_vis = th.clip(((vb_map - vb_map.min()) /(vb_map.max()-vb_map.min())*255).cpu().permute(0, 2, 3, 1).mean(-1), 0, 255).numpy()[0].astype(np.uint8)
            cv2.imwrite('./img_{:04d}_vbmap.jpg'.format(self.id), vb_map_vis)
            cv2.imwrite('./img_{:04d}_diffmap.jpg'.format(self.id), diff_map_vis)
            self.id += 1

        # pred_mask = vb_map.mean(dim=1, keepdim=True) * 10
        pred_mask = vb_map.mean(dim=1, keepdim=True) 
        # pred_mask = diff_map.mean(dim=1, keepdim=True) * 10
        print(pred_mask.mean(), pred_mask.max())
        return {
            "pred_mask": pred_mask,
            'single_masks': results,
        }


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

from torchvision.models import resnet50, resnet18, wide_resnet50_2
class FeatureExtractor(th.nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # self.model = resnet50(pretrained=True)
        self.model = wide_resnet50_2(pretrained=True)
        self.model.eval()
        self.model.cuda()
    
    def forward(self, img):
        x = self.model.conv1(img)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x1 = self.model.layer1(x)
        x2 = self.model.layer2(x1)
        x3 = self.model.layer3(x2)
        x4 = self.model.layer4(x3)
    
        xg = self.model.avgpool(x4)

        x11 = x1
        x21 = F.interpolate(x2, size=x1.shape[-2:], mode='nearest')
        x31 = F.interpolate(x3, size=x1.shape[-2:], mode='nearest')
        x41 = F.interpolate(x4, size=x1.shape[-2:], mode='nearest')

        return [x1, x2, x3, x4], xg
    
class Padim(th.nn.Module):
    def __init__(self, image_size=224):
        super().__init__()
        self.height = image_size
        self.width = image_size

        self.stages = [0, 1, 2]
        self.online_encoder = FeatureExtractor()
        self.rand_dims = th.randperm((256+512+1024)//1)[:500].cuda()
    
    def train_padim(self, data_loader, alpha_bar_t, noise):
        
        outputs = []
        with th.no_grad():
            idx = 0 
            for data in data_loader:
                img, mask, _ = data
                img = img.cuda()

                # noise = th.randn_like(img)
                img_noisy = np.sqrt(alpha_bar_t) * img + np.sqrt(1.0 - alpha_bar_t) * noise

                idx += 1
                xl, xg = self.online_encoder(img_noisy)

                xl = [f.mean(dim=0, keepdim=True) for f in xl]
                    
                xl.append(xg)

                outputs.append(xl)

            x1, x2, x3, x4, xg = map(list, zip(*outputs))
            self.feat1 = th.cat(x1, dim=0).mean(dim=0, keepdim=True)
            self.feat2 = th.cat(x2, dim=0).mean(dim=0, keepdim=True)
            self.feat3 = th.cat(x3, dim=0).mean(dim=0, keepdim=True)
            self.feat4 = th.cat(x4, dim=0).mean(dim=0, keepdim=True)

        with th.no_grad():
            outputs = []
            H, W, C = self.height//4, self.width//4, len(self.rand_dims)

            S = th.zeros((H, W, C, C)).cuda()
            n_count = 0
            for data in data_loader:
                img, mask, _ = data
                img = img.cuda()

                # noise = th.randn_like(img)
                img_noisy = np.sqrt(alpha_bar_t) * img + np.sqrt(1.0 - alpha_bar_t) * noise

                n_count += img.shape[0]

                if img.size(1) == 1:
                    img = img.expand(-1, 3, -1, -1)

                xl, xg = self.online_encoder(img_noisy)

                xl = self.fuse_feats(xl, self.stages)
                fm = self.fuse_feats(self.featmaps, self.stages)

                fi = self.reduce_dimension(xl)
                fm = self.reduce_dimension(fm)

                N, C, H, W = fi.size()
                assert N == 1
                fi = fi - fm
                fi = fi.permute(0, 2, 3, 1).reshape(-1, C, 1)
                    
                Si = fi.bmm(fi.permute(0, 2, 1)) # NHW, C, C
                # Note: sum will cause pytorch bug
                # S += Si.view(N, H, W, C, C).sum(dim=0, keepdim=False)
                S += Si.view(H, W, C, C)

            S /= (n_count - 1)
            S += 0.01 * th.eye(S.size(-1)).expand_as(S).cuda()

            S_inv = th.inverse(S)
            self.S = S
            self.S_inv = S_inv
    
    @property
    def featmaps(self):
        return [self.feat1, self.feat2, self.feat3, self.feat4]

    def reduce_dimension(self, f):
        return f[:, self.rand_dims, ...]

    def fuse_feats(self, featmaps, stages=[0, 1, 2]):
        target_size = featmaps[0].size()[-2:]
        results = []
        for idx, f in enumerate(featmaps):
            if idx in stages:
                #f_resize = F.interpolate(f, size=target_size, mode='nearest')
                f_resize = F.interpolate(f, size=target_size, mode='bilinear')
                results.append(f_resize)

        return th.cat(results, dim=1)

    def forward(self, img):
        te_featmaps, te_feat = self.online_encoder(img)

        te_featmaps = self.fuse_feats(te_featmaps, self.stages)
        te_featmaps = self.reduce_dimension(te_featmaps)
        N, C, H, W = te_featmaps.size()

        #with torch.autograd.profiler.profile(use_cuda=True) as prof:
        # with th.no_grad():
        with th.enable_grad():
            # N, C, H, W
            f = te_featmaps - self.reduce_dimension(self.fuse_feats(self.featmaps, self.stages))
            # H*W, C   H*W, C, C
            f = f.permute(0, 2, 3, 1)
            # H*W, 1, 1
            anoms = f.reshape(N*H*W, 1, C).bmm(self.S_inv.view(H*W, C, C).repeat([N, 1, 1]).bmm(f.reshape(N*H*W, C, 1))).sqrt()
            anoms = anoms.reshape(N, 1, H, W)
        anoms = F.interpolate(anoms, size=img.size()[-2:], mode='bilinear')
        return anoms
    
    def get_score(self, anoms):
        score = anoms.flatten(1).topk(dim=-1, k=64)[0].mean(-1)
        return score

