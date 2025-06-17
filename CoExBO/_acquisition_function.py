import torch
from botorch.acquisition import AnalyticAcquisitionFunction
from botorch.utils import t_batch_mode_transform
from ._utils import TensorManager


class CoExBO_UCB(AnalyticAcquisitionFunction, TensorManager):
    def __init__(
        self,
        model,
        prior_pref,
        beta,
        gamma,
        pi_augment=True,
    ):
        AnalyticAcquisitionFunction.__init__(
            self, 
            model=model,
            posterior_transform=None,
        )
        TensorManager.__init__(self)
        self.register_buffer("beta", self.tensor(beta))
        self.register_buffer("gamma", self.tensor(gamma))        
        self.pi_augment = pi_augment
        self.initialise(prior_pref, model)
        
    def initialise(self, prior_pref, model):
        if not hasattr(prior_pref, "y_test_mean"):
            prior_pref.normalising_constant()
        
        self.E_y_pref = prior_pref.y_test_mean.mean()
        self.std_y_pref = prior_pref.y_test_mean.std()
        self.E_y_obs = model.train_targets.mean()
        self.std_y_obs = model.train_targets.std()
        self.prior_pref = prior_pref
        
    def prior_gp(self, X):
        prior_mean, prior_std = self.prior_pref.probability(X, both=True)

        prior_mean_conv = (prior_mean - self.E_y_pref) / self.std_y_pref * self.std_y_obs + self.E_y_obs
        prior_std_conv = prior_std / self.std_y_pref * self.std_y_obs
        return prior_mean_conv, prior_std_conv
    
    def posterior_gp(self, X, likelihood_gp_mean, likelihood_gp_std):
        prior_gp_mean, prior_gp_std = self.prior_gp(X)
        prior_gp_std_max = (
            self.gamma * likelihood_gp_std.pow(2) + prior_gp_std.pow(2)
        ).sqrt()
        posterior_gp_std = (
            prior_gp_std_max.pow(2) * likelihood_gp_std.pow(2) / (
                prior_gp_std_max.pow(2) + likelihood_gp_std.pow(2)
            )
        ).sqrt()
        posterior_gp_mean = (
            posterior_gp_std.pow(2) / prior_gp_std_max.pow(2)
        ) * prior_gp_mean + (
            posterior_gp_std.pow(2) / likelihood_gp_std.pow(2)
        ) * likelihood_gp_mean
        return posterior_gp_mean, posterior_gp_std
        
    @t_batch_mode_transform(expected_q=1)
    def forward(self, X):
        likelihood_gp_mean, likelihood_gp_std = self._mean_and_sigma(X)
        if self.pi_augment:
            posterior_gp_mean, posterior_gp_std = self.posterior_gp(X, likelihood_gp_mean, likelihood_gp_std)
            return posterior_gp_mean + self.beta.sqrt() * posterior_gp_std
        else:
            return likelihood_gp_mean + self.beta.sqrt() * likelihood_gp_std


class PiBO_UCB(AnalyticAcquisitionFunction, TensorManager):
    def __init__(
        self,
        model,
        prior_pref,
        beta,
        gamma,
        pi_augment=True,
    ):
        AnalyticAcquisitionFunction.__init__(
            self, 
            model=model,
            posterior_transform=None,
        )
        TensorManager.__init__(self)
        self.prior_pref = prior_pref
        self.pi_augment = pi_augment
        self.register_buffer("beta", self.tensor(beta))
        self.register_buffer("gamma", self.tensor(gamma))
        
    def prior_gp(self, X):
        prior_mean = self.prior_pref.pdf(X)
        return prior_mean
        
    @t_batch_mode_transform(expected_q=1)
    def forward(self, X):
        mean, sigma = self._mean_and_sigma(X)
        ucb = mean + self.beta.sqrt() * sigma
        ucb_norm = (ucb - torch.min(ucb)) / (torch.max(ucb) - torch.min(ucb))

        if self.pi_augment:
            prior_mean = self.prior_gp(X)
            prior_norm = (prior_mean - torch.min(prior_mean)) / (torch.max(prior_mean) - torch.min(prior_mean))
            ucb_norm *= prior_norm.pow(self.gamma)
        return ucb_norm

class AlphaPiBO_UCB(AnalyticAcquisitionFunction, TensorManager):
    '''
    Possible issues: Because you need to normalize the ucb values and prior values, you might need 
    to change all other acquisition function classes so that they're all min-max normalized.
    '''
    def __init__(
        self,
        model,
        prior_pref,
        beta,
        alpha,
        pi_augment=True,
    ):
        AnalyticAcquisitionFunction.__init__(
            self, 
            model=model,
            posterior_transform=None,
        )
        TensorManager.__init__(self)
        self.prior_pref = prior_pref
        self.pi_augment = pi_augment
        self.alpha = alpha
        self.register_buffer("beta", self.tensor(beta))
        self.register_buffer("alpha", self.tensor(alpha))
        
    def prior_gp(self, X):
        prior_mean = self.prior_pref.pdf(X)
        return prior_mean
        
    @t_batch_mode_transform(expected_q=1)
    def forward(self, X):
        mean, sigma = self._mean_and_sigma(X)
        ucb = mean + self.beta.sqrt() * sigma
        ucb_norm = (ucb - torch.min(ucb)) / (torch.max(ucb) - torch.min(ucb))
        af_val = ucb_norm

        if self.pi_augment:
            prior_mean = self.prior_gp(X)
            prior_norm = (prior_mean - torch.min(prior_mean)) / (torch.max(prior_mean) - torch.min(prior_mean))
            af_val = ucb_norm * ((prior_norm*self.alpha) + (1 - self.alpha))

        return af_val