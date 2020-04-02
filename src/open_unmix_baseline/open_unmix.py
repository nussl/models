from nussl.separation import SeparationBase
import gin
from . import test

@gin.configurable
class OpenUnmix(SeparationBase):
    def __init__(self, input_audio_signal, targets, model_name='umxhq',
                 niter=1, softmask=False, alpha=1.0, model_path=None,
                 residual_model=False, device='cpu'):

        self.targets = targets
        self.model_name = model_name
        self.niter = 1
        self.softmask = False
        self.alpha = alpha
        self.residual_model = residual_model
        self.device = device

        super().__init__(input_audio_signal)

    def forward(self):
        estimates = test.separate(
            self.audio_signal.audio_data.T,
            self.targets,
            self.model_name,
            niter=self.niter,
            softmask=self.softmask,
            alpha=self.alpha,
            residual_model=self.residual_model,
            device=self.device
        )
        return estimates
    
    def run(self, estimates=None):
        if estimates is None:
            estimates = self.forward()
        self.estimates = estimates
        return self.estimates

    def make_audio_signals(self):
        self.keys = sorted(list(self.estimates.keys()))
        estimates = []

        for k in self.keys:
            audio_data = self.estimates[k]
            _estimate = self.audio_signal.make_copy_with_audio_data(
                audio_data.T
            )
            estimates.append(_estimate)
        return estimates
