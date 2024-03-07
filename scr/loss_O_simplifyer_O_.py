import segmentation_models_multi_tasking as smp  # Assuming this is a correct import

class LossSimplifyer:
    def __init__(self, config):
        self.config = config
        # Set default values if not specified in config
        self.config.alpha_min = getattr(config, 'alpha_min', 0.1)
        self.config.alpha_max = getattr(config, 'alpha_max', 1.0)
        self.config.beta_min = getattr(config, 'beta_min', 0.1)
        self.config.beta_max = getattr(config, 'beta_max', 1.0)
        self.config.gamma_min = getattr(config, 'gamma_min', 0.1)
        self.config.gamma_max = getattr(config, 'gamma_max', 3.0)

        # Load loss functions and their parameter suggestion methods
        self.loss_functions = self.load_loss_functions()

    def load_loss_functions(self):

        loss_functions = {
            'diceloss': (smp.utils.losses.DiceLoss, 
                lambda trial: 
                    {'beta': trial.suggest_float('diceloss_beta', self.config.beta_min, self.config.beta_max, log=True)}),
            'tverskyloss': (smp.utils.losses.TverskyLoss, 
                lambda trial: 
                    {'alpha': trial.suggest_float('tverskyloss_alpha', self.config.alpha_min, self.config.alpha_max, log=True),
                     'beta': trial.suggest_float('tverskyloss_beta', self.config.beta_min, self.config.beta_max, log=True)}),
            'focaltverskyloss': (smp.utils.losses.FocalTverskyLoss, 
                lambda trial: 
                    {'gamma': trial.suggest_float('focaltverskyloss_gamma', self.config.gamma_min, self.config.gamma_max, log=True),
                     'alpha': trial.suggest_float('focaltverskyloss_alpha', self.config.alpha_min, self.config.alpha_max, log=True),
                     'beta': trial.suggest_float('focaltverskyloss_beta', self.config.beta_min, self.config.beta_max, log=True)}),
            'focaltverskyplusplusloss': (smp.utils.losses.FocalTverskyPlusPlusLoss, 
                lambda trial: 
                    {'gamma': trial.suggest_float('focaltverskyloss_gamma', self.config.gamma_min, self.config.gamma_max, log=True),
                     'alpha': trial.suggest_float('focaltverskyloss_alpha', self.config.alpha_min, self.config.alpha_max, log=True),
                     'beta': trial.suggest_float('focaltverskyloss_beta', self.config.beta_min, self.config.beta_max, log=True)}),
            'comboloss': (smp.utils.losses.ComboLoss, 
                lambda trial: 
                    {'alpha': trial.suggest_float('comboloss_alpha', self.config.alpha_min, self.config.alpha_max, log=True), 
                    'beta': trial.suggest_float('comboloss_beta', self.config.beta_min, self.config.beta_max, log=True)}),
            'dscplusplusloss': (smp.utils.losses.DSCPlusPlusLoss, 
                lambda trial: 
                    {'beta': trial.suggest_float('dscplusplusloss_beta', self.config.beta_min, self.config.beta_max, log=True),
                     'gamma': trial.suggest_float('dscplusplusloss_gamma', self.config.gamma_min, self.config.gamma_max, log=True)})
        }
        return loss_functions

    def suggest_loss_params(self, trial, lossfn):
        loss_name = lossfn.lower()

        # Check if the loss function requires alpha and beta parameter suggestions
        if loss_name not in ['dsc++','comboloss','diceloss']:
            alpha_value = trial.suggest_float(f'{loss_name}_alpha', self.config.alpha_min, self.config.alpha_max, log=True)
            beta_value = trial.suggest_float(f'{loss_name}_beta', self.config.beta_min, self.config.beta_max, log=True)
        else:
            alpha_value = None
            beta_value = None

        # Update the parameter suggestion methods for losses that require alpha and beta
        if loss_name in ['tverskyloss', 'focaltverskyloss', 'focaltverskyplusplusloss']:
            self.loss_functions[loss_name][1] = lambda trial: {'alpha': alpha_value, 'beta': beta_value}

        if loss_name in self.loss_functions:
            loss_class, params_func = self.loss_functions[loss_name]
            params = params_func(trial)
            return loss_class(**params)

        raise ValueError(f"Unknown loss function: {lossfn}")
