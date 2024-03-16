
from utils import Config

class TrialParameters:
    """
    Class representing the parameters for hypertuner trials.

    Attributes:
        loss (str): The name of the loss function.
        loss_weights (dict): The weights for specific loss parameters.
        lr_min_max (tuple): The range for the learning rate.

    Methods:
        initialize_loss_weights(): Initializes the loss weights based on the configured loss function.
        get_trial_parameters(): Returns the trial parameters for the loss function.
        get_trial_learning_rate(): Returns the trial learning rate.
    """
    def __init__(self, loss: str):
        self.loss = loss
        self.loss_weights = None
        self.lr_min_max = (0.0001, 0.009) # Default range for learning rate
        self.initialize_loss_weights()
        self.config = Config()

    def initialize_loss_weights(self):
        """
        Initializes the loss weights for study_queue and also learning-rate range for optuna to use based on the configured loss function.
        """
        loss_configurations = {
            "DSCPlusPlusLoss": ({'beta': 0.8, 'gamma': 2.6}, (0.001, 0.009)),
            "FocalTverskyLoss": ({'alpha': 0.7, 'beta': 0.3, 'gamma': 0.75}, (0.0001, 0.001)),
            "TverskyLoss": ({'alpha': 0.7, 'beta': 0.3}, (0.0001, 0.001)),
            "DiceLoss": ({'beta': 1.0}, (0.0001, 0.009)),
            "JaccardLoss": (None, (0.0001, 0.009)),
            "FocalTverskyPlusPlusLoss": ({'alpha': 0.3, 'beta': 0.7, 'gamma': 2.5}, (0.0001, 0.009)),
            "BCEWithLogitsLoss": (None, (0.0001, 0.009)),
            "ComboLoss": ({'alpha': 0.5, 'CE_RATIO': 0.5}, (0.0001, 0.009))
        }
        
        # Apply the configuration if the loss function is recognized
        if self.loss in loss_configurations:
            self.loss_weights, self.lr_min_max = loss_configurations[self.loss]
        else:
            print(f"Loss function {self.loss} not recognized. Using default loss weights and learning rate range.")

    def get_trial_parameters(self):
        """
        Returns the study_queue trial parameters for the loss function and learningrate.

        Returns:
            dict: The study_queue trial parameters.
        """
        trial_params = {
            "optimizer": self.config.optimizer,
            "loss": self.loss,
            "activation_function": self.config.activation_function,
            "lr": self.lr_min_max[0]
        }

        # Add specific loss parameters if available
        if self.loss_weights:
            for param, value in self.loss_weights.items():
                trial_params[f"{self.loss.lower()}_{param}"] = value

        return trial_params
    
