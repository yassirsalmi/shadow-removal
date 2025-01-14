import os
import numpy as np
import tensorflow as tf


class BaseModel:
    def __init__(self):
        self.device = "/cpu:0"  # Use CPU explicitly
        self.model = None
        self.save_dir = None
        self.input = None
        self.image_paths = None

    def name(self):
        return "BaseModel"

    def initialize(self):
        # Placeholder for any initialization logic
        print("Initialized model on CPU.")

    def forward(self, input_data):
        """
        Placeholder for the forward pass.
        Subclasses should override this method to define model predictions.
        """
        raise NotImplementedError("The forward method must be implemented in the subclass.")

    def get_image_paths(self):
        """
        Returns the paths of images being processed.
        """
        return self.image_paths

    def optimize_parameters(self):
        """
        Placeholder for optimization logic.
        Subclasses should override this to implement training steps.
        """
        raise NotImplementedError("The optimize_parameters method must be implemented in the subclass.")

    def get_current_visuals(self):
        """
        Returns the input for visual inspection.
        """
        return self.input

    def get_current_errors(self):
        """
        Returns the current errors during training or validation.
        """
        return {}

    def save(self, label):
        """
        Placeholder for saving model states. Subclasses can override this method.
        """
        raise NotImplementedError("The save method must be implemented in the subclass.")

    # Helper method to save the model
    def save_network(self, model, path, model_label, epoch_label):
        save_filename = f"{epoch_label}_net_{model_label}.h5"
        save_path = os.path.join(path, save_filename)
        model.save(save_path)
        print(f"Model saved to {save_path}")

    # Helper method to load the model
    def load_network(self, model, path, model_label, epoch_label):
        save_filename = f"{epoch_label}_net_{model_label}.h5"
        save_path = os.path.join(path, save_filename)
        print(f"Loading model from {save_path}")
        model = tf.keras.models.load_model(save_path)
        return model

    def update_learning_rate(self, optimizer, new_lr):
        """
        Update the learning rate of the optimizer.
        """
        optimizer.learning_rate.assign(new_lr)
        print(f"Updated learning rate to {new_lr}")

    def save_done(self, flag=False):
        """
        Saves a "done flag" as .npy and .txt to the save directory.
        """
        if not self.save_dir:
            raise ValueError("Save directory is not set.")
        flag_file = os.path.join(self.save_dir, "done_flag")
        np.save(flag_file, flag)
        np.savetxt(flag_file + ".txt", [flag], fmt="%i")
        print(f"Done flag saved to {flag_file} and {flag_file}.txt")
