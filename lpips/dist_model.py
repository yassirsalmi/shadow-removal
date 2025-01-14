import tensorflow as tf
import numpy as np
from collections import OrderedDict
from scipy.ndimage import zoom
from tqdm import tqdm

class DistModel:
    def __init__(self, model='net-lin', net='alex', colorspace='Lab', model_path=None, 
                 printNet=False, spatial=False, is_train=False, lr=0.0001, beta1=0.5):
        """
        TensorFlow version of the DistModel.
        """
        self.model = model
        self.net = net
        self.is_train = is_train
        self.spatial = spatial
        self.model_name = f"{model} [{net}]"
        self.lr = lr

        if self.model == 'net-lin':
            # Pretrained net + linear layer
            self.net = self.build_net(net)
            if not is_train:
                print(f"Loading model from: {model_path}")
                self.net.load_weights(model_path)
        elif self.model in ['L2', 'l2']:
            self.net = self.L2_loss(colorspace)
        elif self.model in ['DSSIM', 'dssim', 'SSIM', 'ssim']:
            self.net = self.SSIM_loss(colorspace)
        else:
            raise ValueError(f"Model [{self.model}] not recognized.")

        if self.is_train:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta1)
        else:
            self.net.trainable = False

        if printNet:
            self.net.summary()

    def build_net(self, net_type):
        """
        Build the network based on the specified type (e.g., 'alex', 'vgg').
        """
        if net_type == 'alex':
            return tf.keras.applications.AlexNet(include_top=False, weights='imagenet', pooling='avg')
        elif net_type == 'vgg':
            return tf.keras.applications.VGG16(include_top=False, weights='imagenet', pooling='avg')
        else:
            raise ValueError(f"Network type [{net_type}] not recognized.")

    def L2_loss(self, colorspace):
        """
        Define an L2 loss in the specified colorspace.
        """
        def loss(y_true, y_pred):
            return tf.reduce_mean(tf.square(y_true - y_pred))
        return loss

    def SSIM_loss(self, colorspace):
        """
        Define an SSIM loss in the specified colorspace.
        """
        def loss(y_true, y_pred):
            return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
        return loss

    def forward(self, in0, in1):
        """
        Compute the distance between image patches `in0` and `in1`.
        """
        return tf.reduce_mean(tf.abs(self.net(in0) - self.net(in1)))

    def optimize_parameters(self, input_ref, input_p0, input_p1, judge):
        """
        Perform a single optimization step.
        """
        with tf.GradientTape() as tape:
            d0 = self.forward(input_ref, input_p0)
            d1 = self.forward(input_ref, input_p1)
            loss = self.compute_loss(d0, d1, judge)
        
        gradients = tape.gradient(loss, self.net.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.net.trainable_variables))
        return loss

    def compute_loss(self, d0, d1, judge):
        """
        Compute the ranking loss.
        """
        judge = tf.cast(judge, tf.float32)
        return tf.reduce_mean((d1 - d0) * (2.0 * judge - 1.0))

    def compute_accuracy(self, d0, d1, judge):
        """
        Compute accuracy of the distance predictions compared to the judge labels.
        """
        d1_lt_d0 = tf.cast(d1 < d0, tf.float32)
        return tf.reduce_mean(d1_lt_d0 * judge + (1 - d1_lt_d0) * (1 - judge))

    def get_current_visuals(self, var_ref, var_p0, var_p1):
        """
        Return current visuals.
        """
        zoom_factor = 256 / var_ref.shape[1]

        ref_img = self.tensor2im(var_ref)
        p0_img = self.tensor2im(var_p0)
        p1_img = self.tensor2im(var_p1)

        ref_img_vis = zoom(ref_img, [zoom_factor, zoom_factor, 1], order=0)
        p0_img_vis = zoom(p0_img, [zoom_factor, zoom_factor, 1], order=0)
        p1_img_vis = zoom(p1_img, [zoom_factor, zoom_factor, 1], order=0)

        return OrderedDict([('ref', ref_img_vis),
                            ('p0', p0_img_vis),
                            ('p1', p1_img_vis)])

    @staticmethod
    def tensor2im(tensor):
        """
        Convert tensor to image format.
        """
        tensor = tf.clip_by_value((tensor + 1.0) * 127.5, 0.0, 255.0)
        return tensor.numpy().astype(np.uint8)
