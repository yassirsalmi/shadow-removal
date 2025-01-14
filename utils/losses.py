import tensorflow as tf
import numpy as np
from PIL import Image

# Force TensorFlow to use CPU
tf.config.set_visible_devices([], 'GPU')

class PerceptLoss:
    def __init__(self):
        pass

    def __call__(self, loss_net, fake_img, real_img):
        real_feature = loss_net(real_img, training=False)
        fake_feature = loss_net(fake_img, training=True)
        perceptual_penalty = tf.reduce_mean(tf.square(fake_feature - real_feature))
        return perceptual_penalty

    def set_ftr_num(self, ftr_num):
        pass

class DiscriminatorLoss:
    def __init__(self, ftr_num=4, data_parallel=False):
        self.ftr_num = ftr_num

    def __call__(self, D, fake_img, real_img):
        _, real_feature = D(real_img, training=False)
        _, fake_feature = D(fake_img, training=True)
        
        D_penalty = 0
        for i in range(self.ftr_num):
            f_id = -i - 1  # i:0123, f_id: -1, -2, -3, -4
            D_penalty += tf.reduce_mean(tf.abs(fake_feature[f_id] - real_feature[f_id]))
        return D_penalty

    def set_ftr_num(self, ftr_num):
        self.ftr_num = ftr_num

class ExclusionLoss(tf.keras.layers.Layer):
    def __init__(self, level=3):
        """Loss on the gradient based on:
        http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Single_Image_Reflection_CVPR_2018_paper.pdf
        """
        super(ExclusionLoss, self).__init__()
        self.level = level
        self.avg_pool = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2)
        
    def get_gradients(self, img1, img2):
        gradx_loss = []
        grady_loss = []

        for l in range(self.level):
            gradx1, grady1 = self.compute_gradient(img1)
            gradx2, grady2 = self.compute_gradient(img2)
            
            alphay = 1
            alphax = 1
            
            gradx1_s = tf.sigmoid(gradx1) * 2 - 1
            grady1_s = tf.sigmoid(grady1) * 2 - 1
            gradx2_s = tf.sigmoid(gradx2 * alphax) * 2 - 1
            grady2_s = tf.sigmoid(grady2 * alphay) * 2 - 1

            gradx_loss.extend(self._all_comb(gradx1_s, gradx2_s))
            grady_loss.extend(self._all_comb(grady1_s, grady2_s))
            
            img1 = self.avg_pool(img1)
            img2 = self.avg_pool(img2)
            
        return gradx_loss, grady_loss

    def _all_comb(self, grad1_s, grad2_s):
        v = []
        for i in range(3):
            for j in range(3):
                v.append(tf.pow(tf.reduce_mean(tf.square(grad1_s[:, :, :, j]) * 
                                             tf.square(grad2_s[:, :, :, i])), 0.25))
        return v

    def call(self, img1, img2):
        gradx_loss, grady_loss = self.get_gradients(img1, img2)
        loss_gradxy = (tf.reduce_sum(gradx_loss) / (self.level * 9) + 
                      tf.reduce_sum(grady_loss) / (self.level * 9))
        return loss_gradxy / 2.0

    def compute_gradient(self, img):
        gradx = img[:, 1:, :, :] - img[:, :-1, :, :]
        grady = img[:, :, 1:, :] - img[:, :, :-1, :]
        return gradx, grady

class GradientLoss(tf.keras.layers.Layer):
    """L1 loss on the gradient of the picture"""
    def __init__(self):
        super(GradientLoss, self).__init__()

    def call(self, a):
        gradient_a_x = tf.abs(a[:, :, :-1, :] - a[:, :, 1:, :])
        gradient_a_y = tf.abs(a[:, :-1, :, :] - a[:, 1:, :, :])
        return tf.reduce_mean(gradient_a_x) + tf.reduce_mean(gradient_a_y)

class ArcFaceLoss:
    def __init__(self, model_path):
        # Note: You'll need to convert the IRSE model to TensorFlow format
        self.model = tf.keras.models.load_model(model_path)
        
    def __call__(self, gen_img, target_img):
        gen_img = resize_image(gen_img, 112)
        target_img = resize_image(target_img, 112)

        emb1 = tf.squeeze(self.model(gen_img, training=False))
        emb2 = tf.squeeze(self.model(target_img, training=False))
        
        sim = tf.reduce_sum(emb1 * emb2) / (tf.norm(emb1) * tf.norm(emb2))
        return sim

def resize_image(img, size):
    return tf.image.resize(img, [size, size])

def g_loss(D, fake_img):
    fake_pred = D(fake_img, training=True)
    loss = tf.reduce_mean(tf.nn.softplus(-fake_pred))
    return loss

def tf_to_pil(img):
    """Convert TensorFlow tensor to PIL Image."""
    img_np = img.numpy()[0]
    ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(ar)

def pil_to_tf(img):
    """Convert PIL Image to TensorFlow tensor."""
    ar = np.array(img)
    if len(ar.shape) == 2:
        ar = ar[..., np.newaxis]
    img_np = ar.astype(np.float32) / 255.
    return tf.convert_to_tensor(img_np[np.newaxis, ...])

def l_loss(img1, img2):
    """Compute luminance loss between two images."""
    # Convert RGB to luminance (Y) using BT.601 coefficients
    img1_y = 0.257 * img1[:, :, :, 0] + 0.564 * img1[:, :, :, 1] + 0.098 * img1[:, :, :, 2] + 16
    img2_y = 0.257 * img2[:, :, :, 0] + 0.564 * img2[:, :, :, 1] + 0.098 * img2[:, :, :, 2] + 16
    
    return tf.reduce_mean(tf.square(img1_y - img2_y))

if __name__ == "__main__":
    # Test the losses
    batch_size = 1
    height = 256
    width = 256
    channels = 3
    
    # Create test inputs
    img1 = tf.random.normal([batch_size, height, width, channels])
    img2 = tf.random.normal([batch_size, height, width, channels])
    
    # Test gradient loss
    grad_loss = GradientLoss()
    loss_value = grad_loss(img1)
    print(f"Gradient loss: {loss_value}")
    
    # Test exclusion loss
    excl_loss = ExclusionLoss()
    loss_value = excl_loss(img1, img2)
    print(f"Exclusion loss: {loss_value}")
    
    # Test luminance loss
    lum_loss = l_loss(img1, img2)
    print(f"Luminance loss: {lum_loss}")
