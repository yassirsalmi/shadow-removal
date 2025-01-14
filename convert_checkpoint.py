import os
import torch
import onnx
import tensorflow as tf
import numpy as np

def convert_pytorch_to_onnx(pytorch_checkpoint_path, onnx_output_path, model_class):
    """
    Convert a PyTorch checkpoint to ONNX format
    
    Args:
        pytorch_checkpoint_path (str): Path to the PyTorch checkpoint
        onnx_output_path (str): Path to save the ONNX model
        model_class (type): The PyTorch model class to instantiate
    """
    try:
        checkpoint = torch.load(pytorch_checkpoint_path, map_location='cpu', weights_only=False)
        
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'generator' in checkpoint:
                state_dict = checkpoint['generator']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'g' in checkpoint: 
                state_dict = checkpoint['g']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        model = model_class()
        
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                cleaned_state_dict[k[7:]] = v
            elif k.startswith('cp.'):
                new_k = k.replace('cp.', '').replace('resnet.', '')
                cleaned_state_dict[new_k] = v
            else:
                cleaned_state_dict[k] = v
        
        model.load_state_dict(cleaned_state_dict, strict=False)
        
        model.eval()
        
        dummy_input = torch.randn(1, 3, 512, 512)
        
        torch.onnx.export(
            model,
            dummy_input,
            onnx_output_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output']
        )
        
        print(f"Successfully converted {pytorch_checkpoint_path} to ONNX: {onnx_output_path}")
    
    except Exception as e:
        print(f"Error converting {pytorch_checkpoint_path} to ONNX: {e}")
        raise

def convert_onnx_to_tensorflow(onnx_path, tf_output_path):
    """
    Convert ONNX model to TensorFlow SavedModel
    
    Args:
        onnx_path (str): Path to the ONNX model
        tf_output_path (str): Path to save the TensorFlow SavedModel
    """
    try:
        import onnx
        import tensorflow as tf
        import onnx_tf.backend
        
        # Load the ONNX model
        onnx_model = onnx.load(onnx_path)
        
        # Convert ONNX to TensorFlow
        tf_rep = onnx_tf.backend.prepare(onnx_model)
        
        # Save the TensorFlow model
        tf_rep.export_graph(tf_output_path)
        
        print(f"Successfully converted {onnx_path} to TensorFlow: {tf_output_path}")
    
    except Exception as e:
        print(f"Error converting ONNX to TensorFlow: {e}")
        raise

import torch
import torch.nn as nn

class StyleGANGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class BiSeNetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
        
        self.resnet.fc = nn.Conv2d(512, 19, kernel_size=1) 
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(0)
        
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        
        x = self.resnet.fc(x)
        return x

def main():
    os.makedirs('checkpoint', exist_ok=True)
    
    stylegan_checkpoint = 'checkpoint/550000.pt'
    if os.path.exists(stylegan_checkpoint):
        convert_pytorch_to_onnx(
            stylegan_checkpoint, 
            'checkpoint/stylegan.onnx', 
            StyleGANGenerator
        )
        
        convert_onnx_to_tensorflow(
            'checkpoint/stylegan.onnx', 
            'checkpoint/stylegan_tf'
        )
    
    face_seg_checkpoint = 'checkpoint/face-seg-BiSeNet-79999_iter.pth'
    if os.path.exists(face_seg_checkpoint):
        convert_pytorch_to_onnx(
            face_seg_checkpoint, 
            'checkpoint/face_seg.onnx', 
            BiSeNetModel
        )
        
        convert_onnx_to_tensorflow(
            'checkpoint/face_seg.onnx', 
            'checkpoint/face_seg_tf'
        )

if __name__ == '__main__':
    main()
