import dlib
import torchvision.transforms as transforms
import sys
import os
from PIL import Image

# Add repo to path for alignment utils
sys.path.append(os.path.join(os.getcwd(), 'encoder4editing'))
from utils.alignment import align_face

class ImageProcessor:
    def __init__(self, predictor_path):
        self.predictor = dlib.shape_predictor(predictor_path)
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def preprocess(self, image_path):
        # Align
        aligned_image = align_face(filepath=image_path, predictor=self.predictor)
        aligned_image = aligned_image.resize((256, 256))
        
        # Transform
        img_tensor = self.transform(aligned_image).unsqueeze(0).to('cuda')
        return aligned_image, img_tensor

    @staticmethod
    def tensor2im(var):
        # Helper to convert result back to image
        var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
        var = ((var + 1) / 2)
        var[var < 0] = 0
        var[var > 1] = 1
        var = var * 255
        return Image.fromarray(var.astype('uint8'))