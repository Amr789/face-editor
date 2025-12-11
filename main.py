import argparse
import yaml
import torch
import matplotlib.pyplot as plt
from src.model_handler import load_model
from src.image_processor import ImageProcessor
from src.latent_editor import LatentEditor

def main(image_path):
    # Load Config
    with open("config/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # 1. Initialize
    net = load_model()
    processor = ImageProcessor(cfg['paths']['dlib_shape_predictor'])
    editor = LatentEditor(net, cfg['paths']['boundary_dir'])

    # 2. Process Image
    print(f"Processing {image_path}...")
    original_pil, img_tensor = processor.preprocess(image_path)

    # 3. Invert (Get Latents)
    with torch.no_grad():
        _, codes = net(img_tensor, randomize_noise=False, return_latents=True)

    # 4. Apply Edits
    edits = {
        "Original": img_tensor[0],
        "Age (+Old)": editor.apply_edit(codes, "age", strength=-12.0),
        "Smile": editor.apply_edit(codes, "smile", strength=-8.0, target_layers=range(4, 18)),
        "Pose": editor.apply_edit(codes, "pose", strength=8.0, target_layers=range(0, 8))
    }

    # 5. Visualize
    fig, ax = plt.subplots(1, len(edits), figsize=(20, 5))
    for i, (name, tensor) in enumerate(edits.items()):
        img = processor.tensor2im(tensor)
        ax[i].imshow(img)
        ax[i].set_title(name)
        ax[i].axis('off')
    
    output_file = "result.png"
    plt.savefig(output_file)
    print(f"✅ Done! Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    args = parser.parse_args()
    main(args.image)