import os
import subprocess
import requests
import bz2
import yaml

# Load config
with open("config/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

def run_command(command):
    subprocess.run(command, shell=True, check=True)

def download_file(url, path):
    if os.path.exists(path):
        print(f"✅ {path} already exists.")
        return
    print(f"⬇️ Downloading {path}...")
    response = requests.get(url, stream=True)
    with open(path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def setup_environment():
    # 1. Clone Repo
    if not os.path.exists(cfg['paths']['repo_dir']):
        print("Cloning encoder4editing...")
        run_command(f"git clone {cfg['urls']['repo']}")
    
    # 2. Download Model Weights
    os.makedirs(os.path.dirname(cfg['paths']['ckpt_path']), exist_ok=True)
    download_file(cfg['urls']['ckpt'], cfg['paths']['ckpt_path'])

    # 3. Download Boundaries
    os.makedirs(cfg['paths']['boundary_dir'], exist_ok=True)
    base_url = cfg['urls']['boundaries']
    for name, filename in cfg['boundaries'].items():
        download_file(f"{base_url}/{filename}", f"{cfg['paths']['boundary_dir']}/{name}_boundary.npy")

    # 4. Download & Extract Dlib Predictor
    if not os.path.exists(cfg['paths']['dlib_shape_predictor']):
        print("Downloading Dlib Predictor...")
        compressed_file = "shape_predictor.dat.bz2"
        download_file(cfg['urls']['dlib'], compressed_file)
        
        print("Extracting Dlib...")
        with bz2.open(compressed_file, "rb") as source, open(cfg['paths']['dlib_shape_predictor'], "wb") as dest:
            dest.write(source.read())
        os.remove(compressed_file)

    print("\n✅ Setup Complete! You can now run main.py")

if __name__ == "__main__":
    setup_environment()