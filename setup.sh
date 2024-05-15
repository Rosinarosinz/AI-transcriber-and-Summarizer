

sudo apt update && sudo apt install -y ffmpeg
sudo apt-get install -y cmake build-essential pkg-config libgoogle-perftools-dev

python -m pip install --upgrade pip
python -m pip install -U openai-whisper
python -m pip install setuptools-rust transformers sentencepiece

# install requirements
python -m pip install -r requirements.txt

# CUDA 11.6
# pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
# CUDA 11.3
# pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
# CUDA 10.2
# pip install torch==1.12.0+cu102 torchvision==0.13.0+cu102 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu102
# CPU only
# pip install torch==1.12.0+cpu torchvision==0.13.0+cpu torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cpu