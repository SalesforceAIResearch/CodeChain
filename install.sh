if conda info --envs | grep -q codechain; then echo "Skip! codechain already exists"
else conda create -n codechain python=3.7 -y
fi
source activate codechain 
pip install --upgrade pip
pip install datasets transformers timeout_decorator pyext openai flask scipy scikit-learn
# install torch for 1.10 for cuda 11.1
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html