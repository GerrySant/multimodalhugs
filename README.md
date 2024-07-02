# Bridging Modalities: Leveraging LLM Knowledge with a General Encoding Framework.

This code is a simplified implementation of this [repository](https://github.com/GerrySant/slt_how2sign_wicv2023/tree/signwritting), originally implemented in [fairseq](https://github.com/facebookresearch/fairseq).

## Intallation

1. **Create a virtual environment named `multimodal-encoder`**:
    ```bash
    python -m venv multimodal-encoder
    ```

2. **Activate the virtual environment**:
    ```bash
    source multimodal-encoder/bin/activate
    ```

3. **Upgrade pip and setuptools**:
    ```bash
    pip install --upgrade pip setuptools
    ```

4. **Install PyTorch with CUDA support**:
    ```bash
    pip3 install torch torchvision torchaudio
    ```

5. **Install the framework dependencies**:
    ```bash
    cd /multimodal-mt
    pip install .
    ```

6. **Verify CUDa for the torch installation**:
    ```bash
    python -c "import torch; print(torch.cuda.is_available())"
    ```

   This should print `True` if CUDA support is properly enabled.