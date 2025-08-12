
Reproduce cotracker by koo

## Environment


Prepare the environment by cloning the repository and installing the required dependencies:

```bash
conda create -y -n cotracker python=3.11
conda activate cotracker

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -U lightning matplotlib mediapy einops wandb peft timm opencv-python
```


#### Training Script


```bash
python experiment.py --config config/{your_script}.ini --mode train_davis --save_path ./path_to_save_checkpoints/
```