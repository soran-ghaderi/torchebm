
import pytest
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import tempfile
import os

from torchebm.utils.training import (
    update_ema,
    requires_grad,
    save_checkpoint,
    load_checkpoint,
)
from torchebm.utils.image import (
    center_crop_arr,
    create_npz_from_sample_folder,
)

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)

@pytest.fixture
def device():
    return torch.device("cpu") # Keep utils tests on CPU for simplicity unless needed

def test_update_ema():
    model = SimpleModel()
    ema_model = SimpleModel()
    
    # Initialize with different weights
    nn.init.constant_(model.linear.weight, 1.0)
    nn.init.constant_(ema_model.linear.weight, 0.0)
    
    decay = 0.5
    # EMA update: ema = decay * ema + (1-decay) * model
    # ema = 0.5 * 0 + 0.5 * 1 = 0.5
    
    update_ema(ema_model, model, decay=decay)
    
    assert torch.allclose(ema_model.linear.weight, torch.tensor(0.5))
    
    # Second update
    # model still 1.0
    # ema = 0.5 * 0.5 + 0.5 * 1.0 = 0.25 + 0.5 = 0.75
    update_ema(ema_model, model, decay=decay)
    assert torch.allclose(ema_model.linear.weight, torch.tensor(0.75))

def test_requires_grad():
    model = SimpleModel()
    
    requires_grad(model, False)
    for p in model.parameters():
        assert p.requires_grad is False
        
    requires_grad(model, True)
    for p in model.parameters():
        assert p.requires_grad is True

def test_center_crop_arr():
    # Create a 100x50 image
    img = Image.new('RGB', (100, 50), color='red')
    
    # Crop to 40
    # Steps:
    # 1. min(100, 50) = 50 >= 2*40? No.
    # 2. scale = 40 / 50 = 0.8
    # 3. resize: (100*0.8, 50*0.8) = (80, 40)
    # 4. crop center 40x40.
    
    cropped = center_crop_arr(img, 40)
    assert cropped.size == (40, 40)
    np_arr = np.array(cropped)
    assert np_arr.shape == (40, 40, 3)

def test_save_load_checkpoint():
    with tempfile.TemporaryDirectory() as tmpdir:
        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        step = 100

        # Save
        path = save_checkpoint(model, optimizer, step, tmpdir, args={"foo": "bar"})
        assert os.path.exists(path)

        # Load
        new_model = SimpleModel()
        new_opt = torch.optim.SGD(new_model.parameters(), lr=0.1)

        # Modify new model weights so we can be sure they are loaded
        nn.init.constant_(new_model.linear.weight, 99.0)

        ckpt = load_checkpoint(path, new_model, optimizer=new_opt)

        # Check step
        assert ckpt["step"] == 100
        assert ckpt["args"]["foo"] == "bar"

        # Check weights matched
        assert torch.allclose(new_model.linear.weight, model.linear.weight)


def test_create_npz_from_sample_folder():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy images
        for i in range(5):
            img = Image.new('RGB', (10, 10), color='blue')
            img.save(f"{tmpdir}/{i:06d}.png")

        npz_path = create_npz_from_sample_folder(tmpdir, num=5)

        assert os.path.exists(npz_path)
        data = np.load(npz_path)
        assert 'arr_0' in data
        assert data['arr_0'].shape == (5, 10, 10, 3)
