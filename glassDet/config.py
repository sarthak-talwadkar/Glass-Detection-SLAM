# config.py
class Config:
    # Dataset
    data_root = './data/'
    img_size = 384
    
    # Model
    backbone = 'resnext101_32x8d'
    pretrained = True
    
    # Training
    batch_size = 10  
    num_epochs = 200
    learning_rate = 1e-4
    weight_decay = 5e-4
    num_workers = 2
    
    # Loss weights
    bce_weight = 1.0
    iou_weight = 0.5
    boundary_weight = 0.5
    
    # Checkpoints
    checkpoint_dir = './checkpoints'
    save_interval = 25
    
    # GPU
    gpu_ids = '0' 