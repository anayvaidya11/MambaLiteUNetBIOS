from utils import *

from datetime import datetime


class setting_config:
    """
    config file for training on the BIOS esophagus OCT dataset (Dataset515_EsoOCT2D).
    Same architecture and hyperparameters as the ISIC config; only the data changes.
    Data must first be prepared with dataprepare/prepare_eso_oct.py.
    """
    network = 'MambaLiteUNet'
    model_config = {
        'num_classes': 1,
        'input_channels': 3,  # grayscale OCT replicated to 3 channels (keeps published arch, enables ISIC pretrained init)
        'c_list': [16, 32, 48, 64, 96, 128]
    }

    datasets = 'EsoOCT'
    data_path = './data/EsoOCT/'

    # set via train_eso.py --pretrained to fine-tune from the shipped ISIC weights
    pretrained_path = None

    print(f'Dataset: {datasets} Selected!!')
    print(f"Channel configuration: {model_config['c_list']} Using!!")

    criterion = BceDiceLoss()

    num_classes = 1
    input_size_h = 256
    input_size_w = 256
    input_channels = 3
    distributed = False
    local_rank = -1
    num_workers = 0
    seed = 42
    world_size = None
    rank = None
    amp = False
    batch_size = 8
    epochs = 300

    work_dir = 'results/' + network + '_' + datasets + '_' + datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss') + '/'

    print_interval = 20
    val_interval = 10
    save_interval = 100
    threshold = 0.5

    opt = 'AdamW'
    lr = 0.001
    betas = (0.9, 0.999)
    eps = 1e-8
    weight_decay = 1e-2
    amsgrad = False

    sch = 'CosineAnnealingLR'
    T_max = 50
    eta_min = 0.00001
    last_epoch = -1
