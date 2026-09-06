import torch
import numpy as np
import logging
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os
from datetime import datetime
import time
import yaml

from mediafusion_dataset import SoccerNetFrames, SoccerNetFramesTesting
from mediafusion_train import trainerAS, testSpotting, LearningRateWarmUP, MediaFusionLoss
from mediafusion_legacy_model import LegacyBaseModel
from mediafusion_base import BaseModel
from mediafusion_model import MediaFusion
#Main code to read data, train the model and make predictions
def main(args, cfg):

    logging.info('Parameters:')
    for arg in cfg:
        if type(cfg[arg]) is dict:
            logging.info(' '*2 + str(arg) + ':')
            for arg2 in cfg[arg]:
                logging.info(' '*4 + str(arg2) + ' : ' + str(cfg[arg][arg2]))
        else:
            logging.info(' '*2 + str(arg) + ' : ' + str(cfg[arg]))

    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])


    #create datasets
    if not cfg['test_only']:
        dataset_Train = SoccerNetFrames(path_labels = cfg['path_labels'], path_store = cfg['path_store'],
                                        path_baidu = cfg['path_baidu'], path_audio = cfg['path_audio'], split = cfg['train_split'], chunk_size = cfg['chunk_size'],
                                        outputrate = cfg['outputrate'], stride = cfg['chunk_size'] // 2, rC = cfg['rC'], rD = cfg['rD'],
                                        store = cfg['store'], max_games = cfg['max_games'])
        
        dataset_Valid = SoccerNetFrames(path_labels = cfg['path_labels'], path_store = cfg['path_store'],
                                        path_baidu = cfg['path_baidu'], path_audio = cfg['path_audio'], split = cfg['val_split'], chunk_size = cfg['chunk_size'],
                                        outputrate = cfg['outputrate'], stride = cfg['chunk_size'] // 2, rC = cfg['rC'], rD = cfg['rD'],
                                         max_games = cfg['max_games'], store = cfg['store'])
        
    
    #model
    model_dict = {
        'MediaFusion': MediaFusion,
        'BaseModel': BaseModel,
        'LegacyBaseModel': LegacyBaseModel,
    }
    # Enable TensorFloat-32 and cuDNN benchmark for acceleration on modern GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    model_class = model_dict.get(cfg['model']['name'], MediaFusion)
    model = model_class(chunk_size = cfg['chunk_size'], n_output = int(cfg['outputrate'] * cfg['chunk_size']), baidu = cfg['baidu'], 
                audio = cfg['audio'], model_cfg = cfg['model'])
    model = model.cuda()

    # Note: torch.compile with reduce-overhead / cudagraphs is disabled due to Triton/CUDA graph compatibility with dynamic feature mixup
    # PyTorch eager mode with native TF32, bfloat16, non_blocking and Fused AdamW runs stably and efficiently.

    # wandb.watch(model, log_freq=1000)  
    
    #logging.info(model)
    total_params = sum(p.numel()
                    for p in model.parameters() if p.requires_grad)
    logging.info("Total number of parameters: " + str(total_params))
    
    # create dataloader
    if not cfg['test_only']:
        num_workers = cfg.get('num_workers', 0)
        persistent_workers = True if num_workers > 0 else False
        
        train_loader = torch.utils.data.DataLoader(dataset_Train, 
                            batch_size=cfg['BS'], shuffle=True,
                            num_workers=num_workers, pin_memory=True,
                            prefetch_factor=2 if num_workers > 0 else None,
                            persistent_workers=persistent_workers,
                            drop_last=True)
            
        val_loader = torch.utils.data.DataLoader(dataset_Valid,
            batch_size=cfg['BS'], shuffle=False,
            num_workers=num_workers, pin_memory=True,
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=persistent_workers,
            drop_last=True)

    # training parameters
    if not cfg['test_only']:        

        criterion = MediaFusionLoss(wC = cfg['wC'], wD = cfg['wD'], focal = cfg['focal'], nw = (cfg['rD'] - cfg['rC']) * cfg['outputrate'] * 2 + 1, 
                            uncertainty = cfg['model']['uncertainty'], uncertainty_mode = cfg['model']['uncertainty_mode'], gamma = cfg['gamma'])

        #Adapt LR to batch size (base BS 8)
        cfg['LR'] = cfg['LR'] * np.sqrt(cfg['BS'] / 8)
        logging.info('Adapted LR: ' + str(cfg['LR']))
        
        #Optimizer and scheduler (try fused AdamW if supported)
        try:
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['LR'],
                                betas=(0.9, 0.999), eps=1e-08,
                                weight_decay=1e-5, amsgrad=True, fused=True)
            logging.info("Using Fused AdamW optimizer.")
        except Exception:
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['LR'],
                                betas=(0.9, 0.999), eps=1e-08,
                                weight_decay=1e-5, amsgrad=True)
        
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,             
                                                                    T_max = (cfg['max_epochs'] - cfg['warmup_iter']),
                                                                    eta_min = 0.00000001)
        scheduler = LearningRateWarmUP(optimizer = optimizer,
                                    warmup_iteration = cfg['warmup_iter'],
                                    target_lr = cfg['LR'],
                                    after_scheduler = scheduler_cosine)
            
        #Training
        trainerAS(train_loader, val_loader, model, optimizer, scheduler, criterion, patience = cfg['patience'], model_name = args.model_name,
                max_epochs = cfg['max_epochs'], chunk_size = cfg['chunk_size'], outputrate = cfg['outputrate'], path_experiments = cfg['path_experiments'])

    #Testing
    ckpt_path = None
    if getattr(args, 'checkpoint', None):
        if os.path.isfile(args.checkpoint):
            ckpt_path = args.checkpoint
        else:
            candidates = [
                os.path.join(cfg['path_experiments'], args.checkpoint),
                os.path.join(cfg['path_experiments'], "type", args.model_name, args.checkpoint),
                os.path.join(cfg['path_experiments'], args.model_name, args.checkpoint),
                os.path.join(cfg['path_experiments'], "ASmodels", args.model_name, args.checkpoint),
            ]
            for c in candidates:
                if os.path.isfile(c):
                    ckpt_path = c
                    break
    if ckpt_path is None:
        candidates = [
            os.path.join(cfg['path_experiments'], "type", args.model_name, "model.pth.tar"),
            os.path.join(cfg['path_experiments'], args.model_name, "model.pth.tar"),
            os.path.join(cfg['path_experiments'], "ASmodels", args.model_name, "model.pth.tar"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                ckpt_path = c
                break
        if ckpt_path is None:
            ckpt_path = candidates[0]

    logging.info(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, weights_only = False)
    model.load_state_dict(checkpoint['state_dict'], strict = False)

    for split in cfg['test_split']:
        dataset_Test  = SoccerNetFramesTesting(path_labels = cfg['path_labels'], path_baidu = cfg['path_baidu'], path_audio = cfg['path_audio'], 
                                        split = split, outputrate = cfg['outputrate'], chunk_size = cfg['chunk_size'], baidu = cfg['baidu'], 
                                        audio = cfg['audio'])
        
        print('Test loader')
        test_loader = torch.utils.data.DataLoader(dataset_Test, batch_size = 1, shuffle = False, num_workers = 1, pin_memory = True)

        #Test spotting
        results_l, results_t = testSpotting(test_loader, model = model, model_name = args.model_name, NMS_threshold = cfg['NMS_threshold'], 
                            outputrate = cfg['outputrate'], chunk_size = cfg['chunk_size'], path_experiments = cfg['path_experiments'],
                            path_labels = cfg['path_labels'])

        a_mAP = results_l["a_mAP"]
        a_mAP_per_class = results_l["a_mAP_per_class"]
        a_mAP_visible = results_l["a_mAP_visible"]
        a_mAP_per_class_visible = results_l["a_mAP_per_class_visible"]
        a_mAP_unshown = results_l["a_mAP_unshown"]
        a_mAP_per_class_unshown = results_l["a_mAP_per_class_unshown"]

        logging.info(f"test/loose_aMAP: {a_mAP}")

        logging.info("Best Performance at end of training (loose metric)")
        logging.info("a_mAP visibility all: " +  str(a_mAP))
        logging.info("a_mAP visibility all per class: " +  str( a_mAP_per_class))
        logging.info("a_mAP visibility visible: " +  str( a_mAP_visible))
        logging.info("a_mAP visibility visible per class: " +  str( a_mAP_per_class_visible))
        logging.info("a_mAP visibility unshown: " +  str( a_mAP_unshown))
        logging.info("a_mAP visibility unshown per class: " +  str( a_mAP_per_class_unshown))

        a_mAP = results_t["a_mAP"]
        a_mAP_per_class = results_t["a_mAP_per_class"]
        a_mAP_visible = results_t["a_mAP_visible"]
        a_mAP_per_class_visible = results_t["a_mAP_per_class_visible"]
        a_mAP_unshown = results_t["a_mAP_unshown"]
        a_mAP_per_class_unshown = results_t["a_mAP_per_class_unshown"]

        logging.info(f"test/tight_aMAP: {a_mAP}")

        logging.info("Best Performance at end of training (tight metric)")
        logging.info("a_mAP visibility all: " +  str(a_mAP))
        logging.info("a_mAP visibility all per class: " +  str( a_mAP_per_class))
        logging.info("a_mAP visibility visible: " +  str( a_mAP_visible))
        logging.info("a_mAP visibility visible per class: " +  str( a_mAP_per_class_visible))
        logging.info("a_mAP visibility unshown: " +  str( a_mAP_unshown))
        logging.info("a_mAP visibility unshown per class: " +  str( a_mAP_per_class_unshown))

        # wandb.finish()

    return

if __name__ == '__main__':


    parser = ArgumentParser(description='MediaFusion', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_name', required=False, type=str, default='MediaFusion', help='name of the model')
    parser.add_argument('--checkpoint', required=False, type=str, default=None, help='path to specific checkpoint to evaluate (e.g. model_41.pth.tar)')
    args = parser.parse_args()

    numeric_level = getattr(logging, 'INFO', None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % 'INFO')
    
    args.config_file = os.path.join('configs', args.model_name + '.yaml')
    if not os.path.exists(args.config_file):
        # Fallback to MediaFusion config if model_name config does not exist
        if os.path.exists('configs/MediaFusion.yaml'):
            args.config_file = 'configs/MediaFusion.yaml'
    with open (args.config_file, 'r') as f:
        cfg = yaml.load(f, Loader = yaml.FullLoader)
    cfg['model_name'] = args.model_name
    
    os.makedirs(os.path.join(cfg['path_experiments'], "type", args.model_name), exist_ok=True)
    log_dir = os.path.join(cfg['path_experiments'], "type", args.model_name)
    log_path = os.path.join(log_dir, datetime.now().strftime('%Y-%m-%d_%H-%M-%S.log'))
    logging.basicConfig(
        level=numeric_level,
        format=
        "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ])

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg['gpu'])

    start=time.time()
    logging.info('Starting main function')
    main(args, cfg)
    logging.info(f'Total Execution Time is {time.time()-start} seconds')