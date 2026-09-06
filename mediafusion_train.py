import logging
import os
import time
from tqdm import tqdm
import torch
import numpy as np
from sklearn.metrics import average_precision_score
from SoccerNet.Evaluation.utils import AverageMeter, INVERSE_EVENT_DICTIONARY_V2
from SoccerNet.Evaluation.ActionSpotting import evaluate
import json
import zipfile
from torch.amp import GradScaler
from torch import autocast
from eval import pred2vec, compute_mAP
import pandas as pd
from dataset import feats2clip

def trainerAS(train_loader,
            val_loader,
            model,
            optimizer,
            scheduler,
            criterion,
            patience,
            model_name,
            max_epochs=1000,
            chunk_size=32,
            outputrate=2,
            path_experiments=None):
    """
    Function to train the action spotting model (with validation early stopping)
    """

    logging.info("start training action spotting")

    best_loss = 0
    best_map = 0
    n_bad_epochs = 0
    
    for epoch in range(max_epochs):
        best_model_path = os.path.join(path_experiments, 'ASmodels', model_name, 'model.pth.tar')

        #Update scheduler
        scheduler.step(epoch+1)
        logging.info(optimizer.param_groups[0]['lr'])

        # train for one epoch
        loss_training = trainAS(train_loader, model, criterion, optimizer, epoch + 1, train=True, 
                                chunk_size=chunk_size, outputrate=outputrate)
        
        # evaluate on validation set
        loss_validation = trainAS(val_loader, model, criterion, optimizer, epoch + 1, train=False,
                                chunk_size=chunk_size, outputrate=outputrate)

        state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
        }
        os.makedirs(os.path.join(path_experiments, "ASmodels", model_name), exist_ok=True)

        # remember best loss and save checkpoint
        is_better = loss_validation >= best_loss
        best_loss = max(loss_validation, best_loss)

        # Save the best model based on loss only if validation improves
        if is_better:
            n_bad_epochs = 0
            torch.save(state, best_model_path)
        
        else:
            n_bad_epochs += 1

        #If doesn't improve reduce LR / finish training
        if n_bad_epochs == patience:
            break

    return

def trainerAS_test(train_loader,
            model,
            optimizer,
            scheduler,
            criterion,
            patience,
            model_name,
            max_epochs=1000,
            chunk_size=32,
            outputrate=2,
            path_experiments=None):
    
    """
    Function to train the action spotting model (when no validation early stopping - for using all data and evaluate on challenge)
    """

    logging.info("start training action spotting")

    best_loss = 0
    n_bad_epochs = 0

    
    for epoch in range(max_epochs):
        best_model_path = os.path.join(path_experiments, 'ASmodels', model_name, 'model.pth.tar')

        #Update scheduler
        scheduler.step(epoch+1)
        logging.info(optimizer.param_groups[0]['lr'])

        # train for one epoch
        loss_training = trainAS(train_loader, model, criterion, optimizer, epoch + 1, train=True, 
                                chunk_size=chunk_size, outputrate=outputrate)
        
        state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
        }
        os.makedirs(os.path.join(path_experiments, "ASmodels", model_name), exist_ok=True)

        # remember best loss and save checkpoint
        is_better = True
        best_loss = max(loss_training, best_loss)

        # Save the best model based on loss only if validation improves
        if is_better:
            n_bad_epochs = 0
            torch.save(state, best_model_path)
        
        else:
            n_bad_epochs += 1

        #If doesn't improve reduce LR / finish training
        if n_bad_epochs == patience:
            break

    return

#Define train for AS part
def trainAS(dataloader,
        model,
        criterion,
        optimizer,
        epoch,
        train=False,
        chunk_size=32,
        outputrate=2):
    
    """
    Function to train 1 epoch of the action spotting model
    """

    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = AverageMeter()
    lossesC = AverageMeter()
    lossesD = AverageMeter()

    if train:
        model.train()
    else:
        model.eval()
        list_preds = []
        list_labels = []

    end = time.time()

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = GradScaler(enabled=not use_bf16) # for mixed precision when float16
    
    #Iterate dataloader
    with tqdm(enumerate(dataloader), total=len(dataloader), ncols=160) as t:
        for i, data in t:
                    
            # measure data loading time
            data_time.update(time.time() - end)
            labels = data['labels'].cuda(non_blocking=True)
            labelsD = data['labels_displ'].cuda(non_blocking=True)

            featB = data['featB'].cuda(non_blocking=True)
            if model.audio:
                featA = data['featA'].cuda(non_blocking=True)
            else:
                featA = None

            #make predictions
            with autocast(device_type='cuda', dtype=amp_dtype):
                if train:
                    output = model(featsB = featB, featsA = featA, labels = labels, labelsD = labelsD, inference = False)
                    lossC, lossD = criterion(output['labels'], output['preds'], output['labelsD'], output['predsD'])

                else:
                    with torch.no_grad():
                        output = model(featsB = featB, featsA = featA, inference = True)
                        lossC, lossD = criterion(labels, output['preds'], labelsD, output['predsD'])

                    if (model.model_cfg['uncertainty']):
                        output['predsD'] = output['predsD'][:, :, :, 0]
                    
                    #To compute MAP (labels + predictions) in validation
                    labs = pred2vec([labels, labelsD], chunk_size = chunk_size, outputrate = outputrate, threshold = 0.2, target = True, window = 4)
                    y = [list_labels.append(lab) for lab in labs]
                    y = [list_preds.append(pred) for pred in pred2vec([output['preds'], output['predsD']], chunk_size = chunk_size, outputrate = outputrate, threshold = 0.2, NMS = True, window = 4)]
                    
                loss = lossC + lossD

            if train:
                train_step = (epoch-1) * len(dataloader) + i
                if train_step % 50 == 0:
                    print(f"train/step: {train_step}, train/ASloss: {loss.item()}, train/ASlossC: {lossC.item()}, train/ASlossD: {lossD.item()}")
            else:
                val_step = (epoch-1) * len(dataloader) + i
                if val_step % 50 == 0:
                    print(f"val/step: {val_step}, val/ASloss: {loss.item()}, val/ASlossC: {lossC.item()}, val/ASlossD: {lossD.item()}")

            losses.update(loss.item(), labels.size(0))
            lossesC.update(lossC.item(), labels.size(0))
            lossesD.update(lossD.item(), labels.size(0))
                
            #compute gradient and backpropagate
            if train:
                if use_bf16:
                    loss.backward()
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                optimizer.zero_grad(set_to_none=True)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if train:
                desc = f'Train {epoch}: '
            else:
                desc = f'Evaluate {epoch}: '
            desc += f'Time {batch_time.avg:.3f}s '
            desc += f'(it:{batch_time.val:.3f}s) '
            desc += f'Data:{data_time.avg:.3f}s '
            desc += f'(it:{data_time.val:.3f}s) '
            desc += f'Loss:{losses.avg:.3e} '
            desc += f'LossC:{lossesC.avg:.3e} '
            desc += f'LossD:{lossesD.avg:.3e} '

            t.set_description(desc)

    if train:
        logging.info(f"per_epoch/train/epoch: {epoch}, per_epoch/train/ASloss: {losses.avg}, per_epoch/train/ASlossC: {lossesC.avg}, per_epoch/train/ASlossD: {lossesD.avg}, per_epoch/train/LR: {optimizer.param_groups[0]['lr']}")        
        return losses.avg
    
    else:
        #Compute mAP
        amap, amap_class = compute_mAP(list_preds, list_labels, metric = "tight")
        dataframe = pd.DataFrame(amap_class).T
        dataframe.columns = ["penalty", "kick-off", "goal", "substitution", "offside", "sh. on targ.", "sh. off targ.", "clearance", "ball oop", "throw in", "foul", "ind. fk", "dir. fk", "corner", "yc", "rc", "2nd yc"]
        # Lưu CSV đơn giản
        try:
            csv_path = f"validation_{model.model_cfg['name']}_{epoch}_amap_{amap:.4f}.csv"
            dataframe.to_csv(csv_path, index=False)
            logging.info(f"✅ Saved: {csv_path}")
        except:
            pass
        logging.info('aMAP:' + str(amap))
        return amap


# def testSpotting(dataloader, model, model_name, overwrite=True, NMS_window = 8, NMS_threshold=0.5, outputrate=2, 
#                 chunk_size=32, stride = 8, postprocessing = 'SNMS', path_experiments = None):

#     """
#     Function for inference of the action spotting model (and evaluation)
#     """

#     #Take split from dataloader
#     split = dataloader.dataset.split

#     #Output results folder
#     output_results = os.path.join(path_experiments, "ASmodels", model_name, 'post_' + str(postprocessing) + 'window_' + str(NMS_window), f"results_spotting_{split}.zip")
#     output_folder = f"outputs_{split}"

#     if not os.path.exists(output_results) or overwrite:
#         batch_time = AverageMeter()
#         data_time = AverageMeter()

#         spotting_grountruth = list()
#         spotting_grountruth_visibility = list()
#         spotting_predictions = list()

#         model.eval()

#         end = time.time()
        
#         #Iterate over games (dataloader)
#         with tqdm(enumerate(dataloader), total=len(dataloader), ncols=120) as t:

#             for i, (game_ID, data) in t:

#                 data_time.update(time.time() - end)
    
#                 game_ID = game_ID[0]

#                 featB_half1 = data['featB1'].reshape(-1, data['featB1'].shape[-1])
#                 sec1 = featB_half1.shape[0]
#                 featB_half1 = feats2clip(featB_half1, stride = stride, clip_length = chunk_size)

#                 featB_half2 = data['featB2'].reshape(-1, data['featB2'].shape[-1])
#                 sec2 = featB_half2.shape[0]
#                 featB_half2 = feats2clip(featB_half2, stride = stride, clip_length = chunk_size)
#                 lenB1 = len(featB_half1)
#                 lenB2 = len(featB_half2)

#                 if model.audio:
#                     featA_half1 = data['featA1'].reshape(-1, data['featA1'].shape[-1])
#                     featA_half2 = data['featA2'].reshape(-1, data['featA2'].shape[-1])
#                     featA_half1 = feats2clip(featA_half1.T, stride = stride * 100, clip_length = chunk_size * 100)
#                     featA_half2 = feats2clip(featA_half2.T, stride = stride * 100, clip_length = chunk_size * 100)

#                 #batch size for testing
#                 BS = 4

#                 json_data = dict()
#                 json_data["UrlLocal"] = game_ID
#                 json_data["predictions"] = list()

#                 #HALF 1 PREDICTIONS
#                 featV = []
                
#                 #Initialize half1 preds
#                 timestamp_long_half_1 = np.zeros((sec1 * outputrate, 17))
#                 q = 0
#                 for b in tqdm(range(lenB1)):
#                     if (b % BS == 0) | (b == lenB1-1):
#                         if b != 0:
#                             featB = featB_half1[b-q:b].clone().cuda()
#                             if model.audio:
#                                 featA = featA_half1[b-q:b].clone().cuda()
#                             else:
#                                 featA = None

#                             with autocast(device_type='cuda', dtype=torch.float16):
#                                 with torch.no_grad():
#                                     output = model(featsB = featB, featsA = featA, inference = True)
#                             predC = output['preds'].cpu().detach().numpy()
#                             if model.model_cfg['uncertainty']:
#                                 predD = output['predsD'][:, :, :, 0].cpu().detach().numpy()
#                             else:
#                                 predD = output['predsD'].cpu().detach().numpy()
#                             batch, nf, nc = predC.shape
#                             for l in range(batch):
#                                 initial_pos = (b - len(predC) + l) * stride * outputrate
#                                 for j in range(nf):
#                                     for k in range(nc-1):
#                                         prob = predC[l, j, k+1]
#                                         if prob > NMS_threshold:
#                                             rel_position = j - predD[l, j, k+1]
#                                             position = min(len(timestamp_long_half_1)-1, max(0, int((initial_pos + rel_position).round())))
#                                             timestamp_long_half_1[position, k] = max(timestamp_long_half_1[position, k], prob)
                            
#                             featV = []
#                             q = 0

#                     q += 1

#                 #HALF 2 PREDICTIONS
#                 featV = []
                
#                 #Initialize half2 preds
#                 timestamp_long_half_2 = np.zeros((sec2 * outputrate, 17))
#                 q = 0
#                 for b in tqdm(range(lenB2)):
#                     if (b % BS == 0) | (b == lenB2-1):
#                         if b != 0:
#                             featB = featB_half2[b-q:b].clone().cuda()
#                             if model.audio:
#                                 featA = featA_half2[b-q:b].clone().cuda()
#                             else:
#                                 featA = None

#                             with autocast(device_type='cuda', dtype=torch.float16):
#                                 with torch.no_grad():
#                                     output = model(featsB = featB, featsA = featA, inference = True)
#                             predC = output['preds'].cpu().detach().numpy()
#                             if model.model_cfg['uncertainty']:
#                                 predD = output['predsD'][:, :, :, 0].cpu().detach().numpy()
#                             else:
#                                 predD = output['predsD'].cpu().detach().numpy()
#                             batch, nf, nc = predC.shape
#                             for l in range(batch):
#                                 initial_pos = (b - len(predC) + l) * stride * outputrate
#                                 for j in range(nf):
#                                     for k in range(nc-1):
#                                         prob = predC[l, j, k+1]
#                                         if prob > NMS_threshold:
#                                             rel_position = j - predD[l, j, k+1]
#                                             position = min(len(timestamp_long_half_2)-1, max(0, int((initial_pos + rel_position).round())))
#                                             timestamp_long_half_2[position, k] = max(timestamp_long_half_2[position, k], prob)
                            
#                             featV = []
#                             q = 0

#                     q += 1
                    
#                 spotting_predictions.append(timestamp_long_half_1)
#                 spotting_predictions.append(timestamp_long_half_2)

#                 batch_time.update(time.time() - end)
#                 end = time.time()
        
#                 desc = f'Test (spot.): '
#                 desc += f'Time {batch_time.avg:.3f}s '
#                 desc += f'(it:{batch_time.val:.3f}s) '
#                 desc += f'Data:{data_time.avg:.3f}s '
#                 desc += f'(it:{data_time.val:.3f}s) '
#                 t.set_description(desc)
                
#                 if postprocessing == 'NMS':
#                     get_spot = get_spot_from_NMS
#                     nms_window = [NMS_window] * 17

#                 elif postprocessing == 'SNMS':
#                     get_spot = get_spot_from_SNMS
#                     nms_window = [5, 7, 9, 12, 10, 14, 14, 5, 8, 8, 8, 8, 13, 5, 6, 6, 6]
        
#                 json_data = dict()
#                 json_data["UrlLocal"] = game_ID
#                 json_data["predictions"] = list()
#                 #nms_window = [NMS_window] * 17
#                 for half, timestamp in enumerate([timestamp_long_half_1, timestamp_long_half_2]):
                            
#                     for l in range(dataloader.dataset.num_classes):
#                         spots = get_spot(
#                             timestamp[:, l], window=nms_window[l]*outputrate, thresh=NMS_threshold)


#                         for spot in spots:
#                             # print("spot", int(spot[0]), spot[1], spot)
#                             frame_index = int(spot[0])
#                             confidence = spot[1]
#                             # confidence = predictions_half_1[frame_index, l]
        
#                             seconds = int((frame_index//outputrate)%60)
#                             minutes = int((frame_index//outputrate)//60)
        
#                             prediction_data = dict()
#                             prediction_data["gameTime"] = str(half+1) + " - " + str(minutes) + ":" + str(seconds)

#                             prediction_data["label"] = INVERSE_EVENT_DICTIONARY_V2[l]

#                             prediction_data["position"] = str(int((frame_index/outputrate)*1000))
#                             prediction_data["half"] = str(half+1)
#                             prediction_data["confidence"] = str(confidence)
#                             json_data["predictions"].append(prediction_data)
                        
#                 os.makedirs(os.path.join(path_experiments, "ASmodels", model_name, 'post_' + str(postprocessing) + 'window_' + str(NMS_window), output_folder, game_ID), exist_ok=True)
#                 with open(os.path.join(path_experiments, "ASmodels", model_name, 'post_' + str(postprocessing) + 'window_' + str(NMS_window), output_folder, game_ID, "results_spotting.json"), 'w') as output_file:
#                     json.dump(json_data, output_file, indent=4)

        

#         def zipResults(zip_path, target_dir, filename="results_spotting.json"):            
#             zipobj = zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED)
#             rootlen = len(target_dir) + 1
#             for base, dirs, files in os.walk(target_dir):
#                 for file in files:
#                     if file == filename:
#                         fn = os.path.join(base, file)
#                         zipobj.write(fn, fn[rootlen:])


#         # zip folder
#         zipResults(zip_path=output_results,
#                 target_dir = os.path.join(path_experiments, "ASmodels", model_name, 'post_' + str(postprocessing) + 'window_' + str(NMS_window), output_folder),
#                 filename="results_spotting.json")

#     if split == "challenge": 
#         print("Visit eval.ai to evalaute performances on Challenge set")
#         return None
#     labels_path = "/home/storage/thiendc/soccernet/"
#     results_l = evaluate(SoccerNet_path=labels_path, 
#                 Predictions_path=output_results,
#                 split="test",
#                 prediction_file="results_spotting.json", 
#                 version=2,
#                 metric="loose")   
    
#     results_t = evaluate(SoccerNet_path=labels_path, 
#                 Predictions_path=output_results,
#                 split="test",
#                 prediction_file="results_spotting.json", 
#                 version=2,
#                 metric="tight")  

#     return results_l, results_t

def testSpotting(dataloader, model, model_name, overwrite=True, NMS_window=8, NMS_threshold=0.001, outputrate=2,
                 chunk_size=32, stride=8, postprocessing='SNMS', path_experiments=None, path_labels=None):
    """
    Function for inference of the action spotting model (and evaluation)
    using a memory-efficient sliding window approach with robust data handling.
    """
    # 1. THIẾT LẬP CÁC ĐƯỜNG DẪN VÀ THƯ MỤC
    # ------------------------------------------------
    split = dataloader.dataset.split
    output_folder_name = f'outputs_{split}'
    postprocessing_folder_name = f'post_{postprocessing}_window_{NMS_window}'
    
    base_results_path = os.path.join(path_experiments, model_name, postprocessing_folder_name)
    output_results_zip = os.path.join(base_results_path, f"results_spotting_{split}.zip")
    output_json_folder = os.path.join(base_results_path, output_folder_name)

    # Chỉ chạy nếu file zip kết quả chưa tồn tại hoặc yêu cầu ghi đè
    if not os.path.exists(output_results_zip) or overwrite:
        batch_time = AverageMeter()
        data_time = AverageMeter()
        model.eval()
        end = time.time()

        # 2. VÒNG LẶP CHÍNH QUA TỪNG TRẬN ĐẤU
        # ------------------------------------------------
        with tqdm(enumerate(dataloader), total=len(dataloader), ncols=120) as t:
            for i, (game_ID, data) in t:
                # ...
                game_ID = game_ID[0]

                # --- BƯỚC 1: Tải và tạo tất cả các clip (giữ nguyên logic cũ) ---
                featB_half1_full = data['featB1'].reshape(-1, data['featB1'].shape[-1])
                sec1 = featB_half1_full.shape[0]
                # Chuyển đổi sang float16 ngay trên CPU để tiết kiệm bộ nhớ khi tạo clip
                featB_clips1 = feats2clip(featB_half1_full, stride=stride, clip_length=chunk_size) #.to(dtype=torch.float16)

                featB_half2_full = data['featB2'].reshape(-1, data['featB2'].shape[-1])
                sec2 = featB_half2_full.shape[0]
                featB_clips2 = feats2clip(featB_half2_full, stride=stride, clip_length=chunk_size)
                
                featA_clips1, featA_clips2 = None, None
                if model.audio:
                    AUDIO_DIM = 128
                    min_audio_len = chunk_size * 100

                    # Xử lý audio1 an toàn
                    audio1 = data.get('featA1')
                    if audio1 is not None and audio1.numel() > 0:
                        if audio1.ndim == 3: audio1 = audio1.squeeze(0)
                        if audio1.ndim == 2:
                            if audio1.shape[1] == AUDIO_DIM and audio1.shape[0] >= min_audio_len:
                                featA_clips1 = feats2clip(audio1, stride=stride * 100, clip_length=min_audio_len)
                            elif audio1.shape[0] == AUDIO_DIM and audio1.shape[1] >= min_audio_len:
                                featA_clips1 = feats2clip(audio1.T, stride=stride * 100, clip_length=min_audio_len)

                    # Xử lý audio2 an toàn
                    audio2 = data.get('featA2')
                    if audio2 is not None and audio2.numel() > 0:
                        if audio2.ndim == 3: audio2 = audio2.squeeze(0)
                        if audio2.ndim == 2:
                            if audio2.shape[1] == AUDIO_DIM and audio2.shape[0] >= min_audio_len:
                                featA_clips2 = feats2clip(audio2, stride=stride * 100, clip_length=min_audio_len)
                            elif audio2.shape[0] == AUDIO_DIM and audio2.shape[1] >= min_audio_len:
                                featA_clips2 = feats2clip(audio2.T, stride=stride * 100, clip_length=min_audio_len)

                # --- BƯỚC 2: Thiết lập vòng lặp suy luận với batch size nhỏ ---
                # Giảm BS xuống giá trị rất nhỏ để thử
                INFERENCE_BS = 8
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                all_halves_data = [
                    (featB_clips1, featA_clips1, sec1),
                    (featB_clips2, featA_clips2, sec2)
                ]
                timestamps_per_half = []

                for featB_clips, featA_clips, num_seconds in all_halves_data:
                    timestamp_long = np.zeros((num_seconds * outputrate, 17))
                    
                    if featB_clips is None or len(featB_clips) == 0:
                        timestamps_per_half.append(timestamp_long)
                        continue

                    num_video_clips = len(featB_clips)

                    # Ensure audio clips match video clips length
                    if model.audio:
                        if featA_clips is None or len(featA_clips) == 0:
                            featA_clips = torch.zeros((num_video_clips, chunk_size * 100, 128), dtype=torch.float16)
                        elif len(featA_clips) < num_video_clips:
                            pad_len = num_video_clips - len(featA_clips)
                            featA_clips = torch.cat([featA_clips, featA_clips[-1:].repeat(pad_len, 1, 1)], dim=0)
                        elif len(featA_clips) > num_video_clips:
                            featA_clips = featA_clips[:num_video_clips]

                    # Vòng lặp suy luận qua các batch
                    for b_start in range(0, num_video_clips, INFERENCE_BS):
                        b_end = min(b_start + INFERENCE_BS, num_video_clips)

                        # Lấy batch hiện tại và đưa lên GPU
                        batch_featB = featB_clips[b_start:b_end].cuda()
                        batch_featA = featA_clips[b_start:b_end].cuda() if (model.audio and featA_clips is not None) else None

                        # Suy luận với AMP
                        with autocast(device_type='cuda', dtype=torch.float16):
                            with torch.no_grad():
                                output = model(featsB=batch_featB, featsA=batch_featA, inference=True)

                        predC = output['preds'].float().cpu().numpy()
                        predD = output['predsD'][:, :, :, 0].float().cpu().numpy() if model.model_cfg['uncertainty'] else output['predsD'].float().cpu().numpy()
                        
                        # Tích lũy kết quả (logic cũ được điều chỉnh)
                        for i, clip_idx in enumerate(range(b_start, b_end)):
                            initial_pos = clip_idx * stride * outputrate
                            nf, nc = predC.shape[1], predC.shape[2]
                            for j in range(nf):
                                for k in range(nc - 1):
                                    prob = predC[i, j, k + 1]
                                    if prob > NMS_threshold:
                                        rel_position = j - predD[i, j, k + 1]
                                        position = min(len(timestamp_long) - 1, max(0, int(round(initial_pos + rel_position))))
                                        timestamp_long[position, k] = max(timestamp_long[position, k], prob)

                        # Giải phóng bộ nhớ "hung hăng"
                        del output, predC, predD, batch_featB, batch_featA
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    timestamps_per_half.append(timestamp_long)

                # Cập nhật thời gian và thanh tiến trình
                batch_time.update(time.time() - end)
                end = time.time()
                desc = f'Test (spot.): Game {i+1}/{len(dataloader)} | Time {batch_time.avg:.3f}s'
                t.set_description(desc)

                # 4. POST-PROCESSING VÀ LƯU KẾT QUẢ JSON
                # ------------------------------------------------
                if postprocessing == 'NMS':
                    get_spot = get_spot_from_NMS
                    nms_window_list = [NMS_window] * 17
                elif postprocessing == 'SNMS':
                    get_spot = get_spot_from_SNMS
                    nms_window_list = [5, 7, 9, 12, 10, 14, 14, 5, 8, 8, 8, 8, 13, 5, 6, 6, 6]
                    # nms_window_list = [9, 9, 9, 12, 14, 14, 14, 5, 8, 8, 8, 10, 13, 5, 6, 6, 6]
                json_data = {"UrlLocal": game_ID, "predictions": []}
                for half, timestamp in enumerate(timestamps_per_half):
                    for l in range(dataloader.dataset.num_classes):
                        spots = get_spot(timestamp[:, l], window=nms_window_list[l] * outputrate, thresh=NMS_threshold)
                        for spot in spots:
                            frame_index = int(spot[0])
                            confidence = spot[1]
                            seconds = int((frame_index / outputrate) % 60)
                            minutes = int((frame_index / outputrate) // 60)
                            
                            prediction_data = {
                                "gameTime": f"{half + 1} - {minutes}:{seconds:02d}",
                                "label": INVERSE_EVENT_DICTIONARY_V2[l],
                                "position": str(int((frame_index / outputrate) * 1000)),
                                "half": str(half + 1),
                                "confidence": str(confidence)
                            }
                            json_data["predictions"].append(prediction_data)
                json_data["predictions"] = sorted(
                    json_data["predictions"],
                    key=lambda x: (
                        int(x["half"]),
                        int(x["position"])  # vì position bạn lưu là mili giây
                    )
                )
                game_json_path = os.path.join(output_json_folder, game_ID)
                os.makedirs(game_json_path, exist_ok=True)
                with open(os.path.join(game_json_path, "results_spotting.json"), 'w') as output_file:
                    json.dump(json_data, output_file, indent=4)

        # 5. NÉN KẾT QUẢ THÀNH FILE ZIP
        # ------------------------------------------------
        def zipResults(zip_path, target_dir, filename="results_spotting.json"):
            zipobj = zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED)
            rootlen = len(target_dir) + 1
            for base, dirs, files in os.walk(target_dir):
                for file in files:
                    if file == filename:
                        fn = os.path.join(base, file)
                        zipobj.write(fn, fn[rootlen:])
        
        logging.info(f"Zipping results to {output_results_zip}")
        zipResults(zip_path=output_results_zip, target_dir=output_json_folder)

    # 6. ĐÁNH GIÁ KẾT QUẢ (NẾU KHÔNG PHẢI CHALLENGE SPLIT)
    # ------------------------------------------------
    if split == "challenge":
        print("Visit eval.ai to evaluate performances on Challenge set")
        return None, None
    
    eval_split = split[0] if isinstance(split, list) else split
    labels_path = path_labels or getattr(dataloader.dataset, 'path_labels', None) or getattr(dataloader.dataset, 'path', None) or "downloads/dataset/"

    logging.info("Evaluating loose metric...")
    results_l = evaluate(SoccerNet_path=labels_path, Predictions_path=output_results_zip,
                         split=eval_split, prediction_file="results_spotting.json", version=2, metric="loose")
    
    logging.info("Evaluating tight metric...")
    results_t = evaluate(SoccerNet_path=labels_path, Predictions_path=output_results_zip,
                         split=eval_split, prediction_file="results_spotting.json", version=2, metric="tight")
    
    return results_l, results_t

def get_spot_from_NMS(Input, window, thresh=0.0, min_window=0):
    """
    Non-Maximum Suppression
    """
    detections_tmp = np.copy(Input)
    # res = np.empty(np.size(Input), dtype=bool)
    indexes = []
    MaxValues = []
    while(np.max(detections_tmp) >= thresh):
        
        # Get the max remaining index and value
        max_value = np.max(detections_tmp)
        max_index = np.argmax(detections_tmp)
                        
        # detections_NMS[max_index,i] = max_value
        
        nms_from = int(np.maximum(-(window/2)+max_index,0))
        nms_to = int(np.minimum(max_index+int(window/2), len(detections_tmp)))
                            
        if (detections_tmp[nms_from:nms_to] >= thresh).sum() > min_window:
            MaxValues.append(max_value)
            indexes.append(max_index)
        detections_tmp[nms_from:nms_to] = -1
        
    return np.transpose([indexes, MaxValues])

def get_spot_from_SNMS(Input, window, thresh=0.0, decay='pow2'):
    """
    Soft Non-Maximum Suppression.
    decay options: 'linear', 'sqrt', 'pow2', 'gaussian'
    'gaussian': suppression tập trung quanh đỉnh, giúp các đỉnh cách xa >1s sống sót
    """
    detections_tmp = np.copy(Input)

    indexes = []
    MaxValues = []
    while(np.max(detections_tmp) >= thresh):

        # Get the max remaining index and value
        max_value = np.max(detections_tmp)
        max_index = np.argmax(detections_tmp)

        nms_from = int(np.maximum(-(window/2)+max_index, 0))
        nms_to = int(np.minimum(max_index+int(window/2), len(detections_tmp)-1)) + 1

        MaxValues.append(max_value)
        indexes.append(max_index)

        offsets = np.arange(nms_from - max_index, nms_to - max_index, dtype=np.float64)

        if decay == 'linear':
            weight = np.abs(offsets) / (window / 2)
        elif decay == 'sqrt':
            weight = np.sqrt(np.abs(offsets)) / np.sqrt(window / 2)
        elif decay == 'pow2':
            weight = np.power(np.abs(offsets), 2) / np.power(window / 2, 2)
        elif decay == 'gaussian':
            # Suppression Gaussian: đỉnh chính = 0 (bị xóa hoàn toàn),
            # vùng xa giảm nhẹ hơn → cho phép đỉnh cạnh nhau sống sót
            sigma = max(window / 4.0, 1.0)
            weight = 1.0 - np.exp(-(offsets ** 2) / (2 * sigma ** 2))
            weight[offsets == 0] = 0.0
        else:
            # fallback: pow2
            weight = np.power(np.abs(offsets), 2) / np.power(window / 2, 2)

        detections_tmp[nms_from:nms_to] = detections_tmp[nms_from:nms_to] * weight
        detections_tmp[nms_from:nms_to][detections_tmp[nms_from:nms_to] < thresh] = -1

    return np.transpose([indexes, MaxValues])

class LearningRateWarmUP(object):
    """
    Class to implement Learning rate warmup
    """
    def __init__(self, optimizer, warmup_iteration, target_lr, after_scheduler=None):
        self.optimizer = optimizer
        self.warmup_iteration = warmup_iteration
        self.target_lr = target_lr
        self.after_scheduler = after_scheduler
        self.step(1)

    def warmup_learning_rate(self, cur_iteration):
        warmup_lr = self.target_lr*float(cur_iteration)/float(self.warmup_iteration)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = warmup_lr

    def step(self, cur_iteration):
        if cur_iteration <= self.warmup_iteration:
            self.warmup_learning_rate(cur_iteration)
        else:
            self.after_scheduler.step(cur_iteration-self.warmup_iteration)
    
    def load_state_dict(self, state_dict):
        self.after_scheduler.load_state_dict(state_dict)