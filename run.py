import os
from threading import Thread
import sys
import time
import cv2
import numpy
import argparse
import math
from glob import glob
import numpy as np
import json
import torch

from task1_utils import match_pairs, ocr, MatchImageSizeTo

def task1(frames, imgs, search_radius=15, ocr_batch_size=6, match_batch_size=7 ):
    frame_total = len(frames)

    ######################### Frame-Image Matching ###########################
    print('start image matching')
    match_results = match_pairs(frames, imgs, match_batch_size, 'cuda')
    torch.cuda.empty_cache()

    print(match_results)
    ######################### mask frames ###########################
    print('masking')
    vid_mask = np.zeros(frame_total).astype(np.int)
    img_idx = []
    for match_res in match_results:
        img_idx.append(match_res[0])
        if match_res[0] == -1:
            continue
        idx = np.arange(search_radius*2+1) - search_radius + match_res[0]
        idx = np.clip(idx,0,frame_total-1).astype(np.int)
        vid_mask[idx] = 1
    masked_frame_idx = np.where(vid_mask==1)[0]
    frames = np.stack(frames, axis=0)[vid_mask==1]

    ######################### OCR ###########################
    print('start ocr')
    texts = []
    text_idx = []
    Iters = math.ceil(masked_frame_idx.shape[0]/ocr_batch_size)
    from tqdm import tqdm
    with tqdm(total=Iters) as pbar:
        for i in range(Iters):
            start =  i*ocr_batch_size
            if i == Iters-1:
                end = masked_frame_idx.shape[0]
            else:
                end = (i+1)*ocr_batch_size
            ocr(frames[start:end], start, masked_frame_idx, texts, text_idx)
            pbar.update(1)
    torch.cuda.empty_cache()

    print(text_idx)
    print(texts)

    answer = []
    for i, i_idx in enumerate(img_idx):
        ans = 'NONE'
        if i_idx == -1:
            answer.append(ans)
            continue
        min_dist = search_radius+1
        for j,t_idx in enumerate(text_idx):
            if abs(t_idx-i_idx) > search_radius:
                continue
            if abs(t_idx-i_idx) < min_dist:
                min_dist = abs(t_idx-i_idx)
                ans = texts[j]
        answer.append(ans)

    print(answer)

    return answer

if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', default='./samples', help='video path')
    parser.add_argument('--img_path', default='./samples', help='image path')
    parser.add_argument('--output_path', default='output.json', help='output path')
    parser.add_argument('--frame_skip', type=int, default=30, help='output path')
    args = parser.parse_args()

    # f = open(args.output_path,'w')
    final_result = {
                        "task1_answer":[{
                            "set_1": [],
                            "set_2": [],
                            "set_3": [],
                            "set_4": [],
                            "set_5": []
                        }]
                    }

    imgs=[]
    img_list = glob(os.path.join(args.img_path, "*.jpg"))
    img_list.sort()
    img_resizer = MatchImageSizeTo()
    for img_ in img_list:
        img = cv2.imread(img_, cv2.IMREAD_GRAYSCALE)
        img = img_resizer(img)
        imgs.append(img)
    
    vid_list = glob(os.path.join(args.video_path, "*.mp4"))
    vid_list.sort()
    for vid_path in vid_list:
        vid_name = vid_path.split('/')[-1].split('.')[0].split('_')
        set_num = "set_{}".format(vid_name[0][-1])
        drone_num = "drone_{}".format(vid_name[1][-1])

        frames = []
        cap = cv2.VideoCapture(vid_path)
        while (cap.isOpened()):
            ret, frame = cap.read()
            frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if(type(frame) == type(None)):
                break
            if frame_pos % args.frame_skip != 0 :
                continue
            frames.append(frame)
        cap.release()
        result = task1(frames, imgs)
        final_result["task1_answer"][0][set_num].append({drone_num:result})

    for i in range(len(img_list)):
        ans_1 = final_result["task1_answer"][0][set_num][0]['drone_1'][i]
        ans_2 = final_result["task1_answer"][0][set_num][1]['drone_2'][i]
        ans_3 = final_result["task1_answer"][0][set_num][2]['drone_3'][i]
        if ans_1 == ans_2 and ans_1 != 'NONE':
            if ans_1 != ans_3 and ans_3 != 'NONE':
                ans_3 = 'NONE'
        else:
            if ans_1 == ans_3 and ans_2 != 'NONE':
                ans_2 = 'NONE'
            elif ans_2 == ans_3 and ans_1 != 'NONE':
                ans_1 = 'NONE'
        final_result["task1_answer"][0][set_num][0]['drone_1'][i] = ans_1
        final_result["task1_answer"][0][set_num][1]['drone_2'][i] = ans_2
        final_result["task1_answer"][0][set_num][2]['drone_3'][i] = ans_3


    print(final_result)
    with open(args.output_path, 'w') as f:
        json.dump(final_result, f)

    print("TIME :", time.time()-start)