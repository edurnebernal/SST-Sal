import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pylab import *
from tqdm import tqdm
import os
import cv2
import config

def read_txt_file(path_to_file):
    """
    The names of the videos to be used for training, they must be in a single line separated
    with ','.
    :param path_to_file: where the file is saved (ex. 'data/file.txt')
    :return: list of strings with the names
    """

    with open(path_to_file) as f:
        for line in f:
            names = line.rsplit('\n')[0].split(',')
    return names

def frames_extraction(path_to_videos):
    """
    Extracts the frames from the videos and save them in the same folder as the original videos.
    :param path_to_videos: path to the videos
    :return:
    """
    samples_per_second = 8
    destination_folder = os.path.join(config.videos_folder, 'frames')
    videos_folder = config.videos_folder

    video_names = os.listdir(videos_folder)
    with tqdm(range(len(video_names)), ascii=True) as pbar:
        for v_n, video_name in enumerate(video_names):
            
            video = cv2.VideoCapture(os.path.join(videos_folder, video_name))
            fps = video.get(cv2.CAP_PROP_FPS)
            step = round(fps / samples_per_second)

            new_video_folder = os.path.join(destination_folder, str(v_n).zfill(3))

            if not os.path.exists(new_video_folder):
                os.makedirs(new_video_folder)

            success, frame = video.read()
            frame_id = 0
            frame_name = str(v_n).zfill(3) + '_' + str(frame_id).zfill(4) + '.png'
            frame = cv2.resize(frame, (2048, 1080))
            cv2.imwrite(os.path.join(new_video_folder, frame_name), frame)
            frame_id += 1

            while success:
                success, frame = video.read()
                if frame_id % step == 0 and success:
                    frame_name = str(v_n).zfill(3) + '_' + str(frame_id).zfill(4) + '.png'
                    cv2.imwrite(os.path.join(new_video_folder, frame_name), frame)
                frame_id += 1
            pbar.update(1)

def blend(sample_sal_map, sample_frame):

    colormap = plt.get_cmap('inferno')
    heatmap = (colormap(sample_sal_map)
               * 2 ** 8).astype(np.uint8)[:, :, :3]

    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
    dst = cv2.addWeighted(np.array(sample_frame).astype(
        np.uint8), 0.5, heatmap, 0.5, 0.0)
    return dst

def save_video(frames_dir, pred_dir, gt_dir, output_vid_name = 'SST-Sal_pred.avi'):
    """
    Saves the video with the predicted and ground truth saliency maps.
    :param frames: list of frames 
    :param pred: list of predicted saliency maps
    :param gt: list of ground truth saliency maps
    """

    out_pred = cv2.VideoWriter(os.path.join( config.results_dir, output_vid_name), cv2.VideoWriter_fourcc(*'DIVX'), 4, (1024, 540))
    if not gt_dir is None:
        out_gt = cv2.VideoWriter(os.path.join( config.results_dir, 'gt.avi'), cv2.VideoWriter_fourcc(*'DIVX'), 4, (1024, 540))

    video_frames_names = os.listdir(pred_dir)
    video_frames_names = sorted(video_frames_names, key=lambda x: int((x.split(".")[0]).split("_")[1]))

    for iFrame in tqdm(video_frames_names):
        
        img_frame = cv2.imread(os.path.join(frames_dir, iFrame))
        img_frame = cv2.resize(img_frame, (1024, 540), interpolation=cv2.INTER_AREA)

        if not gt_dir is None:
            gt_salmap = cv2.imread(os.path.join(gt_dir, iFrame.split('_')[1]), cv2.IMREAD_GRAYSCALE)
            gt_salmap = cv2.resize(gt_salmap, (1024, 540), interpolation=cv2.INTER_AREA)


        pred_salmap = cv2.imread(os.path.join(pred_dir, iFrame), cv2.IMREAD_GRAYSCALE)
        pred_salmap = cv2.resize(pred_salmap, (1024, 540), interpolation=cv2.INTER_AREA)


        pred_blend = blend(pred_salmap, img_frame)
        out_pred.write(pred_blend)
        if not gt_dir is None:
            gt_blend = blend(gt_salmap, img_frame)
            out_gt.write(gt_blend)
    out_pred.release()
    if not gt_dir is None:
        out_gt.release()
