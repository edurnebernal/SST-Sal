import torch
import os
import cv2
import math
from torch.utils.data import Dataset
import numpy as np
import time
from tqdm import tqdm


class RGB_and_OF(Dataset):
    def __init__(self, path_to_frames, path_to_flow_maps, path_to_saliency_maps, video_names, frames_per_data=20, split_percentage=0.2, split='train', resolution = [240, 320], skip=20, load_names=False, transform=False, inference=False):
        self.sequences = []
        self.correspondent_sal_maps = []
        self.frames_per_data = frames_per_data
        self.path_frames = path_to_frames
        self.path_sal_maps = path_to_saliency_maps
        self.resolution = resolution
        self.flow_maps = path_to_flow_maps
        self.load_names = load_names
        self.transform = transform

        # Different videos for each split
        sp = int(math.ceil(split_percentage * len(video_names)))
        if split == "validation":
            video_names = video_names[:sp]
        elif split == "train":
            video_names = video_names[sp:]
        
        for name in video_names:
            video_frames_names = os.listdir(os.path.join(self.path_frames, name))
            video_frames_names = sorted(video_frames_names, key=lambda x: int((x.split(".")[0]).split("_")[1]))
            # Frame number and saliency name must be the same (ex. frame name: 0001_0023.png, saliency map: 0023.png)

            # Skip the first frames to avoid biases due to the eye-tracking capture procedure 
            # (Observers are usually asked to look at a certain point at the beginning of each video )
            sts = skip

            # Split the videos in sequences of equal lenght
            initial_frame = frames_per_data + skip
            
            if inference:
                frames_per_data = frames_per_data - 4
                
            for end in range(initial_frame, len(video_frames_names), frames_per_data):
                # Check if exist the ground truth saliency map for all the frames in the sequence
                valid_sequence = True

                if not self.path_sal_maps is None:

                    for frame in video_frames_names[sts:end]:
                        # if not os.path.exists(os.path.join(self.path_sal_maps, frame.split("_")[0], frame.split("_")[1])) or not os.path.exists(os.path.join(self.flow_maps, frame.split("_")[0], frame.split("_")[1])):
                        if not os.path.exists(os.path.join(self.flow_maps, frame.split("_")[0], frame)):
                            print(os.path.join(self.flow_maps, frame.split("_")[0], frame.split("_")[1]))

                            valid_sequence = False
                            print("Saliency map not found for frame: " + frame)
                            break
                
                if valid_sequence: self.sequences.append(video_frames_names[sts:end])
            
                sts = end
                if inference: sts = sts - 4 # To overlap sequences while inference for smooth predictions (4 frames) 

    
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        frame_img = []
        label = []
        flow_map = []
        frame_names = []

        # Read the RGB images, optical flow, and saliency maps for each frame in the sequence
        for frame_name in self.sequences[idx]:

            # Obtain the name of the frame
            fn = os.path.splitext(os.path.basename(frame_name))[0]
            frame_names.append(fn)

            frame_path = os.path.join(self.path_frames, frame_name.split("_")[0], frame_name)
            assert os.path.exists(frame_path), 'Image frame has not been found in path: ' + frame_path
            img_frame = cv2.imread(frame_path)

            if img_frame.shape[1] != self.resolution[1] or img_frame.shape[0] != self.resolution[0]:
                img_frame = cv2.resize(img_frame, (self.resolution[1], self.resolution[0]),
                                        interpolation=cv2.INTER_AREA)
            img_frame = img_frame.astype(np.float32)
            img_frame = img_frame / 255.0

            img_frame = torch.FloatTensor(img_frame)
            img_frame = img_frame.permute(2, 0, 1)

            frame_img.append(img_frame.unsqueeze(0))  # Adding dimension: (n_frames, ch, h, w)

            if not self.path_sal_maps == None:
                sal_map_path = os.path.join(self.path_sal_maps, frame_name.split("_")[0], frame_name.split("_")[1])
                assert os.path.exists(sal_map_path), 'Saliency map has not been found in path: ' + sal_map_path
                

                saliency_img = cv2.imread(sal_map_path, cv2.IMREAD_GRAYSCALE)
                # Assert if the saliency map could not be read
                assert saliency_img is not None, 'Saliency map could not be read in path: ' + sal_map_path
                if saliency_img.shape[1] != self.resolution[1] or saliency_img.shape[0] != self.resolution[0]:  
                    saliency_img = cv2.resize(saliency_img, (self.resolution[1], self.resolution[0]),
                                                interpolation=cv2.INTER_AREA)
                saliency_img = saliency_img.astype(np.float32)
                saliency_img = (saliency_img - np.min(saliency_img)) / (np.max(saliency_img) - np.min(saliency_img))
                saliency_img = torch.FloatTensor(saliency_img).unsqueeze(0)
                label.append(saliency_img.unsqueeze(0))

            flow_map_path = os.path.join(self.flow_maps, frame_name.split("_")[0],
                                        frame_name)
            assert os.path.exists( flow_map_path), 'Flow map has not been found in path: ' + flow_map_path

            flow_img = cv2.imread(flow_map_path)
            if flow_img.shape[1] != self.resolution[1] or flow_img.shape[0] != self.resolution[0]:
                flow_img = cv2.resize(flow_img, (self.resolution[1], self.resolution[0]), interpolation=cv2.INTER_AREA)
            flow_img = flow_img.astype(np.float32)
            flow_img = flow_img / 255.0

            flow_img = torch.FloatTensor(flow_img)
            flow_img = flow_img.permute(2, 0, 1)

            flow_map.append(flow_img.unsqueeze(0))  # Adding dimension: (n_frames, ch, h, w)


        if self.load_names:
            if self.path_sal_maps is None:
                sample = [torch.cat((torch.cat(frame_img, 0), torch.cat(flow_map, 0)), dim=1), frame_names]
            else:
                sample = [torch.cat((torch.cat(frame_img, 0), torch.cat(flow_map, 0)), dim=1), torch.cat(label, 0), frame_names]

        else:
            if self.path_sal_maps is None:
                sample = [torch.cat((torch.cat(frame_img, 0), torch.cat(flow_map, 0)), dim=1)]
            else:
                sample = [torch.cat((torch.cat(frame_img, 0), torch.cat(flow_map, 0)), dim=1), torch.cat(label, 0)]

        if self.transform:
            tf = Rotate()
            return tf(sample)
        return sample

class RGB(Dataset):
    def __init__(self, path_to_frames,path_to_saliency_maps, video_names, frames_per_data=20, split_percentage=0.2, split='train', resolution = [240, 320], skip=20, load_names=False, transform=False, inference=False):
        self.sequences = []
        self.correspondent_sal_maps = []
        self.frames_per_data = frames_per_data
        self.path_frames = path_to_frames
        self.path_sal_maps = path_to_saliency_maps
        self.resolution = resolution
        self.load_names = load_names
        self.transform = transform



        # Different videos for each split
        sp = int(math.ceil(split_percentage * len(video_names)))
        if split == "validation":
            video_names = video_names[:sp]
        elif split == "train":
            video_names = video_names[sp:]
        
        for name in video_names:
            video_frames_names = os.listdir(os.path.join(self.path_frames, name))
            video_frames_names = sorted(video_frames_names, key=lambda x: int((x.split(".")[0]).split("_")[1]))

            # Frame number and saliency name must be the same (ex. frame name: 0001_0023.png, saliency map: 0023.png)

            # Skip the first frames to avoid biases due to the eye-tracking capture procedure 
            # (Observers are usually asked to look at a certain point at the beginning of each video )
            sts = skip

            # Split the videos in sequences of equal lenght
            initial_frame = frames_per_data + skip
            
            if inference:
                frames_per_data = frames_per_data - 4

            # Split the videos in sequences of equal lenght
            for end in range(initial_frame, len(video_frames_names), frames_per_data):

                # Check if exist the ground truth saliency map for all the frames in the sequence
                valid_sequence = True

                if not self.path_sal_maps is None:

                    for frame in video_frames_names[sts:end]:
                        if not os.path.exists(os.path.join(self.path_sal_maps, frame.split("_")[0], frame.split("_")[1])):
                            valid_sequence = False
                            break
                
                if valid_sequence: self.sequences.append(video_frames_names[sts:end])
                sts = end
                if inference: sts = sts - 4 # To overlap sequences while inference for smooth predictions (4 frames) 
                    

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        frame_img = []
        label = []
        frame_names = []

        # Read the RGB images and saliency maps for each frame in the sequence
        for frame_name in self.sequences[idx]:

            # Obtain the name of the frame
            fn = os.path.splitext(os.path.basename(frame_name))[0]
            frame_names.append(fn)

            frame_path = os.path.join(self.path_frames, frame_name.split("_")[0], frame_name)
            assert os.path.exists(frame_path), 'Image frame has not been found in path: ' + frame_path
            img_frame = cv2.imread(frame_path)

            if img_frame.shape[1] != self.resolution[1] or img_frame.shape[0] != self.resolution[0]:
                img_frame = cv2.resize(img_frame, (self.resolution[1], self.resolution[0]),
                                        interpolation=cv2.INTER_AREA)
            img_frame = img_frame.astype(np.float32)
            img_frame = img_frame / 255.0

            img_frame = torch.FloatTensor(img_frame)
            img_frame = img_frame.permute(2, 0, 1)

            frame_img.append(img_frame.unsqueeze(0))  # Adding dimension: (n_frames, ch, h, w)

            if not self.path_sal_maps == None:
                sal_map_path = os.path.join(self.path_sal_maps, frame_name.split("_")[0], frame_name.split("_")[1])
                assert os.path.exists(sal_map_path), 'Saliency map has not been found in path: ' + sal_map_path
                

                saliency_img = cv2.imread(sal_map_path, cv2.IMREAD_GRAYSCALE)
                if saliency_img.shape[1] != self.resolution[1] or saliency_img.shape[0] != self.resolution[0]:  
                    saliency_img = cv2.resize(saliency_img, (self.resolution[1], self.resolution[0]),
                                                interpolation=cv2.INTER_AREA)
                saliency_img = saliency_img.astype(np.float32)
                saliency_img = (saliency_img - np.min(saliency_img)) / (np.max(saliency_img) - np.min(saliency_img))
                saliency_img = torch.FloatTensor(saliency_img).unsqueeze(0)
                label.append(saliency_img.unsqueeze(0))


        if self.load_names:
            if self.path_sal_maps is None: sample = [torch.cat(frame_img, 0), frame_names]
            else: sample = [torch.cat(frame_img, 0), torch.cat(label, 0), frame_names]
        else:
            if self.path_sal_maps is None: sample = [torch.cat(frame_img, 0)]
            else: sample = [torch.cat(frame_img, 0), torch.cat(label, 0)]
            

        if self.transform:
            tf = Rotate()
            return tf(sample)
        return sample

class Rotate(object):
    """
    Rotate the 360º image with respect to the vertical axis on the sphere.
    """

    def __call__(self, sample):
        
        input = sample[0]
        sal_map = sample[1]

        t = np.random.randint(input.shape[-1])

        new_sample = sample
        new_sample[0] = torch.cat((input[:,:,:,t:], input[:,:,:,0:t]),dim=3)
        new_sample[1] = torch.cat((sal_map[:,:,:,t:], sal_map[:,:,:,0:t]),dim=3)

        return new_sample
