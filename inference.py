import os
import numpy as np
import torch
import config
from DataLoader360Video import RGB_and_OF, RGB
from torch.utils.data import DataLoader
import cv2
import tqdm
from utils import frames_extraction

from utils import save_video 
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def eval(test_data, model, device, result_imp_path):

    model.to(device)
    model.eval()

    with torch.no_grad():

        for x, names in tqdm.tqdm(test_data):

            pred = model(x.to(device))

            batch_size, Nframes, _,_ = pred[:, :, 0, :, :].shape
            
            for bs in range(batch_size):
                for iFrame in range(4,Nframes):
     
                    folder = os.path.join(result_imp_path, names[iFrame][bs].split('_')[0])
                    if not os.path.exists(folder):
                        os.makedirs(folder)

                    sal = pred[bs, iFrame, 0, :, :].cpu()
                    sal = np.array((sal - torch.min(sal)) / (torch.max(sal) - torch.min(sal)))
                    cv2.imwrite(os.path.join(folder, names[iFrame][bs] + '.png'), (sal * 255).astype(np.uint8))


if __name__ == "__main__":

    # Extract video frames if hasn't been done yet
    if not os.path.exists(os.path.join(config.videos_folder, 'frames')):
        frames_extraction(config.videos_folder)

    # Obtain video names from the new folder 'frames'
    inference_frames_folder = os.path.join(config.videos_folder, 'frames')
    video_test_names = os.listdir(inference_frames_folder)

    # Select the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device") 

    # Load the model
    model =  torch.load(config.inference_model, map_location=device)

    # Load the data. Use the appropiate data loader depending on the expected input data
    if config.of_available:
        test_video360_dataset = RGB_and_OF(inference_frames_folder, config.optical_flow_dir, None, video_test_names, config.sequence_length, split='test', load_names=True)
    else:
        test_video360_dataset = RGB(inference_frames_folder, None, video_test_names, config.sequence_length, split='test', load_names=True)

    test_data = DataLoader(test_video360_dataset, batch_size=config.batch_size, shuffle=False)

    eval(test_data, model, device, config.results_dir)

    # Save video with the results

    for video_name in video_test_names:
        save_video(os.path.join(inference_frames_folder, video_name), 
                os.path.join(config.results_dir, video_name),
                None,
                'SST-Sal_pred_' + video_name +'.avi')
