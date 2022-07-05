# SST-Sal: A spherical spatio-temporal approach for saliency prediction in 360º videos

Code and models for *“SST-Sal: A spherical spatio-temporal approach for saliency prediction in 360º videos”* ([PDF](https://graphics.unizar.es/papers/2022_Bernal_SSTSal.pdf))

Edurne Bernal, Daniel Martín, Diego Gutierrez, and Belen Masia.
**Computers & Graphics**

## Abstract
Virtual reality (VR) has the potential to change the way people consume content, and has been predicted to become the next big computing paradigm. However, much remains unknown about the grammar and visual language of this new medium, and understanding and predicting how humans behave in virtual environments remains an open problem. In this work, we propose a novel saliency prediction model which exploits the joint potential of spherical convolutions and recurrent neural networks to extract and model the inherent spatio-temporal features from 360º videos. We employ Convolutional Long Short-Term Memory cells (ConvLSTMs) to account for temporal information at the time of feature extraction rather than to post-process spatial features as in previous works. To facilitate spatio-temporal learning, we provide the network with an estimation of the optical flow between 360º frames, since motion is known to be a highly salient feature in dynamic content. Our model is trained with a novel spherical Kullback–Leibler Divergence (KLDiv) loss function specifically tailored for saliency prediction in 360º content. Our approach outperforms previous state-of-the-art works, being able to mimic human visual attention when exploring dynamic 360º videos.

Visit our [website](https://graphics.unizar.es/projects/SST-Sal_2022/) for more information and supplementary material.

## Requirements
The code has been tested with:

```
matplotlib==3.3.4 
numba==0.53.1 
numpy==1.20.1
opencv_python==4.5.4.58 
Pillow==9.1.1 
scipy==1.6.2 
torch==1.5.1+cu92 
torchvision==0.6.1+cu92 tqdm
```

## Download our model
Our model can be found in the `models` folder. SST-Sal predicts saliency maps for 360º videos from their RGB frames and their optical flow estimations. A less accurate variation of our model that does not require optical flow information can also be found in the same folder. Please refer to section 4.4.4 of our paper for more information regarding the performance of our model without optical flow.

## Perform inference with your own videos
### SST-Sal
To perform inference with our model, modify the inference parameters and data loader sections in `config.py`, and use:

```
inference_model = 'models/SST_Sal.pth'
of_available = True
```
SST-Sal requires that you first obtain the optical flow estimations from your own videos. You can use `utils.py/frames_extraction(config.videos_folder)` to extract the different frames, and employ [RAFT](https://github.com/princeton-vl/RAFT) to obtain the optical flow estimation associated with each frame. Please make sure that the name of the frame and its optical flow estimation are identical. Both RGB image names are expected to have the following numerical format: videonumber_framenumber.png (i.g., 0001_1023.png).

Once you have the extracted frames and the associated optical flow estimations, you can run inference.py to obtain the predicted saliency in `config.results_dir` folder.

```
python inference.py 
```
### SST-Sal without optical flow estimations
If you do not have the optical flow estimations of your videos, you can try our less accurate alternative of SST-Sal. Modify the inference parameters and data loader sections in `config.py`, and use:

```
inference_model = 'models/SST_Sal_wo_OF.pth'
of_available = False
```
Then run `inference.py` to obtain the saliency predictions
```
python inference.py
```
