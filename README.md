# SST-Sal: A spherical spatio-temporal approach for saliency prediction in 360º videos

Code and models for *“SST-Sal: A spherical spatio-temporal approach for saliency prediction in 360º videos”* ([PDF](https://graphics.unizar.es/papers/2022_Bernal_SSTSal.pdf))
Edurne Bernal, Daniel Martín, Diego Gutierrez, and Belen Masia.
**Computers & Graphics**

## Abstract
Virtual reality (VR) has the potential to change the way people consume content, and has been predicted to become the next big computing paradigm. However, much remains unknown about the grammar and visual language of this new medium, and understanding and predicting how humans behave in virtual environments remains an open problem. In this work, we propose a novel saliency prediction model which exploits the joint potential of spherical convolutions and recurrent neural networks to extract and model the inherent spatio-temporal features from 360º videos. We employ Convolutional Long Short-Term Memory cells (ConvLSTMs) to account for temporal information at the time of feature extraction rather than to post-process spatial features as in previous works. To facilitate spatio-temporal learning, we provide the network with an estimation of the optical flow between 360º frames, since motion is known to be a highly salient feature in dynamic content. Our model is trained with a novel spherical Kullback–Leibler Divergence (KLDiv) loss function specifically tailored for saliency prediction in 360º content. Our approach outperforms previous state-of-the-art works, being able to mimic human visual attention when exploring dynamic 360º videos.

Visit our [website](https://graphics.unizar.es/projects/SST-Sal_2022/) for more information and supplementary material.
