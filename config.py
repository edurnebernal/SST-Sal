
#####################################################################################
# Data Loader parameters
#####################################################################################
# Sequence length
sequence_length = 20
# Image resolution (height, width)
resolution = (240, 320)
# Path to the folder containing the RGB frames
frames_dir = 'data/frames'
# Path to the folder containing the optical flow
optical_flow_dir = 'data/optical_flow'
# Path to the folder containing the ground truth saliency maps
gt_dir = 'data/saliency_maps'
# Txt file containing the list of video names to be used for training
videos_train_file = 'data/train_split_VRET.txt'
# Txt file containing the list of video names to be used for testing
videos_test_file = 'data/test_split_VRET.txt'

#####################################################################################
# Training parameters
#####################################################################################
# Batch size
batch_size = 1
# Nº of epochs
epochs = 240
# Learning rate
lr = 0.8
# Hidden dimension of the model (SST-Sal uses hidden_dim=36)
hidden_dim = 36
# Percentage of training data intended to validation
validation_split = 0.2
# Name of the model ( for saving pruposes)
model_name = 'SST-Sal'
# Path to the folder where the checkpoints will be saved
ckp_dir = 'checkpoints'
# Path to the folder where the model will be saved
models_dir ='models'
# Path to the folder where the training logs will be saved
runs_data_dir = 'runs'

#####################################################################################
# Inference parameters
#####################################################################################
# Path to the folder containing the model to be used for inference
inference_model = 'models/SST_Sal.pth'
# Path to the folder where the inference results will be saved
results_dir = 'results'
# Path to the folder containing the videos to be used for inference
videos_folder = 'data/videos'
# Indicates if the model used for inference is trained with or without optical flow
of_available = True



