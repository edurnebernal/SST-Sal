import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from DataLoader360Video import RGB_and_OF
from sphericalKLDiv import  KLWeightedLossSequence
from torch.utils.tensorboard import SummaryWriter
import datetime
import os
import time
from torch.utils.data import DataLoader
import models
from utils import read_txt_file

# Import config file
import config

def train(train_data, val_data, model, device, criterion, lr = 0.001, EPOCHS=10, model_name='Model'):

    writer = SummaryWriter(os.path.join(config.runs_data_dir, model_name +'_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '/'))
    path = os.path.join(config.models_dir, model_name + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    ckp_path = os.path.join(config.ckp_dir, model_name + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) 
    os.mkdir(path)
    os.mkdir(ckp_path)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr) 

    model.train()

    model.to(device)
    criterion.cuda(device)
    print("Training model ...")
    epoch_times = []
    
    # Training loop
    for epoch in range(EPOCHS):
        start_time = time.time()
        avg_loss_train = 0.
        avg_loss_val = 0.
        counter_train = 0
        counter_val = 0
            
        for x, y in train_data:
            

            model.zero_grad()
        
            pred = model(x.to(device))
            
            loss = criterion(pred[:, :, 0, :, :], y[:, :, 0, :, :].to(device))

            loss.sum().backward()
            optimizer.step()

            avg_loss_train += loss.sum().item()

            counter_train += 1
            if counter_train % 20 == 0:
                print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(epoch, counter_train, len(train_data),
                                                                                        avg_loss_train / counter_train))

        current_time = time.time()
        print("Epoch {}/{} , Total Spherical KLDiv Loss: {}".format(epoch, EPOCHS, avg_loss_train / counter_train))
        print("Total Time: {} seconds".format(str(current_time - start_time)))
        epoch_times.append(current_time - start_time)

        # Evaluate on validation set
        with torch.no_grad():
            for x, y in val_data:
                counter_val += 1
                pred = model(x.to(device))
                loss = criterion(pred[:, :, 0, :, :], y[:, :, 0, :, :].to(device))
                avg_loss_val += loss.sum().item()

        writer.add_scalars('Loss', {'train': avg_loss_train / counter_train, 'val': avg_loss_val / counter_val}, epoch)
        
        # Save checkpoint and model every 50 epochs
        if epoch % 50 == 0:
            torch.save(model, path + '/'+ str(epoch)+ '_model.pth')
            ckp_path = os.path.join(config.ckp_dir,model_name + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) 
            os.mkdir(ckp_path)
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, ckp_path + '/model.pt')
    
    # Save final model and checkpoints
    torch.save(model, path + '/model.pth')
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, ckp_path + '/model.pt')

    return model


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")

    # Train SST-Sal

    model = models.SST_Sal(hidden_dim=config.hidden_dim)
    criterion = KLWeightedLossSequence()

    video_names_train = read_txt_file(config.videos_train_file)

    train_video360_dataset = RGB_and_OF(config.frames_dir, config.optical_flow_dir, config.gt_dir, video_names_train, config.sequence_length, split='train', resolution=config.resolution)
    val_video360_dataset = RGB_and_OF(config.frames_dir, config.optical_flow_dir, config.gt_dir, video_names_train, config.sequence_length, split='validation', resolution=config.resolution)

    train_data = DataLoader(train_video360_dataset, batch_size=config.batch_size, num_workers=8, shuffle=True)
    val_data = DataLoader(val_video360_dataset, batch_size=config.batch_size, num_workers=8, shuffle=True)

    print(model)
    model = train(train_data, val_data, model, device, criterion, lr=config.lr, EPOCHS=config.epochs, model_name=config.model_name)

    print("Training finished")