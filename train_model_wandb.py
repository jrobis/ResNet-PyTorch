import wandb
import os
import logging
import time
import datetime
from tqdm import tqdm
import sys, getopt
import torch.optim as optim
import torch
import torch.nn as nn

import resnet
from load_data import trainloader
from omegaconf import OmegaConf


logging.basicConfig(level=logging.INFO)

def validate_model(model, valid_dl, loss_func, device, log_images=False, batch_idx=0):
    # Compute performance of the model on the validation dataset and log a wandb.Table
    model.eval()
    val_loss = 0.0
    
    with torch.inference_mode():
        correct = 0
        for i, (images, labels) in enumerate(valid_dl, 0):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            val_loss += loss_func(outputs, labels).item()

            # Compute accuracy and accumulate
            _, predictions = torch.max(outputs.data, 1)
            correct += (predictions == labels).sum().item()

    return val_loss / len(valid_dl.dataset), correct / len(valid_dl.dataset)


def main(argv):

    # Create config and logger
    cfg = OmegaConf.load('conf/config.yml')

    try:
        opts, args = getopt.getopt(argv,"m:e:",["model=", "epochs="])
    except:
        print("No arguments passed.")

    for opt, arg in opts:
        if opt in ('-m', '--model'):
            cfg.model.model_name = model_name = arg.lower()
        if opt in ('-e', '--epochs'):
            cfg.training.num_epochs = int(arg)

    with wandb.init(project=cfg.project, config=OmegaConf.to_container(cfg, resolve=True)):
    
        model = getattr(resnet, model_name)(3,10) if hasattr(resnet, model_name) else resnet.resnet18(3, 10)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=cfg.training.lr, momentum=cfg.training.momentum, weight_decay=cfg.training.weight_decay)
        
        logging.info(f"Config: {cfg}")
        logging.info('Device: %s', device)
        
        completed_steps = 0
        epoch_durations = []
        model.train()
        
        logging.info('Starting training...')
        for epoch in tqdm(range(cfg.training.num_epochs)):   
            start_epoch_time = time.time()
            
            logging.info(f"Starting epoch {epoch}")
            running_loss = 0.0

            for step, data in tqdm(enumerate(trainloader, 0), total=len(trainloader)):
                if step > cfg.training.max_train_steps_per_epoch:
                    break
                start_batch_time = time.time()

                # move batch to device
                inputs, labels = data[0].to(device), data[1].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                completed_steps +=1

                # Log metrics
                wandb.log({"train_loss": running_loss/(step+1), "step_duration": time.time()-start_batch_time}, step=completed_steps)

                # print checkpoints every x num of batches or at the end of an epoch
                if step % cfg.training.checkpoint_every == 0 or step==len(trainloader)-1:
                    logging.info(f"epoch: {epoch}, batch: {step}, loss: {running_loss / (step+1)}")
                    # TODO: Add validation step - validate_model()

            epoch_duration = time.time() - start_epoch_time
            epoch_durations.append(epoch_duration)
            wandb.log({"epoch_duration (seconds)": epoch_duration}, step=completed_steps)

        avg_epoch_runtime = sum(epoch_durations) / len(epoch_durations)
        wandb.log({"avg epoch runtime (seconds)": avg_epoch_runtime})
    

    logging.info('Finished training')

    model_dir=cfg.model.output_dir
    os.makedirs(model_dir, exist_ok = True)
    path = os.path.join(model_dir, cfg.model.model_name, '_', datetime.datetime.now(), '.pt')
    torch.save(model.state_dict(), path)
    ## TODO: Save_pretrained?

    return None


if __name__ == "__main__":
   main(sys.argv[1:])