import wandb
import os
import logging

import sys, getopt
import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
import time
from tqdm import tqdm

import resnet
from load_data import trainloader, testloader, valloader
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO)




def validate_model(model, valid_dl, loss_func, device, log_images=False, batch_idx=0):
    "Compute performance of the model on the validation dataset and log a wandb.Table"
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

    # wandb.login(key=os.getenv('wandb_login_key'))

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

        logging.info(f"Config: {cfg}")

        model = getattr(resnet, model_name)(3,10) if hasattr(resnet, model_name) else resnet.resnet18(3, 10)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        logging.info('Device: %s', device)
        # logging.info("Model Name: %s", cfg.model.model_name)
        # logging.info("Epochs: %s", model_config['epochs'])
        # logging.info("Batch size: %s", model_config['batch_size'])
        # logging.info("Learning rate: %s", model_config['lr'])

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=cfg.training.lr, momentum=cfg.training.momentum, weight_decay=cfg.training.weight_decay)
        
        

        completed_steps = 0
        epoch_durations = []
        model.train()
        # ckpt_batch = 150
        
        logging.info('Starting training...')
        # TODO: Add tqdm readout
        # epoch_bar = tqdm()
        for epoch in tqdm(range(cfg.training.num_epochs)):   
            start_epoch_time = time.time()
            
            logging.info(f"Starting epoch {epoch}")
            running_loss = 0.0
            train_loss = 0

            for step, data in tqdm(enumerate(trainloader, 0), total=len(trainloader)):
            # for step, data in enumerate(trainloader, 0):
                if step > cfg.training.max_train_steps_per_epoch:
                    break
                start_batch_time = time.time()

                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                # print statistics
                running_loss += loss.item()
                completed_steps +=1

                # Log to wandb
                wandb.log({"train_loss": running_loss/(step+1), "step_duration": time.time()-start_batch_time}, step=completed_steps)
                
                
                # logging.info(f"epoch: {epoch}, batch: {step}, loss: {(running_loss / (step+1)):.2%}")

                # print checkpoints every 100 batches or at the end of an epoch
                if step % cfg.training.checkpoint_every == 0 or step==len(trainloader)-1:
                    logging.info(f"epoch: {epoch}, batch: {step}, loss: {running_loss / (step+1)}")

                #     if step==len(trainloader)-1:
                #         logging.info('Finished epoch: %d, loss: %.3f' % (epoch + 1, running_loss / (step % cfg.training.eval_every + 1)))
                #         wandb.log({"train_loss": running_loss/cfg.training.eval_every, "epoch": epoch + ((step+1)/len(trainloader))}, step=completed_steps)
                    
                #     elif step % cfg.training.eval_every == cfg.training.eval_every-1:    
                #         logging.info('epoch: %d, batch: %5d, loss: %.3f' % (epoch + 1, step + 1, running_loss / (step % cfg.training.eval_every + 1)))
                #         wandb.log({"train_loss": running_loss/cfg.training.eval_every, "epoch": epoch + ((step+1)/len(trainloader))}, step=completed_steps)
                    
                #     train_loss += running_loss / (step % cfg.training.eval_every + 1)
                #     running_loss = 0.0
            
            # Log validation metrics
            # val_loss, accuracy = validate_model(model, valloader, criterion, device)
            # wandb.log({"val_loss": val_loss, "val_accuracy": accuracy}, step=completed_steps)
            # logging.info('Validation loss: %.3f, Validation accuracy: %.2f', val_loss, accuracy*100)
            epoch_duration = time.time() - start_epoch_time
            wandb.log({"epoch_duration (seconds)": epoch_duration}, step=completed_steps)
            # wandb.log({"epoch_runtime (seconds)": epoch_duration}, step=completed_steps)

            epoch_durations.append(epoch_duration)

        avg_epoch_runtime = sum(epoch_durations) / len(epoch_durations)
        wandb.log({"avg epoch runtime (seconds)": avg_epoch_runtime})
    

    logging.info('Finished training')

    # model_dir='/outputs/model'
    model_dir='models'
    os.makedirs(model_dir, exist_ok = True) 
    path = os.path.join(model_dir, cfg.model.model_name, '.pt')
    torch.save(model.state_dict(), path)
    ## TODO: Save_pretrained?

    return None


if __name__ == "__main__":
   main(sys.argv[1:])