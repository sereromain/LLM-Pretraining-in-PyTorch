import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group
from data_loader import custom_data_loader, custom_distributed_data_loader
from model import LargeLanguageModel
from config import TrainingConfig

from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.console import Console
from contextlib import contextmanager

import tiktoken

def ddp_setup(rank: int, world_size: int):
   """
   Args:
       rank: Unique identifier of each process
      world_size: Total number of processes
   """
   os.environ["MASTER_ADDR"] = "localhost"
   os.environ["MASTER_PORT"] = "12355"
   torch.cuda.set_device(rank)
   init_process_group(backend="nccl", rank=rank, world_size=world_size)

def epoch_iteration(model,dataloader,mode,accum_steps,steps,criterion,optimizer,lr_scheduler,progress_bar,progress_task,device):

    # Create an iterator from the DataLoader
    data_iter = iter(dataloader)

    model.module.train()
    training = mode=="train"

    average_loss       = 0
    average_loss_count = 0
    average_acc        = 0
    average_acc_count  = 0
    current_lr         = 0

    @contextmanager
    def dummy_context():
        yield

    with progress_bar if progress_bar is not None else dummy_context():

        for i in range(steps):

            if training : optimizer.zero_grad()

            accum_loss = 0
            accum_acc  = 0

            average_loss_count += 1
            average_acc_count  += 1

            # Create a loop to accumulate gradients of sub-batches
            for j in range(accum_steps):

                # Load sub-batch of size batch_size/accum_steps
                inputs, labels = next(data_iter) # shape (sub-batch, seq_size) , (sub-batch, seq_size)
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

                outputs = model(inputs) # shape (sub-batch, seq_size, vocab_size)

                _, predicted_token = torch.max(outputs,-1) # find the argmax of the vocabulary list
                accum_acc += (predicted_token == labels).float().mean().item()/accum_steps

                ls = outputs.shape
                outputs = outputs.view(ls[0]*ls[1], ls[2]) # shape (sub-batch*seq_size, vocab_size)
                labels = labels.view(ls[0]*ls[1])          # shape (sub-batch*seq_size)

                loss = criterion(outputs, labels) # compute loss
                loss /= accum_steps               # Normalize loss
                if training : loss.backward()     # Accumulate gradients of sub-batch

                accum_loss+=loss.item()

            if training : optimizer.step()

            average_loss += (accum_loss - average_loss) / average_loss_count
            average_acc  += (accum_acc  - average_acc)  / average_acc_count

            # Update the progress_bar with new values
            if training:
                current_lr = lr_scheduler.get_last_lr()[0]
                if progress_bar is not None: progress_bar.update(progress_task, advance=1, loss=average_loss, acc=average_acc*100, lr="{:.2e}".format(current_lr))
            else:
                if progress_bar is not None: progress_bar.update(progress_task, advance=1, val_loss=average_loss, val_acc=average_acc*100)

        if not training : lr_scheduler.step(average_acc)

    if progress_bar is not None: progress_bar.reset(progress_task)


    return (average_loss, average_acc, current_lr) if training else (average_loss, average_acc)


def main(rank, world_size, config):

    # ---- Setup distributed GPU training ----
    ddp_setup(rank, world_size)

    print("Experiment launched on GPU:",rank)

    # ---- Model bfloat16, Loss, Optimizer ----
    model = LargeLanguageModel().bfloat16().to(rank)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-1)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                                   factor=config.lr_multiplier,
                                                                   patience=config.lr_patience,
                                                                   threshold=1e-5,
                                                                   threshold_mode='rel',
                                                                   cooldown=0,
                                                                   min_lr=0,
                                                                   eps=1e-08)


    # ---- Create the dataloader ----
    train_dataloader = custom_distributed_data_loader(config.data_dir,
                                                      dataset_type='train',
                                                      block_size=model.config.seq_size,
                                                      batch_size=config.batch_size//config.accum_steps,
                                                      batches_to_load=config.epoch_train_steps*config.accum_steps,
                                                      num_workers=8, pin_memory=True)

    val_dataloader  = custom_distributed_data_loader(config.data_dir,
                                                     dataset_type='val',
                                                     block_size=model.config.seq_size,
                                                     batch_size=config.batch_size//config.accum_steps,
                                                     batches_to_load=config.epoch_val_steps*config.accum_steps,
                                                     num_workers=8, pin_memory=True)

    if rank == 0:
        # Creation of the console
        console = Console()

        # Create a custom training progress bar
        train_progress = Progress(
            TextColumn("[bold white]{task.completed}/{task.total}"),
            BarColumn(bar_width=40, style="rgb(120,80,67)",complete_style="green",finished_style="rgb(102,255,94)"),
            TextColumn("[bold white]Loss: {task.fields[loss]:.4f}"),
            TextColumn("[bold rgb(255,237,194)]Acc: {task.fields[acc]:.2f}%"),
            TextColumn("[bold rgb(255,200,194)]lr: {task.fields[lr]}"),
            TimeRemainingColumn(compact=True,elapsed_when_finished=True),
            console=console)

        # Create a custom evaluation progress bar
        val_progress = Progress(
            TextColumn("-> [bold white]{task.completed}/{task.total}"),
            BarColumn(bar_width=10,style="magenta", complete_style="white",finished_style="white"),
            TextColumn("[bold rgb(212,194,255)]Val_loss: {task.fields[val_loss]:.4f}"),
            TextColumn("[bold rgb(249,194,255)]Val_acc: {task.fields[val_acc]:.2f}%"),
            TimeRemainingColumn(compact=True,elapsed_when_finished=True),
            console=console)

        train_task = train_progress.add_task("Train", total=config.epoch_train_steps, loss=0.0, acc=0.0, lr=0.0)
        val_task   = val_progress.add_task("Val", total=config.epoch_val_steps, val_loss=0.0, val_acc=0.0)
    else:
        train_progress=None
        val_progress=None
        train_task=None
        val_task=None


    # ---- Load weights from previous training ----
    if os.path.isfile("llm.pt"):
        model.load_state_dict(torch.load("llm.pt", weights_only=True))


    # ---- Make the model compatible with distributed training ----
    model = DistributedDataParallel(model, device_ids=[rank])


    # ---- Training loop ----
    if config.num_epochs and rank==0 : print(f"Training started ...")

    for epoch in range(config.num_epochs):

        if rank==0 : 
            # Model weights saving
            torch.save(model.module.state_dict(), "llm.pt")
            print(f"Epoch {epoch+1}/{config.num_epochs}")

        train_loss, train_acc, current_lr = epoch_iteration(model=model,
                                                            dataloader=train_dataloader,
                                                            mode="train",
                                                            accum_steps=config.accum_steps,
                                                            steps=config.epoch_train_steps,
                                                            criterion=criterion,
                                                            optimizer=optimizer,
                                                            lr_scheduler=lr_scheduler,
                                                            progress_bar=train_progress,
                                                            progress_task=train_task,
                                                            device=rank)

        val_loss, val_acc = epoch_iteration(model=model,
                                            dataloader=val_dataloader,
                                            mode="val",
                                            accum_steps=config.accum_steps,
                                            steps=config.epoch_val_steps,
                                            criterion=criterion,
                                            optimizer=optimizer,
                                            lr_scheduler=lr_scheduler,
                                            progress_bar=val_progress,
                                            progress_task=val_task,
                                            device=rank)

        if epoch==0:
            if rank==0 :
                history = np.array([[epoch,train_loss,train_acc,current_lr,val_loss,val_acc]])
                np.savetxt("history.csv", history, fmt=['%u,','%f,','%f,','%f,','%f,','%f'] , header="epoch,train_loss,train_acc,lr,val_loss,val_acc")
        else:
            if rank==0 :
                history = np.concat([history,[[epoch,train_loss,train_acc,current_lr,val_loss,val_acc]]],axis=0)
                with open("history.csv", 'a') as f:
                    np.savetxt(f, [[epoch,train_loss,train_acc,current_lr,val_loss,val_acc]], fmt=['%u,','%f,','%f,','%f,','%f,','%f'])

    destroy_process_group()



if __name__ == "__main__":

    prompt = "Albert Einstein (14 March 1879 – 18 April 1955) was a German-born theoretical physicist who is best known for developing the theory of relativity. Einstein also made important contributions to quantum mechanics. His mass–energy equivalence formula E = mc2, which arises from special relativity, has been called the worlds most famous equation. He received the 1921 Nobel "

    # acceleration for NVIDIA Ampere architectures and beyond
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 matmul
    torch.backends.cudnn.allow_tf32 = True       # allow tf32 on cudnn
    
    world_size = torch.cuda.device_count()

    config = TrainingConfig()

    mp.spawn(main, args=(world_size,config,), nprocs=world_size, join=True)