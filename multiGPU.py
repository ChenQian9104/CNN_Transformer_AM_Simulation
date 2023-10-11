import os
import sys
import copy

import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, LambdaLR, CosineAnnealingLR, CosineAnnealingWarmRestarts

from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data import random_split 

import torch.distributed as dist
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler



from module import * 
from utils import *
from customDataset import AMDataset, collate_fn_padd


def warmup(current_step):
    warmup_steps = 10 
    training_steps = 500 
    
    if current_step < warmup_steps: 
        return float(current_step/warmup_steps) 
    else: 
        return max(0.0, float(training_steps - current_step) / float(max(1, training_steps - warmup_steps)))

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    
def demo_basic(rank, world_size, dataset, validation_dataloader, batch_size=16):
   
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    # Prepare dataloader on each GPU
    train_data_loader = DataLoader(dataset, 
                              batch_size = batch_size, 
                              collate_fn = collate_fn_padd,
                              pin_memory=True, 
                              shuffle=False, 
                              sampler=sampler )
    """                          
    for data, label in data_loader: 
        print(f"on rank: {rank}, | shape of data: ", data.shape)
        print(f"on rank: {rank}    ", label)
    """                         
                             

    # create model and move it to GPU with id rank
    loadCheckPoint = False
    #model = AMTransformer().to(rank)
    model = TransformerFlashAttention(embed_dim=2304, n_heads=9, L=2, expansion_factor=1).to(rank)
    if loadCheckPoint: 
        model.load_state_dict({k.replace('module.', '') : v for k, v in torch.load('model_20.pt').items()})
        print(f"Model load successfully on rank {rank}.")
    dist.barrier()  
    ddp_model = DDP(model, device_ids=[rank],find_unused_parameters=True)
    dist.barrier()  
    criterion = nn.MSELoss() 
    #optimizer = optim.Adam(ddp_model.parameters(), lr=2e-5, weight_decay=0.1)
    #optimizer = optim.AdamW(ddp_model.parameters(), lr=5e-6,weight_decay=0.1)
    optimizer = optim.AdamW(ddp_model.parameters(), lr=2e-5,weight_decay=0.1)
    #optimizer = optim.SGD(ddp_model.parameters(), lr=1e-3, momentum=0.1, weight_decay=0.01)
    #optimizer = optim.SGD(ddp_model.parameters(), lr=1e-4, momentum=0.01)
    scheduler = StepLR(optimizer, step_size = 10, gamma=0.8)
    #scheduler = LambdaLR(optimizer, lr_lambda=warmup)

    num_epoch = 50

    for epoch in range(num_epoch): 

        train_loss = 0 
        
        train_data_loader.sampler.set_epoch(epoch)
        
        #dist.barrier()  
        
        for data, label in train_data_loader: 
        
            

            data = data.to(rank)
            label = label.view(-1,1).to(rank)

            pred = ddp_model(data) 
            loss = criterion(pred, label) 
            #if train_loss == 0 and rank == 1:
                #print((pred-label)[:10])

            optimizer.zero_grad() 
            loss.backward() 

            optimizer.step() 

            train_loss += loss.item()/len(train_data_loader)
            
        scheduler.step()
        

        if epoch % 1 == 0 and rank == 1: 

            with torch.no_grad(): 

                val_loss = 0 

                for  data, label in validation_dataloader:

                    data = data.to(rank)
                    label = label.view(-1,1).to(rank)
                    
                    pred_val = ddp_model(data) 
                    loss = criterion(pred_val, label)  
                    val_loss += loss.item()/len(validation_dataloader)
            
            
            cur_lr = scheduler.get_last_lr()[0]
            print(f"epoch: {epoch:>3} | training loss: {train_loss:.5f} | validation loss: {val_loss:.5f}  | learning rate :{cur_lr:.6f}")
            if epoch >= 0 and epoch % 1 == 0: 

                best_model_wts = copy.deepcopy( ddp_model.state_dict() )
                torch.save( best_model_wts,f"model_{epoch}.pt")

    dist.barrier()  
    if rank == 1:
      best_model_wts = copy.deepcopy( ddp_model.state_dict() )
      torch.save( best_model_wts,f"model_final.pt")
    cleanup()    



if __name__ == '__main__':


    # Read these setup from config file 
    batch_size = 512

    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))

    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

    print('Available memory:', round(torch.cuda.get_device_properties(0).total_memory/1024**3, 1), 'GB')


 
    # Load the simulation data 
    
    dataset = AMDataset()
    
    
    num = len(dataset) 
    print("the size of AM Dataset:", num)
    train_num = int(num * 0.95)
    validation_num = num - train_num
    
    train_dataset, validation_dataset = random_split(dataset, [train_num, validation_num]) 
    
    validation_dataloader = DataLoader(validation_dataset, batch_size = batch_size, collate_fn = collate_fn_padd) 
    

    
    # Train the model 
    
    world_size = torch.cuda.device_count()
    mp.spawn(demo_basic,
             args=(world_size,train_dataset,validation_dataloader, batch_size),
             nprocs=world_size,
             join=True)




    

