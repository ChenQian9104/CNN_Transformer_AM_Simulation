import os
import sys
import copy

import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data import random_split 

import torch.distributed as dist
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler



from module import * 
from utils import *
from customDataset import AMDataset, collate_fn_padd

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
    model = AMTransformer().to(rank)
    
    if loadCheckPoint: 
        model.load_state_dict({k.replace('module.', '') : v for k, v in torch.load('model_100.pt').items()})
        print(f"Model load successfully on rank {rank}.")
    dist.barrier()  
    ddp_model = DDP(model, device_ids=[rank],find_unused_parameters=True)
    dist.barrier()  
    criterion = nn.MSELoss() 
    #optimizer = optim.Adam(ddp_model.parameters(), lr=3e-3, weight_decay=0.1)
    optimizer = optim.Adam(ddp_model.parameters(), lr=1e-3)
    #optimizer = optim.SGD(ddp_model.parameters(), lr=1e-3, momentum=0.1, weight_decay=0.01)
    #optimizer = optim.SGD(ddp_model.parameters(), lr=1e-5, weight_decay=0.01)
    scheduler = StepLR(optimizer, step_size = 10, gamma=0.9)

    num_epoch = 1000

    for epoch in range(num_epoch): 

        train_loss = 0 
        
        train_data_loader.sampler.set_epoch(epoch)
        
        #dist.barrier()  
        
        for data, label in train_data_loader: 
        
            

            data = data.to(rank)
            label = label.view(-1,1).to(rank)

            pred = ddp_model(data) 
            loss = criterion(pred, label) 

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
            
            print('epoch: %d, loss: %.5f, validation loss: %.5f' % (epoch, train_loss, val_loss))
            
            if epoch > 0 and epoch % 10 == 0: 

                best_model_wts = copy.deepcopy( ddp_model.state_dict() )
                torch.save( best_model_wts,f"model_{epoch}.pt")

    dist.barrier()  
    if rank == 1:
      best_model_wts = copy.deepcopy( ddp_model.state_dict() )
      torch.save( best_model_wts,f"model_final.pt")
    cleanup()    



if __name__ == '__main__':


    # Read these setup from config file 
    batch_size = 256

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
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, collate_fn = collate_fn_padd)
    validation_dataloader = DataLoader(validation_dataset, batch_size = batch_size, collate_fn = collate_fn_padd) 
    device = 'cuda'
    #model = AMTransformer().to(device)
    #model = TransformerFlashAttention(embed_dim=2304, n_heads=9, L=2, expansion_factor=1).to(device)
    model =  BaselineLSTM().to(device)
    criterion = nn.MSELoss() 
    #optimizer = optim.Adam(ddp_model.parameters(), lr=3e-3, weight_decay=0.1)
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-2)
    #optimizer = optim.SGD(ddp_model.parameters(), lr=1e-3, momentum=0.1, weight_decay=0.01)
    #optimizer = optim.SGD(ddp_model.parameters(), lr=1e-5, weight_decay=0.01)
    scheduler = StepLR(optimizer, step_size = 100, gamma=0.9)
    

    num_epoch = 100

    for epoch in range(num_epoch): 

        train_loss = 0 
              
        for data, label in train_dataloader: 
            model.train()
            data = data.to(device)
            label = label.view(-1,1).to(device)

            pred = model(data) 
            loss = criterion(pred, label) 

            optimizer.zero_grad() 
            loss.backward() 

            optimizer.step() 

            train_loss += loss.item()/len(train_dataloader)
            
            
        scheduler.step()
        
        
        
        with torch.no_grad():
            val_loss = 0  
            for data, label in validation_dataloader: 
            
                data = data.to(device) 
                label = label.view(-1,1).to(device)

                pred = model(data) 
            
                loss = criterion(pred, label) 
            
                val_loss += loss.item()/len(validation_dataloader)
            
        if epoch > 0 and epoch % 10 == 0:
            best_model_wts = copy.deepcopy( model.state_dict() )
            torch.save( best_model_wts,f"./Baseline_rnn_2_layer_dim_256/model_{epoch}.pt")
        print(f"epoch: {epoch:>3} | training loss: {train_loss:.5f} | validation loss: {val_loss:.5f} ")    
            



    

