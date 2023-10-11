import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, Dataset, DataLoader 
import numpy as np
import os
import scipy.io

from backbone import BackBoneResNet

class AMDataset(): 
    def __init__(self, path='./AMSim'): 
        
        assert os.path.exists(path + '/Geometry'), 'Geometry folder is missed!'
        assert os.path.exists(path + '/Data'), 'Data folder is missed!'
        
        self.root = path
        self.path_geometry = self.root + '/Geometry/'    # path for the geometry files  (.mat format)
        self.path_data = self.root + '/Data/'            # path for the simulation data (.npy format)
        
        self.cnn = BackBoneResNet()                      # cnn model used to generate layer-wise embedding
        self.pooling = nn.AvgPool2d(kernel_size=7)       # pooliny layer for dimension reduction

        self.geometry_file_list = os.listdir(path + '/Geometry') 
        assert len(self.geometry_file_list) != 0, 'Geometry file is empty!'
        
        self.simulation_data_folder_list = []
        for file in self.geometry_file_list: 
            folder_path = path + '/Data/' + file[:-4]
            self.simulation_data_folder_list.append(folder_path)
            assert os.path.exists(folder_path), f'file[:-4] folder is missed'
        
        self.simulation_data = self.data_loading_processing()
        self.geometry_images = self.load_geometry_image()
    
    def load_geometry_image(self):
        
        images = []
        for file in self.geometry_file_list:
            
            mat = scipy.io.loadmat(self.path_geometry + file) 
            val = np.array(mat['result'], dtype = float)

            # convert gray scale image to RGB 
            val_ = np.expand_dims(val, 2)
            val_3_channels = val_.repeat(3, axis=2)

            # convert ndarray to tensor

            image = torch.from_numpy(val_3_channels).type(torch.float32)

            image = torch.permute(image,(3, 2, 0, 1))
            #image = torch.unsqueeze(image, 1)

            out = self.cnn(image) 
            out = self.pooling(out).detach()

            out = out.view(image.shape[0], -1)


            images.append(out)
            
        return images
        
    
    def load_simulation_data(self, num_points=1000, shuffle=True):
        
        """
        Args: 
            num_points: number of selected points from each geometry case 
            shuffle: shuffle before selection 
        """
        
        data = []
        
        np.random.seed(seed=12345)
        
        for case in self.simulation_data_folder_list: 
            
            
            npy_file_list = os.listdir(case)
            npy_data = None
            
            for npy_file in npy_file_list: 
                file_path = case + '/' + npy_file 
                
                temp = np.load(file_path)
                
                
                if npy_data is None: 
                    npy_data = temp 
                else: 
                    npy_data = np.concatenate((npy_data, temp), axis=0)
                
                
                np.random.shuffle(npy_data) 

            #data.append(npy_data[:num_points, :])
            data.append(npy_data)
            
        return data
            
    def normalize_scale_data(self, data, img_size=224, img_domain=200, scale_height=100, scale_time=100000, scale_temperature=100):
        """
        Function:
            -  scale and normalize the data for training and prediction 
            -  transfer the spatial location (x, y) into pixel index; need image size and image domain region
            -  scale the height z (0, 100) to a value between 0.0 and 1.0 
            -  scale the time to a value between 0.0 and 1.0 
            -  dived the temperature value from simulation by 100
            
        Args: 
            data: a n x 5 matrix; each row represents a single point (x, y, z), its time stamp, and the 
                  temperature value from simulation; (x, y, z, time, temperature)
            img_size: the image size of the geotry file
            
        Return:
            a list of scaled data 
        
        """
        dx = img_domain / (img_size - 1)
        data_ = []
        for npy_data in data: 
            
            npy_data[:, 0 ] = (npy_data[:, 0] + img_domain/2) // dx / img_size 
            npy_data[:, 1] = (npy_data[:, 1] + img_domain/2) // dx / img_size 
            
            npy_data[:, 2] = npy_data[:, 2]/scale_height
            
            npy_data[:, 3] = npy_data[:, 3]/scale_time 
            
            npy_data[:, 4] = npy_data[:, 4]/scale_temperature
            
            data_.append(npy_data)
            
        return data_
    
    def positional_temperal_encoding(slef, data, freq_num=32, query_dim=4): 
        """
        Function: 
            -   Map the (x, y, z, time) query vector to a high dimension space determined by freq_num, L
            -   For example: 
                    x -> {sin(2^0*pi*x), cos(2^0*pi*x), sin(2^1*pi*x), cos(2^1*pi*x), ...., sin(2^(L-1)*pi*x), cos(2^(L-1)*pi*x)}
            -   Poisitonal encoding for x, y, z; This positional encoding is different to the one used in transformer
            -   Temperal encoding for time
            -   Add the geometry label to the first column 
        """
        data_ = []
        cnt = 0
        for npy_data in data: 
            
            v = np.zeros((npy_data.shape[0], 2*freq_num*query_dim))

            query_v = npy_data[:, :-1]*np.pi

            for L in range(freq_num): 

                v_sin = np.sin(2**L/100000*query_v)
                v_cos = np.cos(2**L/100000*query_v)

                v[:, 2*L : : 2*freq_num] = v_sin 
                v[:, 2*L + 1 : : 2*freq_num] = v_cos
                
            label = np.ones((npy_data.shape[0], 1))*cnt 
            temp = np.concatenate((label, v, npy_data[:, -1].reshape(npy_data.shape[0], 1)), axis=1).astype(np.float32)
            
            data_.append(temp)
            
            cnt += 1
            
        return data_
            
        
    
    def data_loading_processing(self, num_points=80000, shuffle=True, img_size=224, img_domain=200, scale_height=100, scale_time=10000, scale_temperature=100, freq_num = 32, query_dim=4): 
        
        data = self.load_simulation_data(num_points, shuffle)
        
        data = self.normalize_scale_data(data, img_size, img_domain, scale_height, scale_time, scale_temperature)
        
        data = self.positional_temperal_encoding(data, freq_num, query_dim)
        
        
        # Concatenate the list of numpy arrays 
        
        return np.concatenate(data, axis=0)
    
    def __len__(self):
        return self.simulation_data.shape[0]
        
    def __getitem__(self, key): 
        data = self.simulation_data[key]
        label = int(data[0])
        data_ = torch.tensor(data[1:-1])

        temperature = data[-1]

        seq_length, embed_dim =  self.geometry_images[label].shape

        n = data_.shape[0]

        data_ = data_.reshape((1,n)).repeat(seq_length, 1)

        return torch.cat((self.geometry_images[label], data_), axis=1), temperature
        
        #return self.geometry_images[label], data_
        
        
def collate_fn_padd(data): 
    """
    Args: 
        data: a list of tuple; each tuple is (image_embedding, temperature)
        image_embedding: [batch_size, seq_length, embedding_dim]
    """
    batch_size = len(data)
    max_length = 0
    
    seq_length = data[0]
    _, embed_dim = data[0][0].shape
    for d in data: 
        
        max_length = max(max_length, d[0].shape[0])
    

    
    padded_images = [torch.cat([d[0],
                                torch.zeros(max_length - d[0].shape[0], embed_dim)]).unsqueeze(0) for d in data]
    padded_data = np.array([d[1] for d in data])
    return torch.vstack(padded_images), torch.tensor(padded_data)