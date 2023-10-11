# AM simulation tool based on transformers and convolutional neural network #

## Notes: 
* The sliced geometry is encoded by the pre-trained convolutional neural network for layer-wise geometry embedding
* The long-range dependencies between layer is processed by the transformer module
* A MLP is used to predict the temperature at (x, y, z, t) inspired by NeRF 

## Usage:
```python
python3 multiGPUs.py
```
![alt text](/Asset/model_architecture.jpg)
