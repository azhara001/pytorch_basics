import torch 
import numpy as np

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())

# playing around with tensors
t = torch.Tensor([[1,2,3],[4,5,6],[7,8,9]])

# rank of a tensor is the number of dimensions present in a tensor
# so, a rank-2 tensor is essentially a 2-d array or a matrix

# axes of a tensor tell you the number of elements present along each axes 
# shape of the tensor tells you the number of elems along each axes

print('shape of the tensor:',t.shape)

# tensors are pytorch datasets 
print(f"Type of the torch tensro: {type(t)}")

# ways of declaring pytorch tensors:
np_array = np.array([[1,2,3],[4,5,6],[7,8,9]])

t1 = torch.tensor(np_array)
print(f"dtype: {t1.dtype}")
print(f"device: {t1.device}")
print(f"layout: {t1.layout}\n")

t2 = torch.Tensor(np_array)
print(f"dtype: {t2.dtype}")
print(f"device: {t2.device}")
print(f"layout: {t2.layout}")

t3 = torch.from_numpy(np_array)
t4 = torch.as_tensor(np_array)

# tensor operations between pytorch tensors must happen between tensors on THE SAME DEVICE otherwise an error is generated!
# tensors contain data of the same type 

# Common ways of creating pytorch tensors are:
# from_numpy
# as_tensor
# tensor
# Tensor 

print(torch.from_numpy(np_array).dtype) # factory function -> more dynamic -> automatically infers the datatype from the array passed 
print(torch.as_tensor(np_array).dtype) # factory function -> more dynamic -> automatically infers the datatype from the array passed 
print(torch.tensor(np_array).dtype) # factory function -> more dynamic -> automatically infers the datatype from the array passed. Class Constructor functions DONOT provide the option of declaring a dtype once an object is instantiated.
print(torch.Tensor(np_array).dtype) # class constuctor -> uses global datatype (float32) to instantiate an object 

# Creation operations without data for pytorch tensors
iden = torch.eye(5,5)
ones = torch.ones(5,5)
zeros = torch.zeros(5,5)
rand = torch.rand(2,2)

print(torch.tensor(np_array,dtype=torch.float64)) # should work
#print(torch.Tensor(np_array,dtype=torch.int64)) # should generate an error

# torch.as_tensor() and torch.from_numpy() are basically pointers pointing to the original numpy array whereas, torch.tensor() and torch.Tensor() generates copies of the original numpy array. readability (creating copies) versus performance (sharing memories via pointers)

# t1 = t1.squeeze() # removes rank with no entries 
# t1 = t1.unsqueeze(dim=0) # adds a dimension with length 1 to the dim specified 
# t1 = t1.flatten() # convers the tensor into a rank-1 tensor 

t1 = torch.tensor([[1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4,4]])
t2 = torch.tensor([[9,9,9,9],[8,8,8,8],[7,7,7,7],[6,6,6,6]])

torch.cat((t1,t2),axis=0)
torch.cat((t1,t2),axis=1)
torch.stack((t1,t2))


# reduction operations on a tensor
t1.sum()
t1.unique()
t1.sum().numel() > t1.sum()

# other reduction operations include:
# t1.argmax(dim=0)
# t1.argmax(dim=1)
# t1.mean(dim=0)
# t1.mean(dim=1)
