### micrograd with a bit more punch

This is my personal playground of writing an auto diff engine a la [micrograd](https://github.com/karpathy/micrograd) but based on tensors instead of single values to pack more punch. The tensor class is using numpy arrays for data and gradients.

### goals

My personal goal is to implement the following:

- [x] tensor based autograd
- [x] linear layers
- [x] train dense network on mnist
- [ ] convolutional layers
- [ ] dropout
- [ ] batchnorm
- [ ] adam
- [ ] train conv network on small CIFAR10 (32x32)
- [ ] optimize with triton for GPU
- [ ] train conv network on larger CIFAR100 (224x224)
- [ ] qkv attention
- [ ] base transformer
- [ ] vision transformer
- [ ] train ImageNet with vision transformer

These will constantly change, depending on mood and time of day. This is all for educational purposes only. I don't have the nerve or the time to write a production grade library. 