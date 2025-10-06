import numpy as np

class Tensor:
    def __init__(self, data, children=()):
        self.data = data
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(children)

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, children=(self, other))

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward
    
        return out
    
    def __mul__(self, other):
        out = Tensor(self.data * other.data, children=(self, other))

        def _backward():
            grad_self = out.grad * other.data
            self.grad += other._unbroadcast(grad_self, self.data.shape)

            grad_other = out.grad * self.data
            other.grad += other._unbroadcast(grad_other, other.data.shape)

        out._backward = _backward

        return out
        

    def __add__(self, other):
        out = Tensor(self.data + other.data, children=(self, other))

        def _backward():
            self.grad += self._unbroadcast(out.grad, self.data.shape)
            other.grad += other._unbroadcast(out.grad, other.data.shape)

        out._backward = _backward

        return out

    def exp(self):
        out = Tensor(np.exp(self.data), children=(self,))

        def _backward():
            self.grad += out.grad * out.data

        out._backward = _backward

        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), children=(self,))
        
        def _backward():
            if axis is None:
                # Summed everything - broadcast gradient back to original shape
                self.grad += np.ones_like(self.data) * out.grad
            elif not keepdims:
                self.grad += np.expand_dims(out.grad, axis=axis)
            else:
                self.grad += out.grad
        
        out._backward = _backward
        return out
    
    def __sub__(self, other):
        out = Tensor(self.data - other.data, children=(self, other))
        
        def _backward():
            self.grad += self._unbroadcast(out.grad, self.data.shape)
            other.grad += other._unbroadcast(-out.grad, other.data.shape)
        out._backward = _backward
        
        return out
    
    def __truediv__(self, other):
        out = Tensor(np.divide(self.data, other.data), children=(self, other))

        def _backward():
            grad_self = out.grad * (1/other.data)
            self.grad += self._unbroadcast(grad_self, self.data.shape)
            
            grad_other = out.grad * (-out.data / other.data)
            other.grad += other._unbroadcast(grad_other, other.data.shape)

        out._backward = _backward

        return out
    
    def softmax(self, axis=-1):
        # Forward pass with numerical stability
        shifted = self.data - self.data.max(axis=axis, keepdims=True)
        exp_vals = np.exp(shifted)
        softmax_output = exp_vals / exp_vals.sum(axis=axis, keepdims=True)
        
        out = Tensor(softmax_output, children=(self,))
        
        def _backward():
            # Softmax gradient: s * (grad_out - (grad_out * s).sum())
            s = out.data  # The softmax output
            grad_output = out.grad
            
            sum_term = (grad_output * s).sum(axis=axis, keepdims=True)
            grad_input = s * (grad_output - sum_term)
            
            self.grad += grad_input
        
        out._backward = _backward
        return out

    def _unbroadcast(self, grad, original_shape):
        ndims_added = grad.ndim - len(original_shape)
        for i in range(ndims_added):
            grad = grad.sum(axis=0)
        
        for i, dim in enumerate(original_shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
        
        return grad
    
    def relu(self):
        out = Tensor(np.maximum(0, self.data), children=(self,))

        def _backward():
            self.grad += out.grad * np.where(out.data > 0, np.ones_like(self.data), np.zeros_like(self.data))

        out._backward = _backward

        return out
    
    def log(self):
        out = Tensor(np.log(self.data), children=(self,))

        def _backward():
            self.grad += out.grad * (1 / self.data)

        out._backward = _backward

        return out

    def backward(self):
        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()


    def __neg__(self):
        out = Tensor(-self.data, children=(self,))
        
        def _backward():
            self.grad += -out.grad 
        out._backward = _backward
        
        return out

    def __repr__(self):
        return f"Tensor {self.data.shape}"