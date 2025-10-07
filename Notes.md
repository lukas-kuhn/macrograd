## Notes on computer vision

### Backprop

Backpropagation is done via a computational graph so we always want to know, how much did a node influence the outcome, in our machine learning case the loss $L$. To do that we basically built a graph of computations, each with its children (the things we added or multiplied or divided or whatever) as nodes and then we backpropapage through them one by one via topological order by calcuating the gradients of each computation for the children. For that we need the calculus' chain rule.

### Chain Rule 

If a variable _z_ depends on the variable _y_, which itself depends on the variable _x_ then _z_ (rather obviously) depends _x_ as well. The chain rule is expressed as 
$$
\frac{dz}{dx} = \frac{dz}{dy} * \frac{dy}{dx}
$$
For our computational graph this means we first have to calculate $\frac{dz}{dy}$ before we then can multiply that by $\frac{dy}{dx}$, therefor the topological sort that we traverse in reverse.

### Matrix multiply

Now what we mostly do in computer vision or machine learning in general is multiply matrices. This is because weights of layers (also called neurons) are most effectively stored in a matrix. 

Let's say we have the following input
$$
\begin{pmatrix}
x_{11} & x_{12} & x_{13} \\
x_{21} & x_{22} & x_{23} \\
\end{pmatrix}
$$
where $x_{11}$ is the first feature of the first sample and $x_{12}$ is the second feature of the first sample. We want to multiply this effectively by multiple neurons. Let's sketch the case of a single neuron with weights $w_{11}, w_{21},  w_{31}$ which we obviosuly need because we need a weight for each feature (columns in input matrix).
$$
\begin{pmatrix}
x_{11} & x_{12} & x_{13} \\
x_{21} & x_{22} & x_{23} \\
\end{pmatrix}
\times
\begin{pmatrix}
w_{11} \\
w_{21} \\
w_{31} \\
\end{pmatrix}
$$
Now this would not be very powerful. We basically weight each input only once and output it into a $2 \times 1$ matrix. So we need multiple neurons which make up the columns of our weight matrix.
$$
\begin{pmatrix}
x_{11} & x_{12} & x_{13} \\
x_{21} & x_{22} & x_{23} \\
\end{pmatrix}
\times
\begin{pmatrix}
w_{11} & w_{12} & w_{13}\\ 
w_{21} & w_{22} & w_{23}\\
w_{31} & w_{32} & w_{33}\\
\end{pmatrix}
$$
Now we have a litle bit more power. We have three different neurons (the columns of $W$) with their own weight (rows in $W$) for each input feature (columns) in $X$. This is now a simple matrix multiplication to get the intermediate representation in the network.

For a standard linear layer we would add on a bias matrix $B$. 
$$
\begin{pmatrix}
x_{11} & x_{12} & x_{13} \\
x_{21} & x_{22} & x_{23} \\
\end{pmatrix}
\times
\begin{pmatrix}
w_{11} & w_{12} & w_{13}\\ 
w_{21} & w_{22} & w_{23}\\
w_{31} & w_{32} & w_{33}\\
\end{pmatrix}
+
\begin{pmatrix}
b_{1} & b_{2} & b_{3}\\ 
\end{pmatrix}
$$
Interestingly enough this bias which has dim $(out\_features, )$ is broadcasted over the rows and not the columns (across the left most dimension). [**Sidenote:** this is actually always the case for numpy - it matches the right most dimension and broadcasts across the leftmost.] This is exactly what we want since each neuron, not each feature, gets its own bias.

Back to computational graphs and the chain rule! We want to figure out how much influence for example $w11$ had on the output (which usually would be a loss $L$ in this setting and therefore is noted as $\frac{dL}{dw_{11}}$). 
$$
\begin{pmatrix}
x_{11} & x_{12} & x_{13} \\
x_{21} & x_{22} & x_{23} \\
\end{pmatrix}
\times
\begin{pmatrix}
w_{11} & w_{12} & w_{13}\\ 
w_{21} & w_{22} & w_{23}\\
w_{31} & w_{32} & w_{33}\\
\end{pmatrix}
= 
\begin{pmatrix}
o_{11} & o_{12} & o_{13}\\ 
o_{21} & o_{22} & o_{23}
\end{pmatrix}
=
\begin{pmatrix}
x_{11}w_{11} + x_{12}w_{21} + x_{13}w_{31} & ... & ...\\ 
x_{21}w_{11} + x_{22}w_{21} + x_{23}w_{31} & ... & ...
\end{pmatrix}
$$
As we can see $w_{11}$ has influence on the output $O$ in $o_{11}$ and $o_{12}$ via $x_{11}$ and $x_{21}$. Since it is multiplication the influence of $w_{11}$ on $O$ ($\frac{dO}{dw_{11}}$) is exactly $x_{11} + x_{21}$. We also want to incorporate any gradient that  $O$ has on the outcome $L$ which in this specific case would be gradients in the form $2 \times 3$. 

So let's quickly get order into this: We have $X$ which is $2 \times 3$ and $\frac{dL}{dO}$ which is another matrix with dimensions $2  \times 3$. We want the gradients $\frac{dL}{do_{11}}$ and   $\frac{dL}{do_{21}}$ multiplied by $x_{11}$ and $x_{21}$ respectively. And another constraint is that the matrix with the gradients for $W$ needs to be $3 \times 3$. 

The solution is to transpose the input matrix ($X^T$) and multiply it with the matrix $\frac{dL}{dO}$ which results in a $(3 \times 2) * (2 \times 3)$ matrix multiplication, so a $3 \times 3$ matrix as an output with the exact multiplication that we want.
$$
\frac{dL}{dW} = X^T * \frac{dL}{dO}
$$
But we also want to know the gradients for the matrix $X$ ($\frac{dL}{dX}$) because $X$ does not have to be an input, it could also be intermediate activations inside the hidden layer (if we assume a dense network). Again we need to incorporate the gradients of $O$ based on the chain rule: $\frac{dL}{dX} = \frac{dL}{dO} * \frac{dO}{dX}$. As above we assume we have them (via topological computation of the gradients these have been computed before we arrived here). If we look again at one specific value such as $x_{11}$ we can see that it influenced $o_{11}, o_{12}, o_{13}$ via $w_{11}, w_{12}, w_{13}$ respectively. To get these multiplied correctly and also get an output matrix of dim $2 \times 3$ we just multiply the gradients $\frac{dL}{dO}$ with the transposed weight matrix ($W^T$). 
$$
\frac{dL}{dX} = \frac{dL}{dO} * W^T
$$
Now we have the gradients for both $X$ and $W$ and all we need is some matrix multiplies. Magic!

### Broadcasting issues in other computations

There are a lot of other computations we can do with matrices that we need to backprop through but most of them are pretty straightforward to implement. Adding the bias for example is relatively simple in theory. The gradient of an addition is 1. So we just calculate $\frac{dL}{dO} * 1$. Sometimes (for example in the bias case as can be seen above) we have some broadcasting which we need to "reverse". The bias is dimension $(out\_features, )$ which then gets broadcasted to be $(n, out\_features)$, so the bias influences all the rows in the weight matrix for the neuron it is targeting. This also means that the bias influences $n$ times and not only once. We therefore need to sum over this dimension to get back to our $(out\_features, )$ dimensional matrix for the gradients.

### Softmax

We need softmax constantly. I like the interpretation that it is a "smooth arg max" [[from Wikipedia]](https://en.wikipedia.org/wiki/Softmax_function) most. It is also theoretically very simple: we exponeniate all the values and then divide them by the sum over which ever axis we care about. This would not warrant a seperate header if there wasn't a catch: overflow and underflow in floating point arithmetic. We therefore need to subtact the maximum value of each axis first, to prevent exponentiating already large numbers. This can be treated as a contant since it does not change the softmax result itself and we don't need to backprop through it.

### Cross Entropy Loss

Another example of something theoretically simple but with computational issues. Usually cross entropy loss is calculated like this:

```python
probs = softmax(logits)
loss = -sum(targets * log(probs))
```

Why? Because ```log(p)``` penalizes confident wrong predictions more then it would without the logarithm (since ```log(0.01) = 4.6``` which is much higher then the simple implemetation of absolute difference betweeen target and probability).

Coming back to the overflow and underflow issues: Softmax creates probabilities that are very small for some of the classes. When we then take the ```log()``` of that we run the risk of getting $nan$. We therefore need to write another implementation that has log already in mind.

```python
log(softmax(x)) = log(exp(x) / sum(exp(x)))
                = log(exp(x)) - log(sum(exp(x)))
                = x - log(sum(exp(x)))
```

Why does that work? ```log(exp(x) / sum(exp(x)))``` can be split into ```log(exp(x)) - log(sum(exp(x)))``` because of the log rules ($log(\frac{a}{b}) = log(a) - log(b)$). ```log(exp(x))``` then simplifies to ```x``` because we take the logarithm of the exponent, so it stays the same.  We utilize the same trick as we did in softmax and subtract the maximum per axis first.



