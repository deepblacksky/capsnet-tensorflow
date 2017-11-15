# capsnet-tensorflow
Capsule net tensorflow
A simple tensorflow implementation of Capsnet on Hitton paper [Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829)
In the paper, Geoffrey Hinton introduces Capsule for:
>A capsule is a group of neurons whose activity vector represents the instantiation parameters of a specific type of entity such as an object or object part.
We use the length of the activity vector to represent the probability that the entity exists and its
orientation to represent the instantiation paramters. Active capsules at one level
make predictions, via transformation matrices, for the instantiation parameters of
higher-level capsules. When multiple predictions agree, a higher level capsule becomes active.

## Capsnet architecture
![Capnet architecture](https://github.com/deepblacksky/capsnet-tensorflow/blob/master/images/arch.png)
There are three key layer in capsnet.
- ### Convolution Layer with ReLU is a traditional CONV Layer. <br>
`conv1 = tf.contrib.layers.conv2d(self.image, num_outputs=256, kernel_size=9, stride=1, padding='VALID')`<br>
input tensor size: `[bathsize, 28, 28, 1]`, output tesnor size: `[batchsize, 20, 20, 256]`
- ### PrimaryCaps<br>
input tensor size: `[bathsize, 20, 20, 256]`, output tesnor size: `[batchsize, 1152, 8, 1]`<br>
The detail of PrimaryCaps as follow:<br>
![primarycaps](https://github.com/deepblacksky/capsnet-tensorflow/blob/master/images/primaryCaps.jpg)<br>
The primarycaps layer can be viewed as performing 8 Conv2d operations of different weights on an input
tensor of dimensions 20 x 20 x 256,
each time operation is a Conv2d with `num_output=32, kernel_size=9, stride=2`.<br>
Since each convolution operation generates a `6 × 6 × 1 × 32` tensor and generates a total of 8 similar tensors,
the 8 tensors (that is, the 8 components of the Capsule input vector) combined dimensions become `6 × 6 × 8 × 32`.<br>
PrimaryCaps is equivalent to a normal depth of 32 convolution layer,
except that each layer from the previous scalar value into a length of 8 vector.
- ### DigitCaps<br>
input tensor size: `[batchsize, 1152, 8, 1]`, output tesnor size: `[batch_size, 10, 16, 1]`<br>
The output of PrimaryCaps is 6x6x32=1152 Capsule units, and every Capasule unit is 8 length vector.
The DigitCaps has 10 standard Capsule units, each of the Capsule's output vectors has 16 elements.
In a word, DigitCaps rounting an updating based on the output vector of last layer.
The length of output vector is the probability of the class.

## Sqaushing and Routing
The length of the output vector of a capsule represents the probability that the entity represented by the capsule is present in the current input.
So its value must be between 0 and 1.
To achieve this compression and to complete the Capsule-level activation, Hinton used a non-linear function called "squashing."
This non-linear function ensures that the length of the short vector can be shortened to almost zero, while the length of the long vector is compressed to a value close to but not more than one.
The following is an expression of this non-linear function:
<center>![squashing](https://github.com/deepblacksky/capsnet-tensorflow/blob/master/images/squashing.png)</center>
where v_j is the output vector of Capsule j, s_j is the weighted sum of all the Capsules output by the previous layer to the Capsule j of the current layer, simply saying that s_j is the input vector of Capsule j.<br>
Since one Capsule unit is the 8D vector not like traditional scalar, how compute the vector inputs and outputs of capsule is the KEY.
Capsule's input vector is equivalent to the scalar input of a neuron in a classical neural network, which is equivalent to the propagation and connection between two Capsules.<br>
The calculation of input vector is divided into two stages, namely linear combination and Routing, this process can be expressed by the following formula:
<center>![](https://github.com/deepblacksky/capsnet-tensorflow/blob/master/images/digit.png)</center>
After determining u_ji hat, we need to use Routing to allocate the second stage to compute output node s_j, which involves iteratively updating c_ij using dynamic routing. Get the next level of Capsule's input s_j through Routing, and put s_j into the "Squashing" non-linear function to get the output of the next Capsule.<br>
Routing algorithm as follow:
<center>![](https://github.com/deepblacksky/capsnet-tensorflow/blob/master/images/Routing_alg.png)</center>
Therefore, the whole level of propagation and distribution can be divided into two parts, the first part is a linear combination of u_i and u_ji hat, the second part is the routing process between u_ji hat and s_j.<br>
Dynamic routing is a kind of algorithm between the PrimaryCaps and DigitCaps to propagae "vevtor in vector output", propagation process as follow:
![routing](https://github.com/deepblacksky/capsnet-tensorflow/blob/master/images/routing.jpg)

## Loss and Optimal
In the paper, the author used Margin loss which is commonly used in SVM. The expression of loss function is:
<center>![](https://github.com/deepblacksky/capsnet-tensorflow/blob/master/images/loss.png)</center>
where T_c = 1 if a digit of class c is present and m_plus = 0.9 and m_minus = 0.1 and lambda = 0.5(suggest). The total loss is simply the sum of the losses of all digit capsules.<br>
The paper used Reconstruction to regularization and improve accuracy.
![](https://github.com/deepblacksky/capsnet-tensorflow/blob/master/images/recong.png)
In addition to improving the accuracy of the model, Capsule's output vector reconstruction improve the interpretability of the model because we can modify the reconstruction of some or all of the components in the reconstructed vector and observe the reconstructed image changes which helps us to understand the output of the Capsule layer.<br>
So, the total loss function is:
<center>![](https://github.com/deepblacksky/capsnet-tensorflow/blob/master/images/total_loss.png)</center>

where, lambda_loss is 0.00005 so that it does not dominate the
margin loss during training.

## Result
- ### mnist
*batchsize : 64<br>
epoch: 30*<br><br>
**train loss:**
![](https://github.com/deepblacksky/capsnet-tensorflow/blob/master/images/result_loss_1.png)
![](https://github.com/deepblacksky/capsnet-tensorflow/blob/master/images/result_loss_2.png)
![](https://github.com/deepblacksky/capsnet-tensorflow/blob/master/images/result_loss_3.png)
<br><br>
**accuracy:**<br>
![](https://github.com/deepblacksky/capsnet-tensorflow/blob/master/images/accuracy.png)

## TODO
- [ ] Capsnet on [fashion-mnist](https://github.com/zalandoresearch/fashion-mnist)
- [ ]Capsnet on [cifar10](http://www.cs.toronto.edu/~kriz/cifar.html)
- [ ]Modify network architecture

*Reference:<br>
https://www.jiqizhixin.com/articles/2017-11-05<br>
https://github.com/naturomics/CapsNet-Tensorflow*
