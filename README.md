# capsnet-tensorflow
Capsule net tensorflow
A simple tensorflow implementation of Capsnet on Hitton paper [Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829)
In the paper, Geoffrey Hinton introduces Capsule for:
>A capsule is a group of neurons whose activity vector represents the instantiation
parameters of a specific type of entity such as an object or object part. We use the
length of the activity vector to represent the probability that the entity exists and its
orientation to represent the instantiation paramters. Active capsules at one level
make predictions, via transformation matrices, for the instantiation parameters of
higher-level capsules. When multiple predictions agree, a higher level capsule
becomes active.
## Capsnet architecture
![Capnet architecture](https://image.jiqizhixin.com/uploads/wangeditor/77f966b8-eba1-4737-a4c2-551e0d4e7610/52501image%20(15).png)
There are three key layer in capsnet. 
- ### Convolution Layer with ReLU is a traditional CONV Layer. <br>
`conv1 = tf.contrib.layers.conv2d(self.image, num_outputs=256, kernel_size=9, stride=1, padding='VALID')`<br>
input tensor size: `[bathsize, 28, 28, 1]`, output tesnor size: `[batchsize, 20, 20, 256]`
- ### PrimaryCaps<br>
input tensor size: `[bathsize, 20, 20, 256]`, output tesnor size: `[batchsize, 1152, 8, 1]`<br>
The detail of PrimaryCaps as follow:<br>
![primarycaps](https://image.jiqizhixin.com/uploads/wangeditor/77f966b8-eba1-4737-a4c2-551e0d4e7610/01507v2-a402c9d94e36bc6bdcf8a6fbe06f78b3_hd.jpg)<br>
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
$ \textbf{v}_j $
Since one Capsule unit is the 8D vector not like traditional scalar, how compute the vector inputs and outputs of capsule is the KEY. 
Capsule's input vector is equivalent to the scalar input of a neuron in a classical neural network, which is equivalent to the propagation and connection between two Capsules. 
The calculation of input vector is divided into two stages, namely linear combination and Routing, this process can be expressed by the following formula:
Dynamic routing is a kind of algorithm between the PrimaryCaps and DigitCaps to propagae "vevtor in vector output", propagation process as follow:
 ![routing](https://image.jiqizhixin.com/uploads/wangeditor/77f966b8-eba1-4737-a4c2-551e0d4e7610/55610%E7%BB%98%E5%9B%BE1.jpg)
