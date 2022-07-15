# Minor1: Image Classification on MNIST dataset
Image classification is the primary domain, in which deep neural networks play the most important role in digit recognition(MNIST) analysis. The image classification accepts the given input images and produces output classification for identifying numbers in range 0-9.

### Objective
To train a neural network to classify different digits in MNIST dataset using C++. 

The field of Deep Learning is vast and we want to contribute to it. The best way to learn is to build it from scratch. With this project we are implementing a Deep neural network from scratch on MNIST dataset in C++.

The Goal of our project is to implement an Image Classification model from scratch with accuracy of above 90%.

# **Dataset**

Dataset used is MNIST [3] dataset(MNIST database is of handwritten digits, with training set of 60,000 images and test set of 10,000 images) Each image in MNIST has 1 colour Channel and size of 28x28 pixels [1,28,28]

 ![image](https://user-images.githubusercontent.com/55954820/179272380-3a25a54f-871b-48df-a555-08144a3ff326.png)

## Architecture design:
the ﬁrst layer is given by :      
###             h(1)= g(1)  (W(1)Tx + b(1) );

the second layer is given by :          
### h(2)= g(2) (W(2)Th(1)+ b(2)); 

Our Neural Network Architecture is:
### 784->32->10
*number represents the nodes in particular layer.

## Design Diagram
![image](https://user-images.githubusercontent.com/55954820/179272262-165f85a3-ac3f-40e7-8952-6386c38b6c7f.png)


## Result
![image](https://user-images.githubusercontent.com/55954820/179272070-01580f69-e758-49f5-bb19-a9be8df07c19.png)


