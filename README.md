# CNN-multi-class-classifier
CNN model implement with Pytorch



Convolutional NN model implemented with Pytorch.

instructions:

1. download the data.tar.gz from the address specified in the address folder.
2. extract the train valid and test folders to the path where the .py files.
3. run the model.py file and get the test_y file with the predictions.



**about the model:**
 the model classifying 30 class by **.wav** 1 second length provided files.
 the train is made with 100 instances batching and shuffling as well as the validation.
 
 
**architecture:**
 each instance passes 2 convolutional layers with 3X3 kernel size after each convolution.
 the output is passing through ReLu activition function after each layer.
 after the convolutionl we are applying max pooling in order to minimize the size.
 for the last step we will pass 4 fully connected linear layers.
