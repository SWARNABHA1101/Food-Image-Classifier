# Food-Image-Classifier
Food Image classification is the process of taking an input like “image”and output us in the form of class like (“food”,”non-food”) or a probability of the particular class (there’s is a probability of 95% that it belongs to class food).

Goal: Building a machine learning model that can distinguish between food and non-food class using CNN for given an input of image.

1)In this model I have used Tensorflow, Keras, and
pretrained MobileNetV2 CNN to make our predictions.

2)Firstly, uploaded the dataset into Kaggle then imported
that into google collab, and then unzipped it and
performed the task.

Input Image→Feature Extraction→Classification→
Output.

3)To use a pre-trained model like VGG-16 without having
to master the skills necessary to tune and train those
models which have been already trained with lots of
images as we can directly use the weights and architecture
obtained and apply the learning on our problem
statement. This method is also known as transfer
learning. We “transfer the learning” of the pre-trained
model to our specific problem statement.

4)The main reason for splitting the dataset — to evaluate
the performance of the classifier. We are interested in how well
the classifier generalizes its recognition capability to unseen data.

5)The main aim of the cross-validation set is to check how
well the model recognizes the pattern on the unseen data in the
training phase itself ,if it has any problems like “overfitting” and
“underfitting”.

6)So,finally our dataset is divided into 70% into Train dataset,10%
into Cross-validation dataset , and remaining 20% into Test
dataset by using sklearn train_test_split() function.
