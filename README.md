# Food-Image-Classifier
1)In this model I have used Tensorflow, Keras, and
pretrained MobileNetV2 CNN to make our predictions.
2)Firstly, uploaded the dataset into Kaggle then imported
that into google collab, and then unzipped it and
performed the task
Input Image→Feature Extraction→Classification→
Output
3) To use a pre-trained model like VGG-16 without having
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
5) The main aim of the cross-validation set is to check how
well the model recognizes the pattern on the unseen data in the
training phase itself ,if it has any problems like “overfitting” and
“underfitting”.
6)So,finally our dataset is divided into 70% into Train dataset,10%
into Cross-validation dataset , and remaining 20% into Test
dataset by using sklearn train_test_split() function.
7)Building Machine Learning Model
Constraints while building the model
-Errors can be very costly
-Low latency
-Probability of each class of data set is needed
Key Performance Indicators(KPI):
1. Accuracy(Link)
2. Confusion Matrix(Link)
3.Log Loss(Link)
—--------------------------------------------------------------------------
Step-1- Import Libraries
The initial step that we demand to do is to import libraries. We want
TensorFlow, NumPy, os, and pandas.
Step-2- Preparing the data
Discussed in point (2) above.
We need to produce the dataframe with columns as the image filename and
the label with that folder arrangement.
Step-3- Train the data
After we produce the batches, now we can train the model by the transfer
learning technique. Because we apply that method, we don’t demand to
perform CNN architecture from scratch. Instead, we will use the present and
previously pretrained architecture.
Step-4- Test the Model
We will practice classification_report from the scikit-learn library to produce a
report regarding model execution. Also, we will fancy the confusion matrix
from it.
At last we plot the model on graph and print the classification report of the
model.
