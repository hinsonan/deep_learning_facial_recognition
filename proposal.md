## Team Name: Apollo

## Project Title: Emotional Face Detection 

<strong>Project summary</strong>

The problem we will address is facial detection and emotional classification based on facial expression. Detecting and predicting people's emotions is a challenging and rewarding project because being able to detect emotions is one small step toward being able to learn more about human emotions and AI. The goal will be to take an image and detect the face in the image then send that detected face through another model to classify the emotion. For this project we will use a pretrained model for face detection and build a model for emotional classification. 

<strong>Approach</strong>

We will utilize a pre-trained MTCNN (Multi-Task Cascaded Convolutional Network) to detect faces in images. These faces will then be classified using a model that is built from the ground up. The main part of this project will focus on the emotional classification network that will be built. Experiments such as Hyper param tuning and model topology will be done in order to see what works best. Different normalization techniques will be used on the data to see what works best. Some experiments will be done on the pre-trained detection model to see what type of data it struggles with and how that affects the project.

Resources / Related Work & Papers
MTCNN is a paper that talks about face detection. This model is available through many python libraries. We will use one of these libraries that contain this model already trained. EfficientNetL2, ResneXt, and VGG are state of the art models that are used to classify images. We will use concepts from these to implement our emotional face classification model.

<strong>Datasets</strong>

Emotional Faces in RGB https://www.kaggle.com/amitprajapati191978/mood-detection?select=Happy
Emotional Faces Greyscale https://www.kaggle.com/sudarshanvaidya/random-images-for-face-emotion-recognition?select=contempt

List your Group members

Andrew Hinson

Christopher Austin