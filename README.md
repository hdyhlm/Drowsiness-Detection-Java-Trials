# Drowsiness-Detection-Java-Trials

# Description
This project is to detect drowsiness of a driver and provide assistance e.g: alarm and suggestion
This is just a trial for java project, and as we progress in building the model and algorihtm for the java, we end up having difficulites in completing the project using java due to time and resources. Due to that reasons, we have move on to using Python as our platform to complete the project.

However, we will provide and possibily continue in updating the code so it would be a good references for us and community to explore on using java an DL4j as the platform for this projects.

# Model of the project
In this java project, we have tried on using LSTM as our model to train and test the features that we have extracted from the dataset. The features extraction was performed using JavaCV to extract the landmark of the face using point. From these collected features, we used it to calculate the Eye Aspect Ratio (EAR), Circularity, Mouth Aspect Ratio (MAR) and lastly, Mouth Over Eye (MOE). From these calculation, we will feed it into the LSTM model to calculate the energy level of the driver to classify whether the driver is drowsy or alert .

# Dataset
UTA Real-Life Drowsiness Dataset
Datasets used:https://sites.google.com/view/utarldd/home

The dataset we used for training the model are taken from the link. However, the download link is unaccessable. We managed to download one folder of the dataset before the link gone and the folder have 12 participants folder and each of the folder has 3 videos that classify between 0 (as alert), 5 (as non-vigilant) and 10 (as drowsy).


By Ch'ng Hou Zhi, Tan Yeung How, Mohd Hazri Muhd Ridhwan, Nurhidayah halim
