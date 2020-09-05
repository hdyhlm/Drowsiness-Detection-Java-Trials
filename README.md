# Java-Based-Drowsiness-Detection

# Description
This capstone project is to detect drowsiness of a driver and provide assistance e.g: alarm and suggestion.
This is just a trial for java project, as we progress in building the model and algorithm for the java, we end up having difficulites in completing the project due to time and resources. Due to that reasons, we have move on to Python as our platform to complete the project: https://github.com/hzchng/drowsiness-detector.

However, we will provide and possibily continue in updating the code so it would be a good references for us and community to explore on using java and DL4j as the platform for this projects.

# Model of the project
In this java project, we have tried on using LSTM to train and test the features that we have extracted from the dataset. The features extraction was performed using JavaCV to extract the landmark of the face point. From these collected features, we calculate the Eye Aspect Ratio (EAR), Circularity, Mouth Aspect Ratio (MAR) and lastly, Mouth Over Eye (MOE). From these calculation, we feed it into the LSTM model to calculate the energy level of the driver to classify whether the driver is drowsy or alert .

# Dataset
UTA Real-Life Drowsiness Dataset.
Datasets used:https://sites.google.com/view/utarldd/home

The dataset used for training the model are taken from the link. However, the download link is unaccessable. We managed to download one folder of the dataset before the link is gone. The folder have 12 participants in total and each of the folder has 3 videos that classify between 0 (as alert), 5 (as non-vigilant) and 10 (as drowsy).

# P/S
We unable to upload our face landmark model, but worry not, here is the link to directly download it.
face_landmark_model.dat: https://drive.google.com/file/d/1hjjUgxZz1IGYA_7rNI7ZetDkpUs4pThW/view?usp=sharing

Have good times in improving (or actually fixing) the code.
:)

By Ch'ng Hou Zhi, Tan Yeung How, Mohd Hazri Muhd Ridhwan, Nurhidayah halim
