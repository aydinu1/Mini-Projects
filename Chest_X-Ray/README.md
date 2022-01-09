This code develops a binary imagine classifier for the pneumonia using the chset xray data from: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia. 
The model is based on transfer learning. As base model VGG19 is used and all the layers are re-trained again for the best results. 

Final model gives on the test data:\
Recall: 0.98\
Precision: 0.93

