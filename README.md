# LocationPredictor
Predicting Location, Based on Wifi Signal Strength

The setup consists of 4 hotel rooms and 7 wifi access points. All the 7 wifi APs are accessible from each room, but with different signal strengths. The signal strengths are in dBm. The task is to train a model which can accurately predict the room number, given the strengths of wifi signals from all the seven APs.

# Evaluation metric
Categorisation accuracy.<br>
Categorisation accuracy is basically the same as sklearn.accuracy_score(), which calculates (the number of correct prediction)/(total number of prediction).

# File Description
train.csv : the training data<br>
test.csv : testing data<br>
sampleSubmission.csv : sample submission format<br>

# Data Field Description
Columns 1-7 : Wifi signal strength in dBm<br>
Column 8 : room number from where the reading was made

# [Contest Link](https://www.kaggle.com/c/logicalrhythm18-wifi/)

# Result
Rank : 3<br>
Accuracy : 98.666%
