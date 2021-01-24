# CSCI3230-Drug-Toxicity-Prediction-Project
- Convolutional Neural Network for detecting drug toxicity
- Drug Input: Simplified Molecular-Input Line-Entry System (SMILES) Expression 
- Example:

![alt text](https://github.com/div1121/CSCI3230-Drug-Toxicity-Prediction-Project/blob/main/example.JPG)

# Implementation
1. The SMILES Expression of drug is transformed into one-hot encoding.
2. Model Representation: implement 4 different architecture of CNN models (using TensorFlow 1.15.4)

Model 1

![alt text](https://github.com/div1121/CSCI3230-Drug-Toxicity-Prediction-Project/blob/main/model1.JPG)

Model 2

![alt text](https://github.com/div1121/CSCI3230-Drug-Toxicity-Prediction-Project/blob/main/model2.JPG)

Model 3

![alt text](https://github.com/div1121/CSCI3230-Drug-Toxicity-Prediction-Project/blob/main/model3.JPG)

Model 4

![alt text](https://github.com/div1121/CSCI3230-Drug-Toxicity-Prediction-Project/blob/main/model4.JPG)

3. Early stopping is used in training progress.

# Conclusion
- After testing, model 4 achieve the best result for predicting drug toxicity without overfitting, 
- Model 4 is used to build the final version of model in final_version folder.
- Some pretrained models are saved at model folder.
