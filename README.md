[![Neural_Networks_Stock_PredictorImage](https://github.com/benjaminweymouth/Neural_Networks_Stock_Predictor/blob/main/Resources/RNNsmaller2.png)](https://github.com/benjaminweymouth/Neural_Networks_Stock_Predictor/blob/main/lstm_stock_predictor_closing.ipynb)

# Deep Learning: Neural Networks BitCoin Predictor

This repo will compare two different deep learning recurrent neural networks to model bitcoin closing prices. Specifically, this GitHub repository highlights 2 Python Jupyter notebooks, for building and evaluating the two deep learning models. 

### Live Deployed Site (Full Comparative Analysis) 

https://benjaminweymouth.github.io/Neural_Networks_Stock_Predictor/ 

### LSTM Model 

RNN, or Recurrent Neural networks provide an analysis typically based on modelling sequence data, thanks to their sequential memory. LSTM (Long Short-Term Memory) RNNs are one solution for longer time windows. An LSTM RNN works like an original RNN, but it selects which types of longer-term events are worth remembering, and which can be discarded.

### Live Site for LSTM Analysis 

https://benjaminweymouth.github.io/Neural_Networks_Stock_Predictor/LSTMRNN/

Here are some screenshots: 

[![image](https://user-images.githubusercontent.com/47256041/153524565-e2682597-eb6b-46a8-ac82-2c076d7c85ea.png)](https://benjaminweymouth.github.io/Neural_Networks_Stock_Predictor/LSTMRNN/)

Loading the Two Datasets for Comparative Analysis 
[![image](https://user-images.githubusercontent.com/47256041/153524710-c0f8be36-4ee8-4139-aeb1-5a7bd32c1dad.png)](https://benjaminweymouth.github.io/Neural_Networks_Stock_Predictor/LSTMRNN/)

 
Building and Training the LSTM / RNN 

[![image](https://user-images.githubusercontent.com/47256041/153524977-daf15a52-a243-47ea-9919-a1b79d3bd239.png)](https://benjaminweymouth.github.io/Neural_Networks_Stock_Predictor/LSTMRNN/)

 ### FNG Model
This model will use the FNG indicators to predict the closing price while the second model will use a window of closing prices to predict the nth closing price. They use the FNG index values and simple closing prices to determine if the FNG indicator provides a better signal for cryptocurrencies than the normal closing price data.

### Live Site for FNG Analysis 

https://benjaminweymouth.github.io/Neural_Networks_Stock_Predictor/FNG/

Screenshots for the FNG Analysis: 

![image](https://user-images.githubusercontent.com/47256041/153525457-941d41a5-5f8c-4032-a4e7-cb0fd7ad604d.png)

Loading the Historical Prices for BitCoin 

![image](https://user-images.githubusercontent.com/47256041/153525500-e87d638c-bcff-4477-92aa-1cf23afdd28a.png)

Training the Model (with a Batch Size of 5) 

![image](https://user-images.githubusercontent.com/47256041/153525592-d54d5d1e-fa8d-46ac-a629-9024dc095cef.png)

Evaluating the Model using the Test Data 

![image](https://user-images.githubusercontent.com/47256041/153530468-f426df05-ffde-40ed-bc40-c18b4850518b.png)



## Conclusions & Analysis (LSTM VS. FNG) 

This is the final visualization from this analysis. However, let us dig deeper with a few questions and answers about what this comparative analysis reveals. 

![image](https://user-images.githubusercontent.com/47256041/153524367-922d66e6-80e0-4f1e-8249-6162ebb5e740.png)


### Question: Which model has a lower loss?
Answer: The model for the lstm_stock_predictor_closing has a significantly lower loss. 
### Question: Which model tracks the actual values better over time?
Answer: The model for the lstm_stock_predictor_closing tracks the actual values better over time
### Question: Which window size works best for the model?
Answer: A lower window size works much better. Specifically for the lstm_stock_predictor_closing model, setting the window_size = 2 worked well.  
