[![Neural_Networks_Stock_PredictorImage](https://github.com/benjaminweymouth/Neural_Networks_Stock_Predictor/blob/main/Resources/RNNsmaller2.png)](https://github.com/benjaminweymouth/Neural_Networks_Stock_Predictor/blob/main/lstm_stock_predictor_closing.ipynb)

# Deep Learning: Neural Networks BitCoin Predictor

This repo will compare two different deep learning recurrent neural networks to model bitcoin closing prices. Specifically, this GitHub repository highlights 2 Python Jupyter notebooks, for building and evaluating the two deep learning models. 

### LSTM Model 

RNN, or Recurrent Neural networks provide an analysis typically based on modelling sequence data, thanks to their sequential memory. LSTM (Long Short-Term Memory) RNNs are one solution for longer time windows. An LSTM RNN works like an original RNN, but it selects which types of longer-term events are worth remembering, and which can be discarded.

Here are some screenshots: 

![image](https://user-images.githubusercontent.com/47256041/153524565-e2682597-eb6b-46a8-ac82-2c076d7c85ea.png)](https://benjaminweymouth.github.io/Neural_Networks_Stock_Predictor/LSTMRNN/)


### FNG Model
This model will use the FNG indicators to predict the closing price while the second model will use a window of closing prices to predict the nth closing price. They use the FNG index values and simple closing prices to determine if the FNG indicator provides a better signal for cryptocurrencies than the normal closing price data.

### Live Deployed Site

https://benjaminweymouth.github.io/Neural_Networks_Stock_Predictor/ 



## Conclusions & Analysis (LSTM VS. FNG) 

This is the final visualization from this analysis. However, let us dig deeper with a few questions and answers about what this comparative analysis reveals. 

![image](https://user-images.githubusercontent.com/47256041/153524367-922d66e6-80e0-4f1e-8249-6162ebb5e740.png)


### Question: Which model has a lower loss?
Answer: The model for the lstm_stock_predictor_closing has a significantly lower loss. 
### Question: Which model tracks the actual values better over time?
Answer: The model for the lstm_stock_predictor_closing tracks the actual values better over time
### Question: Which window size works best for the model?
Answer: A lower window size works much better. Specifically for the lstm_stock_predictor_closing model, setting the window_size = 2 worked well.  
