[![Neural_Networks_Stock_PredictorImage](https://github.com/benjaminweymouth/Neural_Networks_Stock_Predictor/blob/main/Resources/RNNsmaller2.png)](https://github.com/benjaminweymouth/Neural_Networks_Stock_Predictor/blob/main/lstm_stock_predictor_closing.ipynb)

# Deep Learning: Neural Networks BitCoin Predictor

This repo will compare two different deep learning recurrent neural networks to model bitcoin closing prices. Specifically, this GitHub repository highlights 2 Python Jupyter notebooks, for building and evaluating the two deep learning models. 

### LSTM Model 

### FNG Model
One model will use the FNG indicators to predict the closing price while the second model will use a window of closing prices to predict the nth closing price. They use the FNG index values and simple closing prices to determine if the FNG indicator provides a better signal for cryptocurrencies than the normal closing price data.

## Conclusions & Analysis

### Question: Which model has a lower loss?
Answer: The model for the lstm_stock_predictor_closing has a significantly lower loss. 
### Question: Which model tracks the actual values better over time?
Answer: The model for the lstm_stock_predictor_closing tracks the actual values better over time
### Question: Which window size works best for the model?
Answer: A lower window size works much better. Specifically for the lstm_stock_predictor_closing model, setting the window_size = 2 worked well.  
