[![Neural_Networks_Stock_PredictorImage](https://github.com/benjaminweymouth/Neural_Networks_Stock_Predictor/blob/main/Resources/RNNsmaller2.png)](https://github.com/benjaminweymouth/Neural_Networks_Stock_Predictor/blob/main/lstm_stock_predictor_closing.ipynb)

# Deep Learning: Neural Networks BitCoin Predictor
This GitHub repository highlights 2 Python Jupyter notebooks. They are for building and evaluating deep learning models. They use the FNG index values and simple closing prices to determine if the FNG indicator provides a better signal for cryptocurrencies than the normal closing price data.

## Conclusions & Analysis

### Question: Which model has a lower loss?
Answer: The model for the lstm_stock_predictor_closing has a significantly lower loss. 
### Question: Which model tracks the actual values better over time?
Answer: The model for the lstm_stock_predictor_closing tracks the actual values better over time
### Question: Which window size works best for the model?
Answer: A lower window size works much better. Specifically for the lstm_stock_predictor_closing model, setting the window_size = 2 worked well.  
