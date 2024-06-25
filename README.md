<h1 align="center">Sentiment Analysis Using Recurrent Neural Networks</h1>
<h2 align="center">Introduction</h2>

<img  align="right" height="290" src="https://www.aimtechnologies.co/wp-content/uploads/2023/08/Arabic-Sentiment-Analysis-2.jpg" />

- 👨‍💻This project implements a sentiment analysis model for movie reviews using RNNs built with Python's libraries.

-  🔭 Sentiment analysis, also known as opinion mining, is a subfield of Natural Language Processing (NLP) that aims to automatically detect the emotional tone of a piece of text.

- ⚡This project demonstrates proficiency in deep learning, data manipulation, text preprocessing, NLP, model building, evaluation and handling sequential data.


<h2 align="center">Requirements</h2>

- Python 3.x
- pandas
- numpy
- Regular expressions (re)
- NLTK
- Scikit-learn
- Tensorflow
- Scipy


  
<h2 align="center">Text Preprocessing</h2>

- After loading the data, perform preprocessing techniques like HTML tag removal, punctuation removal, lowercasing, stop word removal using libraries nltk, re and spacy.

- Make Tokens for the preprocessed text using Tokenizer. The Tokenizer class from Keras tokenizes the text, converting words into sequences of integer indices.

- Sequences are padded with zeros to ensure a fixed length for all reviews using pad_sequences.

- Sentiment labels ("positive" and "negative") are converted to numerical labels (0 and 1) using LabelEncoder.

<h2 align="center">Model Building and Training </h2>

- Complex Bidirectional LSTM-GRU has been used in model building. This is a more complex model with multiple stacked Bidirectional LSTM and GRU layers with dropout regularization. Total trainable parameters were 37379651.

- The data is split into training and testing sets using train_test_split.
- Each model is compiled with the Adam optimizer, binary cross-entropy loss function, and accuracy metric.
- Models are trained for a specified number of epochs and batch size.
- Training and validation accuracy and loss are monitored during training.
  
<h2 align="center">Working</h2>

- The code demonstrates how to use the trained model to predict the sentiment of new movie reviews.

- Text pre-processing steps are applied to the new review text given as input.
- The preprocessed text is converted to a sequence of integers using the fitted tokenizer.
- The sequence is padded to match the maximum review length.
- The model predicts the sentiment probability (positive or negative) for the review.
- Based on the predicted probability, the review is classified as "flop movie", "average movie", "good movie", or "blockbuster movie".
- Overall this this code provides a basic framework for sentiment analysis using RNNs. You can experiment with different RNN architectures, hyperparameters.

<h2 align="center">Use Cases</h2>

- Analyze customer reviews and social media posts to understand customer satisfaction with products, services, and brand interactions.
- Identify spam emails by analyzing the language and sentiment used in the message content.
- Track brand sentiment across social media platforms to identify potential issues or crises early on.
- Analyze employee satisfaction surveys and feedback forms to identify potential morale issues or areas for improvement in the workplace.
- Analyze student feedback on online courses to understand their learning experience and identify areas for improvement in course content and delivery.
