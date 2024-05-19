# Mental-Health-Chatbot

The chatbot is designed to respond to user inputs related to mental health topics by predicting
appropriate tags and providing corresponding responses.

## System Description:
#### Data Processing:
• A JSON file with mental health-related intents is loaded and processed.
• The dataset contains training patterns and their accompanying labels.
• Label encoding is used to translate text labels into numerical formats.
• Training patterns are tokenized and padded before being sent into the neural network.
## Model Architecture
• The model architecture includes an embedding layer, a LSTM layer, and a final dense
layer with softmax activation for multi-class classification.
• The Embedding layer learns dense word embeddings for each word in the lexicon and
maps them to a high-dimensional vector space.
• The Long Short-Term Memory (LSTM) layer is a specialized recurrent neural
network (RNN) layer. LSTMs are designed to capture long-term dependencies in
sequential data, making them particularly suitable for tasks such as natural language
processing (NLP), time series analysis, and speech recognition.
• Dense layers: These layers apply nonlinear changes to the incoming data, capturing
intricate patterns and relationships.
• Softmax activation translates the last layer's output into probability scores, which
indicate the likelihood of each class.
Training
• The model is trained using the compiled architecture and rmsprop optimizer, with
sparse categorical cross-entropy loss.
• Training is carried out over 500 epochs, with the model parameters optimized to
minimize the loss function.
User’s End
• Users can communicate with the chatbot by typing messages.
• The model uses the trained model to predict the appropriate tag for the input message.
• The predicted tag is then utilized to extract a response from the dataset and return a
suitable response to the user.
Techniques Used:
### Natural language processing (NLP):
• Tokenization is the process of breaking down text into individual words or tokens.
• Padding: Making sure that all input sequences are the same length by adding zeros or
truncating as needed.
• Word Embeddings: Representing words in a continuous vector space while capturing
semantic links between them.
Machine Learning:
• Sequential Model: Uses sequential neural network architecture for text classification.
• Label encoding is the conversion of categorical labels into a numerical format for
model training.
• To increase performance, hyperparameter tuning involves experimenting with various
model topologies, embedding dimensions, and learning rates.
• LSTM networks are intended to identify long-term dependencies in sequential data. In
the case of a chatbot, this implies it can successfully interpret and generate responses
based on the context of prior messages, resulting in more coherent and contextually
relevant discussions.
• LSTM networks can handle variable-length input sequences. LSTMs have the ability
to selectively remember or forget information over time, which is crucial for
maintaining context in conversations. The hidden states acquired by LSTM layers can
extract valuable semantic information from incoming text. This allows the model to
learn more meaningful representations of the input data, perhaps yielding more
accurate predictions and responses.
• The vocab size is the number of input features that the model will be learning on and
it needed to be divisible by 16 that’s why addition of 1 was done.
• The number of LSTM units or neurons in the LSTM layer was 32.
• RMSprop is an adaptive learning rate optimization algorithm that adjusts the learning
rate for each parameter based on the magnitude of recent gradients. It is well-suited
for recurrent neural networks like LSTMs.
• Sparse categorical cross-entropy is commonly used for multi-class classification tasks
where the target labels are integers (sparse). It computes the cross-entropy loss
between the predicted probabilities and the true labels.
• Accuracy measures the proportion of correctly classified samples. It is a common
metric for classification tasks but may not be sufficient for evaluating model
performance in imbalanced datasets.
• Dense Layer (Dense(number_of_classes, activation='softmax')): The dense layer at
the output of the model. The dense layer maps the LSTM layer's output to the number
of classes in the classification task. The softmax activation function ensures that the
output probabilities sum to 1, making it suitable for multi-class classification.
Dataset:
Dataset was obtained from Kaggle
https://www.kaggle.com/code/jocelyndumlao/chatbot-for-mental-health-conversations.
There was no need to clean the dataset as it was already cleaned and stored in json format
with patterns for training, the tags for category of class it fall under and suggested response.

## Tokenization:
• Tokenization is the process of breaking down preprocessed text into individual words,
or tokens.
• Tokenized sequences are padded or trimmed to ensure uniform length using technique
zero-padding.
## Model Prediction:
• To predict, padded tokenized sequences are input into a trained neural network model.
• The model predicts the appropriate tag for the input based on the training data's learnt
patterns and correlations.
## Sample Dialog:
Start messaging with the bot (type quit to stop)!
User: Hello
ChatBot: Hi there. How are you feeling today?
User: I feel great today
ChatBot: Oh i see. That's great.
User: Thanks
ChatBot: You're most welcome!
User: Who are you?
ChatBot: I'm Pandora, a Therapeutic AI Assitant designed to assist you. Tell me about yourself.
User: What can you do?
ChatBot: I can provide general advice regarding anxiety and depression, answer questions related to
mental health and make daily conversations. Do not consider me as a subsitute for an actual mental
healthcare worker. Please seek help if you don't feel satisfied with me.
User: okay
ChatBot: Tell me more
User: I was feeling sad yesterday
ChatBot: I'm here for you. Could you tell me why you're feeling this way?
User: I was a little depressed and had some anxiety
ChatBot: I understand how you feel. Don't put yourself down because of it.
User: Why is mental health important?
ChatBot: Maintaining mental health is crucial to stabilizing constructive behaviors, emotions, and
thoughts. Focusing on mental health care can increase productivity, enhance our self-image, and
improve relationships.
User: Is it bad to go to a therapist?
ChatBot: A therapist is a broad designation that refers to professionals who are trained to provide
treatment and rehabilitation. The term is often applied to psychologists, but it can include others who
provide a variety of services, including social workers, counselors, life coaches, and many others.
User: I feel so lonely sometimes?
ChatBot: I'm here for you. Could you tell me why you're feeling this way?
User: Might be due to exam pressure
ChatBot: Hello there. Glad to see you're back. What's going on in your world right now?
User: You didn't get me!
ChatBot: I'm sorry to hear that. I'm doing my best to help
User: Nevermind
ChatBot: Please go on.
User: Thanks for your help
ChatBot: Happy to help!
User: quit
