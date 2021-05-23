# Named Entity Recognition for Unstructured Healthcare Data

The following is an exploration into Named Entity Recognition and its applications in unstructured Healthcare Data

We will take the following approach to understanding the problem and get up to date with recent works

- Literature Review
- Exploratory Data Analysis
- Experiment Setup
- Model Selection & Evaluation

Challenges:

- Significant lack of annotated datasets
- Heavily unbalanced datasets
- The data consists of significantly varying sentence length
- Requires context and results of the training significantly depend on the training data and domain of the corpus

# Recent Work

### Clinical Named Entity Recognition Using Deep Learning Models

([https://www.researchgate.net/publication/325532104_Clinical_Named_Entity_Recognition_Using_Deep_Learning_Models](https://www.researchgate.net/publication/325532104_Clinical_Named_Entity_Recognition_Using_Deep_Learning_Models))

- Wu et al. attempt to outperform baseline CRF based NER models that were considered quite successful at the time
- They do this by considering two approaches: a CNN and an RNN (LSTM)
- The model uses word embeddings with 50 dimensions, which was pre-trained on a MIMIC II corpus. In order to train the word embeddings they used a neural network with negative sampling to train the embeddings - which gave results comparable to the word2vec algorithm

Takeaways: Due to significant class imbalances, using negative sampling in some manner during either training or pre-training may significantly benefit our model. Likewise, transfer learning for a particular domain seems to provide a good performance boost

### BioBERT: a pre-trained biomedical language representation model for biomedical text mining

([https://arxiv.org/abs/1901.08746](https://arxiv.org/abs/1901.08746))

- Lee et al. outperform current BERT model by using the same architecture and pre-training it with  biomedical domain corpora
- The BioBERT has shown good results in biomedical named entity recognition, biomedical relation extraction, and biomedical QnA

Takeaways: Most NER models generally use word embeddings that have been trained on either PubMed or PMC corpora. For the evaluation metrics of NER, we used entity level precision, recall and F1 score. BioBERT has a good range of datasets for NER which we can make use of for our training : [https://github.com/dmis-lab/biobert](https://github.com/dmis-lab/biobert)

### Named-Entity Recognition using Keras BiLSTM

([https://towardsdatascience.com/named-entity-recognition-ner-using-keras-bidirectional-lstm-28cd3f301f54](https://towardsdatascience.com/named-entity-recognition-ner-using-keras-bidirectional-lstm-28cd3f301f54))

- Snehal uses Keras in order to build a NER model with a Bidirectional LSTM
- Results seem decent but no in-depth analysis of the results to gain further inference

Takeaways: Model architecture seems good and generic for a NER model. There is a mistake with the activation function of the final layer however necessary corrections have been done to my model. Embeddings were not pre-trained in this model. Snehal uses a TimeDistributedLayer which I personally have not worked with but on further investigation seems like an appropriate selection for sequential models in a many-to-many task.

### Bidirectional LSTM-CRF Models for Sequence Tagging

([https://arxiv.org/pdf/1508.01991.pdf](https://arxiv.org/pdf/1508.01991.pdf))

Have a look at the considered spelling and context features used during training

### End-to-end Sequence Labelling via Bi-directional LSTM-CNNs-CRF

([https://arxiv.org/pdf/1603.01354.pdf](https://arxiv.org/pdf/1603.01354.pdf))

### Optimal Hyperparameters for Deep LSTM-Networks for Sequence Labelling Tasks

([https://www.researchgate.net/publication/318652600_Optimal_Hyperparameters_for_Deep_LSTM-Networks_for_Sequence_Labeling_Tasks](https://www.researchgate.net/publication/318652600_Optimal_Hyperparameters_for_Deep_LSTM-Networks_for_Sequence_Labeling_Tasks))

# Literature Review

### Conditional Random Fields (CRF)

Uses potentials to model to the conditional probabilities of events given other variables hence, making no naive independence assumptions - has proven successful in sequence labelling and segmentation

As opposed to using a softmax classifier which makes independent tagging decisions for each output based on the feature vector given, a CRF models the features jointly. In an NLP setting we can suggest that the CRF takes into account neighbouring tags and yields final predictions for each word using a decoding process computed with dynamic programming (Viterbi Algorithm)

# Exploratory Data Analysis

There are two main datasets which I will be considering throughout the initial stages of this experimenting:

Medical Transcriptions ([https://www.kaggle.com/tboyle10/medicaltranscriptions](https://www.kaggle.com/tboyle10/medicaltranscriptions)) - This dataset consists of medical transcriptions from multiple specialties. Given that we have relatively interesting results in the "Surgery" specialty it may be worth considering another specialty - possibly "Orthopaedic". My main concern with this dataset is due to the lack of annotations we must manually label the words of each sentence based on our own custom rule. Due to this, one approach is to consider latin prefixes and suffixes commonly used in a surgical domain and give such terms one binary label while all other terms are labeled with the other label. With each run we can improve with our own annotations and improve training which seems appropriate and easily automated (need to research this further). 

NER Dataset from BioBERT ([https://github.com/dmis-lab/biobert](https://github.com/dmis-lab/biobert)) - This dataset consists of 6 biomedical named entity recognition datasets each from different sources. Each word/symbol in the sentences are labeled with one of three possible labels. "O" refers to non-medical terminology, "B" refers to the beginning of a medical entity, "I" refers to the inside of a medical entity, i.e. any sequence of I's will always be preceded by a B. In comparison to using binary labels, this seems like the type of data that is commonly available with more than one entity type. On further analysis, the text also seems relatively comparable to the Medical Transcripts dataset hence for the initial experiments we will use this dataset in order to better understand the main problems and concerns that others come across when working with NER models. If we arrive at successful results we will evaluate our model on the Medical Transcript dataset and compare the results with those when using binary labels.

Therefore going forward we will be discussing the NCBI Disease dataset from the NER Dataset used to train and evaluate BioBERT - we will refer to it as the NCBI dataset for convenience. 

The NCBI dataset consists of 6347 sentences each with a sequence of labels for each word/symbol in the sentence. For e.g.

[NCBI Data](https://www.notion.so/bdac5baf470a465980af5f4bea7f60aa)

In comparison to most NLP tasks, what is important to note is that we generally keep punctuation and capitalisation as these features are important differentiators in a medical context for multi-word entities.  

With some minor work (build parser, feature dictionaries) we arrive at the following dataframe which summarises all the necessary information (both raw and encoded) we will use during the training phase.

![Named%20Entity%20Recognition%20for%20Unstructured%20Healthca%209e4a9cf894164a35b169ed1822326b1c/Untitled.png](Named%20Entity%20Recognition%20for%20Unstructured%20Healthca%209e4a9cf894164a35b169ed1822326b1c/Untitled.png)

Given that "B" and "I" are the main classes in our data that we are interested with, it is worth having a look into the class distribution of our data beforehand.

![Named%20Entity%20Recognition%20for%20Unstructured%20Healthca%209e4a9cf894164a35b169ed1822326b1c/Untitled%201.png](Named%20Entity%20Recognition%20for%20Unstructured%20Healthca%209e4a9cf894164a35b169ed1822326b1c/Untitled%201.png)

As expected, we have a heavily imbalanced dataset, where 91.8% of the labels in our dataset is an "O". Keeping this in mind we will make the following suggestions : 

- Use F1, Precision, Recall on the "B" and "I" classes in order to better compare our results fairly
- Use of accuracy either for early stopping or model checkpoint is a bad idea and will result in superficially good results.
- During training it may be appropriate to consider oversampling the minority class - will need to research further on the best practices to do so in an NLP context.

Using a 80:20 train test split, I padded each sentence to the maximum length of 123 dimensions and considered embedding vectors of 64 dimensions.

# Experiments

## 1. Word Level Model

Given recent literature, I strongly believe that a neural approach to this problem has the potential of producing strong results hence I will mainly focus on using a Neural Network. However, I would like to look further into the applications of CRF's and how well suited they are for this task as well. 

We will use a many-to-many sequential model which consists of 3 main layers: Embedding, Bidirectional LSTM, and a Time Distributed Dense Layer.

- Embedding Layer - The vectors retrieved from this layer are randomly initialised and parameters are updated during training. This layer retrieves a 64-dimensional sentiment vector for each word token.
- Bidirectional LSTM Layer - The embedding vectors retrieved from the previous layer are then passed through an LSTM in two different directions resulting in two output vectors: the forward hidden state and the backward hidden state which are both centred around the current word token. Therefore we must choose a merging mode for the two hidden states produced before passing them to the next layer. For this case I have selected "concat" however we can also take the sum, avg, etc.
- Time Distributed Dense Layer - This layer is suitable for the many-to-many case and is simply a wrapper allowing us to apply a single layer to every temporal slice of an input, in our case each word token representation from the BiLSTM layer. We use this wrapper in order to apply a Dense layer with 3 outputs and a softmax activation function to each word token representation.

Further details : 

- The loss function we use is the Categorical CrossEntropy function which is appropriate for the task of multiclass classification
- The optimiser used is Adam with varying learning rates, $\beta_1$ = 0.9, $\beta_2$ = 0.999
- Batch size was 32
- The model state is saved based on the best validation loss during training
- The early stopping condition is if the validation loss does not decrease over 5 epochs then we stop training
- The learning rate will is set to reduce on plateau of validation loss with a minimum learning rate of 0.00001 and a patience of 3 epochs
- Model is trained on 2538 samples and validated on 1904 samples (note the sample sizes are exceptionally small for a task such as this however, initial experiments were more exploratory and not significantly performance driven)

---

### Experiment 1 - Word Level BiLSTM

Our first model was trained with an initial learning rate of 0.00001 for 100 epochs (early stopping condition was not met) The following is the plot of the train and validation loss through training

![Named%20Entity%20Recognition%20for%20Unstructured%20Healthca%209e4a9cf894164a35b169ed1822326b1c/Untitled%202.png](Named%20Entity%20Recognition%20for%20Unstructured%20Healthca%209e4a9cf894164a35b169ed1822326b1c/Untitled%202.png)

The results are a little bit concerning based off this graph alone and when evaluating the model on the test set we arrive at a 98.35% accuracy which is tempting to celebrate but it is an expected overestimation due to the class imbalance.

On further analysis, we see that from the test set predictions, ALL predictions were simply "O" which basically means nothing informative was learnt by our model. The model simply learnt that predicting "O" more often reduces the validation loss and hence based on the task given to the model, assumed it was performing better.

Takeaway : Evident overfitting in this approach

---

### Experiment 2 - Word Level BiLSTM (Parameter variation)

After training the model over 20 epochs, starting at a learning rate of 0.001 (once again early stopping condition is not met) we arrive at the following plot of the loss

![Named%20Entity%20Recognition%20for%20Unstructured%20Healthca%209e4a9cf894164a35b169ed1822326b1c/Untitled%203.png](Named%20Entity%20Recognition%20for%20Unstructured%20Healthca%209e4a9cf894164a35b169ed1822326b1c/Untitled%203.png)

This seems to be a more satisfying result however we still do observe a level of overfitting after around 10 epochs.

On further analysis into the test predictions we arrive at the following metrics,

![Named%20Entity%20Recognition%20for%20Unstructured%20Healthca%209e4a9cf894164a35b169ed1822326b1c/Untitled%204.png](Named%20Entity%20Recognition%20for%20Unstructured%20Healthca%209e4a9cf894164a35b169ed1822326b1c/Untitled%204.png)

Note that given we are more concerned of the results produced from the "B" and "I" classes, I have removed the majority class when comparing the model metrics.

One example of the predictions can be visualised as follows,

[Prediction of Experiment 2](https://www.notion.so/e81cd65b6ee04c96822072044de80eca)

---

### 2.Word + Char Representation

Here we attempt to address the evident lexical patterns between characters in words from medical jargon. As mentioned earlier, medical jargon uses consistent prefixes and suffixes but when considering a word level model all this information is lost in a numeric encoding of the entire word. Therefore we will use a CNN to encode character level information of the word into its character-level representation. We then combine the character and word level representations  and feed this joint representation into the bidirectional LSTM.

![Named%20Entity%20Recognition%20for%20Unstructured%20Healthca%209e4a9cf894164a35b169ed1822326b1c/Untitled%205.png](Named%20Entity%20Recognition%20for%20Unstructured%20Healthca%209e4a9cf894164a35b169ed1822326b1c/Untitled%205.png)

The above figure shows how character-level representations are produced from a single word. In conclusion, we will use the same BiLSTM-Softmax model as earlier but use these new character and word joint representations as input.

The complete model is as follows:

![Named%20Entity%20Recognition%20for%20Unstructured%20Healthca%209e4a9cf894164a35b169ed1822326b1c/Untitled%206.png](Named%20Entity%20Recognition%20for%20Unstructured%20Healthca%209e4a9cf894164a35b169ed1822326b1c/Untitled%206.png)

Further details:

- Each character is mapped to an input embedding vector of 64 dimensions. The embedding matrix at this time is randomly initialised. Each word embedding is also 64 dimensions
- We apply a convolutional filter of length 4 along the time dimension of the sequence of character embedding vectors and use a

# Current Challenges and Work

Snowmed CT

Autocoding

Redaction

Temporal Tags