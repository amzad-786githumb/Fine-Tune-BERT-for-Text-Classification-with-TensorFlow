# Fine-Tune-BERT-for-Text-Classification-with-TensorFlow

<h2>Learning Objectives</h2>
<p>By the time you complete this project, you will be able to:</p>
<ol>Build TensorFlow Input Pipelines for Text Data with the tf.data API</ol>
<ol>Tokenize and Preprocess Text for BERT</ol>
<ol>Fine-tune BERT for text classification with TensorFlow 2 and TF Hub</ol>

<h1>BERT</h1>
<P>BERT (Bidirectional Encoder Representations from Transformers) is a natural language processing (NLP) model developed by Google. It is designed to understand the context of words in a sentence by analyzing them in both directions (left and right) simultaneously.BERT is widely used for text classification tasks such as sentiment analysis, spam detection, and topic classification. </P>

<H3>Key Features of BERT:</H3>
<ol>1. <b>Bidirectional Understanding – </b>Unlike earlier models that read text in one direction (left to right or right to left), BERT reads in both directions to gain a deeper understanding of word meaning.</ol>
<ol>2. <b>Transformer-Based Architecture –</b> Uses the Transformer model, which relies on self-attention mechanisms to process words in relation to all other words in a sentence.</ol>
<ol><b>3. Pretrained on Large Text Data –</b>3. Initially trained on Wikipedia and BooksCorpus, then fine-tuned for specific NLP tasks.</ol>
<ol><b>4. Improves NLP Tasks –</b> Enhances search engine results, text classification, question answering, and sentiment analysis by better understanding natural language.</ol>

<h2><b>BERT Algorithm for Text Classification</b></h2>
<h3><b>1. Input Processing</b></h3>
<p><b>Tokenization</b></p>
<p>BERT uses WordPiece Tokenization, which breaks words into subwords if necessary. It also adds special tokens:</p>

<ol>[CLS] → Placed at the beginning of the text (used for classification tasks).</ol>
<ol>[SEP] → Used to separate two sentences (if applicable).</ol>

<h3><b>2. Embedding Layer</b></h3>
<p>BERT generates three embeddings for each token:</p>

<ol>Token Embeddings → Represents each token.</ol>ol>
<ol>Segment Embeddings → Differentiates sentence pairs (if applicable).</ol>
<ol>Position Embeddings → Adds positional context, as Transformers do not have a built-in order mechanism.</ol>
<p>These embeddings are added together and passed to the Transformer layers.</p>

<h3><b>3. Transformer Encoder (Self-Attention)</b></h3>
<p>BERT uses multiple Transformer layers (e.g., BERT-Base has 12 layers). Each layer consists of:</p>

<ol><b>Self-Attention Mechanism →</b> Helps understand relationships between words.</ol>
<ol><b>Feedforward Neural Network →</b> Processes the self-attention outputs.</ol>
<ol><b>Layer Normalization & Dropout →</b> Stabilizes training and prevents overfitting.</ol>

<h3><b>4. Output Representation</b></h3>
<p>After passing through Transformer layers, we extract the embedding of the [CLS] token. This embedding represents the entire text and is used for classification.
</p>

<h3><b>5. Classification Layer</b></h3>
<p>The [CLS] embedding is passed to a fully connected (dense) layer with a softmax activation function to predict the class label.</p>

<p>For binary classification (e.g., spam vs. not spam), the softmax function produces probabilities for each class:<p></p>

<p>𝑃(spam)=0.85 , 𝑃(not spam)=0.15 </p>
<p>P(spam)=0.85,P(not spam)=0.15</p>
<p>For multi-class classification (e.g., topic classification with 3+ categories), softmax assigns probabilities across multiple labels.</p>

<h3><b>6. Fine-Tuning with a Loss Function</b><h3>
<p>During training, BERT uses cross-entropy loss to minimize classification errors. The model learns by adjusting weights using backpropagation and gradient descent.</p>


<h2>Applications of BERT:</h2>
<ol>- Google Search (improves query understanding)</ol>
<ol>- Chatbots & Virtual Assistants</ol>
<ol>- Text Summarization & Sentiment Analysis</ol>
<ol>- Question Answering (e.g., answering questions based on a given passage)</ol>

<h2><b>Dataset:</b></h2><p>Quora Insincere Questions Classification</p>
-"https://archive.org/download/fine-tune-bert-tensorflow-train.csv/train.csv.zip"

<h2><b>Tensorhub Model:</b></h2>
-"https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2"

<h2><b>Data fields</b></h2>h2>
<ol>qid - unique question identifier</ol>
<ol>question_text - Quora question text</ol>
<ol>target - a question labeled "insincere" has a value of 1, otherwise 0</ol>
