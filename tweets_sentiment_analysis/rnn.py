
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
from string import punctuation
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
from sklearn.model_selection import train_test_split
import csv


# In[2]:


# Preprocessing functions
def get_wordnet_pos(treebank_tag):
    """
    Cf. https://stackoverflow.com/questions/15586721/wordnet-lemmatization-and-pos-tagging-in-python
    :param treebank_tag: a tag from nltk.pos_tag treebank
    """
    
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def preprocess(sentence):
    lemmatizer = nltk.WordNetLemmatizer()
    sentence = sentence.lower()
    tokenized_sentence = word_tokenize(sentence)
    
    result = []
    
    # part of speech for the tokens
    documents_pos = nltk.pos_tag(tokenized_sentence)
    
    # Lemmatize words based on their part of speech
    for word, tag in documents_pos:
        wntag = get_wordnet_pos(tag)
        # not supply tag in case of None
        if wntag is None:
            lemma = lemmatizer.lemmatize(word) 
        else:
            result.append(lemmatizer.lemmatize(word, pos=wntag))
    
    result = " ".join(result)
    result = ''.join([char for char in result if char not in punctuation])
    
    return result


# In[3]:


with open('./datasets/train_neg_full.txt', 'r') as f:
    tweets_neg = f.read()
with open('./datasets/train_pos_full.txt', 'r') as f:
    tweets_pos = f.read()

with open('./datasets/test_data.txt', 'r') as f:
    tweets_test = f.read()
    
    
tweets_neg = tweets_neg.split("\n")
tweets_pos = tweets_pos.split("\n")
tweets_test = tweets_test.split("\n")

tweets = []
labels = []
for tweet in tweets_neg[:500]:
    tweets.append(preprocess(tweet))
    labels.append(0)
    
for tweet in tweets_pos[:500]:
    tweets.append(preprocess(tweet))
    labels.append(1)
    
for tweet in tweets_test[:500]:
    tweets.append(preprocess(tweet))
    labels.append(2)
    
words = []
for tweet in tweets:
    for w in tweet.split():
        words.append(w)


# In[4]:


from collections import Counter
counts = Counter(words)
vocab = sorted(counts, key=counts.get, reverse=True)
vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}

tweets_ints = []
for tweet in tweets:
    tweets_ints.append([vocab_to_int[word] for word in tweet.split()])
    
labels = np.array(labels)


# In[5]:


tweet_lens = Counter([len(x) for x in tweets_ints])
print("Zero-length tweets: {}".format(tweet_lens[0]))
print("Maximum tweet length: {}".format(max(tweet_lens)))


# In[6]:


non_zero_idx = [ii for ii, tweet in enumerate(tweets_ints) if len(tweet) != 0]
len(non_zero_idx)


# In[7]:


tweets_ints[-1]


# In[8]:


tweets_ints = [tweets_ints[ii] for ii in non_zero_idx]
labels = np.array([labels[ii] for ii in non_zero_idx])


# In[9]:


seq_len = 200
features = np.zeros((len(tweets_ints), seq_len), dtype=int)
for i, row in enumerate(tweets_ints):
    features[i, -len(row):] = np.array(row)[:seq_len]


# In[10]:


test_index_list = []
for index, val in enumerate(labels):
    if val == 2:
        test_index_list.append(index)

test_x = []
test_x_id = []
counter = 1
for index,val in enumerate(test_index_list):
    test_x.append(features[val])
    test_x_id.append(counter)
    counter += 1
    
features = np.delete(features,test_index_list,axis=0)
labels = np.delete(labels,test_index_list,axis=0)


# In[11]:


seed = 1
train_x, val_x, train_y, val_y = train_test_split(features, labels, test_size=0.2, random_state=seed)


# In[12]:


train_y = np.array(train_y)
val_y = np.array(val_y)


# # Recurrent Neural Network

# In[13]:


lstm_size = 256
lstm_layers = 1
batch_size = 500
learning_rate = 0.001


# In[14]:


n_words = len(vocab_to_int) + 1

# Create the graph object
graph = tf.Graph()
# Add nodes to the graph
with graph.as_default():
    inputs_ = tf.placeholder(tf.int32, [None, None], name='inputs')
    labels_ = tf.placeholder(tf.int32, [None, None], name='labels')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')


# In[15]:


# Size of the embedding vectors (number of units in the embedding layer)
embed_size = 300 

with graph.as_default():
    embedding = tf.Variable(tf.random_uniform((n_words, embed_size), -1, 1))
    embed = tf.nn.embedding_lookup(embedding, inputs_)


# In[16]:


with graph.as_default():
    # Your basic LSTM cell
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    
    # Add dropout to the cell
    drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
    
    # Stack up multiple LSTM layers, for deep learning
    cell = tf.contrib.rnn.MultiRNNCell([drop] * lstm_layers)
    
    # Getting an initial state of all zeros
    initial_state = cell.zero_state(batch_size, tf.float32)


# In[17]:


with graph.as_default():
    outputs, final_state = tf.nn.dynamic_rnn(cell, embed, initial_state=initial_state)


# In[18]:


with graph.as_default():
    predictions = tf.contrib.layers.fully_connected(outputs[:, -1], 1, activation_fn=tf.sigmoid)
    cost = tf.losses.mean_squared_error(labels_, predictions)
    
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


# In[19]:


with graph.as_default():
    correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels_)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# In[20]:


def get_batches(x, y, batch_size=100):
    n_batches = x.shape[0]//batch_size
    x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]
    for ii in range(0, x.shape[0], batch_size):
        yield x[ii:ii+batch_size], y[ii:ii+batch_size]


# In[21]:


epochs = 10

with graph.as_default():
    saver = tf.train.Saver()
    
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


# In[22]:


with tf.Session(config=config,graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    iteration = 1
    for e in range(epochs):
        state = sess.run(initial_state)
        
        for ii, (x, y) in enumerate(get_batches(train_x, train_y, batch_size), 1):
            feed = {inputs_: x,
                    labels_: y[:, None],
                    keep_prob: 0.5,
                    initial_state: state}
            loss, state, _ = sess.run([cost, final_state, optimizer], feed_dict=feed)
            
            if iteration%5==0:
                print("Epoch: {}/{}".format(e, epochs),
                      "Iteration: {}".format(iteration),
                      "Train loss: {:.3f}".format(loss))

            if iteration%25==0:
                val_acc = []
                val_state = sess.run(cell.zero_state(batch_size, tf.float32))
                for x, y in get_batches(val_x, val_y, batch_size):
                    feed = {inputs_: x,
                            labels_: y[:, None],
                            keep_prob: 1,
                            initial_state: val_state}
                    batch_acc, val_state = sess.run([accuracy, final_state], feed_dict=feed)
                    val_acc.append(batch_acc)
                print("Val acc: {:.3f}".format(np.mean(val_acc)))
            iteration +=1
    saver.save(sess, "checkpoints/sentiment.ckpt")


# In[23]:


def get_batches_test(x, x_id,batch_size=100):
    n_batches = len(x)//batch_size
    x,x_id = x[:n_batches*batch_size], x_id[:n_batches*batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii+batch_size], x_id[ii:ii+batch_size]


# In[24]:


with tf.Session(config=config,graph=graph) as sess:
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
    test_state = sess.run(cell.zero_state(batch_size, tf.float32))
    for ii, (x,x_id) in enumerate(get_batches_test(test_x, test_x_id ,batch_size), 1):
        feed = {inputs_: x,
                keep_prob: 1,
                initial_state: test_state}
        prediction = sess.run([predictions], feed_dict=feed)
        prediction = prediction[0]
        # Get the probability for each statement
        prediction = prediction[:, 0]
        
f = open("tweets_kaggle.csv", "w")
for i in range(len(x_id)):
    f.write("{},{}\n".format(x_id[i], prediction[i]))
f.close()

