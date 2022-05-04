############################################
# Text Translation using attention         #
############################################
# Dataset:  http://www.manythings.org/anki/#
############################################
import numpy as np
import typing
from typing import Any, Tuple
import tensorflow as tf
import tensorflow_text as tf_text
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
#print("----TensorFlow version:", tf.__version__)
#print("----Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Download the file
import pathlib
path_to_zip = r'C:\Users\pmspr\Documents\Machine Learning\Courses\Tensorflow Cert\Data\nlp'
path_to_file = pathlib.Path(path_to_zip)/'spa-eng/spa.txt'

# Load the file and print sample text
text = path_to_file.read_text(encoding = 'utf-8')
lines = text.splitlines()
pairs = [line.split('\t') for line in lines]

inp = [inp for targ, inp in pairs]
targ = [targ for targ, inp in pairs]

# print(inp[-1])
# print(targ[-1])

# Create tensorflow dataset
BUFFER_SIZE = len(inp)
BATCH_SIZE = 64

dataset = tf.data.Dataset.from_tensor_slices((inp, targ)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE)

# Print the sample from dataset
print('Number of examples: {}'.format(len(inp)))
print('Number of batches: {}'.format(len(list(dataset.as_numpy_iterator()))))

for inpBatch,targBatch in dataset.take(1):
    for input,target in zip(inpBatch,targBatch):
        print('Input spanish text:{}'.format(input))
        print('Target English Translation:{}'.format(target))
        break
    break

##################
# Pre-Processing #
###############################################################################
# First step - Text normalization -
# The standard also defines a text normalization procedure,
# called Unicode normalization, that replaces equivalent sequences of
# characters so that any two texts that are equivalent will be reduced
# to the same sequence of code points, called the normalization form or
# normal form of the original text.

# We use Normalization Form KD (NFKD) - Compatibility Decomposition
# Example
ex_string = tf.constant('¿Todavía está en casa?')
print('Raw string:{}'.format(ex_string.numpy()))
norm_string = tf_text.normalize_utf8(ex_string,'NFKD')
print('Normalized string:{}'.format(norm_string.numpy()))

# tf.strings should be used as this method will be used
# inside a tensorflow layer.
def tf_lower_and_split_punct(text):
  # Split accecented characters.
  text = tf_text.normalize_utf8(text, 'NFKD')
  text = tf.strings.lower(text)
  # Keep space, a to z, and select punctuation.
  text = tf.strings.regex_replace(text, '[^ a-z.?!,¿]', '')
  # Add spaces around punctuation.
  text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')
  # Strip whitespace.
  text = tf.strings.strip(text)

  text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
  return text

print(ex_string.numpy().decode())
print(tf_lower_and_split_punct(ex_string).numpy().decode())
###############################################################################
# Text Vectorization -
# Text Vectorization is the process of converting text into numerical representation.

# Input text processor. This include normalization and Vectorization.
max_vocab_size = 5000

input_text_processor = tf.keras.layers.TextVectorization(
    standardize=tf_lower_and_split_punct,
    max_tokens=max_vocab_size)

# Output text processor. This include normalization and Vectorization.

output_text_processor = tf.keras.layers.TextVectorization(
    standardize=tf_lower_and_split_punct,
    max_tokens=max_vocab_size)

# Adapt the processors on input and output
input_text_processor.adapt(inp)
output_text_processor.adapt(targ)
example_tokens = input_text_processor(inpBatch)

print('Example batch vocabulary size:{}'.format(input_text_processor.vocabulary_size()))
print('Shape of the tokens:{}'.format(tf.shape(example_tokens).numpy()))
print('Rank of the tokens:{}'.format(tf.rank(example_tokens).numpy()))

plt.subplot(1, 2, 1)
plt.pcolormesh(example_tokens)
plt.title('Token IDs')

plt.subplot(1, 2, 2)
plt.pcolormesh(example_tokens != 0)
plt.title('Mask')
plt.show()
###############################################################################
# Encoder architecture
# For text translation, we use encoder-decoder model with attention.
# Because, for translation, the size of the input need not be same as output
# Takes the input sequence and provies us with a context
class Encoder(tf.keras.layers.Layer):
    def __init__(self, input_vocab_size, embedding_dim, enc_units, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.enc_units = enc_units # Number of RNN units
        self.input_vocab_size = input_vocab_size # Input vocabulary size

        # The embedding layer converts sequence of tokens to sequence of vectors
        # tokens = [1, 2, 3] -> embedded to = [[2,4,5..embedding_dim], [4,1,3..embedding_dim], [4,5,2..embedding_dim]]
        self.embedding = tf.keras.layers.Embedding(self.input_vocab_size,
                                                   embedding_dim)

        # The GRU RNN layer processes those vectors sequentially.
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       # Return the sequence and state
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, tokens, state=None):

        #1 tokens shape - ('batch', 's')

        # 2. The embedding layer looks up the embedding for each token.
        vectors = self.embedding(tokens) # ('batch', 's', 'embed_dim')

        # 3. The GRU processes the embedding sequence.
        #    output shape: (batch, s, enc_units)
        #    state shape: (batch, enc_units)
        output, state = self.gru(vectors, initial_state=state)

        # 4. Returns the new sequence and its state.
        return output, state , vectors

# Test the encoder class on the example input batch
# Convert the input text to tokens.

embedding_dim = 256
units = 1024

# Encode the input sequence.
encoder = Encoder(input_text_processor.vocabulary_size(),
                  embedding_dim, units)
example_enc_output, example_enc_state, embedded_vectors = encoder(example_tokens)

print('######## Encoder shapes ########')
print(f'Input batch, shape (batch): {inpBatch.shape}')
print(f'Input batch tokens, shape (batch, s): {example_tokens.shape}')
print('Embedded tokens shape, shape (batch, s, embedded_dim):{}'.format(tf.shape(embedded_vectors).numpy()))
print(f'Encoder output, shape (batch, s, units): {example_enc_output.shape}')
print(f'Encoder state, shape (batch, units): {example_enc_state.shape}')

###############################################################################
# Attention layer - This used to give attention to important tokens for
# translation. Before we got for decoder, we create this layer and use in decoder
# The query: This will be generated by the decoder, later.
# The value: This Will be the output of the encoder.
# The mask: To exclude the padding, example_tokens != 0
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()

        # Bahdanau attention is additive
        self.W1 = tf.keras.layers.Dense(units, use_bias=False)
        self.W2 = tf.keras.layers.Dense(units, use_bias=False)
        self.attention = tf.keras.layers.AdditiveAttention()

    def call(self, query, value, mask):
        print('######## Attention shapes ########')
        print(f'Query, shape (batch, t, query_units): {query.shape}')
        print(f'Value, shape (batch, t, query_units): {value.shape}')
        print(f'Mask, shape (batch, t, query_units): {mask.shape}')

        w1_query = self.W1(query)
        w2_key = self.W2(value)

        query_mask = tf.ones(tf.shape(query)[:-1], dtype=bool)
        value_mask = mask

        context_vector, attention_weights = self.attention(
            inputs=[w1_query, value, w2_key],
            mask=[query_mask, value_mask],
            return_attention_scores=True,
        )
        print(f'context_vector, (batch, t, value_units):{context_vector.shape}')
        print(f'attention_weights, (batch, t, s):{attention_weights.shape}')

        return context_vector, attention_weights

# Test the attention layer with an example
# Initialize the attention layer
attention_layer = BahdanauAttention(units)
# Later, the decoder will generate this attention query
example_attention_query = tf.random.normal(shape=[len(example_tokens), 2, 10])
print('Example token shape:{}'.format(example_attention_query.shape))

# Attend to the encoded tokens
context_vector, attention_weights = attention_layer(
    query=example_attention_query,
    value=example_enc_output,
    mask=(example_tokens != 0))

print(f'Attention result shape: (batch_size, query_seq_length, units):           {context_vector.shape}')
print(f'Attention weights shape: (batch_size, query_seq_length, value_seq_length): {attention_weights.shape}')

###############################################################################
# Decoder - The decoder's job is to generate predictions for the next output token.
# The decoder receives the complete encoder output.
# It uses an RNN to keep track of what it has generated so far.
# It uses its RNN output as the query to the attention over the encoder's output, producing the context vector.
# It combines the RNN output and the context vector using Equation 3 (below) to generate the "attention vector".
# It generates logit predictions for the next token based on the "attention vector".

# The decoder takes 4 inputs.
# new_tokens - The last token generated. Initialize the decoder with the "[START]" token.
# enc_output - Generated by the Encoder.
# mask - A boolean tensor indicating where tokens != 0
# state - The previous state output from the decoder (the internal state of the decoder's RNN). '
# 'Pass None to zero-initialize it. The original paper initializes it from the encoder's final RNN state.

# Function annotations
class DecoderInput(typing.NamedTuple):
    new_tokens: Any
    enc_output: Any
    mask: Any

class DecoderOutput(typing.NamedTuple):
    logits: Any
    attention_weights: Any

class Decoder(tf.keras.layers.Layer):
    def __init__(self, output_vocab_size, embedding_dim, dec_units):
        super(Decoder, self).__init__()
        self.dec_units = dec_units
        self.output_vocab_size = output_vocab_size
        self.embedding_dim = embedding_dim

        # For Step 1. The embedding layer convets token IDs to vectors
        self.embedding = tf.keras.layers.Embedding(self.output_vocab_size,
                                                   embedding_dim)

        # For Step 2. The RNN keeps track of what's been generated so far.
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

        # For step 3. The RNN output will be the query for the attention layer.
        self.attention = BahdanauAttention(self.dec_units)

        # For step 4. converting `ct` to `at`
        self.Wc = tf.keras.layers.Dense(dec_units, activation=tf.math.tanh,
                                        use_bias=False)

        # For step 5. This fully connected layer produces the logits for each
        # output token.
        self.fc = tf.keras.layers.Dense(self.output_vocab_size)

    def call(self,
             inputs: DecoderInput,
             state=None) -> Tuple[DecoderOutput, tf.Tensor]:
        print('######## Decoder shapes ########')
        # Step 1. Lookup the embeddings
        vectors = self.embedding(inputs.new_tokens)
        print('Embedded tokens shape, shape (batch, t, embedding_dim):{}'.format(tf.shape(vectors).numpy()))

        # Step 2. Process one step with the RNN
        rnn_output, state = self.gru(vectors, initial_state=state)
        print(f'RNN output shape (batch, t, dec_units):{rnn_output.shape}')
        print(f'RNN State shape (batch, t, dec_units):{state.shape}')

        # Step 3. Use the RNN output as the query for the attention over the
        # encoder output.
        context_vector, attention_weights = self.attention(
            query=rnn_output, value=inputs.enc_output, mask=inputs.mask)
        print(f'Context vector shape (batch, t, dec_units):{context_vector.shape}')
        print(f'attention_weights shape (batch, t, s):{attention_weights.shape}')

        # Step 4. Eqn. (3): Join the context_vector and rnn_output
        #     [ct; ht] shape: (batch t, value_units + query_units)
        context_and_rnn_output = tf.concat([context_vector, rnn_output], axis=-1)
        print(f'context+rnn output shape:{context_and_rnn_output.shape}')

        # Step 4. Eqn. (3): `at = tanh(Wc@[ct; ht])`
        attention_vector = self.Wc(context_and_rnn_output)
        print(f'Attention vector shape (batch, t, dec_units) :{attention_vector.shape}')

        # Step 5. Generate logit predictions:
        logits = self.fc(attention_vector)
        print(f'Logits shape (batch, t, output_vocab_size) :{logits.shape}')

        return DecoderOutput(logits, attention_weights), state

# Test the decoder
# Convert the target sequence, and collect the "[START]" tokens
example_output_tokens = output_text_processor(targBatch)
start_index = output_text_processor.get_vocabulary().index('[START]')
first_token = tf.constant([[start_index]] * example_output_tokens.shape[0])
print(f'example_output_tokens:{example_output_tokens}')
print(f'First token:{first_token}')

# Initialize the decoder
decoder = Decoder(output_text_processor.vocabulary_size(),
                  embedding_dim, units)

# Run the decoder
dec_result, dec_state = decoder(
    inputs = DecoderInput(new_tokens=first_token,
                          enc_output=example_enc_output,
                          mask=(example_tokens != 0)),
    state = example_enc_state # Final state of the encoder
)

print(f'logits shape: (batch_size, t, output_vocab_size) {dec_result.logits.shape}')
print(f'state shape: (batch_size, dec_units) {dec_state.shape}')

#Sample a token according to the logits:
sampled_token = tf.random.categorical(dec_result.logits[:, 0, :], num_samples=1)
#Decode the token as the first word of the output:
vocab = np.array(output_text_processor.get_vocabulary())
first_word = vocab[sampled_token.numpy()]
print(first_word[:5])

#Now use the decoder to generate a second set of logits.
#Pass the same enc_output and mask, these haven't changed.
#Pass the sampled token as new_tokens. Pass the decoder_state the
# decoder returned last time, so the RNN continues with a
# memory of where it left off last time.
dec_result, dec_state = decoder(
    DecoderInput(sampled_token,
                 example_enc_output,
                 mask=(example_tokens != 0)),
    state=dec_state)

sampled_token = tf.random.categorical(dec_result.logits[:, 0, :], num_samples=1)
first_word = vocab[sampled_token.numpy()]
print(first_word[:5])