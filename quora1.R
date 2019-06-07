library(readr)
library(keras)
library(purrr)
library(dplyr)

# params for further tuning

FLAGS <- flags(
  flag_integer("vocab_size", 50000),
  flag_integer("max_len_padding", 20),

  #word embedding size, i. e. vector dimension
  flag_integer("embedding_size", 300),
  flag_numeric("regularization", 0.0001),
  flag_integer("seq_embedding_size", 512)
)

# data for download available at:
# https://raw.githubusercontent.com/MLDroid/quora_duplicate_challenge/master/data/quora_duplicate_questions.tsv
# modify the location w. r. t. your environment
# using readr functions is more fast than base functions
df <- read_tsv("quora_duplicate_questions.tsv")

# tokenization -- we will obtain a word index w. r. t. word frequency
tokenizer <- text_tokenizer(num_words = FLAGS$vocab_size)
fit_text_tokenizer(tokenizer, x = c(df$question1, df$question2))

# we will use FastText word embeddings, analogously we can use word2vec/GloVe embeddings -- do not forget modify the dimension in FLAGS!
# https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.en.vec
# readr functions used again -- read_table does not work well, use read_table2 instead
fast.text.en <- read_table2("wiki.multi.en.vec", col_names = F, skip = 1, progress = T)

word.index <- tokenizer$word_index

# setting the number of words used in real -- minimum of words provided by tokenizer and no. of words we wanted to use (defined in flagging)
voc.size <- min(length(word.index), FLAGS$vocab_size) 

# real word types:
words <- names(word.index)[1:voc.size]

# preparing lookup table: key-value, e. g. word index and word type -- one row added for padding symbol
aux.index.table <- data.frame(Indx=1:(voc.size+1), Wrd=c(words, "PAD_VAL"), stringsAsFactors = F)

# FastText data -- we will set up the name of the first column to "Wrd" -- first column contain words, other columns are doubles
colnames(fast.text.en)[1] <- "Wrd"

# we will enrich the lookup table with corresponding vectors, i. e. we will obtain "index-word-Vectors" table:
aux.join <- left_join(aux.index.table, fast.text.en, by = "Wrd")

# building embedding matrix: i-th word from the tokenizer has its vector on the i-th row of the embedding matrix (3rd and further cols of aux.join)
# more efficient implementation than https://keras.rstudio.com/articles/examples/pretrained_word_embeddings.html -- but we use dplyr
emb.mat <- as.matrix(aux.join[,3:ncol(aux.join)])

# NA values are set to all-zero vectors (words not covered by the FastText pretrained embeddings as well as padded values)
emb.mat[is.na(emb.mat)] <- 0

# texts, i. e. questions are transformed into sequences of numbers -- w. r. t. tokenizer results
question1 <- texts_to_sequences(tokenizer, df$question1)
question2 <- texts_to_sequences(tokenizer, df$question2)

# question representations are padded into maxlen, padding value is 50001
question1 <- pad_sequences(question1, maxlen = FLAGS$max_len_padding, value = voc.size + 1)
question2 <- pad_sequences(question2, maxlen = FLAGS$max_len_padding, value = voc.size + 1)

#############
#
# KERAS MODEL
# inspiration: https://blogs.rstudio.com/tensorflow/posts/2018-01-09-keras-duplicate-questions-quora/
# but we will use pretrained word embedings!
#
#############

input1 <- layer_input(shape = c(FLAGS$max_len_padding))
input2 <- layer_input(shape = c(FLAGS$max_len_padding))

embedding <- layer_embedding(
  input_dim = FLAGS$vocab_size + 1, 
  output_dim = FLAGS$embedding_size, 
  input_length = FLAGS$max_len_padding, 
  
  # weights are stored in the embedding matrix
  weights = list(emb.mat),
  
  # word embeddings are not trained
  trainable = F
)

# encoding of sequences by LSTMs -- we may experiment with GRU, ...
seq_emb <- layer_lstm(
  units = FLAGS$seq_embedding_size, 
  recurrent_regularizer = regularizer_l2(l = FLAGS$regularization)
)

vector1 <- embedding(input1) %>%
  seq_emb()
vector2 <- embedding(input2) %>%
  seq_emb()

out <- layer_dot(list(vector1, vector2), axes = 1) %>%
  layer_dense(1, activation = "sigmoid")

model <- keras_model(list(input1, input2), out)
model %>% compile(
  optimizer = "nadam", 
  loss = "binary_crossentropy", 
  metrics = list(
    acc = metric_binary_accuracy
  )
)

# for possible replications
set.seed(1257)
val_sample <- sample.int(nrow(question1), size = 0.1*nrow(question1))

# training
model %>%
  fit(
    list(question1[-val_sample,], question2[-val_sample,]),
    df$is_duplicate[-val_sample], 
    batch_size = 128, 
    epochs = 30, 
    validation_data = list(
      list(question1[val_sample,], question2[val_sample,]), df$is_duplicate[val_sample]
    ),
    callbacks = list(
      callback_early_stopping(patience = 5),
      callback_reduce_lr_on_plateau(patience = 3)
    )
  )

# let us save the model for the further use
save_model_hdf5(model, "model-question-pairs-weights.hdf5", include_optimizer = TRUE)
save_text_tokenizer(tokenizer, "tokenizer-question-pairs-weights.hdf5")

# predicting...
predict_question_pairs <- function(model, tokenizer, q1, q2) {
  q1 <- texts_to_sequences(tokenizer, list(q1))
  q2 <- texts_to_sequences(tokenizer, list(q2))
  
  q1 <- pad_sequences(q1, maxlen = FLAGS$max_len_padding, value = voc.size + 1)
  q2 <- pad_sequences(q1, maxlen = FLAGS$max_len_padding, value = voc.size + 1)
  
  as.numeric(predict(model, list(q1, q2)))
}

# example of usage
# round(predict_question_pairs(model, tokenizer, "Do you have a pet?", "Do you like dogs?"))
