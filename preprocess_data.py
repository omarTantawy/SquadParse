import pickle
import spacy
import numpy as np
import pandas as pd

# Load up spacy model and import stop words
nlp = spacy.load('en_core_web_sm')
from spacy.lang.en import STOP_WORDS

for word in STOP_WORDS:
    lexeme = nlp.vocab[word]
    lexeme.is_stop = True


def save_pickle(data, filename):
    save_document = open('data/' + filename + '.pickle', 'wb')
    pickle.dump(data, save_document)
    save_document.close()


def load_pickle(filepath):
    document = open('data/' + filepath + '.pickle', 'rb')
    data = pickle.load(document)
    document.close()
    return data


def preprocess_data(data):
    parsed_data = []
    for index in range(len(data)):
        # tolower
        text = data[index]
        text = text.lower()
        #todo: remove Stopwords
        # remove punctuation , stopwords later
        text = nlp(text)
        new_text = []
        for token in text:
            if not token.is_punct:
                new_text.append(str(token.orth_))
        text = ' '.join(new_text)
        parsed_data.append(text)

        if index % 1000 == 0 and index > 0:
            print('Pickling progress so far.')
            save_pickle(parsed_data, 'parsed_data')

    return parsed_data


def load_embeddings(embeddings_index, filepath):
    print('Loading Conceptnet Numberbatch word embeddings')
    with open(filepath, encoding='utf-8') as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            embedding = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = embedding
    print('Word embeddings:', len(embeddings_index))


def count_word_frequency(word_frequency, data):
    for text in data:
        for token in text.split():
            if token not in word_frequency:
                word_frequency[token] = 1
            else:
                word_frequency[token] += 1


def create_conversion_dictionaries(word_frequency, embeddings_index, threshold=15):
    print('Removing token which frequency in the corpus is under specified threshold')
    missing_words = 0

    for token, freq in word_frequency.items():
        if freq > threshold:
            if token not in embeddings_index:
                missing_words += 1

    missing_ratio = round(missing_words / len(word_frequency), 4) * 100
    print('Number of words missing from Conceptnet Numberbatch:', missing_words)
    print('Percent of words that are missing from vocabulary: ', missing_ratio, '%')

    # word to int dict
    vocab2int = {}
    value = 0
    for token, freq in word_frequency.items():
        if freq >= threshold or token in embeddings_index:
            vocab2int[token] = value
            value += 1

    # Seq2seq special tokens
    vocab2int['<UNK>'] = len(vocab2int)
    vocab2int['<PAD>'] = len(vocab2int)
    vocab2int['<EOS>'] = len(vocab2int)
    vocab2int['<GO>'] = len(vocab2int)

    # int to word dict
    int2vocab = {}
    for token, index in vocab2int.items():
        int2vocab[index] = token

    usage_ratio = round(len(vocab2int) / len(word_frequency), 4) * 100
    print("Total number of unique words:", len(word_frequency))
    print("Number of words we will use:", len(vocab2int))
    print("Percent of words we will use: {}%".format(usage_ratio))

    return vocab2int, int2vocab


def create_embedding_matrix(vocab2int, embeddings_index, embedding_dimensions=300):
    # Number of words in total in the corpus
    num_words = len(vocab2int)

    print('Creating word embedding matrix with all the tokens and their corresponding vectors.')
    word_embedding_matrix = np.zeros((num_words, embedding_dimensions), dtype=np.float32)
    for token, index in vocab2int.items():
        if token in embeddings_index:
            # If the token is pretrained in CN's vectors, use that
            word_embedding_matrix[index] = embeddings_index[token]
        else:
            # Else, create a random embedding for it
            new_embedding = np.array(np.random.uniform(-1.0, 1.0, embedding_dimensions))
            word_embedding_matrix[index] = new_embedding

    return word_embedding_matrix


def convert_data_to_ints(data, vocab2int, word_count, unk_count, eos=True):
    converted_data = []
    for text in data:
        converted_text = []
        for token in text.split():
            word_count += 1
            if token in vocab2int:
                # Convert each token in the paragraph to int and append it
                converted_text.append(vocab2int[token])
            else:
                # If it's not in the dictionary, use the int for <UNK> token instead
                converted_text.append(vocab2int['<UNK>'])
                unk_count += 1
        if eos:
            # Append <EOS> token if specified
            converted_text.append(vocab2int['<EOS>'])

        converted_data.append(converted_text)

    assert len(converted_data) == len(data)
    return converted_data, word_count, unk_count


data_inputs = load_pickle('train_squad_paragraphs')
data_targets = load_pickle('train_squad_questions')
assert len(data_targets) == len(data_inputs)
print('Loaded {} question/answer pairs.'.format(len(data_inputs)))

print('Preprocessing inputs')
# remove punctuation + tolower
try:
    parsed_inputs = load_pickle('parsed_inputs')
except:
    parsed_inputs = preprocess_data(data_inputs)
    save_pickle(parsed_inputs, 'parsed_inputs3')
try:
    parsed_targets = load_pickle('parsed_targets')
except:
    parsed_targets = preprocess_data(data_targets)
    save_pickle(parsed_targets, 'parsed_targets3')

filepath = 'numberbatch-en-17.06.txt'
embeddings_index = {}
load_embeddings(embeddings_index, filepath)

word_frequency = {}
count_word_frequency(word_frequency, parsed_targets)
count_word_frequency(word_frequency, parsed_inputs)

vocab2int, int2vocab = create_conversion_dictionaries(word_frequency, embeddings_index)
save_pickle(vocab2int, 'vocab2int')
save_pickle(int2vocab, 'int2vocab')

word_embedding_matrix = create_embedding_matrix(vocab2int, embeddings_index)
del embeddings_index
save_pickle(word_embedding_matrix, 'word_embedding_matrix')

word_count = 0
unk_count = 0

print('Converting text to integers')
converted_inputs, word_count, unk_count = convert_data_to_ints(parsed_inputs,
                                                               vocab2int,
                                                               word_count,
                                                               unk_count)
converted_targets, word_count, unk_count = convert_data_to_ints(parsed_targets,
                                                                vocab2int,
                                                                word_count,
                                                                unk_count)

unk_percent = round(unk_count / word_count, 4) * 100
print('Total number of words:', word_count)
print('Total number of UNKs:', unk_count)
print('Percent of words that are UNK:', unk_percent)

save_pickle(converted_inputs, 'sorted_inputs')
save_pickle(converted_targets, 'sorted_targets')

assert len(converted_inputs) == len(converted_targets)

print('Preprocessing data finished!')
#todo: remove wronge lenght data and sort