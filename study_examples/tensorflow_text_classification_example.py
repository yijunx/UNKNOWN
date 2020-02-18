import tensorflow as tf
from tensorflow import keras
import numpy as np
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

data = keras.datasets.imdb


(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=10000)

# data here are frequency of words
# train_data[0] = [1, 14, 22 ....]
# lets find the mapping of these words

word_index = data.get_word_index()  # => give tuples

# turn that into a dictionary
word_to_num_dic = {word: (value + 3) for word, value in word_index.items()}
word_to_num_dic['<PAD>'] = 0
word_to_num_dic['<START>'] = 1
word_to_num_dic['<UNK>'] = 2
word_to_num_dic['<UNUSED>'] = 3

# reverse the dic
num_to_word_dic = dict([(value, key) for (key, value) in word_to_num_dic.items()])

# now lets use the dic...

# make all data same length, more than 250, chop, less then 250, appending with pad (number 0) at back
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_to_num_dic['<PAD>'],
                                                        padding='post',
                                                        maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_to_num_dic['<PAD>'],
                                                       padding='post',
                                                       maxlen=250)
# just for have a look, haha
def decode_review(review):
    return " ".join([num_to_word_dic.get(i, "?") for i in review])

# print(decode_review(test_data[0]))
# # time to create the model
# amodel = keras.Sequential()
#
# # add layers one by one here
# amodel.add(keras.layers.Embedding(10000, 16))
# # what is the embedding layer?
# #
#
# amodel.add(keras.layers.GlobalAveragePooling1D())
# amodel.add(keras.layers.Dense(16, activation='relu'))    # linear rectify
# amodel.add(keras.layers.Dense(1, activation='sigmoid'))  # good or bad, so sigmoid, for the 1,0 label
#
# amodel.summary()
#
# amodel.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
#
# # a further split
# x_val = train_data[:10000]  # val => validation
# x_train = train_data[10000:]
#
# y_val = train_labels[:10000]  # val => validation
# y_train = train_labels[10000:]
#
# # now lets fit the model
#
# fitModel = amodel.fit(x_train,
#                       y_train,
#                       epochs=40,
#                       batch_size=512,  # how many review being loaded in one shot, we got too many reviews..
#                       validation_data=(x_val, y_val),
#                       verbose=1)
#
# results = amodel.evaluate(test_data, test_labels)
# # (loss, accuracy)
# print(results)
#
#
# # now lets use the model on the test data
#
# a_review = test_data[3]
# prediction = amodel.predict([a_review])
# print(len(a_review))
# print(prediction)
# print(len(test_labels))
# print('review: ')
# print(decode_review(a_review))
# print("prediction: " + str(prediction[3]))
# print("prediction0: " + str(prediction[0]))
# print("actual: " + str(test_labels[3]))
# amodel.save("model_text_classification.h5")

amodel = keras.models.load_model("model_text_classification.h5")

def review_encode(s):
    encoded = [1]  # 1 is the <START>
    for word in s:
        if word.lower() in word_to_num_dic:
            encoded.append(word_to_num_dic[word.lower()])
        else:
            encoded.append(2)  # 2 is <UNKNOWN>

    return encoded

# test on real data
with open("test.txt", encoding="utf-8") as f:
    for line in f.readlines():
        nline = line.replace(",", "").replace(".", "").replace("'", "").replace('"', "").replace("(", "").replace(")", "").replace(":", "").split(' ')
        encode = review_encode(nline)
        encode = keras.preprocessing.sequence.pad_sequences([encode],
                                                       value=word_to_num_dic['<PAD>'],
                                                       padding='post',
                                                       maxlen=250)
        # word_to_num_dic
        predict = amodel.predict(encode)
        print(line)
        print(encode)
        print(predict[0])
        print(predict)

print('i dont understand why, now i understand why')
print(test_data[0])
predict = amodel.predict(np.array([test_data[0]]))
print(predict[0])
print(predict)
print(test_labels[0])
