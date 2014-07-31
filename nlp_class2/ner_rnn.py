# Course URL:
# https://deeplearningcourses.com/c/natural-language-processing-with-deep-learning-in-python
# https://udemy.com/natural-language-processing-with-deep-learning-in-python
from ner_baseline import get_data
from pos_rnn import RNN

def main():
    Xtrain, Ytrain, Xtest, Ytest, word2idx, tag2idx = get_data(split_sequences=True)
    V = len(word2idx)
    rnn = RNN(10, [10], V)
    rnn.fit(Xtrain, Ytrain, epochs=70)
    print "train f1 score:", rnn.f1_score(Xtrain, Ytrain)
    print "test f1 score:", rnn.f1_score(Xtest, Ytest)
    

if __name__ == '__main__':
    main()
