from ner_baseline import get_data, get_data2
from pos_rnn import RNN

def main():
    Xtrain, Ytrain, Xtest, Ytest, word2idx, tag2idx = get_data2(split_sequences=True)
    V = len(word2idx)
    rnn = RNN(50, [50], V)
    rnn.fit(Xtrain, Ytrain, epochs=30)
    print "train f1 score:", rnn.f1_score(Xtrain, Ytrain)
    print "test f1 score:", rnn.f1_score(Xtest, Ytest)
    

if __name__ == '__main__':
    main()
