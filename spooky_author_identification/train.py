import pandas as pd
import nltk

def model():
    train = pd.read_csv('./train.csv')
    test = pd.read_csv('./test.csv')

    by_author = train.groupby('author')

    fre_word_by_author = nltk.probability.ConditionalFreqDist()

    for name, group in by_author:
        sentences = group['text'].str.cat(sep=' ')
        sentences = sentences.lower()
        tokens = nltk.tokenize.word_tokenize(sentences)
        frequency = nltk.FreqDist(tokens)
        fre_word_by_author[name] = frequency

    '''
    for i in fre_word_by_author.keys():
        print('blood ' + i)
        print(fre_word_by_author[i].freq('blood'))
        print(fre_word_by_author[i])
    '''

    test_sentence = 'It was a dark and stormy night'
    pre_process_tokens = nltk.tokenize.word_tokenize(test_sentence.lower())
    test_prob = pd.DataFrame(columns=['author', 'word', 'probability'])

    for i in fre_word_by_author:
        for j in pre_process_tokens:
            word_freq = fre_word_by_author[i].freq(j) + 0.000001
            output = pd.DataFrame([[i, j, word_freq]], columns=['author', 'word', 'probability'])

            test_prob = test_prob.append(output, ignore_index=True)

    print(test_prob)

    for i in fre_word_by_author.keys():
        one_author = test_prob.query('author=="' + i + '"')
        print(one_author)

        joint_prob = one_author.product(numeric_only=True)[0]
        print(joint_prob)


if __name__ == '__main__':
    model()