import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries (size number of models) where each key a word and value is Log Liklihood
           [{SOMEWORD_Model': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """

    # TODO implement the recognizer
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    for index, word in enumerate(test_set.wordlist):
        Y = test_set._hmm_data[index][0]
        lengths =  test_set._hmm_data[index][1]
        test_word_probabilities = {}
        max_logLvalue = float("-inf")
        best_guess = None
        for word_model, model in models.items():
            try:
                logLvalue = model.score(Y, lengths)
                if logLvalue > max_logLvalue:
                    max_logLvalue = logLvalue
                    best_guess = word_model
            except:
                logLvalue = None
                pass
            test_word_probabilities[word_model] = logLvalue

        probabilities.append(test_word_probabilities)
        guesses.append(best_guess)



    return probabilities, guesses

