import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    where L is the likelihood of the fitted model, p is the number of parameters,
    and N is the number of data points
    p is the sum of these 4 terms:
    - transition probabilities: n_components*(n_components-1)  the last row can be calculated from the others because they add to 1.0
    - starting probabilities: n_components-1  the  starting probability for the last component can be calculated since they add up to 1.0
    - number of means: n_components * n_features
    - number of variances: n_components * n_features
    and therefore after simplifying -->  p = n_components^2 + 2*n_components*n_features - 1 = n_components^2 + 2*n_means - 1
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        best_model = None
        lowest_BIC = float('inf')
        # TODO implement model selection based on BIC scores
        for constant in range(self.min_n_components,self.max_n_components):
            try:
                model = self.base_model(constant)
                log_LH =  model.score(self.X, self.lengths)

                # p is the sum of these 4 terms:
                # - transition probabilities: n_components*(n_components-1)  the last row can be calculated from the others because they add to 1.0
                # - starting probabilities: n_components-1  the  starting probability for the last component can be calculated since they add up to 1.0
                # - number of means: n_components * n_features
                # - number of variances: n_components * n_features
                # and therefore after simplifying -->  p = n_components^2 + 2*n_components*n_features - 1 = n_components^2 + 2*n_means - 1
                #
                p = constant ** 2 +  (2*model.means_.size) - 1
                n = len(self.X)
                BIC = -2*log_LH + p* np.log(n)
                if (lowest_BIC > BIC  ):
                    lowest_BIC = BIC
                    best_model = model
            except :
                pass

        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        best_model = None
        highest_DIC = float('-inf')

        for constant in range(self.min_n_components,self.max_n_components):
            try:
                model = self.base_model(constant)
                log_LH =  model.score(self.X, self.lengths)
                other_logs = []
                for word in self.words:
                    if word != self.this_word:
                        X, lengths = self.hwords[word]
                        other_logs.append(model.score(X, lengths))

                DIC = log_LH  - np.average(other_logs)
                if (highest_DIC < DIC  ):
                    highest_DIC = DIC
                    best_model = model
            except :
                pass

        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV

        avg_logLikelihoods = {}
        words = self.sequences
        from sklearn.model_selection import KFold

        n_splits = len(words) if len(words) <3 else 3
        for constant in range(self.min_n_components,self.max_n_components):
            logs = []
            if n_splits == 1: #k-fold cross-validation not possible: it requires at least one train/test split by setting n_splits=2 or more, as there is only one sample it score on the training sample
                try:
                    model = self.base_model(constant)
                    log_LH =  model.score(self.X, self.lengths)
                    logs.append(log_LH)
                except:
                    pass
            else:
                split_method = KFold(n_splits)
                for cv_train_idx, cv_test_idx in split_method.split(words):
                    self.X, self.lengths = combine_sequences(cv_train_idx,words)
                    Y, y_lengths = combine_sequences(cv_test_idx,words)
                    try:
                        model = self.base_model(constant)
                        log_LH =  model.score(Y, y_lengths)
                        logs.append(log_LH)
                    except:
                        pass

            avg_logLikelihoods[constant] = np.mean(np.array([logs]))
        best_num_components = max(avg_logLikelihoods, key = lambda key: avg_logLikelihoods[key])
        return self.base_model(best_num_components)


