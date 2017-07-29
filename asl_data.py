import os

import numpy as np
import pandas as pd


class AslDb(object):
    """ American Sign Language database drawn from the RWTH-BOSTON-104 frame positional data

    This class has been designed to provide a convenient interface for individual word data for students in the Udacity AI Nanodegree Program.

    For example, to instantiate and load train/test files using a feature_method 
	definition named features, the following snippet may be used:
        asl = AslDb()
        asl.build_training(tr_file, features)
        asl.build_test(tst_file, features)

    Reference for the original ASL data:
    http://www-i6.informatik.rwth-aachen.de/~dreuw/database-rwth-boston-104.php
    The sentences provided in the data have been segmented into isolated words for this database
    """

    def __init__(self,
                 hands_fn=os.path.join('data', 'hands_condensed.csv'),
                 speakers_fn=os.path.join('data', 'speaker.csv'),
                 ):
        """ loads ASL database from csv files with hand position information by frame, and speaker information

        :param hands_fn: str
            filename of hand position csv data with expected format:
                video,frame,left-x,left-y,right-x,right-y,nose-x,nose-y
        :param speakers_fn:
            filename of video speaker csv mapping with expected format:
                video,speaker

        Instance variables:
            df: pandas dataframe
                snippit example:
                         left-x  left-y  right-x  right-y  nose-x  nose-y  speaker
            video frame
            98    0         149     181      170      175     161      62  woman-1
                  1         149     181      170      175     161      62  woman-1
                  2         149     181      170      175     161      62  woman-1

        """
        self.df = pd.read_csv(hands_fn).merge(pd.read_csv(speakers_fn),on='video')
        self.df.set_index(['video','frame'], inplace=True)

    def build_training(self, feature_list, csvfilename =os.path.join('data', 'train_words.csv')):
        """ wrapper creates sequence data objects for training words suitable for hmmlearn library

        :param feature_list: list of str label names
        :param csvfilename: str
        :return: WordsData object
            dictionary of lists of feature list sequence lists for each word
                {'FRANK': [[[87, 225], [87, 225], ...], [[88, 219], [88, 219], ...]]]}
        """
        return WordsData(self, csvfilename, feature_list)

    def build_test(self, feature_method, csvfile=os.path.join('data', 'test_words.csv')):
        """ wrapper creates sequence data objects for individual test word items suitable for hmmlearn library

        :param feature_method: Feature function
        :param csvfile: str
        :return: SinglesData object
            dictionary of lists of feature list sequence lists for each indexed
                {3: [[[87, 225], [87, 225], ...]]]}
        """
        return SinglesData(self, csvfile, feature_method)


class WordsData(object):
    """ class provides loading and getters for ASL data suitable for use with hmmlearn library

    """

    def __init__(self, asl:AslDb, csvfile:str, feature_list:list):
        """ loads training data sequences suitable for use with hmmlearn library based on feature_method chosen

        :param asl: ASLdata object
        :param csvfile: str
            filename of csv file containing word training start and end frame data with expected format:
                video,speaker,word,startframe,endframe
        :param feature_list: list of str feature labels
        """
        self._data = self._load_data(asl, csvfile, feature_list)
        self._hmm_data = create_hmmlearn_data(self._data)
        self.num_items = len(self._data)
        self.words = list(self._data.keys())

    def _load_data(self, asl, fn, feature_list):
        """ Consolidates sequenced feature data into a dictionary of words

        :param asl: ASLdata object
        :param fn: str
            filename of csv file containing word training data
        :param feature_list: list of str
        :return: dict
        """
        tr_df = pd.read_csv(fn)
        dict = {}
        for i in range(len(tr_df)):
            word = tr_df.ix[i,'word']
            video = tr_df.ix[i,'video']
            new_sequence = [] # list of sample lists for a sequence
            for frame in range(tr_df.ix[i,'startframe'], tr_df.ix[i,'endframe']+1):
                vid_frame = video, frame
                sample = [asl.df.ix[vid_frame][f] for f in feature_list]
                if len(sample) > 0:  # dont add if not found
                    new_sequence.append(sample)
            if word in dict:
                dict[word].append(new_sequence) # list of sequences
            else:
                dict[word] = [new_sequence]
        return dict

    def get_all_sequences(self):
        """ getter for entire db of words as series of sequences of feature lists for each frame

        :return: dict
            dictionary of lists of feature list sequence lists for each word
                {'FRANK': [[[87, 225], [87, 225], ...], [[88, 219], [88, 219], ...]]],
                ...}
        """
        return self._data

    def get_all_Xlengths(self):
        """ getter for entire db of words as (X, lengths) tuple for use with hmmlearn library

        :return: dict
            dictionary of (X, lengths) tuple, where X is a numpy array of feature lists and lengths is
            a list of lengths of sequences within X
                {'FRANK': (array([[ 87, 225],[ 87, 225], ...  [ 87, 225,  62, 127], [ 87, 225,  65, 128]]), [14, 18]),
                ...}
        """
        return self._hmm_data

    def get_word_sequences(self, word:str):
        """ getter for single word series of sequences of feature lists for each frame

        :param word: str
        :return: list
            lists of feature list sequence lists for given word
                [[[87, 225], [87, 225], ...], [[88, 219], [88, 219], ...]]]
        """
        return self._data[word]

    def get_word_Xlengths(self, word:str):
        """ getter for single word (X, lengths) tuple for use with hmmlearn library

        :param word:
        :return: (list, list)
            (X, lengths) tuple, where X is a numpy array of feature lists and lengths is
            a list of lengths of sequences within X
                (array([[ 87, 225],[ 87, 225], ...  [ 87, 225,  62, 127], [ 87, 225,  65, 128]]), [14, 18])
        """
        return self._hmm_data[word]


class SinglesData(object):
    """ class provides loading and getters for ASL data suitable for use with hmmlearn library

    """

    def __init__(self, asl:AslDb, csvfile:str, feature_list):
        """ loads training data sequences suitable for use with hmmlearn library based on feature_method chosen

        :param asl: ASLdata object
        :param csvfile: str
            filename of csv file containing word training start and end frame data with expected format:
                video,speaker,word,startframe,endframe
        :param feature_list: list str of feature labels
        """
        self.df = pd.read_csv(csvfile)
        self.wordlist = list(self.df['word'])
        self.sentences_index  = self._load_sentence_word_indices()
        self._data = self._load_data(asl, feature_list)
        self._hmm_data = create_hmmlearn_data(self._data)
        self.num_items = len(self._data)
        self.num_sentences = len(self.sentences_index)

    # def _load_data(self, asl, fn, feature_method):
    def _load_data(self, asl, feature_list):
        """ Consolidates sequenced feature data into a dictionary of words and creates answer list of words in order
        of index used for dictionary keys

        :param asl: ASLdata object
        :param fn: str
            filename of csv file containing word training data
        :param feature_method: Feature function
        :return: dict
        """
        dict = {}
        # for each word indexed in the DataFrame
        for i in range(len(self.df)):
            video = self.df.ix[i,'video']
            new_sequence = [] # list of sample dictionaries for a sequence
            for frame in range(self.df.ix[i,'startframe'], self.df.ix[i,'endframe']+1):
                vid_frame = video, frame
                sample = [asl.df.ix[vid_frame][f] for f in feature_list]
                if len(sample) > 0:  # dont add if not found
                    new_sequence.append(sample)
            if i in dict:
                dict[i].append(new_sequence) # list of sequences
            else:
                dict[i] = [new_sequence]
        return dict

    def _load_sentence_word_indices(self):
        """ create dict of video sentence numbers with list of word indices as values

        :return: dict
            {v0: [i0, i1, i2], v1: [i0, i1, i2], ... ,} where v# is video number and
                            i# is index to wordlist, ordered by sentence structure
        """
        working_df = self.df.copy()
        working_df['idx'] = working_df.index
        working_df.sort_values(by='startframe', inplace=True)
        p = working_df.pivot('video', 'startframe', 'idx')
        p.fillna(-1, inplace=True)
        p = p.transpose()
        dict = {}
        for v in p:
            dict[v] = [int(i) for i in p[v] if i>=0]
        return dict

    def get_all_sequences(self):
        """ getter for entire db of items as series of sequences of feature lists for each frame

        :return: dict
            dictionary of lists of feature list sequence lists for each indexed item
                {3: [[[87, 225], [87, 225], ...], [[88, 219], [88, 219], ...]]],
                ...}
        """
        return self._data

    def get_all_Xlengths(self):
        """ getter for entire db of items as (X, lengths) tuple for use with hmmlearn library

        :return: dict
            dictionary of (X, lengths) tuple, where X is a numpy array of feature lists and lengths is
            a list of lengths of sequences within X; should always have only one item in lengths
                {3: (array([[ 87, 225],[ 87, 225], ...  [ 87, 225,  62, 127], [ 87, 225,  65, 128]]), [14]),
                ...}
        """
        return self._hmm_data

    def get_item_sequences(self, item:int):
        """ getter for single item series of sequences of feature lists for each frame

        :param word: str
        :return: list
            lists of feature list sequence lists for given word
                [[[87, 225], [87, 225], ...]]]
        """
        return self._data[item]

    def get_item_Xlengths(self, item:int):
        """ getter for single item (X, lengths) tuple for use with hmmlearn library

        :param word:
        :return: (list, list)
            (X, lengths) tuple, where X is a numpy array of feature lists and lengths is
            a list of lengths of sequences within X; lengths should always contain one item
                (array([[ 87, 225],[ 87, 225], ...  [ 87, 225,  62, 127], [ 87, 225,  65, 128]]), [14])
        """
        return self._hmm_data[item]


def combine_sequences(sequences):
    '''
    concatenates sequences and return tuple of the new list and lengths
    :param sequences:
    :return: (list, list)
    '''
    sequence_cat = []
    sequence_lengths = []
    # print("num of sequences in {} = {}".format(key, len(sequences)))
    for sequence in sequences:
        sequence_cat += sequence
        num_frames = len(sequence)
        sequence_lengths.append(num_frames)
    return sequence_cat, sequence_lengths

def create_hmmlearn_data(dict):
    seq_len_dict = {}
    for key in dict:
        sequences = dict[key]
        sequence_cat, sequence_lengths = combine_sequences(sequences)
        seq_len_dict[key] = np.array(sequence_cat), sequence_lengths
    return seq_len_dict

if __name__ == '__main__':
    asl= AslDb()
    print(asl.df.ix[98, 1])
    # collect the features into a list
    features_ground = ['grnd-rx','grnd-ry','grnd-lx','grnd-ly']
    asl.df['grnd-ry'] = asl.df['right-y'] - asl.df['nose-y']
    asl.df['grnd-rx'] = asl.df['right-x'] - asl.df['nose-x']
    asl.df['grnd-ly'] = asl.df['left-y'] - asl.df['nose-y']
    asl.df['grnd-lx'] = asl.df['left-x'] - asl.df['nose-x']

    # TODO add features for normalized by speaker values of left, right, x, y
    # Name these 'norm-rx', 'norm-ry', 'norm-lx', and 'norm-ly'
    # using Z-score scaling (X-Xmean)/Xstd

    features_norm = ['norm-rx', 'norm-ry', 'norm-lx','norm-ly']
    # TODO Create a dataframe named `df_std` with standard deviations grouped by speaker
    df_std = asl.df.groupby('speaker').std()
    df_means = asl.df.groupby('speaker').mean()
    asl.df['left-x-mean']= asl.df['speaker'].map(df_means['left-x'])
    asl.df['left-y-mean']= asl.df['speaker'].map(df_means['left-y'])
    asl.df['right-x-mean']= asl.df['speaker'].map(df_means['right-x'])
    asl.df['right-y-mean']= asl.df['speaker'].map(df_means['right-y'])
    asl.df['left-x-std']= asl.df['speaker'].map(df_std['left-x'])
    asl.df['left-y-std']= asl.df['speaker'].map(df_std['left-y'])
    asl.df['right-x-std']= asl.df['speaker'].map(df_std['right-x'])
    asl.df['right-y-std']= asl.df['speaker'].map(df_std['right-y'])

    asl.df['norm-rx'] = (asl.df['right-x']-asl.df['right-x-mean'])/asl.df['right-x-std']
    asl.df['norm-ry'] = (asl.df['right-y']-asl.df['right-y-mean'])/asl.df['right-y-std']
    asl.df['norm-lx'] = (asl.df['left-x']-asl.df['left-x-mean'])/asl.df['left-x-std']
    asl.df['norm-ly'] = (asl.df['left-y']-asl.df['left-y-mean'])/asl.df['left-y-std']

    features_polar = ['polar-rr', 'polar-rtheta', 'polar-lr', 'polar-ltheta']
    import numpy as np
    asl.df['polar-rr'] = np.sqrt(asl.df['grnd-rx']**2 + asl.df['grnd-ry']**2 )
    asl.df['polar-lr'] = np.sqrt(asl.df['grnd-lx']**2 + asl.df['grnd-ly']**2 )
    asl.df['polar-rtheta'] = np.arctan2(asl.df['grnd-rx'],asl.df['grnd-ry'])
    asl.df['polar-ltheta'] = np.arctan2(asl.df['grnd-lx'],asl.df['grnd-ly'])

    # TODO add features for left, right, x, y differences by one time step, i.e. the "delta" values discussed in the lecture
    # Name these 'delta-rx', 'delta-ry', 'delta-lx', and 'delta-ly'

    features_delta = ['delta-rx', 'delta-ry', 'delta-lx', 'delta-ly']
    asl.df['delta-rx'] = asl.df['grnd-rx'].diff()
    asl.df['delta-rx'] = asl.df['delta-rx'].fillna(value=0.0 )
    asl.df['delta-ry'] = asl.df['grnd-ry'].diff()
    asl.df['delta-ry'] = asl.df['delta-ry'].fillna(value=0.0 )
    asl.df['delta-lx'] = asl.df['grnd-lx'].diff()
    asl.df['delta-lx'] = asl.df['delta-lx'].fillna(value=0.0 )
    asl.df['delta-ly'] = asl.df['grnd-ly'].diff()
    asl.df['delta-ly'] = asl.df['delta-ly'].fillna(value=0.0 )



    # TODO add features of your own design, which may be a combination of the above or something else
    # Name these whatever you would like

    # TODO define a list named 'features_custom' for building the training set
    features_custom = [ 'norm-polar-lr', 'norm-polar-rr', 'polar-ltheta', 'polar-rtheta',
                        'delta-polar-lr', 'delta-polar-rr', 'delta-ltheta', 'delta-rtheta']
    df_desc = asl.df.groupby('speaker').describe()
    df_means = asl.df.groupby('speaker').mean()
    df_std = asl.df.groupby('speaker').std()
    asl.df['polar-lr-mean']= asl.df['speaker'].map(df_means['polar-lr'])
    asl.df['polar-rr-mean']= asl.df['speaker'].map(df_means['polar-rr'])
    asl.df['polar-lr-std']= asl.df['speaker'].map(df_std['polar-lr'])
    asl.df['polar-rr-std']= asl.df['speaker'].map(df_std['polar-rr'])

    asl.df['norm-polar-lr'] = (asl.df['polar-lr']-asl.df['polar-lr-mean'])/asl.df['polar-lr-std']
    asl.df['norm-polar-rr'] = (asl.df['polar-rr']-asl.df['polar-rr-mean'])/asl.df['polar-rr-std']

    asl.df['delta-polar-lr'] = asl.df['polar-lr'].diff()
    asl.df['delta-polar-lr'] = asl.df['delta-polar-lr'].fillna(value=0.0 )
    asl.df['delta-polar-rr'] = asl.df['polar-rr'].diff()
    asl.df['delta-polar-rr'] = asl.df['delta-polar-rr'].fillna(value=0.0 )
    asl.df['delta-ltheta'] = asl.df['polar-ltheta'].diff()
    asl.df['delta-ltheta'] = asl.df['delta-ltheta'].fillna(value=0.0 )
    asl.df['delta-rtheta'] = asl.df['polar-rtheta'].diff()
    asl.df['delta-rtheta'] = asl.df['delta-rtheta'].fillna(value=0.0 )

    print(asl.df['norm-polar-lr'].head())
    print(asl.df['delta-polar-lr'].head())
    print(asl.df['polar-lr'].head())

    pd.set_option('display.max_columns', 50)
    print(asl.df['norm-polar-rr'].describe())
    print(asl.df.describe() )


    #show a single set of features for a given (video, frame) tuple
    #print([asl.df.ix[98,1][v] for v in features_ground] )
    #words_to_train = ['FISH', 'BOOK', 'VEGETABLE', 'FUTURE', 'JOHN']
    import timeit
    from my_model_selectors import SelectorCV
    from my_model_selectors import SelectorBIC
    from my_model_selectors import SelectorDIC
    from my_model_selectors import SelectorConstant


    #training = asl.build_training(features_ground)  # Experiment here with different feature sets defined in part 1
    #sequences = training.get_all_sequences()
    #Xlengths = training.get_all_Xlengths()
    #for word in words_to_train:
        #start = timeit.default_timer()
        #model = SelectorCV(sequences, Xlengths, word,
        #                   min_n_components=2, max_n_components=15, random_state = 14).select()
        #model = SelectorBIC(sequences, Xlengths, word,
        #                    min_n_components=2, max_n_components=15, random_state = 14).select()
        #model = SelectorDIC(sequences, Xlengths, word,
        #                    min_n_components=2, max_n_components=15, random_state = 14).select()

        #end = timeit.default_timer()-start
        #if model is not None:
            #print("Training complete for {} with {} states with time {} seconds".format(word, model.n_components, end))
        #else:
            #print("Training failed for {}".format(word))



    def train_all_words(features, model_selector):
        training = asl.build_training(features)  # Experiment here with different feature sets defined in part 1
        sequences = training.get_all_sequences()
        Xlengths = training.get_all_Xlengths()
        model_dict = {}
        for word in training.words:
            model = model_selector(sequences, Xlengths, word,
                                   n_constant=3).select()
            model_dict[word]=model
        return model_dict

    # TODO implement the recognize method in my_recognizer -- remove code below
    from my_recognizer import recognize
    from asl_utils import show_errors
    # TODO Choose a feature set and model selector
    features = features_custom # change as needed
    model_selector = SelectorConstant # change as needed
    model_selectors = [SelectorBIC, SelectorDIC]

    # TODO Recognize the test set and display the result with the show_errors method
    for model_selector in model_selectors:
        models = train_all_words(features, model_selector)
        test_set = asl.build_test(features)
        probabilities, guesses = recognize(models, test_set)
        show_errors(guesses, test_set)

