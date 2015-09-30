# -*- coding: utf-8 -*-
from __future__ import absolute_import, division

import os, pickle, tempfile, zipfile

cimport numpy as cnp
import numpy as np

import sklearn.ensemble, sklearn.neighbors

from ..common.structs cimport FLOAT_t
from ..common.constants import FLOAT_dtype, Detection_dtype


class Storage(object):
    """Class for storing vocabulary data of specified category and recognizing it with a specified regressor

    The class is aware of the special *'negative'* category.
    """

    def __init__(self, dimensions, category, preprocessors = None, regressor = None):
        """Initialize storage object

        :Parameters:
            dimensions: int
                Number of feature vector dimensions
            category: str
                Category of stored features
            preprocessors: list, optional
                List of preprocessors which support fit() and transform() functions, empty by default
            regressor: object, optional
                Regressor to fit and use for recognition, ``sklearn.neighbors.KNeighborsRegressor(n_neighbors=1)`` by default
        """
        self.GROWTH_FACTOR = 2  # factor to use when resizing internal storages for storing new entries

        self._dims = dimensions
        self._category = category

        # initialize internal arrays
        self._entries = 0       # number of features/detections
        self._features = np.empty((0, self.dims), dtype = FLOAT_dtype)
        if self.category != 'negative':
            # initialize array for (positive) detections
            self._detections = np.empty(0, dtype = Detection_dtype)
        else:
            # no need to store detections for the negative category
            self._detections = None

        # call to property setters, sets self._learned
        self.preprocessors = preprocessors
        self.regressor = regressor

    @property
    def dims(self):
        return self._dims
    @property
    def category(self):
        return self._category
    @property
    def entries(self):
        return self._entries
    @property
    def preprocessors(self):
        return self._preprocessors
    @preprocessors.setter
    def preprocessors(self, preprocessors):
        if self.category != 'negative':
            self._learned = False
        else:
            # negative storages can always produce negative detections
            self._learned = True
        if preprocessors is None:
            self._preprocessors = []
        else:
            self._preprocessors = preprocessors
    @property
    def regressor(self):
        return self._regressor
    @regressor.setter
    def regressor(self, regressor):
        if self.category != 'negative':
            self._learned = False
        else:
            # negative storages can always produce negative detections
            self._learned = True
        if regressor is None:
            self._regressor = sklearn.neighbors.KNeighborsRegressor(n_neighbors=1)
        else:
            self._regressor = regressor
    @property
    def isEmpty(self):
        return (self.entries == 0)
    @property
    def isPreparedForRecognition(self):
        """Return *True* if storage can be used for recognition"""
        return self._learned

    def save(self, file):
        """Save storage data to file or file-like object

        Compression will be applied if possible.

        file: str or file-like object
            If file is a string, the file will be created
        """
        try:
            # try compression
            import zlib
            compression = zipfile.ZIP_DEFLATED
        except:
            # save uncompressed
            compression = zipfile.ZIP_STORED

        # reduce array sizes
        self._shrink()

        params = self._getParams()
        # write NumPy arrays to temporary file to avoid memory overflows
        with tempfile.NamedTemporaryFile(delete = False) as tf:
            if self.category != 'negative':
                np.savez(tf, features = self._features, detections = self._detections)
            else:
                np.savez(tf, features = self._features)
            tf_name = tf.name
        # temporary file has to be closed on Windows plattforms to be opened again (by ZipFile)
        # create final zip file or file-like object
        with zipfile.ZipFile(file, 'w', compression) as zf:
            zf.writestr('params.dict', pickle.dumps(params, protocol = pickle.HIGHEST_PROTOCOL))
            zf.write(tf_name, 'data.npz')
        # delete temporary file
        os.remove(tf_name)

    @classmethod
    def load(cls, file):
        """Load storage from file or file-like object and return the new instance

        file: str or file-like object
            If file is a string, the file will be opened and read
        """
        with zipfile.ZipFile(file, 'r') as zf:
            # read params
            params = pickle.load(zf.open('params.dict', 'r'))
            result = cls._fromParams(params)

            # read data
            data_file = zf.open('data.npz', 'r')
            # NumPy can't use the object returned by ZipFile.open(), so write it to temporary file
            with tempfile.NamedTemporaryFile(delete = False) as tf:
                tf.write(data_file.read())
                tf_name = tf.name
            # create np.lib.npyio.NpzFile object to read the data
            with np.load(tf_name) as data:
                # load data
                # convert to correct float format in case we are loading a storage saved in a different one
                result._features = data['features'].astype(FLOAT_dtype, copy = False)
                if result.category != 'negative':
                    # in case the category is negative, result._detections was already initialized to None
                    result._detections = data['detections'].astype(Detection_dtype, copy = False)
            # remove temporary file
            os.remove(tf_name)
        return result

    def addWords(self, cnp.ndarray[FLOAT_t, ndim=2] features, cnp.ndarray detections = None):
        """Add words to storage

        The internal storage will grow at least by factor *self.GROWTH_FACTOR* or more if there is more data to store.

        :Parameters:
            features: np.ndarray[FLOAT_dtype, ndim=2]
                Features to learn
            detections: np.ndarray[:const:`~pydriver.Detection_dtype`], optional
                Corresponding detections to learn, only required for the non-negative category
        """
        if self.category == 'negative':
            assert detections is None, "Can't pass detections to a storage for the negative category"
        else:
            assert features.shape[0] == detections.shape[0], "Numbers of features (%d) and detections (%d) do not match" % (features.shape[0], detections.shape[0])
        assert features.shape[1] == self.dims, "Features have wrong number of dimensions, expected %d, got %d" % (self.dims, features.shape[1])

        if self.category != 'negative':
            # reset learned flag (negative storages can always produce negative detections)
            self._learned = False

        # number of new words
        newWords = features.shape[0]

        # resize features and detections arrays to be able to store at least the new elements
        # use self.GROWTH_FACTOR to reduce memory copy operations
        # compute new minimal required size
        minSize = self.entries + newWords
        if self._features.shape[0] < minSize:
            # we don't have enough space, resize internal storages
            newSize = max(int(self._features.shape[0] * self.GROWTH_FACTOR), minSize)
            self._features.resize((newSize, self.dims), refcheck = False)
            if self.category != 'negative':
                self._detections.resize(newSize, refcheck = False)

        # copy new words to internal arrays
        self._features[self.entries : self.entries+newWords] = features
        if self.category != 'negative':
            self._detections[self.entries : self.entries+newWords] = detections
        self._entries += newWords

    def prepareForRecognition(self, nMaxRandomSamples=0, keepData=False, verbose=False):
        """Prepare storage for recognition"""
        if self.category == 'negative':
            # negative storages don't need to learn anything
            return

        if self.isEmpty:
            raise RuntimeError("Can't prepare non-negative storage for recognition if empty.")
        # get training features and detections
        X, y = self._getTrainingData()
        if verbose:
            print("Storage learning, category: %s, samples: %d, dimensions: %d" % (self.category, X.shape[0], X.shape[1]))
        if nMaxRandomSamples > 0:
            if verbose:
                print("Drawing %d random samples..." % min(nMaxRandomSamples, X.shape[0]))
            randoms = np.random.choice(X.shape[0], min(nMaxRandomSamples, X.shape[0]), replace = False)
            X = X[randoms]
            y = y[randoms]
        # fit and utilize preprocessors
        for preprocessor in self.preprocessors:
            if verbose:
                print("Fitting storage preprocessor...")
            X = preprocessor.fit_transform(X)
        # train regressor
        if verbose:
            print("Fitting storage regressor...")
        self.regressor.fit(X, y)
        # set flag indicating we are ready for recognition
        self._learned = True
        if not keepData:
            # delete training data since it's not required for recognition
            # we need NumPy arrays (with 0 elements) to remain compatible with loading and saving functions
            self._entries = 0
            self._features = np.empty((0, self.dims), dtype = FLOAT_dtype)
            if self.category != 'negative':
                self._detections = np.empty(0, dtype = Detection_dtype)
            else:
                self._detections = None

    def recognizeFeatures(self, cnp.ndarray[FLOAT_t, ndim=2] features):
        """Produce detections for given features

        The result will be a list of arrays with the same number of list elements as features.

        Detection weights will be set to 1.0.

        The current implementation produces exactly one detection per feature.

        :Parameters:
            features: np.ndarray[FLOAT_dtype, ndim=2]
                Array with features

        :Returns: detections
        :Returntype: list(np.ndarray[:const:`~pydriver.Detection_dtype`])
        """
        cdef:
            int i

        assert features.shape[1] == self.dims, "Features have wrong number of dimensions, expected %d, got %d" % (self.dims, features.shape[1])

        # resulting detections
        result = []
        if features.shape[0] == 0:
            # no features, nothing to do
            return result

        # initialize array with detections
        detections = np.zeros(features.shape[0], dtype = Detection_dtype)
        # set Detection category
        detections['category'] = self.category.encode('ascii')
        if self.category == 'negative':
            # weight has no meaning for negative category, set to 1.0
            detections['weight'] = 1.0
        else:
            if not self.isPreparedForRecognition:
                raise ValueError("Can't recognize features, storage is not prepared for recognition")

            # utilize preprocessors
            X = features    # X can change its data type depending on preprocessing
            for preprocessor in self.preprocessors:
                X  = preprocessor.transform(X)
            # utilize regressor
            predictions = self.regressor.predict(X)
            # convert output to Detection_dtype
            detections['position']['x'] = predictions[:, 0]
            detections['position']['y'] = predictions[:, 1]
            detections['position']['z'] = predictions[:, 2]
            detections['position']['rotation_y'] = predictions[:, 3]
            detections['height'] = predictions[:, 4]
            detections['width'] = predictions[:, 5]
            detections['length'] = predictions[:, 6]
            # weight is currently not used here, set to 1.0
            detections['weight'] = 1.0
        # for each feature add its detection as array with one element to overall list of detections
        for i in range(features.shape[0]):
            result.append(detections[i].reshape(1))
        return result

    def _getTrainingData(self):
        """Get tuple (X, y) with features and detections for training"""
        # shrink internal arrays since we currently don't expect them to grow anymore
        self._shrink()
        X = self._features
        # note: 'position' has 4 dimensions
        y = self._detections[['position', 'height', 'width', 'length']].view(FLOAT_dtype).reshape(self.entries, 7)
        return X, y

    def _getParams(self):
        """Get parameters dictionary to save"""
        params = {
                  'dims': self.dims,
                  'category': self.category,
                  'preprocessors': self.preprocessors,
                  'regressor': self.regressor,
                  'learned': self._learned,
                  'entries': self.entries,
                  }
        return params

    @classmethod
    def _fromParams(cls, params):
        """Create instance from loaded parameters dictionary

        The NumPy arrays of the result remain uninitialized.
        """
        result = cls(
            dimensions = params['dims'],
            category = params['category'],
            preprocessors = params['preprocessors'],
            regressor = params['regressor'],
        )
        result._learned = params['learned']
        result._entries = params['entries']
        return result

    def _shrink(self):
        """Resize internal arrays to be exactly as large as needed"""
        self._features.resize(self.entries, self.dims)
        if self.category != 'negative':
            self._detections.resize(self.entries)


class Vocabulary(object):
    """Vocabulary class to perform classifiction and recognition

    The data for each category will be stored in its :class:`Storage` instance.
    """

    def __init__(self, dimensions, preprocessors = None, classifier = None, storageGenerator = None, balanceNegatives = True):
        """Initialize vocabulary object

        :Parameters:
            dimensions: int
                Number of feature vector dimensions
            preprocessors: list, optional
                List of preprocessors which support fit() and transform() functions, empty by default
            classifier: object, optional
                Classifier object to use for category classification, ``sklearn.ensemble.AdaBoostClassifier(n_estimators=50)`` by default
            storageGenerator: callable, optional
                Function to use for creating new internal storage objects, must accept dimensions and category as first positional arguments, ``Storage(dimensions, category)`` by default
            balanceNegatives: bool, optional
                Flag whether to restrict the number of negative samples to maximum number of available samples for positive category, *True* by default
        """
        self._dims = dimensions
        self._balanceNegatives = balanceNegatives

        # flag whether vocabulary is prepared for recognition
        self._learned = False
        # dictionary with storage instances
        # key: category, value: storage instance
        self._storages = {}

        # calls to property setters
        self.preprocessors = preprocessors
        self.classifier = classifier
        self.storageGenerator = storageGenerator

    @property
    def dims(self):
        return self._dims
    @property
    def preprocessors(self):
        return self._preprocessors
    @preprocessors.setter
    def preprocessors(self, preprocessors):
        self._learned = False
        if preprocessors is None:
            self._preprocessors = []
        else:
            self._preprocessors = preprocessors
    @property
    def classifier(self):
        return self._classifier
    @classifier.setter
    def classifier(self, classifier):
        self._learned = False
        if classifier is None:
            self._classifier = sklearn.ensemble.AdaBoostClassifier(n_estimators=50)
        else:
            self._classifier = classifier
    @property
    def storageGenerator(self):
        return self._storageGenerator
    @storageGenerator.setter
    def storageGenerator(self, storageGenerator):
        if storageGenerator is None:
            # assign default storage generator function
            self._storageGenerator = _default_storageGenerator
        else:
            self._storageGenerator = storageGenerator
    @property
    def isEmpty(self):
        """Return *True* if vocabulary has not enough words in it or *False* otherwise

        The vocabulary must contain negative examples and at least one other category.
        """
        if 'negative' not in self._storages or len(self._storages) < 2:
            # need at least the negative storage and a positive one
            return True
        if self._storages['negative'].isEmpty:
            # negative storage must have words
            return True

        for category in self._storages:
            if category != 'negative' and not self._storages[category].isEmpty:
                # at least one non-negative storage is not empty
                return False
        # all non-negative storages are empty
        return True
    @property
    def isPreparedForRecognition(self):
        """Return *True* if vocabulary can be used for recognition"""
        return self._learned and all([s.isPreparedForRecognition for s in self._storages.values()])

    def save(self, file):
        """Save vocabulary to file or file-like object

        No further compression is applied, but sub-objects (like :class:`storages <Storage>`) can compress their data.

        .. warning:: The storageGenerator function will NOT be saved. Only Storages which were already created will be preserved.

        file: str or file-like object
            If file is a string, the file will be created
        """
        params = self._getParams()
        with zipfile.ZipFile(file, 'w') as zf:
            zf.writestr('params.dict', pickle.dumps(params, protocol = pickle.HIGHEST_PROTOCOL))

            for category, storage in self._storages.items():
                # write storage to temporary file to avoid memory overflows
                with tempfile.NamedTemporaryFile(delete = False) as tf:
                    storage.save(tf)
                    tf_name = tf.name
                # temporary file has to be closed on Windows plattforms to be opened again (by ZipFile)
                # write storage data to final zip file or file-like object
                zf.write(tf_name, '%s.str' % category)
                # delete temporary file
                os.remove(tf_name)

    @classmethod
    def load(cls, file):
        """Load vocabulary from file or file-like object and return the new instance

        .. warning:: The storageGenerator function will NOT be loaded. Only Storages which were already created can be used.

        file: str or file-like object
            If file is a string, the file will be opened and read
        """
        with zipfile.ZipFile(file, 'r') as zf:
            # read params
            params = pickle.load(zf.open('params.dict', 'r'))
            result = cls._fromParams(params)

            for category in params['categories']:
                # read storage file into temporary file since ZipFile has problems reading from ZipFile.open() directly
                with tempfile.NamedTemporaryFile(delete = False) as tf:
                    tf.write(zf.open('%s.str' % category, 'r').read())
                    tf_name = tf.name
                # load storage
                result._storages[category] = Storage.load(tf_name)
                # remove temporary file
                os.remove(tf_name)
        return result

    def addWords(self, cnp.ndarray[FLOAT_t, ndim=2] features, cnp.ndarray detections = None):
        """Add words to vocabulary

        All detections must be of the same category.

        :Parameters:
            features: np.ndarray[FLOAT_dtype, ndim=2]
                Features to learn
            detections: np.ndarray[:const:`~pydriver.Detection_dtype`], optional
                Corresponding detections to learn, only required for the non-negative category
        """
        if features.shape[0] == 0:
            assert detections is None or detections.shape[0] == 0, "No features given, but number of detections is %d" % detections.shape[0]
            # nothing to do
            return

        # reset learned flag
        self._learned = False

        # determine category
        if detections is None:
            category = 'negative'
        else:
            assert features.shape[0] == detections.shape[0], "Numbers of features (%d) and detections (%d) do not match" % (features.shape[0], detections.shape[0])
            # Python 3 note: "categories" contains bytes (and not strings)
            categories = np.unique(detections['category'])
            assert categories.shape[0] == 1, "Detections contain more than one category. Encountered categories: %s" % categories
            # get unicode string
            category = categories[0].decode('ascii')
            # prevent accidental use of bytes
            del categories
        # add words to the category storage, create the storage if necessary
        if category == 'negative' and self._balanceNegatives:
            # only add data if number of negative samples is under the maximum number of samples for a positive category
            negStorage = self._getStorage(category)
            if negStorage._entries <= max([0]+[s._entries for s in self._storages.values() if s.category != 'negative']):
                negStorage.addWords(features, detections)
        else:
            self._getStorage(category).addWords(features, detections)

    def prepareForRecognition(self, nMaxRandomSamples=0, nStorageMaxRandomSamples=0, keepData=False, verbose=False):
        """Prepare vocabulary for recognition"""
        if self.isEmpty:
            raise RuntimeError("Can't prepare for recognition if empty.")
        self._prepareVocabulary(nMaxRandomSamples=nMaxRandomSamples, keepData=keepData, verbose=verbose)
        self._prepareStorages(nMaxRandomSamples=nStorageMaxRandomSamples, keepData=keepData, verbose=verbose)
        # set flag indicating we are ready for recognition
        self._learned = True

    def _prepareVocabulary(self, nMaxRandomSamples=0, keepData=False, verbose=False):
        # get training features and categories
        X, y = self._getTrainingData()
        if verbose:
            print("Vocabulary learning, samples: %d, dimensions: %d" % (X.shape[0], X.shape[1]))
        if nMaxRandomSamples > 0:
            if verbose:
                print("Drawing %d random samples..." % min(nMaxRandomSamples, X.shape[0]))
            randoms = np.random.choice(X.shape[0], min(nMaxRandomSamples, X.shape[0]), replace = False)
            X = X[randoms]
            y = y[randoms]
        # fit and utilize preprocessors
        for preprocessor in self.preprocessors:
            if verbose:
                print("Fitting vocabulary preprocessor...")
            X = preprocessor.fit_transform(X)
        # train classifier
        if verbose:
            print("Fitting vocabulary classifier...")
        self.classifier.fit(X, y)

    def _prepareStorages(self, nMaxRandomSamples=0, keepData=False, verbose=False):
        for storage in self._storages.values():
            storage.prepareForRecognition(nMaxRandomSamples=nMaxRandomSamples, keepData=keepData, verbose=verbose)

    def recognizeFeatures(self, cnp.ndarray[FLOAT_t, ndim=2] features, suppressNegatives):
        """Recognize given features

        The result will contain a list of arrays with detections with the same number of list elemnents as features.

        Detection weights will be multiplied with the probability of recognizing the correct category.

        :Parameters:
            features: np.ndarray[FLOAT_dtype, ndim=2]
                Array with features
            suppressNegatives: bool
                Flag whether to produce only positive detections

        :Returns: detections
        :Returntype: list(np.ndarray[:const:`~pydriver.Detection_dtype`])
        """
        cdef:
            cnp.ndarray[cnp.float64_t, ndim=2] probabilities
            cnp.ndarray categories
            int i, j
        if not self.isPreparedForRecognition:
            raise RuntimeError("Vocabulary not prepared for recognition, try to call prepareForRecognition() first.")

        # resulting detections
        result = []
        if features.shape[0] == 0:
            # no features, nothing to do
            return result

        # classify feature categories
        # utilize preprocessors
        X = features    # X can change its data type depending on preprocessing
        for preprocessor in self.preprocessors:
            X = preprocessor.transform(X)
        # category probabilities, array of shape (nFeatures, nCategories)
        probabilities = self.classifier.predict_proba(X)
        if suppressNegatives:
            # get id of negative category
            idNegative = np.where(self.classifier.classes_ == 'negative')[0][0]
            # set probability for negative detection to a negative value so the best positive category wins
            probabilities[:, idNegative] = -1
        # ids of most probable categories
        categories = np.argmax(probabilities, axis = 1)
        for i in range(features.shape[0]):
            # recognize feature using the storage for this category
            # _storages already contain every category which can be classified, no need to call _getStorage()
            detections_list = self._storages[self.classifier.classes_[categories[i]]].recognizeFeatures(features[i].reshape(1, self.dims))
            for detections in detections_list:
                detections['weight'] *= probabilities[i, categories[i]]
            result += detections_list
        return result

    def _getParams(self):
        """Get parameters dictionary to save"""
        params = {
                  'dims': self.dims,
                  'preprocessors': self.preprocessors,
                  'classifier': self.classifier,
                  'categories': list(self._storages.keys()),
                  'learned': self._learned,
                  }
        return params

    @classmethod
    def _fromParams(cls, params):
        """Create instance from loaded parameters dictionary

        The storages remain uninitialized.
        """
        result = cls(
            dimensions = params['dims'],
            preprocessors = params['preprocessors'],
            classifier = params['classifier'],
        )
        result._learned = params['learned']
        return result

    def _getStorage(self, category):
        """Get storage for the given classification category"""
        if category not in self._storages:
            # create a new storage for the category if not already done, create a new regressor object for it
            self._storages[category] = self.storageGenerator(self.dims, category)
        return self._storages[category]

    def _getTrainingData(self):
        """Get tuple (X, y) with features and categories for classification training"""
        # lists with arrays
        features = []
        categories = []
        for category, storage in self._storages.items():
            cur_features = storage._features[:storage.entries]
            if cur_features.shape[0] > 0:
                # append array of features for this category
                features.append(cur_features)
                # append array with categories, one entry for each feature, all entries have the same category
                categories.append(np.array([category]*cur_features.shape[0]))
        # concatenate features and categories arrays
        X = np.concatenate(features)
        y = np.concatenate(categories)
        return X, y


# --- internal functions ---

def _default_storageGenerator(dimensions, category):
#    """Default storage generator function"""
    return Storage(dimensions = dimensions, category = category)
