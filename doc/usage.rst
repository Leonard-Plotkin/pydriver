Usage Example
-------------

This example shows PyDriver training and evaluation steps using the "Objects" dataset of
the `KITTI Vision Benchmark Suite <http://www.cvlibs.net/datasets/kitti/eval_object.php>`_.
You will need Velodyne point clouds, camera calibration matrices, training labels and
optionally both left and right color images if you set USE_IMAGE_COLOR to True. The only
required adjustment is to set the correct path to KITTI training data on your system.

You can find this code in the "examples" directory.


.. code-block:: python

    # example for using the PyDriver framework with KITTI objects dataset
    # this code performs classifier training on one half of the dataset and evaluates it on the other

    # adjust this path to point to the training directory of KITTI objects dataset
    PATH_TO_KITTI = 'path_to_KITTI/objects/training'

    import copy, datetime
    import sklearn, sklearn.cluster, matplotlib.pyplot as plt

    # import the PyDriver framework
    import pydriver

    # training and testing
    TRAINING_FRAMES = 3740
    TESTING_FRAMES = 3741
    SHOT_RADIUS = 2.0
    # point cloud coloring (False: reflectance, True: camera)
    USE_IMAGE_COLOR = False

    # evaluation
    # mandatory ground truth categories
    DETECTION_CATEGORIES = ['car',]
    # optional ground truth categories
    DETECTION_CATEGORIES_OPT = ['van',]
    MIN_OVERLAP = 0.7               # minimal overlap between 2D boxes
    EVALUATION_MODE = 'moderate'    # mode (easy, moderate, hard)
    VISUALIZE3D = False             # show 3D visualization of detections

    # initialize reader for KITTI objects dataset
    reader = pydriver.datasets.kitti.KITTIObjectsReader(PATH_TO_KITTI)
    # initialize lidar reconstructor
    reconstructor = pydriver.preprocessing.LidarReconstructor(
        useImageColor=USE_IMAGE_COLOR,
        removeInvisible=True,
        )
    # default preprocessor
    preprocessor = pydriver.preprocessing.Preprocessor(reconstructor)
    keypointExtractor = pydriver.keypoints.ISSExtractor(salientRadius=0.25, nonMaxRadius=0.25)
    featureExtractor = pydriver.features.SHOTColorExtractor(shotRadius=SHOT_RADIUS, fixedY=-1.0)
    # list with feature types (arbitrary unique name) to use and their dimensionality
    featureTypes = [('myfeature', featureExtractor.dims),]

    # function the vocabularies will use to create category storages
    def storageGenerator(dims, category):
        sto = pydriver.detectors.vocabularies.Storage(dims, category,
            preprocessors=[],
            regressor=sklearn.neighbors.KNeighborsRegressor(n_neighbors=1),
            )
        return sto

    # function the detector will use to create vocabularies
    def vocabularyGenerator(dimensions, featureName):
        voc = pydriver.detectors.vocabularies.Vocabulary(
            dimensions,
            preprocessors=[
                sklearn.cluster.MiniBatchKMeans(n_clusters=100, batch_size=1000, max_iter=100),
                ],
            classifier=sklearn.ensemble.AdaBoostClassifier(n_estimators=75),
            storageGenerator=storageGenerator,
            balanceNegatives=True,
            )
        return voc

    # initialize detector that will perform learning and recognition
    detector = pydriver.detectors.Detector(featureTypes, vocabularyGenerator=vocabularyGenerator)

    # perform training with some frames in the dataset
    timeStart = datetime.datetime.now()
    for frame in reader.getFramesInfo(0, TRAINING_FRAMES):
        print('Training on frame %d...' % frame['frameId'])
        # reconstruct scene
        scene = preprocessor.process(frame)
        # get keypoints
        keypointCloud = keypointExtractor.getKeypointCloud(scene)
        # get lists with labels containing mandatory/optional ground truth
        groundTruth, groundTruthOpt = pydriver.datasets.kitti.getKITTIGroundTruth(
            frame['labels'],
            DETECTION_CATEGORIES,
            DETECTION_CATEGORIES_OPT,
            mode='moderate',    # use moderate mode for training
            )

        # initialize list of boxes that will contain non-negative examples
        # it will be used later to extract negative examples
        boxes3D_exclude = []
        for label in groundTruth + groundTruthOpt:
            # adjust labeled 3D box to applied scene transformations
            # in this setup the only transformation is ground plane adjustment
            box3D = pydriver.geometry.transform3DBox(label['box3D'], scene['transformation'])

            # avoid training with non-negative examples
            # the box is made bigger so SHOT features used as negatives
            # don't capture parts of the object
            box3D_exclude = copy.deepcopy(box3D)
            box3D_exclude['dimensions']['height'] += 2*SHOT_RADIUS
            box3D_exclude['dimensions']['width'] += 2*SHOT_RADIUS
            box3D_exclude['dimensions']['length'] += 2*SHOT_RADIUS
            boxes3D_exclude.append(box3D_exclude)

            # only use mandatory ground truth for training
            if label in groundTruth:
                # get keypoints which lie inside the labeled object box
                boxKeypointCloud = keypointCloud.extractOrientedBoxes([box3D])
                # extract features at these keypoints (and get new keypoints
                # which depend on the feature extractor)
                fkeypoints, features = featureExtractor.getFeatures(scene, boxKeypointCloud)
                # learn new features and relations between features and objects
                detector.addWords(label['category'], 'myfeature', features, fkeypoints, box3D)
        # get keypoints which lie outside of labeled object boxes
        negativeKeypointCloud = keypointCloud.extractOrientedBoxes(boxes3D_exclude, invert=True)
        # extract features at these keypoints
        fkeypoints, features = featureExtractor.getFeatures(scene, negativeKeypointCloud)
        # learn features associated with absence of objects
        detector.addWords('negative', 'myfeature', features)
    timeTraining = datetime.datetime.now() - timeStart

    # perform learning on stored data
    print('Learning...')
    timeStart = datetime.datetime.now()
    detector.learn(nStorageMaxRandomSamples=25000)
    timeLearning = datetime.datetime.now() - timeStart

    # initialize evaluator
    evaluator = pydriver.evaluation.Evaluator(minOverlap=MIN_OVERLAP, nPoints=100)
    # perform testing with frames which were not used for training
    firstFrame = TRAINING_FRAMES
    lastFrame = TRAINING_FRAMES + TESTING_FRAMES
    timeStart = datetime.datetime.now()
    for frame in reader.getFramesInfo(firstFrame, lastFrame):
        print('Testing on frame %d...' % frame['frameId'])
        # see the training part above
        scene = preprocessor.process(frame)
        keypointCloud = keypointExtractor.getKeypointCloud(scene)
        groundTruth, groundTruthOpt = pydriver.datasets.kitti.getKITTIGroundTruth(
            frame['labels'],
            DETECTION_CATEGORIES,
            DETECTION_CATEGORIES_OPT,
            mode=EVALUATION_MODE,
            )

        # extract keypoints and features for the whole scene
        fkeypoints, features = featureExtractor.getFeatures(scene, keypointCloud)
        # perform recognition on extracted features
        detections = detector.recognize({'myfeature': (fkeypoints,features)})

        # convert 3D detections (NumPy array) to labels (list of
        # dictionaries) that include 2D box projections used for evaluation
        # and revert the transformation of the scene, so they have the same
        # coordinate system as the original KITTI labels
        detections_labels = pydriver.datasets.detections2labels(
            detections,
            scene['transformation'].I,    # inverse matrix
            frame['calibration']['projection_left'],
            scene['img_left'].shape,
            )
        # exclude detections which are always considered optional by
        # KITTI (i.e. in 'hard' mode) and will not positively contribute
        # to performance
        detections_labels = [l for l in detections_labels if \
                              l['info']['truncated'] <= 0.5 and \
                              l['box2D']['bottom']-l['box2D']['top'] >= 25.0
                            ]

        # add frame recognition results to evaluator
        evaluator.addFrame(groundTruth, groundTruthOpt, detections_labels)
        if VISUALIZE3D:
            # perform visualization in the transformed cloud
            # convert ground truth labels to ground truth detections
            gtd = pydriver.datasets.labels2detections(groundTruth, scene['transformation'])
            gtdOpt = pydriver.datasets.labels2detections(groundTruthOpt, scene['transformation'])
            scene['cloud'].visualizeDetections(detections, gtd, gtdOpt)
    timeEvaluation = datetime.datetime.now() - timeStart

    # show evaluation results
    print("Training time: %s" % timeTraining)
    print("Learning time: %s" % timeLearning)
    print("Evaluation time: %s" % timeEvaluation)
    print("Average precision: %.2f" % evaluator.aprecision)
    print("Average orientation similarity: %.2f" % evaluator.aos)
    values = evaluator.getValues()
    plt.figure()
    plt.plot(values['recall'], values['precision'],
        label='Precision (AP %0.2f)' % evaluator.aprecision)
    plt.plot(values['recall'], values['OS'],
        label='Orientation similarity (AOS %0.2f)' % evaluator.aos)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision / OS')
    plt.legend(loc="upper right")
    plt.show()
