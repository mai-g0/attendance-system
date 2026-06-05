import os
import pickle
import sys

import numpy as np
from sklearn import metrics, neighbors, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import _classification as sklearn_neighbors_classification
from sklearn.neighbors import _dist_metrics as sklearn_neighbors_dist_metrics
from sklearn.neighbors import _kd_tree as sklearn_neighbors_kd_tree

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

sys.modules['sklearn.neighbors.classification'] = sklearn_neighbors_classification
sys.modules['sklearn.neighbors.kd_tree'] = sklearn_neighbors_kd_tree
sys.modules['sklearn.neighbors.dist_metrics'] = sklearn_neighbors_dist_metrics

BASE_DIR = os.path.dirname(__file__) + '/'
PATH_TO_PKL = 'trained_classifier.pkl'


class FaceClassifier(object):
    def __init__(self, model_path=None):
        self.model = None

        if model_path is None:
            return
        elif model_path == 'default':
            model_path = BASE_DIR + PATH_TO_PKL

        print("Loading classifier:", model_path)
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f, encoding='latin1')
            print("Classifier loaded")
        except BaseException as e:
            print("Classifier load failed:", repr(e))
            raise

    def my_rmse(self, labels, predictions):
        pred_values = predictions['predictions']
        return {'rmse': tf.metrics.root_mean_squared_error(labels, pred_values)}

    def train(self, X, y, model='random-forests', knn_neighbours=3, save_model_path=None):
        if model.lower() in ['knn', 'svm', 'random-forests', 'svc-poly', 'svc-rbf']:
            if model.lower() == 'knn':
                self.model = neighbors.KNeighborsClassifier(knn_neighbours, weights='uniform')
            elif model.lower() == 'svm':
                self.model = svm.SVC(kernel='linear', probability=True)
            elif model.lower() == 'random-forests':
                self.model = RandomForestClassifier()
            elif model.lower() == 'svc-rbf':
                self.model = svm.SVC(kernel='rbf', probability=True)
            elif model.lower() == 'svc-poly':
                self.model = svm.SVC(kernel='poly', probability=True)

            self.model.fit(X, y)
            print(metrics.accuracy_score(y, self.model.predict(X)))

            if save_model_path is not None:
                with open(save_model_path, 'wb') as f:
                    pickle.dump(self.model, f)
        else:
            feature_columns = [tf.feature_column.numeric_column("X", shape=[512])]

            eval_interval = 300
            run_config = tf.estimator.RunConfig(
                save_checkpoints_secs=eval_interval,
                keep_checkpoint_max=3
            )

            classifier = tf.estimator.DNNClassifier(
                feature_columns=feature_columns,
                hidden_units=[256, 128],
                n_classes=len(X),
                model_dir="NN_model",
                config=run_config
            )

            train_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"X": np.array(X)},
                y=np.arange(len(y)),
                num_epochs=None,
                shuffle=True
            )

            classifier = tf.contrib.estimator.add_metrics(classifier, self.my_rmse)

            train_spec = tf.estimator.TrainSpec(
                input_fn=train_input_fn,
                max_steps=2000
            )

            eval_spec = tf.estimator.EvalSpec(
                input_fn=train_input_fn,
                steps=200,
                start_delay_secs=60,
                throttle_secs=eval_interval
            )

            tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
            accuracy_score = classifier.evaluate(input_fn=train_input_fn)["accuracy"]

            print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

    def classify(self, descriptor, model_type="SVM"):
        if self.model is None:
            print('Train the model before doing classifications.')
            return None, None

        if model_type.lower() in ['knn', 'svm', 'random-forests', 'svc-poly', 'svc-rbf']:
            prediction = self.model.predict([descriptor])[0]
            probabilities = self.model.predict_proba([descriptor])[0]
            print("sklearn model", prediction, probabilities)
            return prediction, probabilities

        return None, None

    def ensemble(self, descriptor):
        with open('./classifier/new_classifiers/trained_svm.pkl', 'rb') as f:
            model1 = pickle.load(f, encoding='latin1')
        with open('./classifier/new_classifiers/knn_7.pkl', 'rb') as f:
            model2 = pickle.load(f, encoding='latin1')
        with open('./classifier/new_classifiers/random_forests.pkl', 'rb') as f:
            model3 = pickle.load(f, encoding='latin1')

        model_avg = (
            model1.predict_proba([descriptor])[0]
            + 0.75 * model2.predict_proba([descriptor])[0]
            + 0.75 * model3.predict_proba([descriptor])[0]
        ) / 2.5

        return np.argmax(model_avg), model_avg
        #return (model1.predict([descriptor][0])+model2.predict([descriptor])[0]+model3.predict([descriptor])[0])/3,
        #    (model1.predict_proba([descriptor])[0]+model2.predict_proba([descriptor])[0]+model3.predict_proba([descriptor])[0])/3