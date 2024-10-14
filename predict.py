import dataset
import betting
import tensorflow as tf
import numpy as np
import csv
import logging

TRAINING_SET_FRACTION = 0.95

# Set up logging
logging.basicConfig(level=logging.INFO)

def main(argv):
    data = dataset.Dataset('data/book.csv')

    train_results_len = int(TRAINING_SET_FRACTION * len(data.processed_results))
    train_results = data.processed_results[:train_results_len]
    test_results = data.processed_results[train_results_len:]

    def map_results(results):
        features = {}

        for result in results:
            for key in result.keys():
                if key not in features:
                    features[key] = []

                features[key].append(result[key])

        for key in features.keys():
            features[key] = np.array(features[key])

        return features, features['result']

    train_features, train_labels = map_results(train_results)
    test_features, test_labels = map_results(test_results)

    # Create dataset input function using tf.data.Dataset
    def make_input_fn(features, labels, batch_size, num_epochs=None, shuffle=True):
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(labels))
        dataset = dataset.batch(batch_size).repeat(num_epochs)
        return dataset

    train_input_fn = lambda: make_input_fn(train_features, train_labels, batch_size=500, num_epochs=None, shuffle=True)
    test_input_fn = lambda: make_input_fn(test_features, test_labels, batch_size=500, num_epochs=1, shuffle=False)

    # Define feature columns
    feature_columns = []

    for mode in ['home', 'away']:
        feature_columns += [
            tf.feature_column.numeric_column(key=f'{mode}-wins'),
            tf.feature_column.numeric_column(key=f'{mode}-draws'),
            tf.feature_column.numeric_column(key=f'{mode}-losses'),
            tf.feature_column.numeric_column(key=f'{mode}-goals'),
            tf.feature_column.numeric_column(key=f'{mode}-opposition-goals'),
            tf.feature_column.numeric_column(key=f'{mode}-shots'),
            tf.feature_column.numeric_column(key=f'{mode}-shots-on-target'),
            tf.feature_column.numeric_column(key=f'{mode}-opposition-shots'),
            tf.feature_column.numeric_column(key=f'{mode}-opposition-shots-on-target'),
        ]

    # Define the DNNClassifier
    model = tf.estimator.DNNClassifier(
        model_dir='model/',
        hidden_units=[10],
        feature_columns=feature_columns,
        n_classes=3,
        label_vocabulary=['H', 'D', 'A'],
        optimizer=tf.optimizers.Adagrad(learning_rate=0.1, initial_accumulator_value=0.1, l1_regularization_strength=0.001)
    )

    # Open CSV log for training results
    with open('training-log.csv', 'w') as stream:
        csvwriter = csv.writer(stream)
        csvwriter.writerow(['Steps', 'Accuracy', 'Average Loss', 'Performance'])  # Add header

        for i in range(0, 200):
            model.train(input_fn=train_input_fn, steps=100)
            evaluation_result = model.evaluate(input_fn=test_input_fn)

            predictions = list(model.predict(input_fn=test_input_fn))
            prediction_result = betting.test_betting_stategy(predictions, test_features, test_labels)

            csvwriter.writerow([(i + 1) * 100, evaluation_result['accuracy'], evaluation_result['average_loss'], prediction_result['performance']])

if __name__ == '__main__':
    # TensorFlow 2.x does not have tf.app.run; simply call the main function.
    main(None)
