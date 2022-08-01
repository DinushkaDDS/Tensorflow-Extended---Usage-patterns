
from scripts.tft_trainer_support import *
import tensorflow_transform as tft
import os

def _gzip_reader_fn(filenames):
    '''
    Helper to read tfrecords compressed using gzip compression
    '''
    return tf.data.TFRecordDataset(filenames,
        compression_type='GZIP')

def input_fn(file_pattern, tf_transform_output, batch_size=32):
    '''
    Helper function to return dataset as batches.
    '''
    transformed_feature_spec = (tf_transform_output.transformed_feature_spec().copy())
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=_gzip_reader_fn,
        label_key=tranformed_name(LABEL_KEY))
    return dataset

def _get_serve_prediction_fn(model, tf_transform_output):
    """Returns a function that parses a serialized tf.Example."""

    # Loading the transform graph from the Transform component output
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        """Returns the output to be used in the serving signature."""
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)

        # Parsing the request data to tf.Example format
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)

        # Do the transformation based on the loaded transformation graph.
        transformed_features = model.tft_layer(parsed_features)

        # Do the inference
        outputs = model(transformed_features)

        # Return output
        return {"outputs": outputs}

    return serve_tf_examples_fn

def run_fn(args):

    # Loading data to the training process
    tf_transform_output = tft.TFTransformOutput(args.transform_output)
    train_dataset = input_fn(args.train_files, tf_transform_output) 
    eval_dataset = input_fn(args.eval_files, tf_transform_output)

    model = get_model()
    
    # Defining callbacks for tensorboard
    log_dir = os.path.join("logs")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, update_freq="batch", histogram_freq= 1
    )
    callbacks = [tensorboard_callback]

    model.fit(
        train_dataset,
        epochs=10,
        steps_per_epoch=args.train_steps,
        validation_data=eval_dataset,
        validation_steps=args.eval_steps,
        callbacks=callbacks, # Adding callback for the tensorboard
    )
   
    signatures = {
        'serving_default':
            _get_serve_prediction_fn(model, tf_transform_output)\
                    .get_concrete_function(
                        tf.TensorSpec(
                            shape=[None],
                            dtype=tf.string,
                            name='examples')
                    )
    }

    model.save(args.serving_model_dir, save_format='tf', signatures=signatures)
