
import tempfile
import tensorflow_transform.beam as tft_beam
import pprint
import tensorflow as tf
from tensorflow_transform.tf_metadata import dataset_metadata, schema_utils

raw_data = [
            {'x': 1.20},
            {'x': 2.99},
            {'x': 4.00},
            ]

raw_data_metadata = dataset_metadata.DatasetMetadata(
    schema_utils.schema_from_feature_spec({
        'x': tf.io.FixedLenFeature([], tf.float32),
    }))

def preprocessing_fn(inputs):
    x = inputs['x']
    x_normalized = tft.scale_to_0_1(x)

    return { 'x_xf': x_normalized }

with tft_beam.Context(temp_dir=tempfile.mkdtemp()):
    transformed_dataset, transform_fn = (  
        (raw_data, raw_data_metadata) | tft_beam.AnalyzeAndTransformDataset(preprocessing_fn))

transformed_data, transformed_metadata = transformed_dataset
pprint.pprint(transformed_data)
