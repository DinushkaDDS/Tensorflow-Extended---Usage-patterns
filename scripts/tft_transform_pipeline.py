
import tensorflow as tf
import tensorflow_transform as tft

# THE CODE IS FROM THE REFERENCE BOOK. SLIGHTLY MODIFIED TO MY LIKING!

LABEL_KEY = "consumer_disputed"

# feature name, feature dimensionality
ONE_HOT_FEATURES = {
    "product": 11,
    "sub_product": 45,
    "company_response": 5,
    "state": 60,
    "issue": 90,
}

# feature name, bucket count
BUCKET_FEATURES = {"zip_code": 10}

# feature name, value is unused
TEXT_FEATURES = {"consumer_complaint_narrative": None}

def transformed_name(key):
    '''
    Helper function to assign names to transformed features. 
    You give the input key and this will return the key modified.
    '''
    return key + "_xf"

def fill_missing_values(x):
    '''
    Helper to fill the missing values of a given input feature.
    If input is a Sparse tensor it would be returned as a Dense tensor.
    '''

    if isinstance(x, tf.sparse.SparseTensor):
        default_value = "" if x.dtype == tf.string else 0

        x = tf.sparse.to_dense(
            tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
            default_value,
        )
    return tf.squeeze(x, axis=1)

def convert_num_to_one_hot(cat_tensor, num_labels = 2):
    """
    Helper to convert categorical features into a one-hot vector
    """
    one_hot_tensor = tf.one_hot(cat_tensor, num_labels)
    return tf.reshape(one_hot_tensor, [-1, num_labels])


def convert_zip_code(zipcode):
    """
    Helper to convert zipcode string to int64 representation. 
    In the dataset some zipcodes have XXX values in the bottom. 
    So we are replaing it with zeros.
    """
    zipcode = tf.strings.regex_replace(zipcode, r"X{0,5}", "0")
    zipcode = tf.strings.to_number(zipcode, out_type=tf.float32)
    return zipcode



def preprocessing_fn(inputs):
    """
    Actuall preprocessing is done here.
    tf.transform's callback function for preprocessing inputs.
    """
    outputs = {} # Remember we need to output a dictionary of features.

    for key in ONE_HOT_FEATURES.keys():
        dim = ONE_HOT_FEATURES[key]
        int_value = tft.compute_and_apply_vocabulary(
            fill_missing_values(inputs[key]), top_k=dim + 1
        )
        outputs[transformed_name(key)] = convert_num_to_one_hot(
            int_value, num_labels=dim + 1
        )

    for key, bucket_count in BUCKET_FEATURES.items():
        temp_feature = tft.bucketize(
            convert_zip_code(fill_missing_values(inputs[key])),
            bucket_count,
        )
        outputs[transformed_name(key)] = convert_num_to_one_hot(
            temp_feature, num_labels=bucket_count + 1
        )

    for key in TEXT_FEATURES.keys():
        outputs[transformed_name(key)] = fill_missing_values(inputs[key])

    outputs[transformed_name(LABEL_KEY)] = fill_missing_values(inputs[LABEL_KEY])

    return outputs
