
import tensorflow as tf
import tensorflow_hub as hub

LABEL_KEY = "consumer_disputed"
ONE_HOT_FEATURES = {
    "product": 11,
    "sub_product": 45,
    "company_response": 5,
    "state": 60,
    "issue": 90,
}
BUCKET_FEATURES = {"zip_code": 10}
TEXT_FEATURES = {"consumer_complaint_narrative": None}

def tranformed_name(key):
    return key + "_xf"

def get_model():

    input_features = []
    
    # These 2 types would go to the broad network segment!
    for key, dim in ONE_HOT_FEATURES.items():
        input_features.append(
            tf.keras.Input(shape=(dim+1,), name=tranformed_name(key))
        )

    for key, dim in BUCKET_FEATURES.items():
        input_features.append(
            tf.keras.Input(shape=(dim+1,), name=tranformed_name(key))
        )

    input_texts = []
    for key in TEXT_FEATURES.keys():
        input_texts.append(
            tf.keras.Input(shape=(1,), name=tranformed_name(key), dtype=tf.string)
        )

    inputs = input_features + input_texts

    # URL to get the sentence embedding model from tensorflow hub
    MODULE_URL = "https://tfhub.dev/google/universal-sentence-encoder/4"
    embed_layer = hub.KerasLayer(MODULE_URL)
    reshaped_text_input = tf.reshape(input_texts[0], [-1])
    embedded_text_inputs = embed_layer(reshaped_text_input)
    reshaped_embeddings = tf.keras.layers.Reshape((512,), input_shape=(1, 512))(embedded_text_inputs)

    deep = tf.keras.layers.Dense(256, activation='relu')(reshaped_embeddings)
    deep = tf.keras.layers.Dense(64, activation='relu')(deep)
    deep = tf.keras.layers.Dense(16, activation='relu')(deep)

    wide_ff = tf.keras.layers.concatenate(input_features)
    wide = tf.keras.layers.Dense(16, activation='relu')(wide_ff)
    
    both = tf.keras.layers.concatenate([deep, wide])
    output = tf.keras.layers.Dense(1, activation='sigmoid')(both)

    keras_model = tf.keras.models.Model(inputs, output) 
    keras_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                        loss='binary_crossentropy',
                        metrics=[
                            tf.keras.metrics.BinaryAccuracy(),
                            tf.keras.metrics.TruePositives()
                        ])
    return keras_model
