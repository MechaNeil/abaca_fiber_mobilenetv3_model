import tensorflow as tf
import re
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

# File paths (update these paths if needed)
test_record_fname = 'abaca-fiber-classifiacation-dataset/test/Normal-Grade.tfrecord'
train_record_fname = 'abaca-fiber-classifiacation-dataset/train/Normal-Grade.tfrecord'
label_map_pbtxt_fname = 'abaca-fiber-classifiacation-dataset/train/Normal-Grade_label_map.pbtxt'

# Step 1: Load the label map from the pbtxt file
def load_label_map(label_map_path):
    """
    Parses the label map pbtxt file and returns a dictionary mapping class names to ids.
    Example pbtxt format:
        item {
          id: 1
          name: 'S2'
        }
    """
    label_map = {}
    with open(label_map_path, 'r') as f:
        content = f.read()
    # Use regex to find each item block and extract id and name
    items = re.findall(r"item\s*{(.*?)}", content, re.DOTALL)
    for item in items:
        id_match = re.search(r"id:\s*(\d+)", item)
        name_match = re.search(r"name:\s*'([^']+)'", item)
        if id_match and name_match:
            label_map[name_match.group(1)] = int(id_match.group(1))
    return label_map

# Load the label map and print it
label_map = load_label_map(label_map_pbtxt_fname)
print("Label map:", label_map)


import tensorflow as tf

tfrecord_path = 'abaca-fiber-classifiacation-dataset/train/Normal-Grade.tfrecord'
raw_dataset = tf.data.TFRecordDataset(tfrecord_path)

for raw_record in raw_dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    print(example)

# Step 2: Define the TFRecord parsing function
# Assumes each TFRecord example has 'image' and 'label' fields
feature_description = {
    'image': tf.io.FixedLenFeature([], tf.string),  # image stored as JPEG bytes
    'label': tf.io.FixedLenFeature([], tf.int64)      # label as an integer id
}

def _parse_function(example_proto):
    # Parse the input tf.train.Example proto using the dictionary above.
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    # Decode the JPEG image, resize it to 224x224 (MobileNetV3 input size), and normalize pixel values.
    image = tf.io.decode_jpeg(parsed['image'], channels=3)
    image = tf.image.resize(image, [224, 224])
    image = image / 255.0
    # Cast label to integer (it should match the ids from the label map)
    label = tf.cast(parsed['label'], tf.int32)
    return image, label

# Step 3: Create a function to build a dataset from a TFRecord file
def get_dataset(tfrecord_fname, batch_size=32, shuffle_buffer=1000):
    dataset = tf.data.TFRecordDataset(tfrecord_fname)
    dataset = dataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(shuffle_buffer).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# Build training and testing datasets
train_dataset = get_dataset(train_record_fname)
test_dataset = get_dataset(test_record_fname)



# Step 4: Build the MobileNetV3-Large model for image classification


# Number of classes (using the label map)
num_classes = len(label_map)

# Load MobileNetV3-Large without the top classification layers.
base_model = tf.keras.applications.MobileNetV3Large(
    input_shape=(224, 224, 3),
    include_top=False,  # Exclude the pre-trained classification head
    weights='imagenet'
)
# Freeze the base model for transfer learning.
base_model.trainable = False

# Add a custom classification head.
x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.3)(x)  # Dropout to help prevent overfitting.
outputs = Dense(num_classes, activation='softmax')(x)  # num_classes = 7

# Build and compile the model.
model = Model(inputs=base_model.input, outputs=outputs)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# Step 5: Train the model using the training dataset and validate with the test dataset.
model.fit(train_dataset, epochs=10, validation_data=test_dataset)
