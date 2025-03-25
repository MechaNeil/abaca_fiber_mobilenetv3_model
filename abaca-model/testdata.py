import tensorflow as tf

tfrecord_path = 'abaca-fiber-classifiacation-dataset/train/Normal-Grade.tfrecord'
raw_dataset = tf.data.TFRecordDataset(tfrecord_path)

for raw_record in raw_dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    print(example)
