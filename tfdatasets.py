#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         : 19-6-12 下午8:50
# @Author       : ding
# @File         : tfdatasets.py
# @Description  : conver your data file into tfrecords and read a tfrecords into batch for train or test
import codecs
import os
import cv2

from tqdm import tqdm
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class TDDataset:
    def __init__(self, save_root, pattern="train_%03d_of_%03d", use_zip=False, use_repeat=True):
        """
        create a dataset
        :param save_root:
        :param pattern:
        :param use_zip:
        :param use_repeat:
        """
        if use_zip:
            self.pattern = pattern + '.zip'
            self.option = tf.io.TFRecordOptions(compression_type='GZIP')
            self.compression_type = 'GZIP'
        else:
            self.pattern = pattern + '.tfrecords'
            self.option = tf.io.TFRecordOptions()
            self.compression_type = None
        self.save_root = save_root
        self.use_repeat = use_repeat
        if not os.path.exists(save_root):
            os.mkdir(save_root)

    def create_features(self, line_string):
        """
        create a feature by line_string
        :param line_string:  line of string
        :return: a feature by tf.train.Features
        """
        img_dir, img_lab = line_string.split()
        image = cv2.imread(img_dir)
        img_buf = image.tostring()
        buf = tf.train.BytesList(value=[img_buf])
        lab = tf.train.BytesList(value=[img_lab.encode('utf-8')])
        shape = tf.train.Int64List(value=image.shape)
        ddir = tf.train.BytesList(value=[img_dir.encode('utf-8')])
        features = tf.train.Features(
            feature={
                "img_raw": tf.train.Feature(bytes_list=buf),
                "img_lab": tf.train.Feature(bytes_list=lab),
                "shape": tf.train.Feature(int64_list=shape),
                "img_dir": tf.train.Feature(bytes_list=ddir)
            }
        )

        return features

    def create_record(self, list_of_file, nfiles=1):
        """
        create a tfrecord file
        :param list_of_file: a list label file
        :param nfiles: split n tfrecord files
        :return:
        """
        lines = []
        for lfile in list_of_file:
            with codecs.open(lfile, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    lines.append(line.strip())
        nline = len(lines)
        nstep = nline // nfiles
        for i in range(nfiles):
            bidx = i * nstep
            eidx = min(bidx + nstep, nline)
            jpath = os.path.join(self.save_root, self.pattern % (i, nfiles))
            with tf.io.TFRecordWriter(jpath, options=self.option) as writer:
                for line in tqdm(lines[bidx:eidx]):
                    features = self.create_features(line)
                    try:
                        example = tf.train.Example(features=features)
                        writer.write(example.SerializeToString())
                    except AttributeError:
                        pass

    def parse_example(self, serialize_example_tensor):
        """
        parse a serialize example to a normal dict example
        :param serialize_example_tensor: a serialize example
        :return: img and lab tensor
        """
        expected_features = {
            "img_raw": tf.io.VarLenFeature(dtype=tf.string),
            "img_lab": tf.io.VarLenFeature(dtype=tf.string),
            "shape": tf.io.FixedLenFeature([3], tf.int64),
            "img_dir": tf.io.VarLenFeature(dtype=tf.string)
        }
        example = tf.io.parse_single_example(serialize_example_tensor, expected_features)
        img_raws = tf.sparse.to_dense(example['img_raw'], default_value=b'')
        img_labs = tf.sparse.to_dense(example['img_lab'], default_value=b'')
        shape = example["shape"]

        img = tf.io.decode_raw(img_raws, tf.uint8)
        img = tf.reshape(img, shape)
        img = tf.image.resize(img, [32, 200])
        img /= 255.0
        lab = img_labs[0]
        return img, lab

    def create_batcher(self, list_of_records, batch_size=32, number_of_reader=5,
                       number_of_thread=5, shuffle_buffer_size=10000):
        """
        create batcher for dataset
        :param list_of_records: record files list
        :param batch_size: batch size
        :param number_of_reader: number of reader for load
        :param number_of_thread: thread to read
        :param shuffle_buffer_size: shuffer cache size
        :return: a batcher dataset
        """
        dataset = tf.data.Dataset.list_files(list_of_records)
        if self.use_repeat:
            dataset = dataset.repeat()
        dataset = dataset.interleave(
            lambda filename: tf.data.TFRecordDataset(filename, compression_type=self.compression_type),
            cycle_length=number_of_reader
        )
        dataset.shuffle(shuffle_buffer_size)
        dataset = dataset.map(self.parse_example,
                              num_parallel_calls=number_of_thread)
        dataset = dataset.batch(batch_size)
        return dataset

    def sample_testing(self, list_of_records):
        """
        just testing for your records file
        :param list_of_records: list of record path
        :return:
        """
        features = {
            "img_raw": tf.io.VarLenFeature(dtype=tf.string),
            "img_lab": tf.io.VarLenFeature(dtype=tf.string),
            "shape": tf.io.FixedLenFeature([3], dtype=tf.int64),
            "img_dir": tf.io.VarLenFeature(dtype=tf.string)
        }
        dataset = tf.data.TFRecordDataset(list_of_records, compression_type=self.compression_type)
        for serialized_example_tensor in dataset:
            example = tf.io.parse_single_example(serialized_example_tensor, features)
            print(example)
            img_lab = tf.sparse.to_dense(example['img_lab'], default_value=b'')
            print("img label", img_lab.numpy()[0].decode('utf-8'))
            img_raw = tf.sparse.to_dense(example['img_raw'], default_value=b'')
            print(img_raw)
            image = tf.io.decode_raw(img_raw, out_type=tf.uint8)
            print("img shape and type", image.shape, image.dtype)
            print("shape", example["shape"].numpy())
            img_dir = tf.sparse.to_dense(example['img_dir'], default_value=b'')
            print("img_dir", img_dir.numpy()[0].decode('utf-8'))


class OCRTextDataset(TDDataset):
    def __init__(self, save_root, size_objs, pattern="train_%03d_of_%03d", use_zip=False, use_repeat=True):
        super(OCRTextDataset, self).__init__(save_root, pattern, use_zip, use_repeat)
        self.size = tf.constant(size_objs, dtype=tf.float32)
        self.height, self.width = tf.split(self.size, 2)

    def create_features(self, line_string):
        content = line_string.split()
        img_dir = content[0]
        img_txt = ' '.join(content[1:])
        image = cv2.imread(img_dir)
        img_buf = image.tostring()
        buf = tf.train.BytesList(value=[img_buf])
        txt = tf.train.BytesList(value=[img_txt.encode('utf-8')])
        shape = tf.train.Int64List(value=image.shape)
        ddir = tf.train.BytesList(value=[img_dir.encode('utf-8')])
        features = tf.train.Features(
            feature={
                "img_raw": tf.train.Feature(bytes_list=buf),
                "img_txt": tf.train.Feature(bytes_list=txt),
                "shape": tf.train.Feature(int64_list=shape),
                "img_dir": tf.train.Feature(bytes_list=ddir)
            }
        )

        return features

    def parse_example(self, serialize_example_tensor):
        expected_features = {
            "img_raw": tf.io.VarLenFeature(dtype=tf.string),
            "img_txt": tf.io.VarLenFeature(dtype=tf.string),
            "shape": tf.io.FixedLenFeature([3], tf.int64),
            "img_dir": tf.io.VarLenFeature(dtype=tf.string)
        }
        example = tf.io.parse_single_example(serialize_example_tensor, expected_features)
        img_raws = tf.sparse.to_dense(example["img_raw"], default_value=b'')
        img_txts = tf.sparse.to_dense(example['img_txt'], default_value=b'')
        shape = example["shape"]

        img = tf.io.decode_raw(img_raws, tf.uint8)
        img = tf.reshape(img, shape)
        # img = tf.image.resize(img, [32, 200])
        fshape = tf.cast(shape, dtype=tf.float32)
        h, w, c = tf.split(fshape, 3)
        # print(h, w, c)

        hrate = h / self.height
        width = tf.cast(hrate * w, dtype=tf.float32)

        print(hrate, width, width > self.width)

        def resize_normal(size, image):
            size = tf.cast(size, tf.int32)
            return tf.image.resize(image, size)

        def resize_special(height, width, origin_width, image):
            # resize image
            size = tf.concat([height, width], axis=0)
            size = tf.cast(size, tf.int32)
            image = tf.image.resize(image, size)

            # padding image
            result = tf.cast(origin_width - width, dtype=tf.int32)
            zero = tf.zeros_like(result)
            paddings = tf.stack([[zero, result], [zero, zero], [zero, zero]], axis=0)
            paddings = tf.squeeze(paddings)
            image = tf.pad(image, paddings=paddings, mode="CONSTANT")

            return image

        img = tf.cond(width > self.width, lambda : resize_normal(self.size, img), lambda : resize_special(self.height, width, self.width, img))

        txt = img_txts[0]

        return img, txt

    def sample_testing(self, list_of_records):
        features = {
            "img_raw": tf.io.VarLenFeature(dtype=tf.string),
            "img_txt": tf.io.VarLenFeature(dtype=tf.string),
            "shape": tf.io.FixedLenFeature([3], dtype=tf.int64),
            "img_dir": tf.io.VarLenFeature(dtype=tf.string)
        }
        dataset = tf.data.TFRecordDataset(list_of_records, compression_type=self.compression_type)
        for serialized_example_tensor in dataset:
            example = tf.io.parse_single_example(serialized_example_tensor, features)
            print(example)
            img_lab = tf.sparse.to_dense(example['img_txt'], default_value=b'')
            print("img label", img_lab.numpy()[0].decode('utf-8'))
            img_raw = tf.sparse.to_dense(example['img_raw'], default_value=b'')
            print(img_raw)
            image = tf.io.decode_raw(img_raw, out_type=tf.uint8)
            print("img shape and type", image.shape, image.dtype)
            print("shape", example["shape"].numpy())
            img_dir = tf.sparse.to_dense(example['img_dir'], default_value=b'')
            print("img_dir", img_dir.numpy()[0].decode('utf-8'))


if __name__ == '__main__':

    """
    data = TDDataset('./result', use_zip=True, use_repeat=False)
    data.create_record(["/home/ding/file.lst"])
    data.sample_testing(['./result/train_000_of_001.zip'])

    data_of_train = data.create_batcher(['./result/train_000_of_001.zip'], batch_size=1)

    for x, y in data_of_train.take(100):
        print(x, x.shape)
        print([label.decode('utf-8') for label in y.numpy()])


    """
    data = OCRTextDataset('./ocrtext_result', [32, 400], use_zip=True, use_repeat=False)
    # data.create_record(["/home/ding/file.lst"])
    # data.sample_testing(['./ocrtext_result/train_000_of_001.zip'])


    data_of_train = data.create_batcher(['./ocrtext_result/train_000_of_001.zip'], batch_size=6)

    for x, y in data_of_train.take(4):
        print(x, y)


