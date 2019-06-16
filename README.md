# tfdatasets
create tfrecords files(include zip), read batch data from tfrecords file

## tips
- if you want to read batch datas from tfrecords, you should make sure your images is same size, or lead to faiiure(set batch_size>1) 
- you can pre-process your single data in parse_example and make sure your opration work in tensor(tensorflow)
