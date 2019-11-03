import numpy as np
import quandl
import tensorflow as tf


class NeuralNetwork():

    def creatematrix():

        random_int_var = tf.get_variable(
            initializer=tf.random_uniform([3, 4],
                                          minval=1,
                                          maxval=10,
                                          dtype=tf.int32))
        print(random_int_var)

    def get_data():
        quandl.ApiConfig.api_key = '8apR-N8gxbjo4M63Q7ik'
        data = quandl.get_table('WIKI/PRICES',
                                qopts={'columns': ['close']},
                                ticker=['TSLA'],
                                date={'gte': '2018-01-01', 'lte': '2018-11-01'}
                                )
        prices = []
        for price in data:
            prices.append(price)
        print(prices)


# let's create a ones 3x3 rank 2 tensor
rank_2_tensor_A = tf.ones([3, 3], name='MatrixA')
print("3x3 Rank 2 Tensor A: \n{}\n".format(rank_2_tensor_A))

# let's manually create a 3x3 rank two tensor and specify the data type as float
rank_2_tensor_B = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]], name='MatrixB', dtype=tf.float32)
print("3x3 Rank 2 Tensor B: \n{}\n".format(rank_2_tensor_B))

# addition of the two tensors
rank_2_tensor_C = tf.add(rank_2_tensor_A, rank_2_tensor_B, name='MatrixC')
print("Rank 2 Tensor C with shape={} and elements: \n{}".format(rank_2_tensor_C.shape, rank_2_tensor_C))
