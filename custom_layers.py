import tensorflow as tf
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(
    package="ANAT_ASD"
)
class CBAMBlock(layers.Layer):
    def __init__(
        self,
        ratio=8,
        spatial_kernel_size=7,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.ratio = int(ratio)
        self.spatial_kernel_size = int(
            spatial_kernel_size
        )

        self.global_average_pooling = (
            layers.GlobalAveragePooling2D()
        )

        self.global_max_pooling = (
            layers.GlobalMaxPooling2D()
        )

    def build(self, input_shape):
        self.channel_count = int(input_shape[-1])

        reduced_channels = max(
            self.channel_count // self.ratio,
            1
        )

        self.shared_dense_1 = layers.Dense(
            reduced_channels,
            activation="relu",
            name="channel_mlp_1"
        )

        self.shared_dense_2 = layers.Dense(
            self.channel_count,
            name="channel_mlp_2"
        )

        self.spatial_convolution = layers.Conv2D(
            filters=1,
            kernel_size=self.spatial_kernel_size,
            padding="same",
            activation="sigmoid",
            name="spatial_attention_conv"
        )

        super().build(input_shape)

    def call(self, inputs):
        average_pool = self.global_average_pooling(
            inputs
        )

        maximum_pool = self.global_max_pooling(
            inputs
        )

        average_attention = self.shared_dense_2(
            self.shared_dense_1(average_pool)
        )

        maximum_attention = self.shared_dense_2(
            self.shared_dense_1(maximum_pool)
        )

        channel_attention = tf.nn.sigmoid(
            average_attention + maximum_attention
        )

        channel_attention = tf.reshape(
            channel_attention,
            (-1, 1, 1, self.channel_count)
        )

        channel_refined = inputs * channel_attention

        spatial_average = tf.reduce_mean(
            channel_refined,
            axis=-1,
            keepdims=True
        )

        spatial_maximum = tf.reduce_max(
            channel_refined,
            axis=-1,
            keepdims=True
        )

        spatial_features = tf.concat(
            [spatial_average, spatial_maximum],
            axis=-1
        )

        spatial_attention = self.spatial_convolution(
            spatial_features
        )

        return channel_refined * spatial_attention

    def get_config(self):
        config = super().get_config()

        config.update({
            "ratio": self.ratio,
            "spatial_kernel_size": (
                self.spatial_kernel_size
            )
        })

        return config