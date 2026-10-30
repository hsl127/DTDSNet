import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv1D, Conv2D, DepthwiseConv2D, AveragePooling2D, BatchNormalization, 
    Activation, Dropout, Dense, Concatenate, Lambda, Flatten, Layer, 
    GlobalAveragePooling2D, Reshape, GaussianNoise
)
from tensorflow.keras.models import Model
from tensorflow.keras.constraints import max_norm


class GatedSpatialFusion(Layer):
    def __init__(self, **kwargs): super().__init__(**kwargs)
    def build(self, input_shape):
        filters = input_shape[0][-1]
        self.alpha = self.add_weight(name='fusion_alpha', shape=(1, 1, 1, filters), initializer='zeros', trainable=True)
        super().build(input_shape)
    def call(self, inputs):
        x_s, x_g = inputs
        gate = tf.nn.sigmoid(self.alpha)
        return x_s * gate + x_g * (1.0 - gate)

class BandPowerScaling(Layer):
    def __init__(self, **kwargs): super().__init__(**kwargs)
    def build(self, input_shape):
        self.scale = self.add_weight(name='band_scale', shape=(1, 1, 1, input_shape[-1]), initializer='ones', trainable=True)
        super().build(input_shape)
    def call(self, x): return x * self.scale


class AdaptiveTemporalSaliency(Layer):
    def __init__(self, name_suffix="", **kwargs):
        super().__init__(**kwargs); self.name_suffix = name_suffix
    def build(self, input_shape):
        filters, self.time_steps = input_shape[-1], input_shape[2]
        
        self.temporal_generator = tf.keras.Sequential([
            Dense(filters // 4, activation='relu', use_bias=False),
            Dense(self.time_steps, activation=None, use_bias=False)
        ], name=f"t_gen_{self.name_suffix}")
        self.smoothing = Conv1D(1, kernel_size=3, padding='same', use_bias=False)
        self.noise = GaussianNoise(0.01)
        super().build(input_shape)
        
    def call(self, x, training=None):
        
        mu = tf.reduce_mean(x, axis=[1, 2]) # (Batch, Channels)
        
        sigma = tf.math.reduce_std(x, axis=[1, 2]) # (Batch, Channels)
        
        
        stats = tf.concat([mu, sigma], axis=-1)
        
        
        t_weights = self.temporal_generator(stats)
        t_weights = tf.expand_dims(t_weights, axis=-1)
        t_weights = self.smoothing(t_weights)
        t_weights = tf.nn.sigmoid(tf.squeeze(t_weights, axis=-1))
        t_weights = self.noise(t_weights, training=training)
        t_weights = tf.clip_by_value(t_weights, 0.0, 1.0)
        return x * tf.reshape(t_weights, (-1, 1, self.time_steps, 1))

class DynamicFrequencyBandGating(Layer):
    def __init__(self, num_bands=3, **kwargs):
        super().__init__(**kwargs); self.num_bands = num_bands
    def build(self, input_shape):
        self.total_filters = input_shape[-1]; self.pool_size = self.total_filters // self.num_bands
        self.gate_dense = Dense(self.num_bands, activation='sigmoid', use_bias=False)
        super().build(input_shape)
    def call(self, inputs):
        y = tf.reduce_mean(inputs, axis=[1, 2], keepdims=False)
        bf = tf.keras.layers.AveragePooling1D(pool_size=self.pool_size)(tf.expand_dims(y, axis=-1))
        bw = self.gate_dense(tf.squeeze(bf, axis=-1))
        ew = tf.repeat(bw, repeats=self.pool_size, axis=1)
        return inputs * tf.reshape(ew, (-1, 1, 1, self.total_filters))

class DepthwiseTemporalPSR(Layer):
    def __init__(self, **kwargs): super().__init__(**kwargs)
    def build(self, input_shape):
        filters = input_shape[0][-1]
        self.gate_conv = DepthwiseConv2D(kernel_size=(1, 3), padding='same', activation='sigmoid', use_bias=False)
        
        self.gamma_vector = self.add_weight(name='dt_psr_gamma_vector', shape=(1, 1, 1, filters), initializer='zeros', trainable=True)
        super().build(input_shape)
    def call(self, inputs):
        e_map, t_map = inputs
        weights = self.gate_conv(e_map) 
        return t_map + self.gamma_vector * (t_map * weights)

class GatedResidualTemporal(Layer):
    def __init__(self, filters=24, **kwargs):
        super().__init__(**kwargs); self.filters = filters
    def build(self, input_shape):
        self.alpha = self.add_weight(name='grt_alpha', shape=(self.filters,), initializer='zeros', trainable=True)
        super().build(input_shape)
    def call(self, inputs):
        t_short, t_long = inputs
        gate = tf.nn.tanh(self.alpha)
        return t_short + t_long * tf.reshape(gate, (1, 1, 1, self.filters))

class IndependentChannelDistillation(Layer):
    def __init__(self, **kwargs): super().__init__(**kwargs)
    def build(self, input_shape):
        self.filters = input_shape[-1]
        self.conv = tf.keras.layers.Conv1D(1, kernel_size=1, padding='same', use_bias=False)
        super().build(input_shape)
    def call(self, inputs):
        y = tf.reduce_mean(inputs, axis=[1, 2], keepdims=False)
        y = tf.expand_dims(y, axis=-1); y = self.conv(y); y = tf.nn.sigmoid(y)
        return inputs * tf.reshape(tf.transpose(y, perm=[0, 2, 1]), (-1, 1, 1, self.filters))

def safe_log(x): return tf.math.log(tf.clip_by_value(x, 1e-7, 1e6))


def build_ortho_model(nb_classes=4, Chans=22, Samples=1125):
    inp = Input((Chans, Samples, 1))

    
    t_branches = [Conv2D(6, (1, k), padding='same', use_bias=False)(inp) for k in [96, 72, 48, 24]]
    x_time = BatchNormalization()(Concatenate(axis=-1)(t_branches))
    x_s = BatchNormalization()(DepthwiseConv2D((Chans, 1), use_bias=False, depth_multiplier=1, depthwise_constraint=max_norm(1.0))(x_time))
    x_g = AveragePooling2D((Chans, 1))(x_time) 
    x = GatedSpatialFusion()([x_s, x_g])
    x = DynamicFrequencyBandGating(num_bands=3)(Activation('elu')(x))
    x.set_shape([None, 1, Samples, 24])

    e_raw = tf.square(x); g_f = Reshape((1, 1, 24))(GlobalAveragePooling2D()(e_raw))
    t_f_all = BatchNormalization()(Activation('elu')(GatedResidualTemporal(filters=24)([
        DepthwiseConv2D(kernel_size=(1, 16), padding='same', use_bias=False)(x), 
        DepthwiseConv2D(kernel_size=(1, 16), dilation_rate=(1, 2), padding='same', use_bias=False)(x)
    ])))


    e_mac_base = AdaptiveTemporalSaliency(name_suffix="emac")(Concatenate(axis=2)([AveragePooling2D((1, 45), strides=(1, 30))(e_raw), g_f]))
    e_map_mac = IndependentChannelDistillation()(BatchNormalization()(Lambda(safe_log)(BandPowerScaling()(e_mac_base))))
    t_mac_base = AdaptiveTemporalSaliency(name_suffix="tmac")(AveragePooling2D((1, 44), strides=(1, 29))(t_f_all))
    t_map_mac = IndependentChannelDistillation()(DepthwiseTemporalPSR()([e_map_mac, BatchNormalization()(t_mac_base)]))


    e_mic_base = AdaptiveTemporalSaliency(name_suffix="emic")(AveragePooling2D((1, 22), strides=(1, 15))(e_raw))
    e_map_mic = IndependentChannelDistillation()(BatchNormalization()(Lambda(safe_log)(BandPowerScaling()(e_mic_base))))
    t_mic_base = AdaptiveTemporalSaliency(name_suffix="tmic")(AveragePooling2D((1, 22), strides=(1, 15))(t_f_all))
    t_map_mic = IndependentChannelDistillation()(DepthwiseTemporalPSR()([e_map_mic, BatchNormalization()(t_mic_base)]))

  
    e_feat = Flatten(name='energy_features')(Concatenate(axis=2)([e_map_mac, e_map_mic]))
    t_feat = Flatten(name='data_features')(Concatenate(axis=2)([t_map_mac, t_map_mic]))
    fused = Dropout(0.5)(Concatenate(axis=-1)([e_feat, t_feat]))
    out = Dense(nb_classes, activation='softmax', kernel_constraint=max_norm(0.25), name='classifier')(fused)

    return Model(inputs=inp, outputs=[out, e_feat, t_feat])