import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np
import os


class CBS(layers.Layer):
    """Conv + BatchNorm + SiLU"""
    def __init__(self, filters, kernel_size=1, strides=1, **kwargs):
        super().__init__(**kwargs)
        self.conv = layers.Conv2D(filters, kernel_size, strides, padding='same', use_bias=False)
        self.bn = layers.BatchNormalization()
        self.act = layers.Activation('swish')
    
    def call(self, x, training=False):
        x = self.conv(x)
        x = self.bn(x, training=training)
        return self.act(x)

class SEBlock(layers.Layer):
    """Squeeze-and-Excitation Block"""
    def __init__(self, filters, reduction=16, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.reduction = reduction
        
    def build(self, input_shape):
        self.gap = layers.GlobalAveragePooling2D(keepdims=True)
        self.fc1 = layers.Dense(max(self.filters // self.reduction, 8), activation='relu')
        self.fc2 = layers.Dense(self.filters, activation='sigmoid')
        
    def call(self, x):
        se = self.gap(x)
        se = self.fc1(se)
        se = self.fc2(se)
        return x * se

class ScConv(layers.Layer):
    """Spatial and Channel Convolution"""
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        
    def build(self, input_shape):
        self.conv_spatial = layers.Conv2D(self.filters // 2, 3, padding='same')
        self.conv_channel = layers.Conv2D(self.filters // 2, 1, padding='same')
        self.concat = layers.Concatenate()
        
    def call(self, x):
        spatial = self.conv_spatial(x)
        channel = self.conv_channel(x)
        return self.concat([spatial, channel])

class ScK2(layers.Layer):
    """Fixed Lightweight Deep Feature Extraction Module"""
    def __init__(self, filters, num_blocks=3, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.num_blocks = num_blocks
        
    def build(self, input_shape):
        self.cbs_in = CBS(self.filters)
        self.sc_convs = [ScConv(self.filters) for _ in range(self.num_blocks)]
        self.cbs_out = CBS(self.filters)
        
    def call(self, x, training=False):
        x = self.cbs_in(x, training=training)
        # Use residual connections instead of averaging
        identity = x
        for sc_conv in self.sc_convs:
            x = sc_conv(x)
        x = x + identity  # Residual connection
        return self.cbs_out(x, training=training)

class SimplifiedLarkBlock(layers.Layer):
    """Simplified Large Kernel Block"""
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        
    def build(self, input_shape):
        # Use depthwise conv instead of complex re-parameterization
        self.dw_conv = layers.DepthwiseConv2D(7, padding='same', use_bias=False)
        self.bn = layers.BatchNormalization()
        self.se_block = SEBlock(self.filters)
        self.pwconv1 = layers.Conv2D(self.filters * 4, 1)
        self.act = layers.Activation('gelu')
        self.pwconv2 = layers.Conv2D(self.filters, 1)
        
    def call(self, x, training=False):
        identity = x
        x = self.dw_conv(x)
        x = self.bn(x, training=training)
        x = self.se_block(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return x + identity  # Residual

class SmaKBlock(layers.Layer):
    """Small Kernel Block for Local Features"""
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        
    def build(self, input_shape):
        self.dw_3x3 = layers.DepthwiseConv2D(3, padding='same', use_bias=False)
        self.bn = layers.BatchNormalization()
        self.se_block = SEBlock(self.filters)
        self.pwconv1 = layers.Conv2D(self.filters * 4, 1)
        self.act = layers.Activation('gelu')
        self.pwconv2 = layers.Conv2D(self.filters, 1)
        
    def call(self, x, training=False):
        identity = x
        x = self.dw_3x3(x)
        x = self.bn(x, training=training)
        x = self.se_block(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return x + identity

class SPPF(layers.Layer):
    """Spatial Pyramid Pooling - Fast"""
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.cbs1 = CBS(filters // 2, 1)
        self.pool = layers.MaxPooling2D(5, strides=1, padding='same')
        self.cbs2 = CBS(filters, 1)
        
    def call(self, x, training=False):
        x = self.cbs1(x, training=training)
        p1 = self.pool(x)
        p2 = self.pool(p1)
        p3 = self.pool(p2)
        out = tf.concat([x, p1, p2, p3], axis=-1)
        return self.cbs2(out, training=training)



class MedLKSCNet(Model):
    """Fixed Detection Model for 4-Class Classification"""
    def __init__(self, num_classes=4, base_filters=64, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.base_filters = base_filters
        
    def build(self, input_shape):
        self.stem = CBS(self.base_filters, 6, 2)
        
        # Stage 1 - Input 64 -> Output 128
        self.stage1_lark = SimplifiedLarkBlock(self.base_filters)     
        self.stage1_smak = SmaKBlock(self.base_filters)               
        self.stage1_down = CBS(self.base_filters * 2, 3, 2)
        
        # Stage 2 - Input 128 -> Output 256
        self.stage2_lark = SimplifiedLarkBlock(self.base_filters * 2) 
        self.stage2_smak = SmaKBlock(self.base_filters * 2)           
        self.stage2_down = CBS(self.base_filters * 4, 3, 2)
        
        # Stage 3 - Input 256
        self.stage3_lark = SimplifiedLarkBlock(self.base_filters * 4) 
        self.stage3_smak = SmaKBlock(self.base_filters * 4)           
        
        # SPPF
        self.sppf = SPPF(self.base_filters * 4)
        
        # Neck - Adjusted to match corrected dimensions
        self.neck_reduce_p5 = CBS(self.base_filters * 4, 1)
        self.neck_reduce_p4 = CBS(self.base_filters * 2, 1)
        self.neck_reduce_p3 = CBS(self.base_filters, 1)
        
        self.upsample_1 = layers.UpSampling2D(2)
        self.upsample_2 = layers.UpSampling2D(2)
        
        self.neck_sck2_1 = ScK2(self.base_filters * 4)
        self.neck_sck2_2 = ScK2(self.base_filters * 2)
        self.neck_sck2_3 = ScK2(self.base_filters)
        
        # Classification head
        self.gap_p3 = layers.GlobalAveragePooling2D()
        self.gap_p4 = layers.GlobalAveragePooling2D()
        self.gap_p5 = layers.GlobalAveragePooling2D()
        
        self.fc1 = layers.Dense(256, activation='relu')
        self.dropout = layers.Dropout(0.5)
        self.classifier = layers.Dense(self.num_classes, activation='softmax')
        
        super().build(input_shape)
    
    def call(self, inputs, training=False):
        # Stem
        x = self.stem(inputs, training=training)
        
        # Stage 1 - P3 features (64 filters)
        lark1 = self.stage1_lark(x, training=training)
        smak1 = self.stage1_smak(x, training=training)
        P3 = lark1 + smak1
        x = self.stage1_down(P3, training=training)
        
        # Stage 2 - P4 features (128 filters)
        lark2 = self.stage2_lark(x, training=training)
        smak2 = self.stage2_smak(x, training=training)
        P4 = lark2 + smak2
        x = self.stage2_down(P4, training=training)
        
        # Stage 3 - P5 features (256 filters)
        lark3 = self.stage3_lark(x, training=training)
        smak3 = self.stage3_smak(x, training=training)
        P5 = lark3 + smak3
        
        # SPPF
        P5 = self.sppf(P5, training=training)
        
        # Neck
        P5_processed = self.neck_sck2_1(P5, training=training)
        
        # Upsample P5 and fuse with P4
        P5_up = self.upsample_1(P5_processed)
        P4_fused = tf.concat([P5_up, P4], axis=-1)
        # Reduce to correct dimension
        P4_fused = self.neck_reduce_p4(P4_fused, training=training)
        P4_processed = self.neck_sck2_2(P4_fused, training=training)
        
        # Upsample P4 and fuse with P3
        P4_up = self.upsample_2(P4_processed)
        P3_fused = tf.concat([P4_up, P3], axis=-1)
        # Reduce to correct dimension
        P3_fused = self.neck_reduce_p3(P3_fused, training=training)
        P3_processed = self.neck_sck2_3(P3_fused, training=training)
        
        # Multi-scale feature fusion for classification
        feat_p3 = self.gap_p3(P3_processed)  # Shape: (batch, 64)
        feat_p4 = self.gap_p4(P4_processed)  # Shape: (batch, 128)
        feat_p5 = self.gap_p5(P5_processed)  # Shape: (batch, 256)
        
        # Concatenate multi-scale features
        combined = tf.concat([feat_p3, feat_p4, feat_p5], axis=-1)
        
        # Classification head
        x = self.fc1(combined)
        x = self.dropout(x, training=training)
        output = self.classifier(x)
        
        return output
