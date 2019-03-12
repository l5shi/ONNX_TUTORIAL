import logging
import numpy as np
import onnx as O
from onnx import TensorProto

def getPadding(size, kernel_size, strides):
	if size[0] % strides[0] == 0:
		pad_h = max(kernel_size[0] - strides[0], 0)
	else:
		pad_h = max(kernel_size[0] - (size[0] % strides[0]), 0)
	if size[1] % strides[1] == 0:
		pad_w = max(kernel_size[1] - strides[1], 0)
	else:
		pad_w = max(kernel_size[1] - (size[1] % strides[1]), 0)
	return [pad_h//2, pad_w//2, pad_h-pad_h//2, pad_w-pad_w//2]

def reluLayer_wrapper(cfgDict):

	def reluLayer(input_str_info, input_relu_shape):

		relu_mode = cfgDict["relu_mode"]
		relu6_clamp = cfgDict["relu6_clamp"]
		node_lst = []
		value_info_lst = []
		output_relu_shape = input_relu_shape
		output_relu_info = O.helper.make_tensor_value_info('output_relu_info', O.TensorProto.FLOAT, 
		list(output_relu_shape))
		if relu6_clamp == "MAX":
			relu6_clamp = 8
		if relu_mode == 3:
			node_relu = O.helper.make_node(
			'Clip',
			inputs=[input_str_info],
			outputs=['output_relu_info'],
			min=0.0,
			max=relu6_clamp,
			name = 'relu6_clamp'
			)
		elif relu_mode == 2:

			slope_relu_info = O.helper.make_tensor_value_info('slope_relu_info', O.TensorProto.FLOAT, list(input_relu_shape[1:]))
			slope_relu_tensor = O.helper.make_tensor('slope_relu_tensor', O.TensorProto.FLOAT, 
				list(input_relu_shape[1:]), np.random.normal(size =input_relu_shape[1]))

			node_slope = O.helper.make_node( 
			op_type='Constant',
			inputs=[],
			outputs=['slope_relu_info'],
			name='Slope',
			value=slope_relu_tensor
			)
			node_lst.append(node_slope)
			value_info_lst.append(slope_relu_info)
			# channel, heigh and width, but want to share axis in w and h
			node_relu = O.helper.make_node(
			op_type='PRelu',
			inputs = [input_str_info, 'slope_relu_info'],
			outputs = ['output_relu_info'],
			name = 'PCONV_prelu'
			)

		elif relu_mode == 1:
			node_relu = O.helper.make_node(
			op_type = 'LeakyRelu',
			inputs=[input_str_info],
			outputs=['output_relu_info'],
			alpha=0.1,
			name = 'PCONV_leakyrelu'
			)


		elif (relu_mode == 0) or (relu_mode == "ReLU"):
			node_relu = O.helper.make_node(
			op_type='Relu',
			inputs = [input_str_info],
			outputs = ['output_relu_info'],
			name = 'PCONV_relu'
			)
		else:
			results = None
			return results
		node_lst.append(node_relu)
		value_info_lst.append(output_relu_info)  
		results = [node_lst, value_info_lst, output_relu_shape,'output_relu_info']
		return results
	return reluLayer

def bnLayer(input_str_info, input_bn_shape):
	node_lst = []
	value_info_lst = []

	# gamma
	scale_bn_info = O.helper.make_tensor_value_info('scale_bn_info', O.TensorProto.FLOAT, [input_bn_shape[1]])
	scale_bn_tensor = O.helper.make_tensor('scale_bn_tensor', O.TensorProto.FLOAT, 
	[input_bn_shape[1]], np.random.normal(size =input_bn_shape[1]))

	node_scale_bn = O.helper.make_node( 
	op_type='Constant',
	inputs=[],
	outputs=['scale_bn_info'],
	name='Scale_BN',
	value=scale_bn_tensor
	)
	node_lst.append(node_scale_bn)
	value_info_lst.append(scale_bn_info)


	# beta
	bias_bn_info = O.helper.make_tensor_value_info('bias_bn_info', O.TensorProto.FLOAT, [input_bn_shape[1]])
	bias_bn_tensor = O.helper.make_tensor('bias_bn_tensor', O.TensorProto.FLOAT, 
	[input_bn_shape[1]], np.random.normal(size =input_bn_shape[1]))
	node_bias_bn = O.helper.make_node( 
	op_type='Constant',
	inputs=[],
	outputs=['bias_bn_info'],
	name='Bias_BN',
	value=bias_bn_tensor
	)
	node_lst.append(node_bias_bn)
	value_info_lst.append(bias_bn_info)


	# mean
	mean_bn_info = O.helper.make_tensor_value_info('mean_bn_info', O.TensorProto.FLOAT, [input_bn_shape[1]])
	mean_bn_tensor = O.helper.make_tensor('mean_bn_tensor', O.TensorProto.FLOAT, 
	[input_bn_shape[1]], np.random.normal(size =input_bn_shape[1]))
	node_mean_bn = O.helper.make_node( 
	op_type='Constant',
	inputs=[],
	outputs=['mean_bn_info'],
	name='Mean_BN',
	value=mean_bn_tensor
	)

	node_lst.append(node_mean_bn)
	value_info_lst.append(mean_bn_info)

	# var
	var_bn_info = O.helper.make_tensor_value_info('var_bn_info', O.TensorProto.FLOAT, [input_bn_shape[1]])
	var_bn_tensor = O.helper.make_tensor('var_bn_tensor', O.TensorProto.FLOAT, 
	[input_bn_shape[1]], np.random.normal(size =input_bn_shape[1]))
	node_var_bn = O.helper.make_node( 
	op_type='Constant',
	inputs=[],
	outputs=['var_bn_info'],
	name='Var_BN',
	value=var_bn_tensor
	)
	node_lst.append(node_var_bn)
	value_info_lst.append(var_bn_info)



	output_bn_shape = input_bn_shape # (N x C x H x W)
	output_bn_info = O.helper.make_tensor_value_info('output_bn_info', O.TensorProto.FLOAT, list(output_bn_shape))
	node_bn = O.helper.make_node( 
	op_type='BatchNormalization',
	inputs=[input_str_info, 'scale_bn_info', 'bias_bn_info', 'mean_bn_info', 'var_bn_info'],
	outputs=['output_bn_info'],
	name='PCONV_BN',
	# epsilon=self.layer.epsilon,
	# momentum=self.layer.momentum,
	#spatial=1,
	#is_test=1
	)
	node_lst.append(node_bn)
	value_info_lst.append(output_bn_info)

	results = [node_lst, value_info_lst, output_bn_shape, 'output_bn_info']
	return results

def poolingLayer_wrapper(cfgDict):
	def poolingLayer(input_str_info, input_pool_shape):
		node_lst = []
		value_info_lst = []

		if cfgDict['pconv_en'] and cfgDict['pconv_pool_en']:
			paddingTop = int(cfgDict['pool_padding_up'])
			paddingBottom = int(cfgDict['pool_padding_dn'])
			paddingLeft = int(cfgDict['pool_padding_left'])
			paddingRight = int(cfgDict['pool_padding_right'])
			options = {}
			if  paddingTop or paddingBottom or paddingLeft or paddingRight:
				# poolingLayerPair.append(ZeroPadding2D(padding = ((paddingTop, paddingBottom),(paddingLeft, paddingRight))))
				options["padding"] = "same"
				pads = [paddingTop, paddingLeft, paddingBottom, paddingRight]
			else:
				pads = [0, 0, 0, 0]

				# Build pooling layer
			#pool_info = cfgDict["pool_size_and_stride"]
			pool_size = cfgDict["pool_size"]
			pool_stride = cfgDict["pool_stride"]
			# if "_" in pool_info:
			#     pool_size, pool_stride = pool_info.split("_")
			#     pool_size = int(pool_size[1])
			#     pool_stride = int(pool_stride[1])
			pool_mode = cfgDict["pool_mode"]
			# poolingLayer = None
			if pool_mode !=3 and pool_mode != "global":
				result_row = int((input_pool_shape[2] - int(pool_size) + paddingLeft + paddingRight)/int(pool_stride) + 1)
				result_col = int((input_pool_shape[3] - int(pool_size) + paddingBottom + paddingTop)/int(pool_stride) + 1)
				output_pool_shape = (input_pool_shape[0], input_pool_shape[1], result_row, result_col)
				output_pool_info = O.helper.make_tensor_value_info('output_pool_info', O.TensorProto.FLOAT, list(output_pool_shape))
				strides = [int(pool_stride), int(pool_stride)]
			if pool_mode == 0 or pool_mode == 1:
				node_pool = O.helper.make_node(
				op_type='MaxPool',
				inputs=[input_str_info],
				outputs=['output_pool_info'],
				name='PCONV_MaxPool',
				kernel_shape=[int(pool_size), int(pool_size)],
				pads=pads,
				strides=strides
				)

			elif pool_mode == 2:
				node_pool = O.helper.make_node(
				op_type='AveragePool',
				inputs=[input_str_info],
				outputs=['output_pool_info'],
				name='PCONV_AveragePool',
				kernel_shape=[int(pool_size), int(pool_size)],
				pads=pads,
				strides=strides
				)

			elif pool_mode == 3 or pool_mode == "global":
				output_toflatten_shape = (input_pool_shape[0], input_pool_shape[1], 1, 1)
				output_toflatten_info = O.helper.make_tensor_value_info('output_toflatten_info', O.TensorProto.FLOAT, list(output_toflatten_shape))
				
				node_flatten = O.helper.make_node(
				op_type='GlobalAveragePool',
				inputs=[input_str_info],
				outputs=['output_toflatten_info'],
				name='PCONV_GlobalAveragePool',
				# kernel_shape=[int(pool_size), int(pool_size)],
				# pads=pads,
				# strides=strides
				)
				node_lst.append(node_flatten)
				value_info_lst.append(output_toflatten_info)


				output_pool_shape = (input_pool_shape[0], input_pool_shape[1])
				output_pool_info = O.helper.make_tensor_value_info('output_pool_info', O.TensorProto.FLOAT, list(output_pool_shape))
				node_pool = O.helper.make_node(
			    'Flatten',
			    inputs=['output_toflatten_info'],
			    outputs=['output_pool_info'],
			    name='flatten',
			    axis=1)
		else:
			results = None
			return results

		node_lst.append(node_pool)
		value_info_lst.append(output_pool_info)
		results = [node_lst, value_info_lst, output_pool_shape,'output_pool_info']
		return results
	return poolingLayer

def construct_weights(output_shape,input_shape,kernel_size,mode='conv'):
	if mode == 'conv':
	    weight_shape = [output_shape[1],input_shape[1],kernel_size[0],kernel_size[1]]
	elif mode == 'dense':
		weight_shape = [input_shape[1], output_shape[1]]
	weight =  O.helper.make_tensor_value_info('weight', O.TensorProto.FLOAT, list(weight_shape))
	weight_tensor = O.helper.make_tensor('weight_tensor',O.TensorProto.FLOAT,weight_shape,np.zeros(weight_shape).ravel())
	wnode = O.helper.make_node(
	"Constant",
	[],
	['weight'],
	name='weight_1',
	value=weight_tensor,)
	return [wnode],[weight]

def construct_bias(output_shape,mode='conv'):
	if mode == 'conv':
		bias_shape = [output_shape[1]]
	elif mode == 'dense':
		bias_shape = [1, output_shape[1]]
	bias = O.helper.make_tensor_value_info('bias', O.TensorProto.FLOAT, bias_shape)
	bias_tensor = O.helper.make_tensor('bias_tensor',O.TensorProto.FLOAT,bias_shape,np.zeros(bias_shape).ravel())
	bnode = O.helper.make_node(
	"Constant",
	[],
	['bias'],
	name='bias_1',
	value=bias_tensor,
	)
	return [bnode],[bias]