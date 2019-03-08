


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
	spatial=1,
	is_test=1
	)
	node_lst.append(node_bn)
	value_info_lst.append(output_bn_info)

	results = [node_lst, value_info_lst, output_bn_shape, 'output_bn_info']
	return results

def poolingLayer_wrapper(cfgDict):
	def poolingLayer(input_str_info, input_pool_shape):
		node_lst = []
		value_info_lst = []

		output_pool_shape = input_pool_shape
		output_pool_info = O.helper.make_tensor_value_info('output_pool_info', O.TensorProto.FLOAT, list(output_pool_shape))

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
			strides = [int(pool_stride), int(pool_stride)]
			if pool_mode == 0 or pooling_mode == 1:
				node_pool = O.helper.make_node(
				op_type='MaxPool',
				inputs=[input_str_info],
				outputs=['output_pool'],
				name='PCONV_MaxPool',
				kernel_shape=[int(pool_size), int(pool_size)],
				pads=pads,
				strides=strides
				)

	      	elif pool_mode == 2:
				node_pool = O.helper.make_node(
				op_type='AveragePool',
				inputs=[input_str_info],
				outputs=['output_pool'],
				name='PCONV_AveragePool',
				kernel_shape=[int(pool_size), int(pool_size)],
				pads=pads,
				strides=strides
				)

	      	elif pool_mode == 3 or pool_mode == "global":
				node_pool = O.helper.make_node(
				op_type='GlobalAveragePool',
				inputs=[input_str_info],
				outputs=['output_pool'],
				name='PCONV_GlobalAveragePool',
				kernel_shape=[int(pool_size), int(pool_size)],
				pads=pads,
				strides=strides
				)

		else:
			results = None
			return results

	    node_lst.append(node_pool)
	    value_info_lst.append(output_pool_info)
	    results = [node_lst, value_info_lst, output_pool_shape,'output_pool_info']
	  	return results

	return poolingLayer

def reluLayer_wrapper(cfgDict):
	relu_mode = cfgDict["relu_mode"]
    relu6_clamp = cfgDict["relu6_clamp"]
	def reluLayer (input_str_info, input_relu_shape):
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
	            min=0,
	            max=relu6_clamp,
	            name = 'relu6_clamp'
	        )

	    elif relu_mode == 2:
	        slope_relu = O.helper.make_tensor_value_info('slope_relu_info', O.TensorProto.FLOAT, list(input_relu_shape[1:]))
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



poolingLayer = poolingLayer_wrapper(cfgDict)
reluLayer = reluLayer_wrapper(cfgDict)

layerDict = OrderedDict()
orderDict = {0:"BRP", 1:"BPR", 2:"RPB", 3:"RBP", 4:"PBR", 5:"PRB"}
layerDict["B"] = bnLayer if cfgDict["pconv_bn_en"] == 1 else None
layerDict["P"] = poolingLayer if cfgDict["pconv_pool_en"] == 1 else None
layerDict["R"] = reluLayer if cfgDict["pconv_relu_en"] == 1 else None
#print(cfgDict["bn_relu_pool_order_(pconv_order)"])

order = orderDict[cfgDict["bn_relu_pool_order_(pconv_order)"]]


input_shape = 
input_str_info = 
for c in order:
    if layerDict[c] != None:

    	output = layerDict[c](input_str_info, input_shape)
    	if output != None:
    		# below two line for next layer assign
	    	input_str_info = output[3]
	    	input_shape = output[2]

	    	# below is for list concatenate
	    	value_info_lst += output[1] # output_value_info_lst = output[1]
	    	node_lst += output[0] # 	output_node_lst = output[0]

value_out = [value_info_lst[-1]]