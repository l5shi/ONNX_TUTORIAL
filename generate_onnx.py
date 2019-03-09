
import keras
from keras.models import load_model
from keras.layers import *
from keras.layers.convolutional import DepthwiseConv2D
from keras.models import Model
from keras import backend as K
from keras.models import Sequential

import onnx as O
import os
import sys
from collections import OrderedDict
from shutil import copyfile

import numpy as np
from IPython import embed
from helper import *


import re
import json
import xlrd



def get_config(ver=5):
    config = OrderedDict()

    if ver == 5:
        config['input_gen'] ='random'
        config['weight_gen'] ='random'
        config['bias_gen'] ='random'
        config['fn_excel'] = "test_cases_red_20181114.xlsx"
        config['row_numbers'] = list(range(1, 102))
        config['sheet_name'] = 'test_case_list'
    elif ver == 111:
        config['input_gen'] ='debug'
        config['weight_gen'] ='debug'
        config['bias_gen'] ='all0'
        config['fn_excel'] = "test_cases_red_20181208_reorder.xlsx"
        config['row_numbers'] = list(range(89,90))
        config['sheet_name'] = 'v11_8bit3x3'    
    else:
        raise NotImplemented

    assert (config['input_gen'] in ['random', 'debug', 'incre'])
    assert (config['weight_gen'] in ['random', 'debug', 'incre', 'all16'])
    assert (config['bias_gen'] in ['random', 'all0'])

    config['version'] = 'v{:03d}'.format(ver)

    return config

def readTestCase(testCaseFileName, row_numbers, sheet_name):
    # Read content of test case table
    data = xlrd.open_workbook(testCaseFileName)
    table = data.sheet_by_name(sheet_name)
    nrows = table.nrows
    ncols = table.ncols
    cfgKey = table.row_values(0)

    # Normalize key values for tables
    for i in range(len(cfgKey)):
        key = cfgKey[i]
        cfgKey[i] = key.lstrip(" ").rstrip(" ").replace(" ", "_").lower()

    # Make configuration list
    cfgList = []
    for r in row_numbers:
        cfgValue = table.row_values(r)
        cfgDict = dict(zip(cfgKey, cfgValue))
        cfgList.append(cfgDict)

    return cfgList

def buildSingleLayerONNX(cfgDict):
    node_list = []
    values_in = []
    values_out = []
    values_info = []
    # Get the core operation
    testType = cfgDict["conv_mode_(in_model_operation)"]
    name = cfgDict["test_case_notes"]

    input_shape = (1, int(cfgDict["input_channel_num"]), int(cfgDict["input_size_h_(row)"]), int(cfgDict["input_size_w_(col)"]))

    if testType == 'add':
        inputs1 = O.helper.make_tensor_value_info('Input1', O.TensorProto.FLOAT, list(input_shape))
        values_info.append(inputs1)
        inputs2 = O.helper.make_tensor_value_info('Input2', O.TensorProto.FLOAT, list(input_shape))
        values_info.append(inputs2)  
        input_name = ['Input1','Input2']  	
        values_in += [inputs1, inputs2]
    else:
        inputs = O.helper.make_tensor_value_info('Input', O.TensorProto.FLOAT, list(input_shape))
        values_info.append(inputs)
        input_name = ['Input']  
        values_in += [inputs]	

    #if testType == "dense":
    #    input_shape = (int(cfgDict["input_size_w_(col)"]) * int(cfgDict["input_size_h_(row)"]),)

    # Generate input layer, if "add", make multiple inputs
    #input = Input(shape=input_shape)

    # Build conv padding layer
    input_pad = input_name[0]
    if cfgDict['conv_en']:
        paddingTop = int(cfgDict['up_padding_t'])
        paddingBottom = int(cfgDict['dn_padding_b'])
        paddingLeft = int(cfgDict['left_padding_l'])
        paddingRight = int(cfgDict['right_padding_r'])
    #########################
    #      build padding    #
    #########################
        padding_info = [paddingTop,paddingLeft,paddingBottom,paddingRight]
        if np.sum(padding_info) > 0:
            node = O.helper.make_node(
            'Pad', # node name
            input_name, # inputs
            ['padding'], # outputs
            #mode='constant', # Attributes
            name='zero_padding2d',
            pads=padding_info,
            )
            node_list.append(node)
            after_pad_row = int(input_shape[2] + paddingLeft + paddingRight)
            after_pad_col = int(input_shape[3] + paddingTop + paddingBottom)
            input_shape = [input_shape[0],input_shape[1],after_pad_row,after_pad_col]
            pad = O.helper.make_tensor_value_info('padding', O.TensorProto.FLOAT, input_shape)
            values_info.append(pad)
            input_pad = 'padding'
 

    #########################
    #     prepare conv      #
    #########################    
    kernel_size = (0, 0)
    channel = input_shape[1]	
    if cfgDict["kernel_size_w"] != "NA":
        kernel_size = (int(cfgDict["kernel_size_w"]), int(cfgDict["kernel_size_h"]))
    strides = (int(cfgDict["conv_stride"]), int(cfgDict["conv_stride"]))
    if cfgDict["padding_mode"] == 'valid':
        padding = [0,0,0,0]
    else:
        padding = helper.getPading(input_shape[2:4],kernel_size[0],strides[0])
    filters = int(cfgDict["concat_channel_end_index"]) - int(cfgDict["concat_channel_start_index"]) + 1
    dense_units = int(cfgDict["concat_channel_end_index"]) - int(cfgDict["concat_channel_start_index"]) + 1
    result_row = int((input_shape[2] - kernel_size[0])/strides[0] + 1)
    result_col = int((input_shape[3] - kernel_size[1])/strides[1] + 1)
    if testType == 'dense':
    	output_shape = (1,  int(cfgDict["output_channel_num"]))
    else:
        output_shape = (1,  int(cfgDict["output_channel_num"]), result_row, result_col)
    

    #########################
    #      build conv       #
    ######################### 

    #logger.debug(testType)
    if cfgDict['conv_en']:
        if testType in[ "conv", "RGBAconv", "RGBA"]: ### named by "RGBA" not "RGBAconv"
            wnode, winfo = construct_weights(output_shape,input_shape,kernel_size)
            node_list += wnode
            values_info += winfo
            bnode, binfo = construct_bias(output_shape)
            node_list += bnode
            values_info += binfo
            node = O.helper.make_node(
              'Conv',
              [input_pad,'weight','bias'],
              [testType+'_out'],
              name = str(testType),
              kernel_shape=list(kernel_size),
              pads=padding,
              strides=list(strides),
              )
            node_list.append(node)

        elif testType == "deconv":
            wnode, winfo = construct_weights(output_shape,input_shape,kernel_size)
            node_list += wnode
            values_info += winfo
            bnode, binfo = construct_bias(output_shape)
            node_list += bnode
            values_info += binfo
            node = O.helper.make_node(
              'ConvTranspose',
              [input_pad,'weight','bias'],
              [testType+'_out'],
              name = str(testType),
              kernel_shape=list(kernel_size),
              pads=padding,
              strides=list(strides),
              )
            node_list.append(node)

        elif testType == "dw":
        	wnode, winfo = construct_weights(output_shape,input_shape,kernel_size)
        	node_list += wnode
        	values_info += winfo
        	bnode, binfo = construct_bias(output_shape)
        	node_list += bnode
        	values_info += binfo
        	node = O.helper.make_node(
              'Conv',
              [input_pad,'weight','bias'],
              [testType+'_out'],
              name = str(testType),
              kernel_shape=list(kernel_size),
              pads=padding,
              strides=list(strides),
              group=channel,
              )
        	node_list.append(node)

        elif testType == "dense":
            new_shape = [input_shape[0],input_shape[1]*input_shape[2]*input_shape[3]]
            fnode = O.helper.make_node(
			    'Flatten',
			    inputs=input_name,
			    outputs=['flatten'],
			    name='flatten',
			    axis=1)
            flatten= O.helper.make_tensor_value_info('flatten', O.TensorProto.FLOAT, list(new_shape))
            node_list.append(fnode)
            values_info.append(flatten)
            wnode, winfo = construct_weights(output_shape,new_shape,kernel_size,mode='dense')
            node_list += wnode
            values_info += winfo
            bnode, binfo = construct_bias(output_shape,mode='dense')
            node_list += bnode
            values_info += binfo
            node = O.helper.make_node(
            'Gemm',
            ['flatten','weight','bias'],
            [testType+'_out'],
            name='dense',
            alpha=1.0,
            beta=1.0,
            broadcast=0,
            transA=0,
            transB=0
            )
            node_list.append(node)

        elif testType == "add":
            output_shape = input_shape
            node = O.helper.make_node(
            'Add',
            input_name,
            [testType+'_out'],
            name="add",
            )
            node_list.append(node)

        outputs = O.helper.make_tensor_value_info(testType+'_out', O.TensorProto.FLOAT, list(output_shape))
        values_info.append(outputs)	
        bypass_conv = testType+'_out'
    else:
        output_shape = input_shape
        bypass_conv = input_pad
    #PCONV

    #########################
    #         TO DO         #
    #########################

    input_shape = output_shape
    input_str_info = bypass_conv

    poolingLayer = poolingLayer_wrapper(cfgDict)
    reluLayer = reluLayer_wrapper(cfgDict)
    layerDict = OrderedDict()
    orderDict = {0:"BRP", 1:"BPR", 2:"RPB", 3:"RBP", 4:"PBR", 5:"PRB"}
    layerDict["B"] = bnLayer if cfgDict["pconv_bn_en"] == 1 else None
    layerDict["P"] = poolingLayer if cfgDict["pconv_pool_en"] == 1 else None
    layerDict["R"] = reluLayer if cfgDict["pconv_relu_en"] == 1 else None
    order = orderDict[cfgDict["bn_relu_pool_order_(pconv_order)"]]
    for c in order:
    	if layerDict[c] != None:
    		output = layerDict[c](input_str_info, input_shape)
    		if output != None:
	    		# below two line for next layer assign
		    	input_str_info = output[3]
		    	input_shape = output[2]

		    	# below is for list concatenate
		    	values_info += output[1] # output_value_info_lst = output[1]
		    	node_list += output[0] # 	output_node_lst = output[0]


    values_out.append(values_info[-1])
    #construct graph
    graph_def = O.helper.make_graph(
	    node_list,
	    name + '_onnx',
	    values_in,
	    values_out,
	    value_info=values_info,
	    )
    # Create the model (ModelProto)

    omodel = O.helper.make_model(graph_def, producer_name='Kneron')

    return omodel

def genTestCase(cfgList):

    for cfgDict in cfgList:

        model = buildSingleLayerONNX(cfgDict)

        O.save(model, 'onnx_model/'+str(int(cfgDict["test_case_number"]))+'_'+cfgDict["test_case_notes"]+'.onnx')


if __name__ == "__main__":
    np.random.seed(8)

    version = 111

    excel_path = "."
    output_path = "."


    #TODO: it should move config to JSON (don't store much data on code), and user get_config to get data from JSON
    configs = get_config(version)
    #configs['weight_gen'] = "random"
    # logger.debug(configs)

    testCasesFileName = "{}/{}".format(excel_path, configs['fn_excel'])
    cfgList = readTestCase(testCasesFileName, row_numbers=configs['row_numbers'], sheet_name=configs['sheet_name'])
    dir_output = "{}/test_cases_{}".format(output_path, configs['version'])
    genTestCase(cfgList)
