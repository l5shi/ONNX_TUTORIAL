def poolingLayer(input_pool_info, input_pool_shape):
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
            inputs=['input_pool'],
            outputs=['output_pool'],
            name='PCONV_MaxPool',
            kernel_shape=[int(pool_size), int(pool_size)],
            pads=pads,
            strides=strides
          )

      elif pool_mode == 2:
          node_pool = O.helper.make_node(
            op_type='AveragePool',
            inputs=['input_pool'],
            outputs=['output_pool'],
            name='PCONV_AveragePool',
            kernel_shape=[int(pool_size), int(pool_size)],
            pads=pads,
            strides=strides
          )

      elif pool_mode == 3 or pool_mode == "global":
          node_pool = O.helper.make_node(
            op_type='GlobalAveragePool',
            inputs=['input_pool'],
            outputs=['output_pool'],
            name='PCONV_GlobalAveragePool',
            kernel_shape=[int(pool_size), int(pool_size)],
            pads=pads,
            strides=strides
          )
    node_lst.append(node_pool)
    value_info_lst.append(output_pool_info)
    poolingLayer = [node_lst, value_info_lst, 'output_pool_info']
    return poolingLayer
  else:
    poolingLayer = None