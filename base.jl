using StatsBase
using Flux
using Images


FILTER_SPACE = [16, 32, 64, 128, 256, 512]
SKIP_CONNECTION_TYPES = ["add", "concat"]

abstract type MGene end
abstract type MBlockGene <: MGene end

###################################
########### Convolution ###########
###################################
mutable struct MConv <: MGene
    in_dims::Tuple
    out_dims::Tuple
    kernel_size::Tuple
    padding::Tuple
end

function MConv(in_dims::Tuple)
    out_dims = in_dims
    out_channels = sample(FILTER_SPACE)
    out_dims[3] = out_channels
    kernel_size = 3, 3
    padding = 1, 1
    MConv(in_dims, out_dims, kernel_size, padding)
end

function MConv(in_dims::Tuple, out_dims::Tuple)
    kernel_size = 3, 3
    padding = 1, 1
    MConv(in_dims, out_dims, kernel_size, padding)
end

function get_phenotype(prev_layer::MGene, cur_layer::MConv)
    return Conv(cur_layer.kernel_size, prev_layer.out_dims[3]=>cur_layer.out_dims[3], pad=cur_layer.padding)
end

###################################
# Depthwise Separable Convolution #
###################################
mutable struct MDepthwiseConv <: MGene
    in_dims::Tuple
    out_dims::Tuple
    kernel_size::Tuple
    padding::Tuple
end

function MDepthwiseConv(in_dims::Tuple)
    out_dims = in_dims
    out_channels = sample(FILTER_SPACE)
    out_dims[3] = out_channels
    kernel_size = 3, 3
    padding = 1, 1
    MDepthwiseConv(in_dims, out_dims, kernel_size, padding)
end

function MDepthwiseConv(in_dims::Tuple, out_dims::Tuple)
    kernel_size = 3, 3
    padding = 1, 1
    MDepthwiseConv(in_dims, out_dims, kernel_size, padding)
end

function get_phenotype(prev_layer::MGene, cur_layer::MDepthwiseConv)
    return DepthwiseConv(cur_layer.kernel_size, prev_layer.out_dims[3]=>cur_layer.out_dims[3], pad=cur_layer.padding)
end

###################################
####### Batch Normalization #######
###################################
mutable struct MBatchNorm <: MGene 
    in_dims::Tuple
    out_dims::Tuple
end

function MBatchNorm(in_dims::Tuple) 
    MBatchNorm(in_dims, in_dims)
end

function get_phenotype(prev_layer::MGene, cur_layer::MBatchNorm)
    return BatchNorm(cur_layer.out_dims[3])
end

###################################
######### ReLU Activation #########
###################################

mutable struct MReLU <: MGene 
    in_dims::Tuple
    out_dims::Tuple
end

function MReLU(in_dims::Tuple) 
    MReLU(in_dims, in_dims)
end

function get_phenotype(prev_layer::MGene, cur_layer::MReLU)
    return x -> relu.(x)
end

###################################
########### Max Pooling ###########
###################################
mutable struct MMaxPool <: MGene
    in_dims::Tuple
    out_dims::Tuple
    kernel_size::Tuple
end

function MMaxPool(in_dims::Tuple)
    out_dims = in_dims
    out_dims[1] = in_dims[1] / 2
    out_dims[2] = in_dims[2] / 2
    kernel_size = 2, 2
    MMaxPool(in_dims, out_dims, kernel_size)
end

function get_phenotype(prev_layer::MGene, cur_layer::MMaxPool)
    return MaxPool(cur_layer.kernel_size)
end

###################################
######### Skip Connection #########
###################################
mutable struct MSkipConnection <: MGene
    in_dims::Tuple
    out_dims::Tuple
    type::String
end

function MSkipConnection(in_dims::Tuple, out_dims::Tuple)
    # type = sample(SKIP_CONNECTION_TYPES)
    type = "concat"
    MSkipConnection(in_dims, out_dims, type)
end

function get_phenotype(prev_layer::MGene, cur_layer::MSkipConnection, last_pheno)
    
    if cur_layer.type == "add"
        return Nothing
    elseif cur_layer.type == "concat"
        # input_dims = 40, 40, 
        # a = randn(Float32, 2, 2, 4, 1)
        # b = padarray(a, Fill(0, (1, 1, 0, 0)))
        return SkipConnection(last_pheno, (mx, x) -> cat(mx, x, dims=3))
    end
end

