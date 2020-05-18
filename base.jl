using StatsBase
using Flux


FILTER_SPACE = [16, 32, 64, 128, 256, 512]
SKIP_CONNECTION_TYPES = ["add", "concat"]

abstract type MGene end
abstract type MBlockGene <: MGene end

###################################
########### Convolution ###########
###################################
mutable struct MConv <: MGene
    kernel_size::Tuple
    n_channels::Int64
    padding::Tuple
end

function MConv()
    kernel_size = 3, 3
    n_channels = sample(FILTER_SPACE)
    padding = 1, 1
    MConv(kernel_size, n_channels, padding)
end

function get_phenotype(input_channels::Int64, layer::MConv)
    return Conv(layer.kernel_size, input_channels=>layer.n_channels, pad=layer.padding)
end

###################################
# Depthwise Separable Convolution #
###################################
mutable struct MDepthwiseConv <: MGene
    kernel_size::Tuple
    n_channels::Int64
    padding::Tuple
end

function MDepthwiseConv()
    kernel_size = 3, 3
    n_channels = sample(FILTER_SPACE)
    padding = 1, 1
    MDepthwiseConv(kernel_size, n_channels, padding)
end

function get_phenotype(input_channels::Int64, layer::MDepthwiseConv)
    return DepthwiseConv(layer.kernel_size, input_channels=>layer.n_channels, pad=layer.padding)
end

###################################
####### Batch Normalization #######
###################################
mutable struct MBatchNorm <: MGene 
    MBatchNorm() = new()
end

# function MBatchNorm() end

function get_phenotype(input_channels::Integer, layer::MBatchNorm)
    return BatchNorm(input_channels)
end

###################################
######### ReLU Activation #########
###################################

mutable struct MReLU <: MGene 
    MReLU() = new()
end

function get_phenotype(input_channels::Integer, layer::MReLU)
    return x -> relu.(x)
end

###################################
########### Max Pooling ###########
###################################
mutable struct MMaxPool <: MGene
    kernel_size::Tuple
end

function MMaxPool()
    kernel_size = 2, 2
    MMaxPool(kernel_size)
end

function get_phenotype(layer::MMaxPool)
    return MaxPool(layer.kernel_size)
end

###################################
######### Skip Connection #########
###################################
mutable struct MSkipConnection <: MGene
    type::String
end

function MSkipConnection()
    type = sample(SKIP_CONNECTION_TYPES)
    MSkipConnection(type)
end

function get_phenotype(layer::MSkipConnection)
    # TODO: implementation
    ErrorException("No SkipConnections yet.")
    return Nothing
end

########################################
########################################
#########      INDIVIDUAL      #########
########################################
########################################
mutable struct Individual
    genotype::Array{MGene}
    fitness::Float64
    model::Chain
    Individual() = new()
end

function construct_model!(ind::Individual, input_channels::Int64, output_layer::Dense)
    layers = []
    last_output_channels = input_channels
    for g in ind.genotype
        if typeof(g) <: MBlockGene
            ph = get_phenotype(last_output_channels, g)
            last_output_channels = g.gene_conv.n_channels
            for p in ph
                push!(layers, p)
            end      
        else   
            push!(layers, get_phenotype(last_output_channels, g))
        end
    end

    ### TODO: placeholder
    ex_data = randn(Float32, 40, 40, 4, 1)
    dense_in_size = size(Chain(layers..., x -> reshape(x, :, size(x, 4)))(ex_data))[1]
    ###

    ind.model = Chain(layers..., x -> reshape(x, :, size(x, 4)), Dense(dense_in_size, 1, relu; initb = ones))
end
