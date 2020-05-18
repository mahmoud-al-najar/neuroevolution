using Random
using Flux
include("base.jl")


GENES = [
    "ConvBlock",
    "DSConvBlock",
    "MaxPool",
    "SkipConnection"
]

LEMONADE_MORPHISMS = [
    "Net2DeeperNet",
    "Net2WiderNet",
    "AddSkipConcat",
    "AddSkipAdd"
]

LEMONADE_APPROXIMATE_MORPHISMS = [
    "Remove",
    "Prune",
    "Conv2DSConv"
]

LAYER_OUTPUT = Dense(256, 1, relu)

########################################
#            LEMONADE GENES            #
########################################

mutable struct MConvBlock <: MBlockGene
    gene_conv::MConv
    gene_bn::MBatchNorm
    gene_relu::MReLU
end

function MConvBlock()
    gene_conv = MConv()
    gene_bn = MBatchNorm()
    gene_relu = MReLU()
    MConvBlock(gene_conv, gene_bn, gene_relu)
end

function get_phenotype(input_channels::Int64, block::MConvBlock)
    layers = Array{Any}(nothing, 3)
    layers[1] = get_phenotype(input_channels, block.gene_conv)
    layers[2] = get_phenotype(block.gene_conv.n_channels, block.gene_bn)
    layers[3] = get_phenotype(block.gene_conv.n_channels, block.gene_relu)
    return layers
end

mutable struct MDepthwiseConvBlock <: MBlockGene
    gene_conv::MDepthwiseConv
    gene_bn::MBatchNorm
    gene_relu::MReLU
end

function MDepthwiseConvBlock()
    gene_conv = MDepthwiseConv()
    gene_bn = MBatchNorm()
    gene_relu = MReLU()
    MDepthwiseConvBlock(gene_conv, gene_bn, gene_relu)
end

function get_phenotype(input_channels::Int64, block::MDepthwiseConvBlock)
    layers = Array{Any}(nothing, 3)
    ### TODO: placeholder?
    if block.gene_conv.n_channels < input_channels || block.gene_conv.n_channels % input_channels != 0
        block.gene_conv.n_channels = input_channels * 2
    end
    ###
    
    layers[1] = get_phenotype(input_channels, block.gene_conv)
    layers[2] = get_phenotype(block.gene_conv.n_channels, block.gene_bn)
    layers[3] = get_phenotype(block.gene_conv.n_channels, block.gene_relu)
    return layers
end
