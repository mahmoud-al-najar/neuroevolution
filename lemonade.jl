using Random
using Flux
include("base.jl")


GENES = [
    "ConvBlock",
    "DSConvBlock",
    "MaxPool",
    "SkipConnection"
]

MUTATIONS = [
    "NM_InsertConvBlock",
    "NM_IncreaseConvFilters",
    "NM_InsertSkipConnection",
    "ANM_RemoveLayer",
    "ANM_PruneConvLayer",
    "ANM_Conv2DSConv"
]

LAYER_OUTPUT = Dense(256, 1, relu)

########################################
#            LEMONADE GENES            #
########################################

mutable struct MConvBlock <: MBlockGene
    gene_conv::MConv
    gene_bn::MBatchNorm
    gene_relu::MReLU
    in_dims::Tuple
    out_dims::Tuple
end

function MConvBlock(in_dims::Tuple, out_dims::Tuple)
    gene_conv = MConv(in_dims, out_dims)
    gene_bn = MBatchNorm(out_dims)
    gene_relu = MReLU(out_dims)
    MConvBlock(gene_conv, gene_bn, gene_relu, in_dims, out_dims)
end

function get_phenotype(prev_layer::MGene, cur_layer::MConvBlock)
    layers = Array{Any}(nothing, 3)
    layers[1] = get_phenotype(prev_layer, cur_layer.gene_conv)
    layers[2] = get_phenotype(cur_layer.gene_conv, cur_layer.gene_bn)
    layers[3] = get_phenotype(cur_layer.gene_bn, cur_layer.gene_relu)
    return layers
end

mutable struct MDepthwiseConvBlock <: MBlockGene
    gene_conv::MDepthwiseConv
    gene_bn::MBatchNorm
    gene_relu::MReLU
    in_dims::Tuple
    out_dims::Tuple
end

function MDepthwiseConvBlock(in_dims::Tuple, out_dims::Tuple)
    gene_conv = MDepthwiseConv(in_dims, out_dims)
    gene_bn = MBatchNorm(out_dims)
    gene_relu = MReLU(out_dims)
    MDepthwiseConvBlock(gene_conv, gene_bn, gene_relu, in_dims, out_dims)
end

function get_phenotype(prev_layer::MGene, cur_layer::MDepthwiseConvBlock)
    layers = Array{Any}(nothing, 3)
    ### TODO: placeholder?
    if cur_layer.gene_conv.out_dims[3] < prev_layer.out_dims[3] || cur_layer.gene_conv.out_dims[3] % prev_layer.out_dims[3] != 0
        println("changing channels")
        # cur_layer.gene_conv.out_dims[3] = prev_layer.out_dims[3] * 2
        cur_layer.gene_conv = MDepthwiseConv(cur_layer.gene_conv.in_dims, (prev_layer.out_dims[1], prev_layer.out_dims[2], prev_layer.out_dims[3] * 2))
        cur_layer.out_dims = cur_layer.gene_conv.out_dims
        cur_layer.gene_bn = MBatchNorm(cur_layer.gene_conv.out_dims)
        cur_layer.gene_relu = MReLU(cur_layer.gene_bn.out_dims)
    end
    ###
    layers[1] = get_phenotype(prev_layer, cur_layer.gene_conv)
    layers[2] = get_phenotype(cur_layer.gene_conv, cur_layer.gene_bn)
    layers[3] = get_phenotype(cur_layer.gene_bn, cur_layer.gene_relu)
    return layers
end

########################################
#          LEMONADE MUTATIONS          #
########################################

function mutate!(ind::Individual)
    # mutation = sample(MUTATIONS)
    mutation = "NM_InsertConvBlock"
    
    if mutation == "NM_InsertConvBlock"
        # genotype = ind.genotype
        
        insert!(ind.genotype, 2, MConv((40, 40, 16), (40, 40, 16)))
        # println(genotype)
        # insert!(ind.model, 2, Conv())
    end

end
