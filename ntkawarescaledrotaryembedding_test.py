from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
import argparse
from llama import Llama, Dialog

""" #### Define patch"""

import transformers

parser = argparse.ArgumentParser()
parser.add_argument("--extend_type", type=str, default="", help = "extend_type, choose from [Linear, NTK]")
args = parser.parse_args()

class ScaledRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        max_position_embeddings = 16384
        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(
            self.max_seq_len_cached,
            device=self.inv_freq.device,
            dtype=self.inv_freq.dtype,
        )
        self.scale = 8
        t /= self.scale
old_init = transformers.models.llama.modeling_llama.LlamaRotaryEmbedding.__init__
def ntk_scaled_init(self, dim, max_position_embeddings=2048, base=10000, device=None):
    
    #The method is just these three lines
    max_position_embeddings = 16384
    a = 8 #Alpha value
    base = base * a ** (dim / (dim-2)) #Base change formula

    old_init(self, dim, max_position_embeddings, base, device)

""" #### Apply NTK-Scaled Init patch"""
if args.extend_type == "NTK":
    transformers.models.llama.modeling_llama.LlamaRotaryEmbedding.__init__ = ntk_scaled_init
elif args.extend_type == "Linear":
    transformers.models.llama.modeling_llama.LlamaRotaryEmbedding = ScaledRotaryEmbedding
# model_path = "TheBloke/OpenAssistant-SFT-7-Llama-30B-HF"
model_path = "/home/yeq6/Research_project/llama/llama-2-7b-chat_hf"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")

""" #### Load research paper as prompt

Taken from https://arxiv.org/pdf/2404.00271
"""

#@title
prompt = '''
You are given this machine learning research paper, please read it carefully and answer the follow up question.

=== BEGIN ===

TG-NAS: Leveraging Zero-Cost Proxies with Transformer and Graph Convolution Networks for nEfficient Neural Architecture Search

Ye Qiao , Haocheng Xu , and Sitao Huang
University of California, Irvine, Irvine CA 92697, USA

Abstract. Neural architecture search (NAS) is an effective method for
discovering new convolutional neural network (CNN) architectures. However, existing approaches often require time-consuming training or intensive sampling and evaluations. Zero-shot NAS aims to create training-free
proxies for architecture performance prediction. However, existing proxies
have suboptimal performance, and are often outperformed by simple
metrics such as model parameter counts or the number of floating-point
operations. Besides, existing model-based proxies cannot be generalized
to new search spaces with unseen new types of operators without golden
accuracy truth. A universally optimal proxy remains elusive. We introduce TG-NAS, a novel model-based universal proxy that leverages a
transformer-based operator embedding generator and a graph convolution
network (GCN) to predict architecture performance. This approach guides
neural architecture search across any given search space without the need
of retraining. Distinct from other model-based predictor subroutines,
TG-NAS itself acts as a zero-cost (ZC) proxy, guiding architecture search
with advantages in terms of data independence, cost-effectiveness, and
consistency across diverse search spaces. Our experiments showcase its
advantages over existing proxies across various NAS benchmarks, suggesting its potential as a foundational element for efficient architecture search.
TG-NAS achieves up to 300× improvements in search efficiency compared
to previous SOTA ZC proxy methods. Notably, it discovers competitive
models with 93.75% CIFAR-10 accuracy on the NAS-Bench-201 space
and 74.5% ImageNet top-1 accuracy on the DARTS space.

Keywords: zero-cost proxy · neural architecture search · transformer


Introduction

Deep convolutional neural networks (CNN) have achieved incredible performance
in computer vision, speech recognition, object detection, and many other fields
[12, 16, 25, 34]. Since then many manually designed CNN topologies that target
either high performance or high computational efficiency have been proposed
[10, 38]. With larger and deeper expert-designed CNN models, people can easily
achieve state-of-the-art performance in image classification, object detection,
and many other applications. However, manual network architecture design

is costly and requires a significant amount of time, expertise, and computing
resources. It becomes even more challenging if the neural network is designed
for hardware platforms with stringent resource and energy constraints. Thus
neural architecture search (NAS) was proposed to automate the model design
and provide an opportunity to explore specialized model architectures within
certain performance and resource constraints [23, 41]. Despite the advancements
in automated search techniques, the majority of neural architecture search (NAS)
approaches still face the challenge of time-consuming training and evaluation. In
the early stage, the prevailing methodology involved iteratively sampling DNN
architectures from the search space, training them, and using their performance
to guide the search process [22]. These approaches were primarily based on
reinforcement learning (RL) or evolutionary algorithms.
In recent years, zero-shot proxies [15, 31, 35, 46] have emerged as a promising
technique in neural architecture search (NAS). Zero-shot proxies use lightweight
computation, typically a single forward pass with one minibatch of data, to
assign a score to each DNN architecture in the search space. The scores obtained
from these proxies are expected to correlate with the final accuracy of candidate
architectures [15,31,46]. This approach not only conserves computational resources
but also accelerates the overall NAS process, enabling researchers to explore a
larger number of architectures. However, recent general zero-shot proxies struggle
to consistently perform well for practical uses. Many of them cannot outperform a
naïve proxy like the number of parameters (Params) and are often data-dependent
and not fast enough. The generally high dependence and correlation of different
zero-cost (ZC) proxies from similar domains, such as gradient information or
kernel methods [7, 18], may also compromise the effectiveness of zero-shot NAS
approaches.
To address these limitations of zero-cost performance indicators, many modelbased performance predictors have emerged. These predictors rely on training
models to forecast the final validation accuracy of an architecture solely based
on its topology and operators. Common models include Gaussian processes [20],
deep neural networks [30, 40], among others. However, these methods are not
zero-shot and cannot be considered zero-cost since they typically require hundreds
of fully-trained architectures as training data, resulting in prolonged search times.
Additionally, model-based prediction has often been used only to accelerate NAS
query times, given their lack of generalization and inability to handle unseen
operators that are not encoded during predictor model training.
This leads us to deep think: how can we develop a pure universal prediction
model capable of predicting any given architecture? Is it possible to develop a
generally applicable model-based predictor that can handle the aforementioned
unseen operator and break the generalization limitations? In other words, why
let model-based predictors settle for a subroutine when it has the potential to
take the center stage? Our response to these questions is TG-NAS, a modelbased predictor-only NAS framework that can be generalized to new search
spaces and unseen operators without retraining. It features a general transformer
operator encoder and a graph convolution network (GCN) trainer. This novel

approach offers an exceptionally efficient and zero-cost solution to general neural
architecture search problems and bridges the gap between model-based predictors
and ZC proxies. We will open source this work to facilitate future research. Link
to the source code can be found in the supplementary materials.
The main contributions of TG-NAS can be summarized as follows:
– We propose a universally applicable, data-independent performance predictor model that can handle unseen operators in new search spaces without
retraining. The model features a graph convolutional network (GCN), an
architecture encoder, and a transformer-based operator embedding generator.
– Our proposed model acts as a zero-cost proxy, guiding architecture search.
This opens up a new space for prediction model-only architecture search, and
we provide a comprehensive analysis on proxy independence.
– We propose a pruning-evolutionary hybrid searching method, enabling quick
and accurate identification of the best architecture within the search space.
– Our analysis of TG-NAS proxies demonstrates remarkable performance on
both NAS-Bench-201 and DARTS spaces with CIFAR-10/ImageNet datasets.
TG-NAS achieves up to 300× faster search compared to other zero-cost proxies
and up to 331, 200× faster than other NAS methods, while maintaining high
accuracy.

Background and Related Work

Neural Architecture Search (NAS)

Conventional NAS algorithms are an important step in automating the design
of neural architectures for a given task [46]. A typical NAS contains a search
strategy that selects the candidate neural architecture from a predefined search
space and an estimation strategy that enables the performance estimation
for the current candidate.
Search space can be categorized as the following types: macro search
space [13], chain-structures search space [38], cell-based search space [8, 28, 53],
and hierarchical search space [27]. Among those, cell-based search space is the
most popular one in NAS [46]. The searchable cells (a directed acyclic graph of operations) make up the microstructure of the search space while the macrostructure
(ways and number of cells to stack together) is fixed. For example, NAS-Bench101 [53] contains 423,624 unique architectures, and each cell consists of 7 nodes
(each node chosen from 3 operators). In NAS-Bench-201 [8], there are 15,625 cell
candidates and each cell consists of 4 nodes (each node chosen from 5 operators).
In contrast, the DARTS search space [28] is more expansive, featuring approximately 1018 architectures. It consists of two cells, each containing seven nodes.
The first two nodes receive inputs from previous layers, and the intermediate
four nodes can assume any DAG structure, each having two incident edges. The
last node serves as the output node, and each edge can take on one of eight
operations. In this work, we perform our experiments on those three search spaces
for different purposes.

The search strategy in NAS has been widely explored. There are some
well-known black-box optimizations, such as random selection with full training,
reinforcement learning, evolution, Bayesian optimization, and Monte Carlo tree
search. With these search strategies, we still need to train the searched architecture
and use the performance result to guide the search, which is a time-consuming
process. To overcome the training time bottleneck, one-shot techniques were
introduced as an estimation strategy. These techniques involve training a
single (one-shot) supernet, which is an over-parameterized architecture that
encompasses all possible architectures in the search space [2, 3, 28]. Once the
supernet has been trained, each architecture in the search space can be evaluated
by inheriting its weights from sampling the subnet within the supernet.


Zero-Cost (ZC) Proxies for NAS

However, the training of the supernet still takes up the majority of the NAS
runtime. To speed up the entire NAS process, a more efficient estimation
strategy to predict the performance of the searched architecture is needed.
Recently, zero-cost proxies have been introduced as a family of performance
prediction techniques. The idea is to avoid training any neural network and
to run a fast computation (one single forward pass over certain data) over
the searched architecture that can assign a ranking or score to each searched
architecture.
The expressivity of a neural network, which relates to the complexity of the
function it can represent, has been widely used as one of the zero-cost proxies.
Several recent works, such as TE-NAS [4], ZEN-NAS [24], and NASWOT [31], approximate the behavior of neural networks by considering the expressivity of ReLU
network and the number of linear regions they can separate. Kernel methods have
also been employed to approximate the convergence and generalization ability of
networks without training. For instance, TE-NAS [4], formulates neural networks
as a Gaussian process [17, 52] and analyzes randomly-initialized architectures by
the spectrum of the neural tangent kernel (NTK) [11,48] and the number of linear
regions in the input space. Zico [21] extends such kernel-based analysis to reveal
the relationships between the gradient properties and the training convergence
and generalization capacity of neural networks. Similarly, gradient with respect
to the parameters of neural networks is proved to be different approximations
of Taylor expansion of deep neural networks [1, 18, 44]. SNIP [18] introduced
a saliency metric computed at initialization using a single minibatch of data;
Grasp [44] improved upon SNIP [18] by approximating the change in gradient
norm instead of loss; Synflow [1] generalized previous approaches and proposed a
modified version computes a loss which is simply the product of all parameters in
the network thus no data is needed. Gradient concerning feature maps combined
with the number of linear regions have been used in Zen-NAS [24]. There are
also other zero-cost proxies considering Jacobian covariances, fisher information,
and other important theoretical indicators [1, 31, 42]. Although zero-cost proxies
can expedite estimation time, they often compromise prediction accuracy and
introduce biases stemming from data dependency [46].


Issue of Model-based Prediction and Unseen Operators

Despite the inherent limitations of zero-cost performance prediction, the integration of model-based prediction has emerged as a pivotal component in guiding
neural architecture search (NAS) algorithms. This approach is particularly useful
when combined with Bayesian optimization and utilized as a subroutine [13, 40].
Various types of predictor models, including Gaussian processes (GP), multi-layer
perceptron (MLP), long short-term memory (LSTM) networks, and graph neural
networks (GNN), have been employed. Typically, as the algorithm progresses
and a set of fully evaluated architectures becomes available, a meta-model is
trained using architecture topology as features and validation accuracies as labels.
This model is then used to predict the validation accuracy of yet-to-be-evaluated
architectures. Notably, White et al. [47] demonstrated that augmenting the model
with Jacobian covariance as an additional feature can enhance its performance
by up to 20%. Shen et al. [39] further extended this approach by integrating
zero-cost proxies into Bayesian optimization, resulting in 3-5× speedups over
previous methods.
Existing model-based predictor approaches exhibit notable biases, limiting
their effectiveness to specific search spaces and requiring fully evaluated architectures as labels. The high initialization time is also a concern. Dudziak et al. [9]
attempted to address this issue by leveraging the model’s binary relation and
training a prediction model through iterative data selection. However, their approach still requires over 6 GPU days to conduct the search process. Furthermore,
due to simplistic architecture and feature embedding methods, such as one-hot encoding, these models cannot process unseen operators. These limitations prompt
us to explore a more robust operator encoding method that enables a pretrained
prediction model to operate effectively in any architecture search space and
accommodate unseen operators. Essentially, we investigate the viability of relying
solely on pretrained model-based prediction as a universal zero-cost proxy for
guiding searches across diverse architecture spaces with negligible search time.


TG-NAS: A General Zero-Cost Proxy

In this work, we propose TG-NAS, a general zero-cost proxy method for neural
architecture search with a transformer embedding generator and a graph neural
network trainer, supporting unseen operators in new search spaces. The embedding generator produces the desired feature embeddings with various techniques
and the GCN trainer functions as a model-based predictor to predict the accuracy (or ranking) of a given architecture. Finally, the data-independent predicted
ranking will guide the search algorithm as shown in Fig. 8.


Architecture Representation

Various neural architecture search work represent their networks in the cell
structure. For instance, DARTS [28], and NATS-Bench (NAS-Bench-201) [6, 8]
define a cell-based search space representing each architecture as a directed acyclic
graph (DAG), with nodes representing the features. NAS-Bench-101 [53] on the
other hand utilizes nodes to represent layers (operators) and edges for forward
dataflow propagation. BANANAS [45] proposed a novel path-based encoding
scheme and claimed it scales better than other methods.
In this work, we represent the DNN architecture candidates using DAG with
nodes representing operators and edges corresponding to model data propagation
flow. We then represent the graphs with adjacent matrix and operator node
embeddings, which becomes data input for our graph convolution network (GCN)
[14] trainer. To make all search space comply with this representation, TGNAS unifies other cell-based search space representations as shown in Fig. 8.
We applied this transformation to both NAS-Bench-201 and DARTS space in
this work. NAS-Bench-101 was provided with the aforementioned architecture
graphs, therefore no transformation is needed. Fig. 8 shows an example on NASBench-201. The final model architectures of each search space are obtained by
stacking multiple repeated cells with some other predefined cells in between. The
differences between different DNN architecture candidates are purely determined
by the cell architecture (represented as graph), so we use the embedding of
individual cell architectures (as graphs) to represent the entire DNN architecture.
Operator node features are encoded using a transformer model with a fixed length
embedding size. The details can be seen in the following section.


Operator Embedding Generator

In our approach, we encode model cell graphs using an adjacency matrix together
with node embeddings. However, encoding operators demand special attention as it involves representing and distinguishing various deep learning operators.
Previous methods, which typically rely on one-hot vectors for operator encoding,
are deemed suboptimal and non-portable, especially when dealing with unseen
search spaces and operators.
Recognizing that the names of operators inherently contain valuable information, we assert that the operator name alone can provide insight into the
operation. For instance, the operator name “CONV3x3-BN-ReLU ” suggests
that it contains a two-dimensional convolution with a 3x3 kernel, followed by
batch normalization and rectified linear activation. Therefore, we propose to
construct a robust embedding model capable of extracting internal semantic information from operator names or their descriptive sentences in natural languages.
For example, in the high-dimensional encoding space, operators like conv3x3
are expected to be closer to conv5x5 than to maxpool3x3. Additionally, if the
embedding model comprehends one type of operator, it should readily extend its
knowledge to similar operators with, for example, different kernel sizes.
Certain existing works have attempted to construct embedding vectors from
words or sentences, such as GloVe [33], or employed character embeddings to
capture fine-grained semantic and syntactic regularities. However, our earlier
experiments indicated that these methods face challenges when dealing with
unseen words, particularly operators in our case. Consequently, we have opted
for Sentence BERT [37] as our primary method for generating desired operator
embeddings. As illustrated in Fig. 8, the Sentence Transformer utilizes siamese and
triplet network structures to derive semantically meaningful sentence embeddings
that can be compared using cosine similarity. A pooling operation is applied to
the output of the pretrained transformer model to obtain a fixed-sized sentence
embedding. We compute the mean of all output word vectors as the pooling
strategy to generate the final operator embedding.
In our experiments, we explore three different sizes of pretrained sentence
transformer models and three distinct operator sentence lengths, crucial for

embedding generation analysis. As outlined in Table 1, we define and experiment
with short, medium, and long operator descriptions as inputs to the embedding
generator model. The test results on NAS-Bench-201 are presented in Table 2.
All three models undergo pretraining on the same 25 dataset collections [37],
containing over 1 billion training sentence pairs in total. The triplet objective
function is employed during the pretraining phase. This function, given an anchor
sentence a, a positive sentence p, and a negative sentence n, tunes the network
to ensure that the distance between a and p is smaller than the distance between
a and n. Mathematically, the loss function can be expressed as:
  \mathcal {L} = max(||s_a - s_p|| - ||s_a - s_n|| + \epsilon ) 

where si ’s are the sentence embeddings for a, n, and p respectively, and || · ||
denotes a distance metric and ϵ represents a margin. The chosen margin ϵ ensures
that sp is at least ϵ closer to sa than sn . In this context, the Euclidean distance
serves as the distance metric, and ϵ is specifically set to 1 during training.
The all-MiniLM-L6-v2-64 entries in Table 2 are downsampled from the
original all-MiniLM-L6-v2 model using principal component analysis (PCA)
to achieve a 64-dimensional embedding vector length. The results of our GCN
model prediction ranking (which we will discuss in the next section), trained with
NAS-Bench-101 and tested on NAS-Bench-201, are presented in Table 2. Notably,
the combination of short sentence embedding and the pretrained all-MiniLM-L6v2-64 sentence transformer model, with embedding length of 384, yields the best
results for both Kendall’s τ and Spearman’s ρ correlation coefficients. Our later
experiments will adopt this combination.
Furthermore, we anticipate that fine-tuning the embedding model with context containing specific knowledge about deep learning operators would further
enhance the performance of our proposed approach. We leave this area as a
potential direction for future research.


GCN Proxy Trainer

After completing the universal architecture encoding and operator feature embedding, we employ a two-layer graph convolution network (GCN) as our prediction
model. With normalization trick, GCN can be defined as
  H = X*_{G}G_{\Theta } = f(\Bar {A}X\Theta ), (2)
where Ā = D̃ ÃD̃ , Ã = A + In , and D˜ii = j Ãij . To prevent overfitting
to a particular training space, we incorporate graph normalization [5] and weight
decay techniques. Our overarching objective is to deliver a universally applicable pretrained predictor model that requires no tuning for new search spaces.
Consequently, the GCN predictor model is subject to heavy regularization. An
additional crucial factor influencing our choice of GCN over other prediction
models is its capability to handle vast differences in architectures. Given the
varying dimensions of the adjacency matrix (from unseen search spaces) and
operator matrix (from unseen operators), GCN emerges as a suitable choice,
demonstrating flexibility in accommodating diverse architectural structures.


Correlation

To assess the predictor’s applicability,
NASBench101 GCN Functional Validation
we partition the NAS-Bench-101 space into
train/validation splits ranging from 90% to
1%. The results, depicted in Fig. 2, demon0.6
strate the predictor’s effectiveness, even when
trained on only 1% of the architecture data.
This is evident in the strong performance in0.2
dicated by Kendall’s τ and Spearman’s ρ correlation coefficients. 
Exploring the use of different graph neural networks (GNNs) such as
Percentage of Data Used for Training
graph attention network (GAT) [43] may yield
different results and behaviors. We consider Fig. 2: GCN Predictor Functional
this as an open area for further investigation. Validation on NAS-Bench-101

Pruning-Evolutionary Hybrid Searching

The search strategy generally falls into two types. The first type involves applying
black-box optimization techniques (reinforcement learning and evolution search
etc.) in conjunction with a sampling-based method to explore the predefined
search space. These methods sample and evaluate architectures to optimize the
searching agent’s reward. The other type is close to one-shot techniques, where
a hypernetwork will be introduced with architecture representation relaxation
to enable searching using gradient descent. It can be applied to all DAG-based
search spaces [28] due to its continuous hyperparameter for the architecture.
However, when considering the cell-based search space, the computational burden
of these sampling-based one-shot methods remains dependent on the scale of
the search space and hence it has limited effectiveness. Consider a network
comprising interconnected cells in a directed acyclic graph structure. Each cell
contains E edges, with each edge selecting a single operator from a set of |O|
potential candidates. Consequently, there are |O|E distinct cells, and during the
sampling-based search process, α|O|E networks must be sampled. The parameter
α represents the sampling process efficiency, where a smaller value can help find
superior architectures faster.
Meanwhile, evolutionary and genetic algorithms have been commonly used
to optimize the NAS [46]. To enhance the efficiency of the search, we propose a
pruning-evolutionary hybrid searching method. The pruning method, inspired
by [19], reduces the search cost from α|O|E to |O| · E. Algorithm 1 shows our
proposed algorithm.
For the pruning search part, we start with a supernetwork Ninit encompassing
all possible operators for each edge. Then we iteratively prune one operator on
each edge until the current supernet Nt transforms into a single-path network,
representing the final searched architecture. For the evolutionary search part, we
first initialize the entire population of architecture candidates with continuous
parameters, where we map all operators evenly into the value between 0 and 1.
Then mutation operation and crossover operation are performed to produce a
new child. After generating all the offspring, the selection process will kick in


using our proposed proxy to select the elite candidates for the next generation.
After finishing both parts of searching, we select the final architecture from the
best of the two search results. This hybrid searching method is valid and efficient
due to the effectiveness of the proposed zero-cost proxy.

Algorithm 1: TG-NAS Pruning-Evolutionary Hybrid Search Algorithm
Data: M0 supernet, E edges in each cell, every edge have O operators
while Mt is not a single path network do
for Cell_type, ct in Mt do
for edge Ei , operator Oj in ct, Ei do
for operator Oj in Ei do
Pt,ei ,oj ← GCN _Inf erence(Mt,ei ,oj );
∆Pt,ei ,oj ← Pt,ei ,oj − Pt,ei \oj ;
end for
end for
Rank ← index of oj in descending of ∆Pt ;
end for
for ei in E do
op ← argmin{S(op ) op ∈ ei };
Mt+1 ← Mt − op ;
end for
t ← t + 1;
end while
Data: N P population size
g ← 0;
while |pop| < N P do
popi ← random_conf iguration();
pop′ i ← discretized_architecture(popi );
f itnessi ← GCN _Inf erence(pop′ i );
end while
while g < gmax do
Vg ← mutate(popg );
Ug ← crossover(Vg , popg );
U ′ g ← discretized_population(Ug );
f itnessg ← GCN _Inf erence(U ′ g );
popg+1 , f itnessg+1 ← select(popg , Ug );
end while
return M ax(Mt , P OPg );


Experiment and Results Analysis


Independent Analysis and Combination with other proxies

Multiple proxies could be combined to enhance the performance prediction
accuracy. However, certain proxies could be highly correlated and do not provide
extra information. Therefore, not all combinations provide good performance.
We conducted assessments of zero-cost proxies on NAS-Bench-201 search spaces
using the aforementioned three datasets. Additionally, by evaluating the same
search space across different tasks, we aimed to determine if data- and taskindependent zero-cost proxies offer universally applicable rankings. Our evaluation
encompassed 14 zero-cost proxies, including our TG-NAS proxy result. We
randomly sampled 1000 architectures from the search space and evaluated each
zero-cost proxy metric for each architecture. Subsequently, we computed the
Spearman′ s rank correlation between each proxy.
Our analysis revealed a trend where the rank correlations of some zero-cost
proxies were highly correlated, such as f lops (number of floating-point operations)
and params (model parameter count), which is expected due to the relationship
between parameters and computation in deep learning. This prompted us to
compute the full correlations between all 14 pairs of zero-cost proxies and analyze
their behaviors.
Fig. 10 shows the heatmap of correlations between pairs of popular proxies
(calculated with CIFAR-10 results; CIFAR-100 and ImageNet16-120 results are
available in the supplementary materials). We observed a consistent trend where
f lops and params exhibited high correlations with each other. Additionally,
synf low [1] often showed high correlations with f lops and params, consistent
with recent work by Ning et al. [32], which demonstrated that synf low′ s value
increases with the number of parameters in the architecture. We also noted
high correlations between grad_norm [1], snip [18], grasp [44], nwot [31], and
f isher [42], occasionally with l2_norm [1] as well, as they all leverage gradient
saliency matrix information. Moreover, epe_nas [29] and jacobian covariance
(jacov) [31] were highly correlated with each other.

It’s intriguing to note that the zen score [24] and our TG-NAS proxy demonstrated the highest level of independence among all proxies evaluated. This
suggests that our method offers a unique perspective on understanding model
architectures. This discovery holds the potential to provide new insights into
the design of zero-cost neural architecture search and the analysis of optimal
combinations of zero-cost proxies. This could lead to further enhancements in
zero-shot NAS techniques overall. We haven’t included some zero-cost proxies due
to various reasons: they either require training of supernet to make an evaluation
or they did not release the source code [3, 9, 40, 51].
Table 3: Kendall’s τ and Spearman’s ρ correlation between various zero-cost proxies
Proxy Name Params FLOPs SNIP Fisher Synflow Zen-score grad-norm TG-NAS (ours)

For testing on NAS-Bench-201, our predictor model was trained using the entire
NAS-Bench-101 architecture set, and the obtained predictions were employed to
guide the search on the new space, NAS-Bench-201, using the proposed method.
The Kendall’s τ and Spearman’s ρ correlation scores displayed in Table 3 indicate
that our TG proxy outperforms other zero-cost proxies when evaluated on NASBench-201 spaces. Fig. 4 demonstrates that our TG proxy is highly positively
correlated with the architecture’s test accuracy. Additionally, as illustrated in
Table 4, our TG-NAS outperforms the majority of zero-shot NAS approaches
and is comparable to the state-of-the-art TE-NAS results, achieving 93.75% top-1
accuracy on CIFAR-10, while consuming only a fraction of the searching time of
prior works.

Notably, despite the efficiency claims of zero-cost NAS methods over conventional NAS due to avoiding the training of sampled architectures, variations
in computational cost and search time persist among them. For example, TENAS [4] required over 4 GPU hours, while ZiCo [21] demanded over 10 GPU
hours for the search on ImageNet on the DARTS space. This discrepancy arises
from the fact that several proxy computations necessitate at least one forward
pass or a forward-backward pass, requiring multiple runs for result stabilization.
Additionally, many are data-dependent, limiting their general applicability and
extending search times.
In contrast, TG-NAS completed the search in about 40 seconds, achieving up
to a 300× speedup compared to other zero-cost methods due to the lightweight
nature of the proposed predictor model. Notably, the comparison of search times
does not factor in the predictor model training time (the model training required
1.5 GPU hours), as it is a one-time training effort, and the pretrained model can
be distributed for uses in various scenarios and search spaces.

For the DARTS search space, our GCN prediction model undergoes a slightly
different training process, incorporating additional data from NAS-Bench-201
into the training set. Consequently, our training dataset comprises a total of
423,000 NAS-Bench-101 architectures and an additional 15,625 architectures from
NAS-Bench-201. The labels for the NAS-Bench-101 portion are derived from the
CIFAR-10 evaluation accuracy. However, the NAS-Bench-201 labels differ slightly
due to additional CIFAR-100 and ImageNet-16-120 results. To mitigate bias
arising from the monolithic dataset, we amalgamate the normalized accuracies of
CIFAR-10, CIFAR-100, and ImageNet-16-120 to construct a unified ground truth
label for NAS-Bench-201 architectures. Indeed, while introducing an additional
proxy is necessary due to the absence of actual results on the ImageNet dataset in
those NAS benchmarks, it’s worth noting that CIFAR-10/100 performance often
serves as a reliable indicator for ImageNet performance. The final discovered cell
architecture can be seen in Fig. 5. After the target cell was determined, the final
network was constructing by stacking 14 cells, with the initial channel number
set to 48. Performance results are shown in Table 5. Our approach achieves a
top-1/5 accuracy of 74.5% and 91.9% respectively, comparable to other works
on ImageNet. Notably, TG-NAS only took less than 2 minutes of search time
on one NVIDIA RTX 4090 GPU. We expect that performance can be further
enhanced if incorporating additional training samples from supplementary NAS
benchmarks.

Conclusion

In this work, We present TG-NAS, a model-based predictor framework that is
generally applicable for new search spaces with unseen new operators. It integrates
a transformer-based operator embedding generator with a graph convolution
network and functions as a ZC proxy. TG-NAS offers advantages in terms of data
independence, cost-effectiveness, and generality. Our experiments demonstrate its
superiority over existing proxies across various NAS benchmarks, positioning it
as a foundational element for efficient architecture search. TG-NAS achieves up
to a 300× improvement in search efficiency compared to previous state-of-the-art
zero-cost proxy methods. Notably, it discovers competitive models with 93.75%
CIFAR-10 accuracy on the NAS-Bench-201 space and 74.5% ImageNet top-1
accuracy on the DARTS space.

=== END OF FILE ===

'''

generation_config = GenerationConfig(
    temperature=0.0,
    top_k=20,
    repetition_penalty=1.2,
    pad_token_id=tokenizer.eos_token_id,
)
# model.generation_config.pad_token_id = tokenizer.pad_token_id
print("Prompt token length:", len(tokenizer(prompt, return_tensors="pt")["input_ids"][0]))

def print_predict(prompt):
    prompt = "<|prompter|>" + prompt + "</s><|assistant|>"
    encoding = tokenizer(prompt, return_tensors="pt")
    input_ids = encoding["input_ids"].to("cuda")

    with torch.inference_mode():
        result = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=False,
            output_scores=False,
            max_new_tokens=512,
        )

    decoded_output = tokenizer.decode(result[0][len(input_ids[0]):])
    print(decoded_output)

prompt_question = prompt + "\nPlease give me a brief summary of this research paper in a few bullet points."
print_predict(prompt_question)

prompt_question = prompt + "\nPlease write me the abstract for this paper."
print_predict(prompt_question)

prompt_question = prompt + "\nWhat make TG-NAS stand out from other model-based rediction works? Give a short answer."
print_predict(prompt_question)

prompt_question = prompt + "\nWhat is the rage of search efficient gain that TG-NAS achieved compared to other NAS works? Give a short answer."
print_predict(prompt_question)

