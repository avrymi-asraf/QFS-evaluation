5
2
0
2

r
p
A
4

]

G
L
.
s
c
[

3
v
1
8
0
0
0
.
2
1
4
2
:
v
i
X
r
a

Task Singular Vectors:
Reducing Task Interference in Model Merging

Antonio Andrea Gargiulo⋆
Simone Scardapane

Donato Crisostomi⋆
Fabrizio Silvestri

Maria Sofia Bucarelli
Emanuele Rodol`a

Sapienza University of Rome

gargiulo.1769185@studenti.uniroma1.it

{crisostomi, rodola}@di.uniroma1.it

simone.scardapane@uniroma1.it

{bucarelli, fsilvestri}@diag.uniroma1.it

Abstract

Task Arithmetic has emerged as a simple yet effective
method to merge models without additional training. How-
ever, by treating entire networks as flat parameter vectors,
it overlooks key structural information and is susceptible to
task interference.
In this paper, we study task vectors at
the layer level, focusing on task layer matrices and their
singular value decomposition.
In particular, we concen-
trate on the resulting singular vectors, which we refer to
as Task Singular Vectors (TSV). Recognizing that layer task
matrices are often low-rank, we propose TSV-Compress
(TSV-C), a simple procedure that compresses them to 10%
of their original size while retaining 99% of accuracy. We
further leverage this low-rank space to define a new mea-
sure of task interference based on the interaction of singu-
lar vectors from different tasks. Building on these findings,
we introduce TSV-Merge (TSV-M), a novel model merg-
ing approach that combines compression with interference
reduction, significantly outperforming existing methods.

1. Introduction

The widespread availability of pre-trained models and pub-
lic repositories has driven the development of techniques for
efficiently combining and reusing existing models. Among
these techniques, model merging approaches enable the cre-
ation of multi-task models without additional training. One
popular approach, Task Arithmetic (TA) [22], stands
out for its simplicity and effectiveness. However, by treat-
ing entire networks as high-dimensional vectors, TA and
subsequent works overlook crucial structural information
[9, 10, 24, 32, 38, 51, 53]. This flattened view limits these
approaches to coarse-grained measures like cosine similar-
ity for assessing inter-task interactions.

⋆ denotes equal contribution.

Figure 1. Mean accuracy of a ViT-L-14 merged over 8, 14, and
20 tasks respectively. By significantly surpassing existing meth-
ods, TSV-M establishes the new state of the art in model merging.

In this work, we instead focus on preserving and lever-
aging the natural structure of neural networks by examining
weight differences at the layer level while retaining their
matrix form wherever possible. Our approach begins by
examining per-layer task matrices, which represent weight
changes for each task, through singular value decomposi-
tion (SVD). This decomposition yields singular vectors and
singular values that capture the most significant directions
of variation within each layer. We term these singular vec-
tors Task Singular Vectors (TSV), as they provide an in-
terpretable basis for assessing task-specific contributions at
each layer. Importantly, in accordance with recent litera-
ture on PEFT (cfr. Sec. 2), our analysis confirms that task
matrices are inherently low-rank, meaning that only a small
portion of TSVs is sufficient to represent the layer’s func-
tion with high fidelity.

Building on this insight, we introduce TSV-Compress
(TSV-C), a simple yet effective procedure to compress
task vectors down to 10% of their original size while
maintaining 99% of the original accuracy. Focusing on
only a fraction of the most relevant TSVs for each task,
TSV-Compress enables efficient storage and processing

8Tasks14Tasks20Tasks60.065.070.075.080.085.090.095.0100.0Accuracy(%)64.7068.2065.2379.5676.7371.6084.9379.4174.0186.3482.2279.0092.9889.1787.72ZeroshotWeightAveragingTaskArithmeticConsensusTATSV-M(Ours)

of task-specific information without waiving performance.
Beyond compression, examining the interplay of TSVs
across different tasks provides a geometrically informed
framework for analyzing task interference at the individual
layer level. By assessing how singular vectors from differ-
ent tasks align or diverge within each layer, this approach
offers a significantly more fine-grained understanding of
inter-task interactions, going beyond global vector similar-
ity metrics like cosine similarity. Building upon these prin-
ciples, we introduce TSV-Merge (TSV-M), a novel model
merging method that combines compression and task inter-
ference reduction. This is achieved by discarding the irrel-
evant singular vectors for each task and then reducing their
inter-task interference with a whitening transformation over
their similarity matrix. We empirically show our approach
to effectively reduce task interference, and the reduction to
be complementary to the compression step. We then eval-
uate our approach across several benchmarks and show it
15% ac-
outperforms existing methods by an average of
curacy points, establishing the new state of the art by a large
margin. We release all our code for research purposes1.

∼

In summary, our contribution is 4-fold:

• We study the singular value decomposition of per-layer
task matrices, showing them to be low-rank and measur-
ing the interplay of singular vectors from different tasks;
• We introduce TSV-C, a method that builds on this insight
while pre-

to compress task vectors by a factor of 10
serving 99% of their original accuracy;

×

• We propose TSV-M, a model merging technique that
complements compression with a task interference reduc-
tion step by applying a whitening transformation to decor-
relate singular vectors across tasks;

• We conduct extensive experiments on multiple computer
vision datasets, showing that TSV-M significantly out-
performs existing model merging methods and provide
in-depth analyses to uncover the underlying factors con-
tributing to its success.

2. Related Work

Model Merging offers an efficient alternative to ensem-
bling by combining existing models without the need for
further training. Several approaches address this by de-
termining the neuron permutations that align the models
into a shared optimization basin, after which the models
are combined using a straightforward averaging method.
[1, 8, 25, 36, 43]. An alternative line of approaches focuses
on the multi-task scenario, where a single pre-trained model
is fine-tuned for different tasks [10, 22, 32, 40, 46, 48, 51,
53, 54]. The prerequisites for methods in this category are
outlined in Table 1. Task Arithmetic [22] introduces
the concept of task vectors, which are the weight differences

1https://github.com/AntoAndGar/task singular vectors

Table 1. Model merging approaches and their requirements.

Method

Additional
training

Extra
storage

Validat. data
inputs

labels

Weight Avg. [48]

Fisher-Merg. [32]
RegMean [24]

EMR-Merging [20]
TALL-Mask [46]
TSV-C (Ours)

Task Arith. [22]
Ties-Merging [51]
AdaMerging [52]
Consensus-TA [46]
TSV-M (Ours)

×

×
×

×
×
×

×
×
✓
×
×

×

×
×

✓
✓
✓

×
×
×
×
×

×

✓
✓

×
×
×

✓
✓
✓
✓
×

×

×
×

×
×
×

✓
✓
×
✓
×

between fine-tuned models and the pre-trained base model.
By averaging these vectors, a merged multi-task model can
be created; conversely, negating a task vector allows for for-
getting a specific task. TIES [51] addresses redundancies
in model parameters by first selecting the top-k most signif-
icant parameter changes and then constructing a sign vector
based on the majority sign across all models. The latter is
used to merge the task vectors disjointly, meaning the av-
erage is not computed when a parameter is zero or when
parameters disagree in sign. Similarly, DARE [53] ran-
domly resets redundant parameters to their pre-trained val-
ues and rescales the remaining parameters by a factor pro-
portional to the dropped ones, aiming to reduce interference
among tasks. Fisher Merging [32] and RegMean [24]
merge models by performing weighted averaging, utiliz-
ing the Fisher information matrix and the inner product of
input vectors, respectively. Model Breadcrumbs [10]
focuses on merging only significant weights by discarding
outliers and both minor and large perturbations in the fine-
tuned parameters. More recently, Wang et al. [46] observed
that tasks often utilize non-overlapping sets of weights.
the first, TALL-Mask, uses
They propose two methods:
binary masks to activate important task-specific weights, re-
quiring extra storage (as does our method TSV-C) for the
masks. The second, Consensus Merging, leverages
these masks to remove parameters that are important to less
than two tasks. Like our approach, TwinMerging [31]
and SMILE [44] apply SVD to layer task matrices. How-
ever, neglecting singular vector interference, they require a
router to selectively activate a subset during inference.

Unlike existing approaches, our methods explicitly ad-
dress task interference by leveraging the geometric structure
of singular vectors to minimize interactions between task-
specific parameters. Rather than averaging or selectively
merging parameters as prior techniques do, we decorrelate
singular vectors to directly reduce interference across tasks.

Additionally, we achieve targeted compression by isolating
each task’s unique components, storing only essential parts,
and ensuring minimal interference, a step beyond traditional
SVD-based compression approaches.

SVD for Model Compression A significant body of work
explores low-rank decompositions for fully-connected and
convolutional layers [11, 12, 16, 23, 27], as well as tensor
decompositions [12, 16, 26]. These approaches typically
achieve a layer-by-layer factorization of a trained network,
focusing on minimizing the difference between the original
and the low-rank approximated weight matrices. This is ac-
complished using techniques like SVD [12, 16, 23, 27] or
iterative optimization [23, 26]. While other methods such
as weight quantization, knowledge distillation, and pruning
exist, we utilize SVD not only for low-rank approximation
and compression but also because its matrix decomposition
properties make it particularly effective in analyzing and
mitigating interference.

3. Task Singular Vectors

In this section, we introduce key concepts for understanding
our approach, including a new measure of task interference
that is central to our method.

3.1. Background

We build on Task Arithmetic (TA), which defines
task vectors capturing the differences in model weights for
individual tasks. Formally, the weights θMT of a multi-task
model for T tasks are computed by aggregating the task-
specific weight differences, or task vectors, as follows:

θMT = θpre + α

(cid:80)T

i=1 τi
T

,

(1)

where θpre is the set of pretrained model weights, α is a
θpre is the task vector for
scaling factor, and τi = θfti −
task i, with θfti being the fine-tuned weights for the task.
Differently from TA, however, we consider these operations
at the layer level. From this perspective, Eq. (1) becomes

θ(l)
MT = θ(l)

pre + α

(cid:80)T

i=1 ∆(l)
T

i

,

(2)

fti −

where θ(l)
pre encodes the pretrained weights for layer l, and
∆(l)
θ(l)
i = θ(l)
pre is the task-specific weight difference for
task i at layer l. When layer l has a matrix structure, we call
l the per-layer task matrix for task i.
its corresponding ∆i
When this is not the case, our framework defaults to stan-
dard TA. For brevity, we will generally omit the layer index
and refer to the layer-l task matrix ∆l

i as ∆i.

Decomposing layer task matrices

Treating layer-wise weights as structured entities rather
than flattened vectors enables us to analyze their SVD,
revealing low-rank properties and deeper insight into the
inter-task interactions. Given two tasks i, j, we consider the
SVD of their task matrices ∆i and ∆j at a generic layer:

∆i = UiΣiV ⊤
i

,

∆j = UjΣjV ⊤
j

where Ui, Uj and Vi, Vj are the matrices of left and right
singular vectors respectively, and Σi, Σj are diagonal ma-
trices of singular values. Due to their role in assessing task-
specific contributions, we term the obtained singular vectors
Task Singular Vectors (TSV).

Importantly for our treatment, we can equivalently write
the aggregated task matrices in Eq. (2) in terms of their sin-
gular vectors and values. By defining U = [U1
UT ] as
the column-wise concatenation of all the left TSVs, V =
VT ] as the row-wise concatenation of the right TSVs,
[V1
T
and Σ as the block diagonal matrix with
i=1 along its
}
diagonal, we can write:

· · ·

· · ·

Σi

{

M = U ΣV ⊤ =

T
(cid:88)

i=1

UiΣiV ⊤

i =

T
(cid:88)

i=1

∆i .

(3)

In the simple case where T = 2, M would be given by:

M = (cid:2) U1 U2

(cid:3)

(cid:20) Σ1

0
0 Σ2

(cid:21) (cid:20) V ⊤
1
V ⊤
2

(cid:21)

= ∆1 + ∆2 .

When concatenating singular components from different
tasks, we violate certain properties of the SVD. Specifically,
the matrices U and V become non-orthogonal because sin-
gular vectors from different tasks may overlap. Addition-
ally, the singular values Σi from different tasks can vary
significantly in magnitude, which may bias the merging pro-
cess toward tasks with larger singular values.

3.2. Low-rank nature of layer task matrices

We start our analysis by studying the low-rank properties of
per-layer task matrices. Interpreting SVD as a sum of rank-
one matrices, for each task i, by Eckart-Young’s theorem
[14] we get the best approximation (in Frobenius norm) of
each task matrix ∆i by retaining only the top-k singular
values and their corresponding vectors:

ˆ∆i =

k
(cid:88)

j=1

jui
σi

jvi⊤
j

.

(4)

As we show in Fig. 2, we find task matrices to be inher-
ently low-rank: a small subset of TSVs is sufficient to rep-
resent the layer’s function with high fidelity. In particular,
even when preserving only 3% of the singular components
per task, the mean accuracy drops by merely 1.5%. This is

4. Approach

We demonstrate below how TSVs can be used for both com-
pression and task interference reduction.

4.1. TSV for compression

When the task is known or inferred (e.g. with a router, as
in Mixture-of-Experts [15, 39] techniques), we can lever-
age the low-rank structure of per-layer task matrices (see
Sec. 3.2) to effectively compress these ones while discard-
ing task singular vectors from different tasks. In particular,
given a known or inferred task index h, we set other task
components to zero, reducing Eq. (5) to:

ˆM =

T
(cid:88)

i=1

1
[i=h]

k
(cid:88)

j=1

jui
σi

jvi⊤

j =

k
(cid:88)

j=1

j uh
σh

j vh⊤

j = ˆ∆i .

(7)

This formula yields a low-rank approximation of ∆h, where
only the top k singular components of the task-specific ma-
trix are retained. Here, the same number of components k is
taken from each task to ensure that each layer is compressed
by a factor of 1
T relative to the original dimension. Increas-
ing k improves approximation but reduces the compression.

4.2. TSV for model merging

In scenarios where the task identity is not known before-
hand, we address the standard model merging problem by
combining Task Singular Vectors (TSVs) to create a single
model that performs well jointly across all tasks.

To reduce our measure of task interference, we decor-
relate the TSVs of different tasks (encoded as columns in
ˆU and ˆV ) by whitening these matrices to minimize their
correlations. This can be achieved by applying the transfor-
2 to both ˆU and ˆV . For improved
mation X
numerical stability, we reformulate this whitening as an or-
thogonal Procrustes problem, seeking the orthogonal matrix
ˆU⊥ that minimizes the projection error:

X(X ⊤X)− 1

(cid:55)→

min
ˆU⊥ ∥

ˆU⊥

ˆU

F
∥

−

s.t. ˆU ⊤
⊥

ˆU⊥ = I ,

(8)

and similarly for ˆV . This problem admits a closed-form
solution via the SVD ˆU = P DQ⊤, yielding ˆU⊥ = P Q⊤
(similarly for ˆV⊥).

Proposition 4.1. The transformations X
(whitening) and X
P DQ⊤ is the SVD of X, are equivalent.

X(X ⊤X)− 1
(cid:55)→
P Q⊤ (Procrustes), where X =

(cid:55)→

2

Proof. Given X = P DQ⊤ and recalling that D⊤D =
D2 for diagonal matrices, simple algebraic manipulation
yields X ⊤X = QD2Q⊤. It follows that (X ⊤X)−1/2 =
QD−1Q⊤, therefore the whitening transformation can be
rewritten as X(X ⊤X)−1/2 = (P DQ⊤)(QD−1Q⊤) =
P DID−1Q⊤ = P Q⊤, completing the proof.

Figure 2. Mean absolute accuracy of the ViT-B-32 model across
increasing fractions of retained singular components, averaged
over 20 tasks. The red line represents the average accuracy of the
original fine-tuned models with full-rank task matrices, while the
green line shows the accuracies using low-rank approximations.

noticeable considering that 97% of singular components in
each layer matrix are discarded. Building on this insight,
we propose in Sec. 4.1 a compression algorithm that main-
tains 99% of the accuracy while shrinking the task vectors
to 10% of their original size.

Given this low-rank structure, it is natural to aggregate
task matrices within their subspaces. We, therefore, obtain
a reduced version of the aggregation matrix M (Eq. (3)) as:

ˆM =

T
(cid:88)

i=1

ˆUi ˆΣi ˆV ⊤

i =

T
(cid:88)

k
(cid:88)

i=1

j=1

jui
σi

jvi⊤
j

,

(5)

where ˆUi and ˆVi contain the top-k left and right singular
vectors for task i, respectively, and ˆΣi is the diagonal ma-
trix of the top-k singular values σi
j. This formulation high-
lights that the low-rank approximation ˆM utilizes the most
relevant singular components from each task to effectively
combine the task-specific weight differences.

3.3. Singular Task interference

We hereby introduce a score of task interference based on
the interplay of TSVs from different tasks, which we term
Singular Task Interference (STI)

(cid:16)

STI

(cid:17)

T
i=1

∆i
{

}

=

(U ⊤U
∥

−

I)Σ(V ⊤V

I)

1 ,
∥

−

(6)

where U , Σ and V are obtained by concatenating the sin-
T
gular value decompositions (
i=1) of
Ui
{
}
}
T
∆i
per-layer task matrices
i=1 as detailed in Sec. 3.1. In
}
{
the expression above, high inner product values for U ⊤U
and V ⊤V imply a higher likelihood of interference, with
minimal interference ideally yielding identity matrices.

T
i=1,
}

T
i=1 {

Σi
{

Vi

The underlying intuition is that overlapping singular vec-
tors suggest shared features in the weight space across tasks.
Such overlap can introduce interference when models are
merged, ultimately degrading performance on individual
tasks. We refer to Fig. 3 for a real example over eight tasks.

0.00.10.20.30.40.5Fractionofnon-zerosingularvalues(rank)86889092AverageAccuracy(%)91.37FullrankLowrankFullrankLowrankFigure 3. Visualization of task interference among 8 tasks computed on the first attention layer of a ViT-B-32. The diagonal blocks dis-
play intra-task similarities, while the off-diagonal blocks illustrate inter-task similarities. The zoomed-in section highlights the interaction
between the right singular vectors of the 3rd and 4th tasks.

Figure 4. Absolute accuracy of a ViT-B-32 merged over 8, 14, and 20 tasks, respectively.

Once the (rank-reduced) TSV matrices ˆU and ˆV are
decorrelated, we reconstruct the merged layer as in Eq. (5);
see Algorithm 1 for the complete steps.

cludes the preceding 14 plus the following six: EMNIST
[7], CIFAR10 [29], Food101 [2], FashionMNIST [49],
RenderedSST2 [41], and KMNIST [5].

5. Results

5.1. Model merging results

We evaluate our approaches over three different suites of
tasks having cardinality 8, 14, and 20, respectively. The
first one, introduced in [22], consists of datasets: Cars
[28], DTD [4], EuroSAT [19], GTSRB [42], MNIST [30],
RESISC45 [3], SUN397 [50], and SVHN [33]. The bench-
mark with 14 tasks builds on the preceding one, incorpo-
rating six additional datasets: CIFAR100 [29], STL10 [6],
Flowers102 [34], OxfordIIITPet [35], PCAM [45],
and FER2013 [18]. Finally, the 20-tasks benchmark in-

We evaluate our method on three variants of the CLIP [37]
model, each employing a different size of ViT [13] vi-
sual encoder: ViT-B-32, ViT-B-16, and ViT-L-14.
The main benchmarks involve merging 8, 14, and 20 tasks,
mirroring the experimental setup described in Wang et al.
[46]. We compare our approach against several training-
free model merging methods, including weight averaging,
Task Arithmetic [22], and Consensus Merging
[46]. For reference, we include the performance of zero-

01002003004005006007000100200300400500600700UTU01002003004005006007000100200300400500600700VTV-0.50-0.40-0.30-0.20-0.100.000.100.200.300.400.50SimilarityCarsDTDEuroSATGTSRBMNISTRESISC45SVHNSUN39720%40%60%80%8TasksCarsDTDEuroSATGTSRBMNISTRESISC45SVHNSUN397STL10OxfordIIITPetFlowers102CIFAR100PCAMFER201320%40%60%80%14TasksTSV-M(Ours)TSV-C(Ours)ConsensusTATaskArithmeticZero-shotCarsDTDEuroSATGTSRBMNISTRESISC45SVHNSUN397STL10OxfordIIITPetFlowers102CIFAR100PCAMFER2013CIFAR10Food101FashionMNISTRenderedSST2EMNISTKMNIST20%40%60%80%20TasksMethod

ViT-B-32

ViT-B-16

ViT-L-14

8 tasks

14 tasks

20 tasks

8 tasks

14 tasks

20 tasks

8 tasks

14 tasks

20 tasks

Zeroshot
Weight Averaging
Task Arithmetic
Consensus TA
TSV-M (Ours)

48.26(53.59)
66.34(72.13)
70.79(76.55)
75.03(80.84)
85.86(92.31)

57.21(63.69)
64.34(71.12)
65.32(72.09)
70.39(77.36)
80.06(87.88)

56.10(62.41)
61.04(67.53)
60.52(66.79)
65.43(71.98)
77.07(84.29)

55.34(59.34)
72.22(76.60)
75.41(79.58)
79.39(83.86)
89.01(93.94)

61.28(66.19)
69.46(74.82)
70.52(75.89)
74.39(79.92)
84.58(91.01)

59.73(64.52)
65.31(70.36)
65.78(70.76)
69.76(74.93)
80.57(86.45)

64.70(68.00)
79.56(83.15)
84.93(88.65)
86.34(90.08)
92.98(96.98)

68.20(72.15)
76.73(81.10)
79.41(83.95)
82.22(86.94)
89.17(94.43)

65.23(68.99)
71.60(75.60)
74.01(78.07)
79.00(83.22)
87.72(92.50)

Table 2. Average absolute accuracy results on model merging benchmarks; subscript (in parentheses) is the normalized average accuracy.

Algorithm 1 TSV-Merge.
Require: Task matrices ∆1, . . . , ∆T , scaling factor α
Ensure: Merged model weights θMT

We report in Fig. 4 the per-dataset accuracies.

5.2. Compression results

T singular components of Ui, Σi, and Vi

|

|

. . .

U2

U
Σ
V

←
←
←

Compute SVD: ∆i = UiΣiV ⊤
i
Retain first 1

1: for i = 1 to T do
2:
3:
4: end for
5: Concatenate the matrices:
UT ]
6:

[U1
|
block-diag(Σ1, Σ2, . . . , ΣT )
[V1

|
U = PU DU Q⊤
U

7:
V2
8:
9: Compute the SVD of U and V :
10:
11: Obtain the orthogonal matrices:
U⊥ = PU Q⊤
12:
U
13: Reconstruct the merged matrix:
U⊥ΣV ⊤
14:
⊥
15: Construct merged model weights:
θpre + α ˆM
θMT
16:
17: return θMT

V⊥ = PV Q⊤
V

VT ]

ˆM

. . .

←

←

|

|

V = PV DV Q⊤
V

shot models in Table 2 to represent the minimum achievable
accuracy, and the average of individually fine-tuned models
in Table 3 to indicate the maximum potential gains. Perfor-
mance metrics were assessed using both the average abso-
lute accuracy and the average normalized accuracy, calcu-
lated as detailed in Appendix B.1.

As presented in Table 2, our method achieves state-
of-the-art results across all benchmarks, regardless of the
ViT size or the number of tasks involved. Notably, the
improvements were observed with the
most significant
smaller ViT-B-32 encoder, where our approach outper-
forms Task Arithmetic and Consensus TA by an
average of +15.45% and +10.71% absolute accuracy, re-
spectively. Furthermore, when utilizing the ViT-L-14
model, our method attains an average normalized accuracy
of 96.98%. This indicates that we can effectively replace
eight individual task-specific models with a single merged
3.02% reduction in average ac-
model, incurring only a
curacy. These results highlight the promise of our model
merging technique as a cost-efficient alternative to mixtures
of experts, multi-task learning, and ensembling methods.

−

In Table 3 we report the results for our compression algo-
rithm TSV-C. We compare our method with TALL-masks
[46], which stores very sparse binary masks for each task.
As shown in Table 3, our method always retains more than
99% of the original accuracy for all considered benchmarks
and models. The results of TSV-C are comparable to those
of TALL-masks [46], with the former slightly underper-
forming TALL-masks for small ViTs and the latter gain-
ing the upper hand in the larger-scale one. In general, how-
ever, the two approaches differ by less than 1% accuracy on
average. Regarding the storage, our method stores only the
top 1
T singular vectors per task, which leads to a fixed stor-
age requirement of approximately twice the size of the orig-
inal model, regardless of the number of tasks. By storing bi-
nary masks, TALL-masks instead results in a storage size
that varies with their compressibility, increasing with the
number of tasks. This variability can lead to unpredictable
storage requirements and may diminish compression bene-
fits when masks are less compressible.

Our results show that TSV-C effectively balances com-
pression and performance by leveraging the most significant
TSVs. Maintaining near-original accuracy, our approach is
particularly suitable for scenarios where storage constraints
are critical but high model performance is still required.

6. Analysis

In this section, we begin by conducting an ablation study to
assess the contributions of interference reduction and low-
rank compression to the overall performance. Next, we
study how task interference varies across layers of differ-
ent depths. Finally, we empirically show that our method
does not require tuning a scaling coefficient.

6.1. Ablation study

TSV-M combines low-rank approximation with task inter-
ference reduction. To evaluate the contribution of each
component, we conducted an ablation study summarized
in Table 4. Applying low-rank approximation alone to
the layer task matrices results in worse performance than
vanilla Task Arithmetic, shown in the first row where

Method

ViT-B-32

ViT-B-16

ViT-L-14

8 tasks

14 tasks

20 tasks

8 tasks

14 tasks

20 tasks

8 tasks

14 tasks

20 tasks

Finetuned
TALL-Mask+TIES
TSV-C (Ours)

92.83(100)
93.13(100.37)
92.62(99.74)

90.88(100)
90.92(100.04)
90.29(99.28)

91.37(100)
91.11(99.70)
90.64(99.14)

94.64(100)
94.68(100.04)
94.47(99.79)

92.76(100)
92.69(99.90)
92.25(99.41)

93.17(100)
93.05(99.87)
92.53(99.27)

95.81(100)
95.96(100.15)
95.68(99.85)

94.29(100)
93.40(99.09)
94.04(99.72)

94.73(100)
93.91(99.16)
94.42(99.66)

Table 3. Average absolute accuracy results across all compression benchmarks for different models and varying number of tasks, subscript
(in parentheses) the normalized average accuracies. Our TSV-C compression method consistently achieves over 99% of the original fine-
tuned models’ accuracy while significantly reducing storage requirements.

Low-rank
approx.

Interf.
reduction

×
✓
×
✓

×
×
✓
✓

8 tasks

76.5 (+0.0)
75.2 (-1.3)
82.6 (+7.4)
92.3 (+9.7)

ViT-B-32
14 tasks

72.1 (+0.0)
71.0 (-1.1)
75.7 (+4.7)
87.9 (+12.2)

20 tasks

66.8 (+0.0)
66.3 (-0.5)
69.9 (+3.6)
84.3 (+14.4)

Comparison of different versions of Task
Table 4.
Arithmetic, comprising either
the low-rank approxima-
tion step, the interference reduction step, or both. The method
performing both corresponds to the proposed TSV-Merge.

Figure 5. Approximation error from the orthogonalization of the
TSVs through Procrustes for the ViT-B-32 model across 8 tasks.
The violin plots represent layer-wise approximation error distribu-
tions for U and V in both full-rank and low-rank cases.

neither component is applied. In contrast, applying task in-
terference reduction alone, while keeping full-rank matri-
ces, significantly improves performance, with gains rang-
ing from +3.1% to +6.1%. However, the best results are
achieved only when both steps are combined, yielding sub-
stantial performance improvements of +15.8% to +17.5%.
To explain why interference reduction applied to low-
rank approximations in TSV-M outperforms its application
to full-rank matrices, we analyze the errors introduced dur-
ing the orthogonalization of the full-rank U and V matrices.
As shown in Fig. 5, orthogonalizing full-rank matrices in-
curs significant approximation errors, measured by the sum
of the Frobenius norms of reconstruction errors for the con-
catenated U and V matrices across all layers.

For example, in the ViT-B-32 model with 8 tasks, the
full-rank setting exhibits a wider error distribution with a

≤
U
∥

higher average, indicating greater variability and larger ap-
proximation discrepancies. In contrast, the low-rank setting
produces a more compact and lower error distribution, sug-
gesting better consistency in approximation across layers.
In the following, we prove that this is not specific to the
chosen model but a general property of our approach. The
proof can be found in Appendix C.3.

∈

N such that T > 4. Define
Theorem 6.1. Let T
U = [U1, . . . , UT ] as the matrix obtained by concatenat-
ing T orthogonal matrices Ui, each of shape n
n. Let
(cid:98)U = [ (cid:98)U1, . . . , (cid:98)UT ] be the matrix formed by truncating each
Ui to its first k columns. Denote by X and (cid:98)X the matrices
resulting from Procrustes orthonormalization of U and (cid:98)U ,
respectively. If k

, then

×

√

T

n T −2
T

X

F
∥

(cid:98)U

(cid:98)X

F .
∥

−

−

≥ ∥
Intuitively, the theorem asserts that the approximation
error introduced by orthogonalization via Procrustes is
smaller when applied to the concatenation of truncated ma-
trices, compared to the concatenation of the original full-
rank matrices, provided the rank of the truncated matrices
satisfies certain conditions. To better understand the state-
ment of the theorem, consider the case T = 10, as in our
scenario. The term T −2
1
3 . In our case,
T
n
we choose k
10 , which is
1
indeed smaller than n
3 n.

≥
n
T . For T = 10, this gives k

3 , satisfying the condition k

= 10−2
10

We emphasize that other orthogonalization methods,
such as Gram-Schmidt, are ineffective in this context be-
cause they do not preserve the overall structure of the orig-
inal data. Gram-Schmidt sequentially orthogonalizes vec-
tors without minimizing deviation from the original set, po-
tentially resulting in significant distortions.

≤

≈

≈

10

√

√

T

Finally, the substantial reduction in approximation error
suggests that, while the advantages of interference reduc-
tion are considerable, they may be offset by the errors in-
In contrast,
troduced when applied to full-rank matrices.
starting with low-rank approximations captures the essen-
tial information of each task, making the orthogonalization
step less costly in terms of approximation error.

6.2. Per-layer task interference

We analyze task interference on a per-layer basis using
Eq. (6). As shown in Fig. 6, interference is highest in the

FullrankLowrank020406080ErrorUApproximationerrorforUFullrankLowrank0102030405060ErrorVApproximationerrorforVFigure 6. Singular Task Interference (STI) across layers in a
ViT-B-32 for 20 tasks. STI is high in early layers sharing com-
mon knowledge and lower in more specialized ones.

Figure 8. Singular Task Interference (STI) and average normal-
ized accuracy for Task Arithmetic and TSV-Merge on the
ViT-B-32 model, evaluated across merges of 8, 14, and 20 tasks.

assessed across three distinct sets of tasks with different car-
dinalities.
In all instances, we observed that a reduction
in interference is closely associated with a significant gain
in accuracy. This provides empirical evidence for the ef-
fectiveness of our approach in minimizing interference and,
ultimately, enhancing the merging process.

7. Conclusions

In this paper, we tackled the problem of model merging by
analyzing the SVD of per-layer task matrices, confirming
their inherent low-rank structure and leveraging their singu-
lar vectors to define task interference.

former

insight, we

Building on the

introduced
TSV-Compress (TSV-C), a model compression al-
gorithm that reduces task vectors to 10% of their original
size while retaining 99% of the original accuracy. Unlike
existing methods, our approach maintains a constant stor-
age requirement, independent of the number of tasks. We
then developed TSV-Merge (TSV-M), combining low-
rank approximation with interference reduction to create
a novel model merging technique. Achieving normalized
accuracy up to 97%, our method offers a storage-efficient
alternative to MoEs, ensembles, and multi-task learning.

Our extensive evaluation revealed that while interference
reduction contributes most to performance improvements,
the optimal results are achieved in combination with com-
pression. We also observed that task interference decreases
in deeper layers, and importantly, that our method does not
require tuning a scaling coefficient.

Exploring alternative methods for task importance and
rank approximation could be a valuable direction for fu-
ture work. Currently, we use a uniform rank across tasks,
approximating each task vector to 1
T of the layer’s dimen-
sionality. However, techniques like those from Idelbayev
and Carreira-Perpin´an [21] or optimal hard thresholds from
Gavish and Donoho [17] could be applied to select the op-
timal rank for each task individually.

Figure 7. Best average normalized accuracy for different alpha
values. The vertical labels indicate different sets of 8 tasks.

initial transformer layers and decreases significantly in the
deeper layers. This aligns with the understanding that early,
generalized layers capture common features across tasks,
increasing the potential for conflict, while deeper layers are
more specialized for specific tasks, reducing shared repre-
sentation and interference. For brevity, we grouped layers
within each transformer block in our analysis; each “Layer
n” in Fig. 6 includes two attention matrices and two MLP
matrices. A layer-by-layer analysis is provided in Fig. 14.

6.3. Choice of the interpolation coefficient

The aggregation in Eq. (2) involves a scaling coefficient α,
typically tuned using a validation set, that can have a signifi-
cant impact on the accuracy of the final model. However, as
shown in Fig. 7, our approach consistently achieves the best
results with α = 1.0, indicating that no additional scaling
is necessary. Although purely empirical, this finding spares
further tuning and validation data, thereby enhancing the
practicality and efficiency of our method.

6.4. Effect of task interference reduction

We report in Fig. 8 the interference of a ViT-B-32 model
both before and after the application of our TSV-M pipeline,

visual.projLayer0Layer1Layer2Layer3Layer4Layer5Layer6Layer7Layer8Layer9Layer10Layer11050001000015000Interference2-LayerMov.Avg.Interference2-LayerMov.Avg.Interference0.60.70.80.91.01.11.21.31.41.5AlphaValueB-160B-161B-162B-163B-164B-165B-166B-320B-321B-322B-323B-324B-325B-326L-140L-143L-144Model+TasksIds89.8091.3692.7393.6193.9893.9894.0693.3092.9091.7885.4687.8289.4990.8891.5492.1292.1291.7690.9689.8993.5994.7395.4495.8596.0095.7495.6395.2394.5093.8191.9193.7595.0896.1896.4496.2695.8594.8293.8192.4287.3190.1492.0493.6894.6695.4695.6295.2294.5693.1992.7193.5294.4394.4394.4394.2993.7493.0092.2691.2191.7993.1993.8694.3494.6394.7694.6294.0193.2392.0488.2389.8490.9591.8592.0392.2291.7191.1690.4989.2286.6688.9490.5991.5891.6891.1590.5289.8188.6787.1991.3292.8893.7493.9394.2194.0293.6092.7691.7890.6690.1392.4093.7694.5095.0294.3993.7692.6490.9188.9686.8590.1292.3993.7394.5794.5393.8392.8791.5290.6190.3291.2491.7792.1091.9791.9991.7891.3590.7889.3191.4792.3492.8393.0493.0293.0192.5091.4390.8189.6092.7793.9795.0995.7996.1496.0396.1095.7695.7695.3895.6897.0797.9398.3698.6598.8798.6598.5198.1397.8495.6697.3098.6799.1999.5299.4699.4799.2098.8698.298084889296100AverageNormalizedAccuracy(%)0.700.750.800.850.90AverageNormalizedAccuracy020000400006000080000InterferenceTaskArithmeticTSV-MergeNumberofTasks8tasks14tasks20tasksAcknowledgments

This work was partially supported by PRIN 2022 project
20227YET9B “AdVVent” CUP code B53D23012830006
and by projects FAIR (PE0000013)
and SERICS
the MUR National Recovery
(PE00000014) under
and Resilience Plan funded by the European Union -
NextGenerationEU. This work was moreover partially
supported by the project NEREO (Neural Reasoning over
Open Data) project funded by the Italian Ministry of
Education and Research (PRIN) Grant no. 2022AEFHAZ.

References

[1] Samuel Ainsworth, Jonathan Hayase, and Siddhartha Srini-
vasa. Git re-basin: Merging models modulo permutation
In The Eleventh International Conference on
symmetries.
Learning Representations, 2023. 2

[2] Lukas Bossard, Matthieu Guillaumin, and Luc Van Gool.
Food-101 – Mining Discriminative Components with Ran-
dom Forests. In Computer Vision – ECCV 2014, pages 446–
461, Cham, 2014. Springer International Publishing. 5
[3] Gong Cheng, Junwei Han, and Xiaoqiang Lu. Remote Sens-
ing Image Scene Classification: Benchmark and State of the
Art. Proceedings of the IEEE, 105(10):1865–1883, 2017.
Conference Name: Proceedings of the IEEE. 5

[4] Mircea Cimpoi, Subhransu Maji, Iasonas Kokkinos, Sammy
Mohamed, and Andrea Vedaldi. Describing Textures in
In 2014 IEEE Conference on Computer Vision
the Wild.
and Pattern Recognition, pages 3606–3613, Columbus, OH,
USA, 2014. IEEE. 5

[5] Tarin Clanuwat, Mikel Bober-Irizar, Asanobu Kitamoto,
Alex Lamb, Kazuaki Yamamoto, and David Ha. Deep learn-
ing for classical japanese literature. CoRR, abs/1812.01718,
2018. 5

[6] Adam Coates, Andrew Ng, and Honglak Lee. An Analysis
of Single-Layer Networks in Unsupervised Feature Learn-
ing. In Proceedings of the Fourteenth International Confer-
ence on Artificial Intelligence and Statistics, pages 215–223.
JMLR Workshop and Conference Proceedings, 2011. ISSN:
1938-7228. 5

[7] Gregory Cohen, Saeed Afshar, Jonathan Tapson, and Andr´e
van Schaik. EMNIST: Extending MNIST to handwritten let-
ters. In 2017 International Joint Conference on Neural Net-
works (IJCNN), pages 2921–2926, 2017. ISSN: 2161-4407.
5

[8] Donato Crisostomi, Marco Fumero, Daniele Baieri, Florian
c2m3: Cycle-consistent
In Advances in Neural Information

Bernard, and Emanuele Rodol`a.
multi-model merging.
Processing Systems, 2025. 2

[9] Nico Daheim, Thomas M¨ollenhoff, Edoardo Ponti, Iryna
Gurevych, and Mohammad Emtiyaz Khan. Model merging
by uncertainty-based gradient matching. In The Twelfth In-
ternational Conference on Learning Representations, 2024.
1

[10] MohammadReza Davari and Eugene Belilovsky. Model
breadcrumbs: Scaling multi-task model merging with sparse

masks. In European Conference on Computer Vision, pages
270–287. Springer, 2025. 1, 2

[11] Misha Denil, Babak Shakibi, Laurent Dinh, Marc’Aurelio
Ranzato, and Nando De Freitas. Predicting parameters in
deep learning. Advances in neural information processing
systems, 26, 2013. 3

[12] Emily L Denton, Wojciech Zaremba, Joan Bruna, Yann Le-
Cun, and Rob Fergus. Exploiting linear structure within
convolutional networks for efficient evaluation. Advances in
neural information processing systems, 27, 2014. 3, 18
[13] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov,
Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner,
Mostafa Dehghani, Matthias Minderer, Georg Heigold, Syl-
vain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is
worth 16x16 words: Transformers for image recognition at
scale, 2021. 5

[14] Carl Eckart and G. Marion Young. The approximation of one
matrix by another of lower rank. Psychometrika, 1:211–218,
1936. 3

[15] David Eigen, Marc’Aurelio Ranzato, and Ilya Sutskever.
Learning factored representations in a deep mixture of ex-
In 2nd International Conference on Learning Rep-
perts.
resentations, ICLR 2014, Banff, AB, Canada, April 14-16,
2014, Workshop Track Proceedings, 2014. 4

[16] Timur Garipov, Dmitry Podoprikhin, Alexander Novikov,
compress-
arXiv preprint

and Dmitry Vetrov. Ultimate tensorization:
ing convolutional and fc layers alike.
arXiv:1611.03214, 2016. 3

[17] Matan Gavish and David L. Donoho. The optimal hard
IEEE Transactions

threshold for singular values is 4/
on Information Theory, 60(8):5040–5053, 2014. 8

√

3.

[18] Ian J. Goodfellow, Dumitru Erhan, Pierre Luc Carrier,
Aaron Courville, Mehdi Mirza, Ben Hamner, Will Cukierski,
Yichuan Tang, David Thaler, Dong-Hyun Lee, Yingbo Zhou,
Chetan Ramaiah, Fangxiang Feng, Ruifan Li, Xiaojie Wang,
Dimitris Athanasakis, John Shawe-Taylor, Maxim Milakov,
John Park, Radu Ionescu, Marius Popescu, Cristian Grozea,
James Bergstra, Jingjing Xie, Lukasz Romaszko, Bing Xu,
Zhang Chuang, and Yoshua Bengio. Challenges in Repre-
sentation Learning: A Report on Three Machine Learning
Contests. In Neural Information Processing, pages 117–124,
Berlin, Heidelberg, 2013. Springer. 5

[19] Patrick Helber, Benjamin Bischke, Andreas Dengel, and
Damian Borth. EuroSAT: A Novel Dataset and Deep Learn-
ing Benchmark for Land Use and Land Cover Classification.
IEEE Journal of Selected Topics in Applied Earth Observa-
tions and Remote Sensing, 12(7):2217–2226, 2019. Con-
ference Name: IEEE Journal of Selected Topics in Applied
Earth Observations and Remote Sensing. 5

[20] Chenyu Huang, Peng Ye, Tao Chen, Tong He, Xiangyu
Yue, and Wanli Ouyang. EMR-merging: Tuning-free high-
In The Thirty-eighth An-
performance model merging.
nual Conference on Neural Information Processing Systems,
2024. 2

[21] Yerlan Idelbayev and Miguel A Carreira-Perpin´an. Low-rank
compression of neural nets: Learning the rank of each layer.
In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pages 8049–8059, 2020. 8

[22] Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman,
Ludwig Schmidt, Hannaneh Hajishirzi, and Ali Farhadi.
Editing models with task arithmetic. In The Eleventh Inter-
national Conference on Learning Representations, 2023. 1,
2, 5, 13

[23] Max Jaderberg, Andrea Vedaldi, and Andrew Zisserman.
Speeding up convolutional neural networks with low rank
expansions. In British Machine Vision Conference, BMVC
2014, Nottingham, UK, September 1-5, 2014. BMVA Press,
2014. 3

[24] Xisen Jin, Xiang Ren, Daniel Preotiuc-Pietro, and Pengxiang
Cheng. Dataless knowledge fusion by merging weights of
language models. In The Eleventh International Conference
on Learning Representations, 2023. 1, 2

[25] Keller Jordan, Hanie Sedghi, Olga Saukh, Rahim Entezari,
and Behnam Neyshabur. REPAIR: REnormalizing permuted
activations for interpolation repair. In The Eleventh Interna-
tional Conference on Learning Representations, 2023. 2
[26] Yong-Deok Kim, Eunhyeok Park, Sungjoo Yoo, Taelim
Choi, Lu Yang, and Dongjun Shin. Compression of deep
convolutional neural networks for fast and low power mobile
applications. In Proc. of the 4th Int. Conf. Learning Repre-
sentations (ICLR 2016), San Juan, Puerto Rico, May 2–4,
2016. 3

[27] Tamara G Kolda and Brett W Bader. Tensor decompositions
and applications. SIAM review, 51(3):455–500, 2009. 3
[28] Jonathan Krause, Michael Stark, Jia Deng, and Li Fei-Fei.
3D Object Representations for Fine-Grained Categorization.
In 2013 IEEE International Conference on Computer Vision
Workshops, pages 554–561, Sydney, Australia, 2013. IEEE.
5

[29] Alex Krizhevsky and Geoffrey Hinton. Learning multiple
layers of features from tiny images. Technical Report 0, Uni-
versity of Toronto, Toronto, Ontario, 2009. 5

[30] Y. Lecun, L. Bottou, Y. Bengio, and P. Haffner. Gradient-
based learning applied to document recognition. Proceed-
ings of the IEEE, 86(11):2278–2324, 1998. 5

[31] Zhenyi Lu, Chenghao Fan, Wei Wei, Xiaoye Qu, Dangyang
Chen, and Yu Cheng. Twin-merging: Dynamic integration
of modular expertise in model merging. Advances in Neural
Information Processing Systems, 37:78905–78935, 2024. 2
[32] Michael S Matena and Colin A Raffel. Merging models with
fisher-weighted averaging. Advances in Neural Information
Processing Systems, 35:17703–17716, 2022. 1, 2

[33] Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bis-
sacco, Bo Wu, and Andrew Y. Ng. Reading digits in natural
images with unsupervised feature learning. In NIPS Work-
shop on Deep Learning and Unsupervised Feature Learning
2011, 2011. 5

[34] Maria-Elena Nilsback and Andrew Zisserman. Automated
Flower Classification over a Large Number of Classes.
In
2008 Sixth Indian Conference on Computer Vision, Graphics
& Image Processing, pages 722–729, 2008. 5

[35] Omkar M Parkhi, Andrea Vedaldi, Andrew Zisserman, and
In 2012 IEEE Conference
C. V. Jawahar. Cats and dogs.
on Computer Vision and Pattern Recognition, pages 3498–
3505, 2012. ISSN: 1063-6919. 5

[36] Fidel A Guerrero Pe˜na, Heitor Rapela Medeiros, Thomas
Dubail, Masih Aminbeidokhti, Eric Granger, and Marco
Pedersoli. Re-basin via implicit sinkhorn differentiation. In
Proceedings of the IEEE/CVF Conference on Computer Vi-
sion and Pattern Recognition, 2023. 2

[37] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya
Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,
Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning
transferable visual models from natural language supervi-
sion. In International conference on machine learning, pages
8748–8763. PMLR, 2021. 5

[38] Alexandre Rame, Matthieu Kirchmeyer, Thibaud Rahier,
Alain Rakotomamonjy, Patrick Gallinari, and Matthieu
Cord. Diverse weight averaging for out-of-distribution gen-
eralization. Advances in Neural Information Processing Sys-
tems, 35:10821–10836, 2022. 1

[39] Noam Shazeer, *Azalia Mirhoseini, *Krzysztof Maziarz,
Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean.
Outrageously large neural networks: The sparsely-gated
In International Conference on
mixture-of-experts layer.
Learning Representations, 2017. 4

[40] Li Shen, Anke Tang, Enneng Yang, Guibing Guo, Yong
Luo, Lefei Zhang, Xiaochun Cao, Bo Du, and Dacheng
Efficient and effective weight-ensembling mixture
Tao.
of experts for multi-task model merging. arXiv preprint
arXiv:2410.21804, 2024. 2

[41] Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang,
Christopher D. Manning, Andrew Ng, and Christopher Potts.
Recursive deep models for semantic compositionality over
In Proceedings of the 2013 Confer-
a sentiment treebank.
ence on Empirical Methods in Natural Language Processing,
pages 1631–1642, Seattle, Washington, USA, 2013. Associ-
ation for Computational Linguistics. 5

[42] Johannes Stallkamp, Marc Schlipsing, Jan Salmen, and
Christian Igel. The German Traffic Sign Recognition Bench-
mark: A multi-class classification competition. In The 2011
International Joint Conference on Neural Networks, pages
1453–1460, 2011. ISSN: 2161-4407. 5

[43] George Stoica, Daniel Bolya, Jakob Brandt Bjorner, Pratik
Ramesh, Taylor Hearn, and Judy Hoffman. Zipit! merg-
In The
ing models from different tasks without training.
Twelfth International Conference on Learning Representa-
tions, 2024. 2

[44] Anke Tang, Li Shen, Yong Luo, Shuai Xie, Han Hu, Lefei
Zhang, Bo Du, and Dacheng Tao. Smile: Zero-shot sparse
mixture of low-rank experts construction from pre-trained
foundation models. arXiv preprint arXiv:2408.10174, 2024.
2

[45] Bastiaan S. Veeling, Jasper Linmans, Jim Winkens, Taco
Cohen, and Max Welling. Rotation Equivariant CNNs for
Digital Pathology. In Medical Image Computing and Com-
puter Assisted Intervention – MICCAI 2018, pages 210–218,
Cham, 2018. Springer International Publishing. 5

[46] Ke Wang, Nikolaos Dimitriadis, Guillermo Ortiz-Jimenez,
Franc¸ois Fleuret, and Pascal Frossard. Localizing task in-
formation for improved model merging and compression. In
Forty-first International Conference on Machine Learning,
2024. 2, 5, 6, 12, 13

[47] B. P. Welford. Note on a method for calculating corrected
sums of squares and products. Technometrics, 4(3):419–420,
1962. 12

[48] Mitchell Wortsman, Gabriel Ilharco, Samir Ya Gadre, Re-
becca Roelofs, Raphael Gontijo-Lopes, Ari S Morcos,
Hongseok Namkoong, Ali Farhadi, Yair Carmon, Simon
Kornblith, and Ludwig Schmidt. Model soups: averaging
weights of multiple fine-tuned models improves accuracy
In Proceedings of the
without increasing inference time.
39th International Conference on Machine Learning, pages
23965–23998. PMLR, 2022. 2

[49] Han Xiao, Kashif Rasul, and Roland Vollgraf. Fashion-
MNIST: a Novel Image Dataset for Benchmarking Machine
Learning Algorithms, 2017. arXiv:1708.07747 [cs, stat]. 5

[50] Jianxiong Xiao, Krista A. Ehinger, James Hays, Antonio
Torralba, and Aude Oliva. SUN Database: Exploring a Large
International Journal of
Collection of Scene Categories.
Computer Vision, 119(1):3–22, 2016. 5

[51] Prateek Yadav, Derek Tam, Leshem Choshen, Colin A Raf-
fel, and Mohit Bansal. Ties-merging: Resolving interference
when merging models. Advances in Neural Information Pro-
cessing Systems, 36, 2024. 1, 2

[52] Enneng Yang, Zhenyi Wang, Li Shen, Shiwei Liu, Guib-
ing Guo, Xingwei Wang, and Dacheng Tao. Adamerging:
In The
Adaptive model merging for multi-task learning.
Twelfth International Conference on Learning Representa-
tions, 2024. 2, 18

[53] Le Yu, Bowen Yu, Haiyang Yu, Fei Huang, and Yongbin Li.
Language models are super mario: Absorbing abilities from
homologous models as a free lunch. In Forty-first Interna-
tional Conference on Machine Learning, 2024. 1, 2

[54] Luca Zhou, Daniele Solombrino, Donato Crisostomi,
Maria Sofia Bucarelli, Fabrizio Silvestri, and Emanuele
Rodol`a. Atm: Improving model merging by alternating tun-
ing and merging, 2024. 2

A. Illustrative example: merging two tasks

with rank-1 approximation

Consider merging two distinct tasks by selecting only the
first singular vector and singular value from the SVD for
each task. This setting yields the following setup for each
layer Li:



|
u1Li



|
u2Li





(cid:20) σ1Li
0

|

|

(cid:21) (cid:34)

0
σ2Li

(cid:35)

vT
1Li −
vT
2Li −

−
−

In this formulation, u1Li
originates from task 1 and u2Li
from task 2, with analogous assignments for the singular
vectors v and singular values σ.

To elucidate the interaction between tasks, we exam-
ine three distinct cases, considering a single layer, thereby
omitting the layer index Li:
1. Orthogonal Singular Vectors: when u1 and u2 (respec-
tively v) are orthogonal, the similarity matrix U ⊤U (re-
spectively V ⊤V ) is given by:

(cid:21)

(cid:20) 1 0
0 1

The zeroes in the off-diagonal elements indicate no in-
terference between the tasks. Consequently, the orthog-
onal components derived from different tasks operate in-
dependently, ensuring that each task does not affect the
other.

2. Collinear Singular Vectors: when u1 and u2 (respec-
tively v) are collinear, either aligned in the same direc-
tion (angle of 0 degrees) or in the opposite direction (an-
gle of 180 degrees), the similarity matrix U ⊤U (respec-
tively V ⊤V ) takes the form:

(cid:20)

1
u, u
⟩

⟨±

(cid:21)

u,
⟨

u
⟩

±
1

If the singular vectors are perfectly aligned (0 degrees),
then u1 = u2 = u, simplifying the diagonal elements
2 = 1. Conversely, if the singular vec-
u
to
∥
tors are oppositely aligned (180 degrees), u1 =
u2,
1 Thus, the similarity matrices
resulting in
becomes:

u, u
⟩

u,
⟨

u
⟩

−

−

=

=

∥

⟨

−
(cid:20) 1
1

±

(cid:21)

1
±
1

This structure reveals complete interference between the
tasks: a double scaling effect when the vectors agree and
complete cancellation when they disagree.

3. Partially Collinear Singular Vectors: when u1 and u2
(respectively v) are partially collinear, with the angle be-
tween them ranging from slightly greater than 0 degrees
to less than 90 degrees or slightly more than 90 degrees

to less than 180 degrees, similarity matrices expressed
as:

(cid:20)

1
u2, u1
⟨

⟩

(cid:21)

u1, u2
⟨
1

⟩

In this case, the overlap between singular vectors induces
a partial interaction between the tasks. The degree of in-
terference, whether it is constructive or destructive, is
proportional to the cosine of the angle between the sin-
gular vectors. This partial collinearity leads to subtle in-
terplay, where the tasks influence each other to a degree
dictated by their vector alignment.
This example underscores the critical

role of sin-
in model merging, highlighting
gular vector alignment
how orthogonality ensures independent task performance,
collinearity leads to maximal
interference and partial
collinearity results in an intermediate level of task interac-
tion.

B. Additional details

B.1. Implementation details and computational re-

sources

Normalized Accuracy To address the varying difficulties
of the task, we report both normalized and absolute accu-
racies in our results. The normalized accuracy provides a
relative performance metric by comparing the accuracy of
the multi-task model to that of individually fine-tuned mod-
els. Specifically, the normalized accuracy is calculated as:

Normalized Accuracy =

1
T

T
(cid:88)

i=1

accuracy(θM T , ti)
accuracy(θf ti, ti)

(9)

where T is the total number of tasks, θM T represents the
multi-task model and θf ti denotes the individually fine-
tuned model for task ti. This metric allows for a more
fair comparison by adjusting for the baseline performance
of each task.

Datasets for tasks All benchmarks were performed by in-
tegrating the codebase provided by Wang et al. [46]. In line
with the principles of PEFT, we reused the already existing
model checkpoints in the codebase for both the models and
classification heads without additional fine-tuning.

Implementation Our method utilizes the SVD, a matrix
decomposition technique applicable to two-dimensional
matrices. For layers that are not represented as matrices
(e.g., normalization layers) we default to standard Task
Arithmetic. In particular, we employ Knut’s algorithm
[47] to compute the average efficiently. This ensures that
all fine-tuned model task layers, regardless of their struc-
ture, are appropriately integrated into the merged model.

Figure 9. Absolute accuracy of a ViT-B-16 merged over 8, 14, and 20 tasks, respectively.

Figure 10. Absolute accuracy of a ViT-L-14 merged over 8, 14, and 20 tasks, respectively.

Compute Resources We utilize PyTorch as deep learn-
ing framework. All the merging and evaluations were con-
ducted on an NVIDIA 4060Ti GPU with 16GB of mem-
ory, and an Intel i7-6800K CPU equipped with 64GB
of RAM. For experiments that need more than 64GB of
RAM, we resort to a shared HTCondor cluster equipped
with NVIDIA P6000 GPUs.

B.2. Hyperparameter Settings

Following Task Arithmetic [22] and Consensus
TA [46], we apply a single scaling factor, α, to adjust the
multi-task vector within the model merging techniques out-
lined in Table 2. This scaling factor is optimized, when
feasible, over the range
, with the
optimal value selected based on the average validation per-
formance across all tasks. However, as discussed in Sec-
tion 6.3, our experimental findings indicate that the pro-
posed TSV-Merge method does not strictly depend on this

0.0, 0.1, ..., 2.9, 3.0
{

}

hyperparameter, as the marginal performance gains from
tuning α do not justify the computational resources re-
quired for the evaluation on the validation datasets. Con-
sequently, this allows us to eliminate the necessity for val-
idation datasets and the corresponding labels from the pre-
requisites of the method, further simplifying the practical-
ity and resource usage of the approach. The evaluation is
performed on batch of 32 images. To produce Fig. 7 we
selected the following subsets of 8 tasks from the 20 avail-
able, using the whitening method to speed up computation,
the number in the image is the index indicating the subset:
OxfordIIITPet,
0. SVHN,

SUN397,

STL10,

Flowers102, CIFAR100, PCAM, FER2013

1. PCAM,

FER2013,

CIFAR10,

FashionMNIST,
KMNIST

RenderedSST2,

Food101,
EMNIST,

2. EuroSAT, GTSRB, MNIST, RESISC45, SVHN,

SUN397, STL10, OxfordIIITPet

CarsDTDEuroSATGTSRBMNISTRESISC45SVHNSUN39720%40%60%80%8TasksCarsDTDEuroSATGTSRBMNISTRESISC45SVHNSUN397STL10OxfordIIITPetFlowers102CIFAR100PCAMFER201320%40%60%80%14TasksTSV-M(Ours)TSV-C(Ours)ConsensusTATaskArithmeticZero-shotCarsDTDEuroSATGTSRBMNISTRESISC45SVHNSUN397STL10OxfordIIITPetFlowers102CIFAR100PCAMFER2013CIFAR10Food101FashionMNISTRenderedSST2EMNISTKMNIST20%40%60%80%20TasksCarsDTDEuroSATGTSRBMNISTRESISC45SVHNSUN39720%40%60%80%8TasksCarsDTDEuroSATGTSRBMNISTRESISC45SVHNSUN397STL10OxfordIIITPetFlowers102CIFAR100PCAMFER201320%40%60%80%14TasksTSV-M(Ours)TSV-C(Ours)ConsensusTATaskArithmeticZero-shotCarsDTDEuroSATGTSRBMNISTRESISC45SVHNSUN397STL10OxfordIIITPetFlowers102CIFAR100PCAMFER2013CIFAR10Food101FashionMNISTRenderedSST2EMNISTKMNIST20%40%60%80%20TasksFigure 11. Mean absolute accuracy of the ViT-B-16 model
across increasing fractions of retained singular components, av-
eraged over 20 tasks. The red line represents the average accu-
racy of the original fine-tuned models with full-rank task matrices,
while the green line shows the accuracies using low-rank approxi-
mations.

Figure 12. Mean absolute accuracy of the ViT-L-14 model
across increasing fractions of retained singular components, av-
eraged over 20 tasks. The red line represents the average accu-
racy of the original fine-tuned models with full-rank task matrices,
while the green line shows the accuracies using low-rank approxi-
mations.

3. Cars, DTD, EuroSAT, GTSRB, MNIST, RESISC45,

SVHN, SUN397

k′ < d×m

d+m+1 . This condition ensures:

4. Cars, DTD, EuroSAT, GTSRB, FashionMNIST,

Params(NN) > Params(TSV)

RenderedSST2, EMNIST, KMNIST

5. MNIST, RESISC45, SVHN, SUN397, STL10,

Substituting the expressions, yields:

OxfordIIITPet, Flowers102, CIFAR100
OxfordIIITPet,

6. STL10,

Flowers102,

CIFAR100, PCAM, FER2013, CIFAR10, Food101

B.3. Storage cost calculation

Suppose we have a neural network comprising of L two-
dimensional layers, each of dimension d
m, and N one-
dimensional layers of size c. The total number of parame-
ters in the network is therefore:

×

Params(NN) = L

(d

m) + N

c.

×

×

×
In standard Task Arithmetic, one must store the same
number of parameters to obtain a task vector. In contrast,
our approach provides the flexibility to select the number of
parameters to preserve based on storage constraints or the
desired needed performance, to adhere to the chosen con-
straints. Under the above assumptions, our method applies
the truncated SVD to each two-dimensional layer. This de-
composition yields two matrices of singular vectors, U and
V , and a vector of singular values, σ, specifically:
• U of size d
• V of size k
• σ of size k,
where k = min(d, m). We select a reduced rank k′
k
to approximate each layer’s task matrix. Consequently, the
total number of parameters for TSV becomes:

k,
m,

×
×

≪

Params(TSV) = L

((d

×

×

k′) + k′ + (k′

m)) + N

c

×

×

To demonstrate that our method results in fewer stored pa-
rameters than the original parameter count, we require that

(d

L

×

×

m)+N

c > L

((d

×

×

×

k′)+k′ +(k′

m))+N

c

×

×

Simplifying, we obtain:

L

×

(d

(d

×

×
d

×

m) > L

×
m) > ((d
m > k′

((d

k′) + k′ + (k′

×

k′) + k′ + (k′
(d + 1 + m)

×

×

×
m))

m))

(10)

k′ <

×
d

m

×
d + m + 1

.

This inequality confirms that our method reduces the stor-
age requirements of a task vector when k′ < d×m
d+m+1 . Em-
pirical evidence from Figures 2, 11 and 12 suggests that
selecting k′ < 0.1
min(d, m) is sufficient
to preserve most of the task performance, preserving the
main requirement of performance. Furthermore, it is easy
to prove that when choosing k′ = k
T , the inequality is al-
ways satisfied for T
3, respecting the main requirement
of limited storage usage.

k = 0.1

≥

×

×

C. Proofs

We hereby prove the claims outlined in the main
manuscript.

C.1. Characterization of the similarity matrices
Proposition C.1. The matrix ˆU ⊤ ˆU is positive definite.

Proof. We define ˆU ⊤ ˆU , where ˆU is a generic d
k rect-
angular matrix. Consequently, ˆU ⊤ ˆU is a k
k square ma-
trix. To establish that ˆU ⊤ ˆU is positive definite, it suffices

×

×

0.00.10.20.30.40.5Fractionofnon-zerosingularvalues(rank)88909294AverageAccuracy(%)93.17FullrankLowrankFullrankLowrank0.00.10.20.30.40.5Fractionofnon-zerosingularvalues(rank)939495AverageAccuracy(%)94.73FullrankLowrankFullrankLowrankto demonstrate that for all non-zero vectors x
following inequality holds:

∈

Rk, the

x⊤ ˆU ⊤ ˆU x > 0.

This expression can be rewritten as:

x⊤( ˆU ⊤ ˆU )x = ( ˆU x)⊤( ˆU x) =

ˆU x
∥
∥

2.

ˆU x
∥
∥

2 denotes the squared Euclidean norm of the
Here,
vector ˆU x, which is always non-negative. Moreover, as-
suming that ˆU has full column rank, the norm
2 is
strictly positive for any non-zero vector x. Therefore, we
have:

ˆU x
∥
∥

ˆU x
2 > 0
∥

∥
This implies that:

for all x

∈

Rk, x

= 0.

x⊤ ˆU ⊤ ˆU x > 0

for all x

Rk, x

= 0,

∈

which confirms that ˆU ⊤ ˆU is positive definite.

Corollary C.2. Since ˆU ⊤ ˆU is positive definite, then ˆU ⊤ ˆU
is invertible.

Proof. From Proposition C.1, we have established that
ˆU ⊤ ˆU is a positive definite matrix. A positive definite ma-
trix, by definition, has all its eigenvalues strictly positive.
Let λ1, λ2, . . . , λk denote the eigenvalues of ˆU ⊤ ˆU . There-
fore, we have:

λi > 0 for all

i = 1, 2, . . . , k.

The determinant of ˆU ⊤ ˆU is the product of its eigenval-

ues:

det( ˆU ⊤ ˆU ) =

k
(cid:89)

i=1

λi.

Since each λi is positive, their product is also positive:

det( ˆU ⊤ ˆU ) > 0.

C.2. Observations
Since ˆU ⊤ ˆU is a real symmetric matrix, it admits an eigen-
decomposition of the form

ˆU ⊤ ˆU = QΛQ−1 = QΛQ⊤,

where:
• Λ is a diagonal matrix containing the real eigenvalues of

ˆU ⊤ ˆU ,

• Q is an orthogonal matrix whose columns are the or-
thonormal eigenvectors of ˆU ⊤ ˆU , satisfying Q⊤ = Q−1.
The inverse of ˆU ⊤ ˆU exists (see Corollary C.2), and can

be expressed using its eigendecomposition as

( ˆU ⊤ ˆU )−1 = QΛ−1Q−1 = QΛ−1Q⊤.

Additionally, since Λ is a diagonal matrix with non-zero di-
agonal entries (Proposition C.1), its inverse Λ−1 is straight-
forward to compute, with each diagonal element given by

Λ−1 = diag

(cid:19)

,

(cid:18) 1
λi

where λi are the eigenvalues of ˆU ⊤ ˆU .

Furthermore, the eigenvalues of ( ˆU ⊤ ˆU )−1 are 1
λi

, each
of which is positive since λi > 0 for all i (following by the
definition in Proposition C.1). Consequently, ( ˆU ⊤ ˆU )−1 is
also a positive definite matrix.

These observations confirm that not only is ˆU ⊤ ˆU posi-
tive definite, but its inverse inherits this property due to the
positivity of its eigenvalues.

Low-rank
approx.

Interf.
reduction

×
✓
×
✓

×
×
✓
✓

8 tasks

88.6 (+0.0)
87.9 (-0.7)
92.1 (+4.2)
97.0 (+4.9)

ViT-L-14
14 tasks

84.0 (+0.0)
83.4 (-0.6)
86.8 (+3.4)
94.4 (+7.6)

20 tasks

78.1 (+0.0)
77.2 (-0.9)
81.0 (+3.8)
92.5 (+11.5)

A matrix is invertible if and only if its determinant is
non-zero. Given that det( ˆU ⊤ ˆU ) > 0, it follows that ˆU ⊤ ˆU
is invertible.

Therefore, ˆU ⊤ ˆU is invertible.

Comparison of different versions of Task
Table 6.
Arithmetic, comprising either
the low-rank approxima-
tion step, the interference reduction step, or both. The method
performing both corresponds to the proposed TSV-Merge.

Low-rank
approx.

Interf.
reduction

×
✓
×
✓

×
×
✓
✓

8 tasks

79.6 (+0.0)
79.6 (+0.0)
84.8 (+5.2)
93.9 (+9.1)

ViT-B-16
14 tasks

75.9 (+0.0)
74.9 (-1.0)
79.0 (+4.1)
91.0 (+12.0)

20 tasks

70.8 (+0.0)
70.0 (-0.8)
73.2 (+3.2)
86.5 (+13.3)

Comparison of different versions of Task
Table 5.
Arithmetic, comprising either
the low-rank approxima-
tion step, the interference reduction step, or both. The method
performing both corresponds to the proposed TSV-Merge.

C.3. Proof of Theorem 6.1

∈

N such that T > 4. Define
Theorem 6.1. Let T
U = [U1, . . . , UT ] as the matrix obtained by concatenat-
n. Let
ing T orthogonal matrices Ui, each of shape n
(cid:98)U = [ (cid:98)U1, . . . , (cid:98)UT ] be the matrix formed by truncating each
Ui to its first k columns. Denote by X and (cid:98)X the matrices
resulting from Procrustes orthonormalization of U and (cid:98)U ,
respectively. If k

, then

×

√

T

n T −2
T

≤

U
∥

X

F
∥

−

(cid:98)U

≥ ∥

(cid:98)X

F .
∥

−

̸
̸
Proof. Let us consider the SVD decomposition of U and
(cid:98)U : U = PuΣuPv and (cid:98)U = Ru (cid:99)ΣuRv. X and (cid:98)X obtain as
X = PuP ⊤
v respectively We first consider
U
F . Notice that the singular
the Frobenius norm of
values of U are the square root of the eigenvalues of Σu =
U U ⊤.

v , (cid:98)X = RuR⊤

X

−

||

||

U U ⊤ = (cid:80)N

i=1 UiU ⊤

i = T In. As a consequence, the
eigenvalues of U U ⊤ are all equal to T and the singular val-
ues are all equal to √T .

X

||

−

U

||

F =

=

=

v −

PuΣuP ⊤
F
v ||
Σu)P ⊤
F
v ||

PuP ⊤
||
Pu(I
In

||

||

−

=

In
= √n(√T

−

||

−
Σu
F
||
√T In

F

||
1).

−

Putting everything together,

(cid:98)X
||

−

(cid:98)U

F =

||

=

(cid:118)
(cid:117)
(cid:117)
(cid:116)

n
(cid:88)

(ˆσi

i=1

(cid:118)
(cid:117)
(cid:117)
(cid:116)n +

1)2

−

n
(cid:88)

i=1

ˆσ2
i −

2

n
(cid:88)

i=1

ˆσi

(cid:118)
(cid:117)
(cid:117)
(cid:116)n + kT

=

n
(cid:88)

i=1

ˆσi

2

−

(cid:113)

n + kT

2√kT .

(14)

(15)

(16)

(17)

(cid:112)

≤

−
So we have to check for what values of k it holds that
n + kT
We have that
(cid:113)

√n(√T

2√kT

1).

−

−

≤

n + kT

2√kT

√n + kT

√n(√T

1)

(18)

−

≤

≤

−

We are now left to compute

F . In this case,
we are not able to compute the exact norm without other
assumptions, but we can provide an upper bound that gives
us a sufficient condition to prove our statement. As before

(cid:98)X
||

(cid:98)U

−

||

Equation 18 is satisifed if
√
T
k

n T −2
T

≤

. This concludes the proof. Since k is
a positive number the inequaility of meaningful for T >
4.

(cid:98)X
||

−

(cid:98)U

F =

||

=

In
||
(cid:118)
(cid:117)
(cid:117)
(cid:116)

−
n
(cid:88)

(cid:98)Σu

F

||

(ˆσi

1)2.

−

i=1

where ˆσ are the singular values of (cid:98)U .

Notice that (cid:80)n

i = tr( (cid:98)U (cid:98)U ⊤) = tr( (cid:98)U ⊤ (cid:98)U ) and (cid:98)U ⊤U
T k matrices with diagonal elements equal to one,

i=1 ˆσ2

is a T k
×
so tr( (cid:98)U ⊤ (cid:98)U ) = kT .
Moreover,

n
(cid:88)

i=1

ˆσi =

≥

=

n
(cid:88)

(cid:113)

i=1
(cid:118)
(cid:117)
(cid:117)
(cid:116)

n
(cid:88)

i=1

λi( (cid:98)U (cid:98)U ⊤)

λi( (cid:98)U (cid:98)U ⊤)

(cid:113)

tr( (cid:98)U (cid:98)U ⊤) = √T k

(11)

(12)

(13)

Notice that this upper bound is tight, indeed the sum

i=1 ˆσi

of the singular values of ˆU must lie within: √kT
≤
(cid:80)n
kT . The minimum √kT is achieved if all ma-
trices Ui are equals, on the other end, the maximum kT is
achieved if the kT columns are orthonormal.

≤

Figure 13. Singular Task Interference (STI) and average normal-
ized accuracy for Task Arithmetic and TSV-Merge on the
ViT-B-16 model, evaluated across merges of 8, 14, and 20 tasks.

D. Additional experimental results

D.1. Per-dataset performance metrics

In Section 5.1, we present comprehensive results for indi-
vidual tasks using the ViT-B-32 model. Here we include
analogous radar plots for the ViT-B-16 model in Figure 9
and the ViT-L-14 model in Figure 10. The analyses of
these models reveal findings consistent with those reported
for ViT-B-32 in the main text.

0.700.750.800.850.900.95AverageNormalizedAccuracy020000400006000080000InterferenceTaskArithmeticTSV-MergeNumberofTasks8tasks14tasks20tasksFigure 14. Detailed view of Singular Task Interference (STI) across layers in a ViT-B-32 for 20 tasks. The interference trend is high in
early layers and decreases later. Here, the pattern for each transformer block is observable, the interference first increases and then drops
in each attention-out layer.

of maintained TSVs. We refer to Figure 17 for a breakdown
of this analysis.

D.2. Extended analysis

D.2.1. Whitening vs. SVD

As we have seen in Section 4.2, applying a whitening trans-
formation to the matrices of task singular vectors is mathe-
matically equivalent to solving the Orthogonal Procrustes
problem. However, implementing these two approaches
may yield different results depending on the distinct matrix
decomposition algorithms employed. In this study, we used
PyTorch to compute both eigendecomposition and SVD,
observing slightly different results that may be attributed
to numerical errors. To more robustly compute the matrix
square root for the eigendecomposition case, we compute

(cid:32)

Λ− 1

2 = diag

(cid:33)

1
λi

(cid:112)
|

+ ϵ

|
where ϵ = 1e
12 prevents division by 0 and the absolute
value avoids numerical errors producing small negative val-
ues in magnitude less than 1e

−

6.

−

D.2.2. Impact of rank

The Section 3.2 shows that
the task matrices of a
ViT-B-32 are inherently low-rank and a small percentage
of TSVs is enough to approximate each layer with satisfy-
ing results. We here provide the same plots for the models
ViT-B-16 (Figure 11) and ViT-L-14 (Figure 12), ob-
In fact, the first shows a de-
serving analogous findings.
crease of 1.3% mean accuracy at 3% of retained TSVs and
the second shows a reduction of 1.1% mean accuracy at 2%

Figure 15. Singular Task Interference (STI) and average normal-
ized accuracy for Task Arithmetic and TSV-Merge on the
ViT-L-14 model, evaluated across merges of 8, 14, and 20 tasks.

D.2.3. Extended Ablation study

In Section 6.1, we reported an ablation study on the
ViT-B-32 model to evaluate the individual contributions
of low-rank approximation and interference reduction to the
overall performance of our TSV-Merge method. To fur-
ther mark our findings and demonstrate the generality of

posembdvis.posembdvis.projvis.trf.res.0.attn.inprojvis.trf.res.0.attn.outprojvis.trf.res.0.mlp.cfcvis.trf.res.0.mlp.cprojvis.trf.res.1.attn.inprojvis.trf.res.1.attn.outprojvis.trf.res.1.mlp.cfcvis.trf.res.1.mlp.cprojvis.trf.res.2.attn.inprojvis.trf.res.2.attn.outprojvis.trf.res.2.mlp.cfcvis.trf.res.2.mlp.cprojvis.trf.res.3.attn.inprojvis.trf.res.3.attn.outprojvis.trf.res.3.mlp.cfcvis.trf.res.3.mlp.cprojvis.trf.res.4.attn.inprojvis.trf.res.4.attn.outprojvis.trf.res.4.mlp.cfcvis.trf.res.4.mlp.cprojvis.trf.res.5.attn.inprojvis.trf.res.5.attn.outprojvis.trf.res.5.mlp.cfcvis.trf.res.5.mlp.cprojvis.trf.res.6.attn.inprojvis.trf.res.6.attn.outprojvis.trf.res.6.mlp.cfcvis.trf.res.6.mlp.cprojvis.trf.res.7.attn.inprojvis.trf.res.7.attn.outprojvis.trf.res.7.mlp.cfcvis.trf.res.7.mlp.cprojvis.trf.res.8.attn.inprojvis.trf.res.8.attn.outprojvis.trf.res.8.mlp.cfcvis.trf.res.8.mlp.cprojvis.trf.res.9.attn.inprojvis.trf.res.9.attn.outprojvis.trf.res.9.mlp.cfcvis.trf.res.9.mlp.cprojvis.trf.res.10.attn.inprojvis.trf.res.10.attn.outprojvis.trf.res.10.mlp.cfcvis.trf.res.10.mlp.cprojvis.trf.res.11.attn.inprojvis.trf.res.11.attn.outprojvis.trf.res.11.mlp.cfcvis.trf.res.11.mlp.cprojtokenembdLayers0100020003000400050006000Interference2-LayerMovingAverageInterference2-LayerMovingAverageInterference0.7750.8000.8250.8500.8750.9000.9250.9500.975AverageNormalizedAccuracy050000100000150000200000250000InterferenceTaskArithmeticTSV-MergeNumberofTasks8tasks14tasks20tasks8 task benchmark, AdaMerging achieves an accuracy
of 85.43%, while our TSV-M attains 88.93%, an im-
provement of approximately 3.5% without requiring any
test-time adaptation. Additionally, when integrating an
AdaMerging-style test-time adaptation into our frame-
work, the accuracy increases to 89.87%, demonstrating the
complementary benefits of combining TSV-M with test-
time adaptation techniques.

E. Theoretical motivations and analysis

E.1. Theoretical foundation - Empirical design

The TSV-C method is grounded in the well-established
framework of low-rank approximation for compression
(e.g., [12]). Instead, TSV-M is motivated by more empir-
ical foundations: it is designed to achieve noise reduction
through low-rank approximation and to eliminate interfer-
ence via orthogonalization. Low-rank truncation serves to
filter out insignificant variations, while orthogonalization
ensures that task-specific singular vectors remain indepen-
dent, preserving individual task performance.

E.2. Heuristic interference measure

Given that a formal definition of interference in model
merging is not yet established, we adopt an operational defi-
nition: interference is any cross-task interaction that hinders
the merging process. Our proposed Singular Task Interfer-
ence measure is empirically validated by the consistent per-
formance improvements observed when its value is mini-
mized. Furthermore, we examine the relationship between
overlapping singular vectors and knowledge sharing. Un-
like multi-task learning (MTL), which enables coordinated
knowledge sharing through joint training, the independent
task-wise finetuning in model merging may evolve in de-
structive overlaps in the activations, resulting in interference
rather than beneficial knowledge sharing. By orthogonaliz-
ing the singular vectors, our approach effectively mitigates
these overlaps, reducing interference and enhancing the per-
formance of the merged model.

Figure 16. Accuracy with varying compression budgets for
ViT-L-14 across 14 tasks.

our approach across different model sizes, we report in Ta-
ble 5 the ablation study for the ViT-B-16 model and in
Table 6 ViT-L-14 model. The experimental setup follows
the one described in Section 6.1. We assess the impact of
the two key components of TSV-M, low-rank approxima-
tion and interference reduction, by considering the follow-
ing four configurations:
1. Baseline Task Arithmetic:
without any modification.

the standard TA method

2. Low-Rank Approximation: apply only low-rank ap-
proximation to task matrices without any interference re-
duction step.

3. Interference Reduction: apply interference reduction
to the full-rank task matrices without any pre-step of
low-rank approximation.

4. TSV-Merge: Combining both low-rank approximation

and interference reduction.

D.2.4. Effect of task interference

We provide here the same plots shown in Figure 8, for the
ViT-B-16 we show it in Figure 13 and respectively for
ViT-L-14 in Figure 15. The finding remains valid also
for these models, all the instances show a significant gain in
accuracy when the interference is removed.

D.2.5. Detailed per-layer task interference

We show in Fig. 14 the per-layer task interference, ex-
tending the block-level analysis in Figure 6 in the main
manuscript.

D.2.6. Compression analysis

Our experimental results (see Table 3 in main manuscript)
demonstrate that TSV outperforms TALL-Mask on the
large-scale ViT-L-14 model for 14 and 20 tasks bench-
marks, signaling a scaling advantage. With a fixed-budget
analysis, we show in Figure 16 that unlike TALL-Mask,
which has a fixed requirement defined by model size and
number of tasks, we allow flexible compression rates by al-
lowing rank selection. This enables more aggressive com-
pression, as highlighted in the green region in the figure.

D.2.7. Test-time adaptation
We compare our method with AdaMerging [52] for
test-time adaptation. On a subset of 7 tasks from the

2.02.53.03.54.04.5Storagebytes×1090.920.94Accuracy(%)TSV-CTALL-MaskFigure 17. Absolute accuracy of the ViT-B-32 model across increasing fractions of retained singular components, for each task. The red
line represents the accuracy of the original fine-tuned models with full-rank task matrices, while the green line shows the accuracies using
low-rank approximations.

Fractionofnon-zerosingularvalues(rank)99.499.599.699.7AverageAccuracy(%)MNISTFractionofnon-zerosingularvalues(rank)65.067.570.072.575.077.5AverageAccuracy(%)CarsFractionofnon-zerosingularvalues(rank)7580859095AverageAccuracy(%)DTDFractionofnon-zerosingularvalues(rank)98.098.599.099.5AverageAccuracy(%)EuroSATFractionofnon-zerosingularvalues(rank)9092949698AverageAccuracy(%)GTSRBFractionofnon-zerosingularvalues(rank)868890929496AverageAccuracy(%)RESISC45Fractionofnon-zerosingularvalues(rank)707274AverageAccuracy(%)SUN397Fractionofnon-zerosingularvalues(rank)96.096.597.097.5AverageAccuracy(%)SVHNFractionofnon-zerosingularvalues(rank)87.087.588.088.5AverageAccuracy(%)PCAMFractionofnon-zerosingularvalues(rank)82848688AverageAccuracy(%)CIFAR100Fractionofnon-zerosingularvalues(rank)97.8597.9097.9598.0098.05AverageAccuracy(%)STL10Fractionofnon-zerosingularvalues(rank)90.290.490.690.891.091.2AverageAccuracy(%)OxfordIIITPetFractionofnon-zerosingularvalues(rank)80.082.585.087.590.0AverageAccuracy(%)Flowers102Fractionofnon-zerosingularvalues(rank)66687072AverageAccuracy(%)FER2013Fractionofnon-zerosingularvalues(rank)96.7597.0097.2597.5097.75AverageAccuracy(%)CIFAR10Fractionofnon-zerosingularvalues(rank)86878889AverageAccuracy(%)Food101Fractionofnon-zerosingularvalues(rank)72737475AverageAccuracy(%)RenderedSST2Fractionofnon-zerosingularvalues(rank)99.599.699.799.8AverageAccuracy(%)EMNIST0.00.10.20.30.40.5Fractionofnon-zerosingularvalues(rank)93.093.594.094.595.0AverageAccuracy(%)FashionMNIST0.00.10.20.30.40.5Fractionofnon-zerosingularvalues(rank)92949698AverageAccuracy(%)KMNISTLowrankFullrankFigure 18. Breakdown of classification accuracy in confusion matrices of a single merged ViT-L-14 model over 20 tasks. The numbers
are omitted when they are too small to display.

050100150PredictedLabels0255075100125150175TrueLabelsCarsTest010203040PredictedLabels010203040TrueLabelsDTDTest02468PredictedLabels02468TrueLabels2371224607017923154003300663302308402617075342119402491613003185601412121391561015239142444730421869130156120111222250240250100198441200000002397EuroSATTest010203040PredictedLabels0510152025303540TrueLabelsGTSRBTest02468PredictedLabels02468TrueLabels97900000100001121207023001209594230134134001100413001005390934001032003087971002115011899390000201403100801000011539622209022132988MNISTTest010203040PredictedLabels0510152025303540TrueLabelsRESISC45Test02468PredictedLabels02468TrueLabels1612191519121917616990434177108721197311962382868384270171463547222545112540151251044463451021016230212138123161341362288275661281727525170338431071291278318641322726136201401560313624152441681511316SVHNTest0100200300PredictedLabels050100150200250300350TrueLabelsSUN397Test02468PredictedLabels02468TrueLabels79900000001008000000000000789000000110007878212000001796030000101217833000000000799001070001179100000000008000002000000798STL10Test0102030PredictedLabels05101520253035TrueLabelsOxfordIIITPetTest020406080100PredictedLabels020406080100TrueLabelsFlowers102Test020406080PredictedLabels020406080TrueLabelsCIFAR100Test−0.50.00.51.01.5PredictedLabels−0.50−0.250.000.250.500.751.001.251.50TrueLabels152971094380412573PCAMTest0246PredictedLabels0123456TrueLabels6043615025656216148184220821566226111112161103216456113127181057788664221562833521363336819128262683378FER2013Test02468PredictedLabels02468TrueLabels990130100140199100000008509772723220214960716432110139810112100062599464721108940974220003041099200600000009940070000005988CIFAR10Test020406080100PredictedLabels020406080100TrueLabelsFood101Test02468PredictedLabels02468TrueLabels92701922111801200992070000102419305191130701911492526110130006019891023070402009022667171892893268060901100000090984076012148897000000010500949FashionMNISTTest−0.50.00.51.01.5PredictedLabels−0.50−0.250.000.250.500.751.001.251.50TrueLabels67024267842RenderedSST2Test02468PredictedLabels02468TrueLabels3991110010042039870070051081039521100169301239820701431106303917002165114039851120203604039549310111280039640510123120398281080920203978EMNISTTest02468PredictedLabels02468TrueLabels5726820308329181681207654803429126187716815513535518632562216411492552628281616778142050413722516427852103519105931123487138216630911114915149374412312772480272117010602260738KMNISTTest