Random Teachers are Good Teachers

Felix Sarnthein 1 Gregor Bachmann 1 Sotiris Anagnostidis 1 Thomas Hofmann 1

Abstract

1. Introduction

3
2
0
2

n
u
J

9
1

]

G
L
.
s
c
[

2
v
1
9
0
2
1
.
2
0
3
2
:
v
i
X
r
a

In this work, we investigate the implicit regular-
ization induced by teacher-student learning dy-
namics in self-distillation. To isolate its effect,
we describe a simple experiment where we con-
sider teachers at random initialization instead of
trained teachers. Surprisingly, when distilling a
student into such a random teacher, we observe
that the resulting model and its representations
already possess very interesting characteristics;
(1) we observe a strong improvement of the dis-
tilled student over its teacher in terms of prob-
ing accuracy. (2) The learned representations are
data-dependent and transferable between different
tasks but deteriorate strongly if trained on random
inputs. (3) The student checkpoint contains sparse
subnetworks, so-called lottery tickets, and lies
on the border of linear basins in the supervised
loss landscape. These observations have interest-
ing consequences for several important areas in
machine learning: (1) Self-distillation can work
solely based on the implicit regularization present
in the gradient dynamics without relying on any
dark knowledge, (2) self-supervised learning can
learn features even in the absence of data augmen-
tation, and (3) training dynamics during the early
phase of supervised training do not necessarily
require label information. Finally, we shed light
on an intriguing local property of the loss land-
scape: the process of feature learning is strongly
amplified if the student is initialized closely to
the teacher. These results raise interesting ques-
tions about the nature of the landscape that have
remained unexplored so far. Code is available at
www.github.com/safelix/dinopl.

1Department of Computer Science, ETH Z¨urich, Switzerland.

Correspondence to: Felix Sarnthein <safelix@ethz.ch>.

Proceedings of the 40 th International Conference on Machine
Learning, Honolulu, Hawaii, USA. PMLR 202, 2023. Copyright
2023 by the author(s).

1

The teacher-student setting is a key ingredient in several
areas of machine learning. Knowledge distillation is a com-
mon strategy to achieve strong model compression by train-
ing a smaller student on the outputs of a larger teacher
model, leading to better performance compared to training
the small model on the original data only (Bucilˇa et al., 2006;
Ba & Caruana, 2013; Hinton et al., 2015; Polino et al., 2018;
Beyer et al., 2022). In the special case of self-distillation,
where the two architectures match, it is often observed in
practice that the student manages to outperform its teacher
(Yim et al., 2017; Furlanello et al., 2018; Yang et al., 2018).
The predominant hypothesis in the literature attests this sur-
prising gain in performance to the so-called dark knowledge
of the teacher, i.e., its logits encode additional information
about the data distribution (Hinton et al., 2015; Wang et al.,
2021; Xu et al., 2018).
Another area relying on a teacher-student setup is self-
supervised learning where the goal is to learn informative
representations in the absence of targets (Caron et al., 2021;
Grill et al., 2020; Chen & He, 2021; Zbontar et al., 2021;
Assran et al., 2022). Here, the two models typically receive
two different augmentations of a sample, and the student
is forced to mimic the teacher’s behavior. Such a learning
strategy encourages representations that remain invariant to
the employed augmentation pipeline, which in turn leads to
better downstream performance.
Despite its importance as a building block, the teacher-
student setting itself remains very difficult to analyze as
its contribution is often overshadowed by stronger compo-
nents in the pipeline, such as dark knowledge in the trained
teacher or the inductive bias of data augmentation. In this
work, we take a step towards simplifying and isolating the
key components in the setup by devising a very simple ex-
periment; instead of working with a trained teacher, we
consider teachers at random initialization, stripping them
from any data dependence and thus removing any dark
knowledge. We also remove augmentations, making the
setting completely symmetric between student and teacher
and further reducing inductive bias. Counter-intuitively, we
observe that even in this setting, the student still manages
to learn from its teacher and even exceed it significantly in
terms of representational quality, measured through linearly
probing the features (see Fig. 1). This result shows the fol-

Random Teachers are Good Teachers

establish risk bounds. Allen-Zhu & Li (2020) on the other
hand, study more realistic width networks and show that if
the data satisfies a certain multi-view property, ensembling
and distilling is provably beneficial. Yuan et al. (2020) study
a similar setup as our work by considering teachers that are
not perfectly pre-trained but of weaker (but still far from
random) nature. They show that the dark knowledge is more
of a regularization effect and that a similar boost in perfor-
mance can be achieved by label smoothing. Stanton et al.
(2021) further question the relevance of dark knowledge
by showing that students outperform their teacher without
fitting the dark knowledge. We would like to point out how-
ever that we study completely random teachers and our loss
function does not provide the hard labels for supervisory
signal, making our task completely independent of the tar-
gets.
Self-supervised learning can be broadly split into two cate-
gories, contrastive and non-contrastive methods. Contrastive
methods rely on the notion of negative examples, where fea-
tures are actively encouraged to be dissimilar if they stem
from different examples (Chen et al., 2020; Schroff et al.,
2015; van den Oord et al., 2018). Non-contrastive meth-
ods follow our setting more closely as only the notion of
positive examples is employed (Caron et al., 2021; Grill
et al., 2020; Chen & He, 2021). While these methods enjoy
great empirical successes, a theoretical understanding is still
largely missing. Tian et al. (2021) investigate the collapse
phenomenon in non-contrastive learning and show in a sim-
plified setting how the stop gradient operation can prevent it.
Wang et al. (2022) extend this work and prove in the linear
setting how a data-dependent projection matrix is learned.
Zhang et al. (2022) explore a similar approach and prove
that SimSiam (Chen & He, 2021) avoids collapse through
the notion of extra-gradients. Anagnostidis et al. (2022)
show that strong representation learning occurs with heavy
data augmentations even if random labels are used. Despite
this progress on the optimization side, a good understanding
of feature learning has largely remained elusive.
The high-dimensional loss landscapes of neural networks re-
main very mysterious, and their properties play a crucial role
in our work. Safran & Shamir (2017) prove that spurious
local minima exist in the teacher-student loss of two-layer
ReLU networks. Garipov et al. (2018); Draxler et al. (2018)
show that two SGD solutions are always connected through
a non-linear valley of low loss. Frankle & Carbin (2018);
Frankle et al. (2019; 2020) investigate the capacity of over-
parameterized networks through pruning of weights. They
find that sparse sub-networks develop already very early in
neural network training. Zaidi et al. (2022); Benzing et al.
(2022) investigate random initializations in supervised loss
landscapes. Still, the field lacks a convincing explanation
as to how simple first-order gradient-based methods such as
SGD manage to navigate the landscape so efficiently.

Figure 1. Linear probing accuracies of representations generated
by teachers, students, and the flattened input images on CIFAR10
as a function of training time. Left: ResNet18. Right: VGG11
without batch normalization.

lowing: (1) Even in the absence of dark knowledge, relevant
feature learning can happen for the student in the setting of
self-distillation. (2) Data augmentation is the main but not
only ingredient in non-contrastive self-supervised learning
that leads to representation learning.
Surprisingly, we find that initializing the student close to the
teacher further amplifies the implicit regularization present
in the dynamics. This is in line with common practices in
non-contrastive learning, where teacher and student are usu-
ally initialized closely together and only separated through
small asymmetries in architecture and training protocol
(Grill et al., 2020; Caron et al., 2021). We study this locality
effect of the landscape and connect it with the asymmetric
valley phenomenon observed in He et al. (2019).
The improvement in probing accuracy suggests that some
information about the data is incorporated into the network’s
weights. To understand how this information is retained,
we compare the behavior of supervised optimization to fine-
tuning student networks. We find that some of the learning
dynamics observable during the early phase of supervised
training also occur during random teacher distillation. In
particular, the student already contains sparse subnetworks
and reaches the border of linear basins in the supervised loss
landscape. This contrasts (Frankle et al., 2020) where train-
ing on a concrete learning task for a few epochs is essential.
Ultimately, these results suggest that label-independent opti-
mization dynamics exist and allow exploring the supervised
loss landscape to a certain degree.

2. Related Work

Several works in the literature aim to analyze self-distillation
and its impact on the student. Phuong & Lampert (2019)
prove a generalization bound that establishes fast decay of
the risk in the case of linear models. Mobahi et al. (2020)
demonstrate an increasing regularization effect through re-
peated distillation for kernel regression. Ji & Zhu (2020)
consider a similar approach and rely on the fact that very
wide networks behave very similarly to the neural tangent
kernel (Jacot et al., 2018) and leverage this connection to

2

0255075100125150epoch0.10.20.30.40.5probingaccuracy0255075100125150epochteacherstudentinputRandom Teachers are Good Teachers

3. Setting

Notation. Let us set up some notation first. We consider a
family of parametrized functions F = {fθ : Rd −→ Rm(cid:12)
(cid:12)θ ∈
Θ} where θ denotes the (vectorized) parameters of a given
model and Θ refers to the underlying parameter space. In
this work, we study the teacher-student setting, i.e., we
consider two models fθT and fθS from the same function
space F. We will refer to fθT as the teacher model and to
fθS as the student model. Notice that here we assume that
both teacher and student have the same architecture unless
otherwise stated. Moreover, assume that we have access
to n ∈ N input-output pairs (x1, y1), . . . , (xn, yn) i.i.d.∼
D distributed according to some probability measure D,
where xi ∈ Rd and yi ∈ {0, . . . , K − 1} encodes the class
membership for one of the K ∈ N classes.

Supervised. The standard learning paradigm in machine
learning is supervised learning, where a model fθ ∈ F is
chosen based on empirical risk minimization, i.e., given a
loss function l, we train a model to minimize

L(θ) :=

n
(cid:88)

i=1

l(fθ(xi), yi).

Minimization of the objective is usually achieved by virtue
of standard first-order gradient-based methods such as
SGD or ADAM (Kingma & Ba, 2014), where parameters
θ ∼ INIT are randomly initialized and then subsequently
updated based on gradient information.

Teacher-Student Loss. A similar but distinct way to per-
form learning is the teacher-student setting. Here we first fix
a teacher model fθT where θT is usually a parameter con-
figuration arising from training in a supervised fashion on
the same task. The task of the student fθS is then to mimic
the teacher’s behavior on the training set by minimizing a
distance function d between the two predictions,

L(θS) :=

n
(cid:88)

i=1

d (fθS (xi), fθT (xi)) .

(1)

We have summarized the setting schematically in Fig. 2. We
experiment with several choices for the distance function
but largely focus on the KL divergence. We remark that
the standard definition of distillation (Hinton et al., 2015)
consider a combination of losses of the form

L(θS) :=

n
(cid:88)

i=1

d (fθS (xi), fθT (xi))+β

n
(cid:88)

i=1

l(fθS (xi), yi),

for β > 0, thus the objective is also informed by the true
labels y. Here we set β = 0 to precisely test how much per-
formance is solely due to the implicit regularization present
in the learning dynamics and the inductive bias of the model.

Figure 2. Schematic drawing of the teacher-student setup. The
model consists of an encoder and projector. The same image is
passed to both student and teacher, and the outputs of the projectors
are compared. The student weights are then adjusted to mimic the
output of the teacher. In this work, we consider a simplified setting
without augmentations and without teacher updates such as EMA.

Somewhat counter-intuitively, it has been observed in many
empirical works that the resulting student often outperforms
its teacher. It has been hypothesized in many prior works
that the teacher logits fθT (x) encode some additional, rel-
evant information for the task that benefits learning (dark
knowledge), i.e., wrong but similar classes might have a
non-zero probability under the teacher model (Hinton et al.,
2015; Wang et al., 2021; Xu et al., 2018). In the following,
we will explore this hypothesis by systematically destroying
the label information in the teacher.

Non-Contrastive. Self-supervised learning is a recently
developed methodology enabling the pretraining of vision
models on large-scale unlabelled image corpora, akin to the
autoregressive loss in natural language processing (Devlin
et al., 2019). A subset of these approaches is formed by non-
contrastive methods. Consider a set of image augmentations
G where any G ∈ G is a composition of standard augmen-
tation techniques such as random crop, random flip, color
jittering, etc. The goal of non-contrastive learning is to learn
a parameter configuration that is invariant to the employed
data augmentations while avoiding simply collapsing to a
constant function. Most non-contrastive objectives can be
summarized to be of the form

L(θS) :=

n
(cid:88)

i=1

EG1,G2 [d (fθS (G1(xi)), fθT (G2(xi)))] ,

where the expectation is taken uniformly over the set of
augmentations G. We summarize this pipeline schematically

3

Random Teachers are Good Teachers

in Fig. 2. While the teacher does not directly receive any
gradient information, the parameters θT are often updated
based on an exponentially weighted moving average,

θT ←− (1 − γ)θT + γθS

which is applied periodically at a fixed frequency. In this
work, we will consider a simplified setting without aug-
mentations and where the teacher remains frozen at random
initialization, γ = 0.

Probing. Since minimizing the teacher-student loss is a
form of unsupervised learning if the teacher itself has not
seen any labels, we need a way to measure the quality of the
resulting features. Here we rely on the idea of probing repre-
sentations, a very common technique from self-supervised
learning (Chen & He, 2021; Chen et al., 2020; Caron et al.,
2021; Bardes et al., 2021; Grill et al., 2020). As illustrated
in Fig. 2, the network is essentially split into an encoder
gψ : Rd −→ Rr and a projector hϕ : Rr −→ Rm where it
holds that fθ = hϕ ◦gψ. The encoder is usually given by the
backbone of a large vision model such as ResNet (He et al.,
2016) or VGG (Simonyan & Zisserman, 2014), while the
projector is parametrized by a shallow MLP. We then probe
the representations gψ by learning a linear layer on top,
where we now leverage the label information y1, . . . , yn.
Notice that the weights of the encoder remain frozen while
learning the linear layer. The idea is that a linear model does
not add more feature learning capacity, and the resulting
probing accuracy hence provides an adequate measure of
the quality of the representations. Unless otherwise stated,
we perform probing on the CIFAR10 dataset (Krizhevsky &
Hinton, 2009) and aggregate mean and standard deviation
over three runs.

4. Random Teacher Distillation

Distillation. Let us denote by θ ∼ INIT a randomly initial-
ized parameter configuration, according to some standard
initialization scheme INIT. Throughout this text, we rely
on Kaiming initialization (He et al., 2015). In standard self-
distillation, the teacher is a parameter configuration θ(l)
T
resulting from training in a supervised fashion for l ∈ N
epochs on the task {(xi, yi)}n

i=1.

θ(l)
T

In a next step, the teacher is then distilled into a student, i.e.,
the student is trained to match the outputs of the pre-trained
teacher f
. In this work, we change the nature of the
teacher and instead consider a teacher at random initializa-
tion θT ∼ INIT (we drop the superscript 0 for convenience).
The teacher has thus not seen any data at all and is hence
of a similar (bad) quality as the student. This experiment,
therefore, serves as the ideal test bed to measure the implicit
regularization present in the optimization itself without rely-
ing on any dark knowledge about the target distribution. Due

to the absence of targets, the setup also closely resembles
the learning setting of non-contrastive methods. Through
that lens, our experiment can also be interpreted as a non-
contrastive pipeline without augmentations and exponential
moving average.
We minimize the objective (1) with the ADAM optimizer
(Kingma & Ba, 2014) using a learning rate η = 0.001. We
analyze two encoder types based on the popular ResNet18
and VGG11 architectures, and similarly to Caron et al.
(2021), we use a 2-hidden layer MLP with an L2 bottle-
neck, as a projector. To assess whether batch-dependent
statistics play a role, we remove the batch normalization lay-
ers (Ioffe & Szegedy, 2015) from the VGG11 architecture.
For more details on the architecture and hyperparameters,
we refer to App. E.

DATASET

MODEL

TEACHER

STUDENT

INPUT

CIFAR10

CIFAR100

STL10

TinyImageNet

ResNet18
VGG11

ResNet18
VGG11

ResNet18
VGG11

ResNet18
VGG11

35.50
36.55

11.58
12.05

24.24
24.67

4.85
5.25

46.02
51.98

21.50
26.62

40.58
46.20

10.40
12.88

39.02

14.07

31.51

3.28

Table 1. Linear probing accuracies (in percentage) of the represen-
tations for various datasets for teacher, student and flattened input
images. Students outperform the baselines in all cases.

We display the linear probing accuracy of both student and
teacher as a function of training time in Fig. 1. We follow
the protocol of non-contrastive learning and initialize the
student closely to the teacher. We will expand more on this
choice of initialization in the next paragraph. Note that while
the teacher remains fixed throughout training, accuracies can
vary due to stochastic optimization of linear probing. The
dashed line represents the linear probing accuracy obtained
directly from the (flattened) inputs. We clearly see that the
student significantly outperforms its teacher throughout the
training. Moreover, it also improves over probing on the raw
inputs, demonstrating that not simply less signal is lost due
to random initialization but rather that meaningful learning
is performed. We expand our experimental setup to more
datasets, including CIFAR100 (Krizhevsky & Hinton, 2009),
STL10 (Coates et al., 2011) and TinyImageNet (Le & Yang,
2015). We summarize the results in Table 1. We observe that
across all tasks, distilling a random teacher into its student
proves beneficial in terms of probing accuracy. For further
ablations on the projection head, we refer to the App. B.
Moreover, we find similar results for more architectures and
k-NN instead of linear probing in App. C.

4

Random Teachers are Good Teachers

Figure 3. Linear probing accuracies as a function of the locality
parameter α on CIFAR10. The color gradient (bright → dark)
reflects the value of α (0 → 1) for ResNet18 in green and VGG11
in red tones. Left: ResNet18. Middle: VGG11. Right: Summary.

Figure 4. Linear probing accuracies of a VGG11 trained on CI-
FAR5M or Gaussian noise inputs and evaluated on CIFAR10 as a
function of sample size n. Representations are data dependent.

Local Initialization.
It turns out that the initialization of
the student and its proximity to the teacher plays a crucial
role. To that end, we consider initializations of the form

θS(α) =

1
δ

(cid:16)

(cid:17)
(1 − α)θT + α ˜θ

,

where ˜θ ∼ INIT is a fresh initialization, α ∈ [0, 1] and
δ = (cid:112)α2 + (1 − α)2 ensures that the variance remains
constant ∀α ∈ [0, 1]. By increasing α from 0 towards 1,
we can gradually separate the student initialization from the
teacher and ultimately reach the more classical setup of self-
distillation where the student is initialized independently
from the teacher. Note, that in the non-contrastive learning
setting, teacher and student are initialized at the same pa-
rameter values (i.e., α = 0), and only minor asymmetries in
the architectures lead to different overall functions.
We now study how the locality parameter α affects the re-
sulting quality of the representations of the student in our
setup. In Fig. 3, we display the probing accuracy as a func-
tion of the training epoch for different choices of alpha.
Furthermore, we summarize the resulting accuracy of the
student as a function of the locality parameter α. Surpris-
ingly, we observe that random teacher distillation behaves
very similarly for all α ∈ [0, 0.6]. Increasing α more slows
down the convergence and leads to worse overall probing
performance. However, even initializing the student inde-
pendently of the teacher (α = 1) results in a considerable
improvement over the teacher. In other words, we show that
representation learning can occur in self-distillation for any
random teacher without dark knowledge. To the best of our
knowledge, we are the first to observe such a locality phe-
nomenon in the teacher-student landscape. We investigate
this phenomenon in more detail in the next section and, for
now, if not explicitly stated otherwise, use initializations
with small locality parameter α ∼ 10−10. Safran & Shamir
(2017) prove that spurious local minima exist in the teacher-
student loss of two-layer ReLU networks. We speculate that
this might be the reason why initializing students close to
the teacher is beneficial, and provide evidence in App. D

Data-Dependence.
In a next step, we aim to understand
better to which degree the learned features are data depen-
dent, i.e., tuned to the particular input distribution x ∼ px.
While the improvement over the raw input probe already
suggests non-trivial learning, we want to characterize the
role of the input data more precisely.
As a first experiment, we study how the improvement of
the student over the teacher evolves as a function of the
sample size n involved in the teacher-student training phase.
We use the CIFAR5M dataset, where the standard CIFAR10
dataset has been extended to 5 million data points using a
generative adversarial network (Nakkiran et al., 2021). We
train the student for different sample sizes in the interval
[5 × 102, 5 × 106] and probe the learned features on the
standard CIFAR10 training and test set. We display the
resulting probing accuracy as a function of sample size in
Fig. 4 (blue line). Indeed, we observe a steady increase in
the performance of the student as the size of the data corpus
grows, highlighting that data-dependent feature learning is
happening.
As further confirmation, we replace the inputs xi ∼ px
with pure Gaussian noise, i.e. xi ∼ N (0, σ21), effectively
removing any relevant structure in the samples. The lin-
ear probing, on the other hand, is again performed on the
clean data. This way, we can assess whether the teacher-
student training is simply moving the initialization in a fa-
vorable way (e.g. potentially uncollapsing it), which would
still prove beneficial for meaningful tasks. We display the
probing accuracy for these random inputs in Fig. 4 as well
(orange line) and observe that such random input training
does not lead to an improvement of the student across all
dataset sizes. This is another indication that data-dependent
feature learning is happening, where in this case, adapting
to the noise inputs of course proves detrimental for the clean
probing.

Transferability. As a final measure for the quality of
the learned features, we test how well a set of represen-
tations obtained on one task transfers to a related but dif-

5

050100150epoch0.10.20.30.40.5probingaccuracy050100150epoch00.51αVGG11ResNet18teachers103104105106datasetsize0.200.300.400.50probingaccuracyCIFAR5MrandomdatainputRandom Teachers are Good Teachers

DATASET

MODEL

TEACHER

STUDENT

CIFAR10

CIFAR100

STL10

ResNet18
VGG11

ResNet18
VGG11

ResNet18
VGG11

35.50
36.55

11.58
12.05

24.24
24.67

46.06
52.45

22.60
27.49

41.42
45.86

Table 2. Linear probing accuracies of the representations for vari-
ous datasets for teacher and student. Students distilled from ran-
dom teachers on TinyImageNet generalize out of distribution.

i=1

i=1

ferent task. More precisely, we are given a source task
i.i.d.∼ DA and a target task B =
A = {(xi, yi)}n
i.i.d.∼ DB and assume that both tasks are re-
{(xi, yi)}˜n
lated, i.e., some useful features on A also prove to be useful
on task B. We first use the source task A to perform random
teacher distillation and then use the target task B to train
and evaluate the linear probe. Clearly, we should only see
an improvement in the probing accuracy over the (random)
teacher if the features learned on the source task encode
relevant information for the target task as well. We use
TinyImageNet as the source task and evaluate on CIFAR10,
CIFAR100, and STL10 as target tasks for our experiments.
We illustrate the results in Table 2 and observe that transfer
learning occurs. This suggests that the features learned by
random teacher distillations can encode common properties
of natural images which are shared across tasks.

5. Loss and Probing Landscapes

Visualization. We now revisit the locality property iden-
tified in the previous section, where initializations with α
closer to zero outperformed other configurations. To gain
further insight into the inner workings of this phenomenon,
we visualize the teacher-student loss landscape as well as the
resulting probing accuracies as a function of the model pa-
rameters. Since the loss function is a very high-dimensional
function of the parameters, only slices of it can be visual-
ized at once. More precisely, given two directions v1, v2 in
parameter space, we form a visualization plane of the form

θ(λ1, λ2) = λ1v1 + λ2v2,

(λ1, λ2) ∈ [0, 1]2

and then collect loss and probing values at a certain reso-
lution. Such visualization strategy is very standard in the
literature, see e.g., Li et al. (2018); Garipov et al. (2018);
Izmailov et al. (2021). Denote by θ∗
S(α) the student trained
until convergence initialized with locality parameter α. We
study two choices for the landscape slices. First, we refer to
a non-local view as the plane defined by the random teacher
θT , the student at a fresh initialization θS(1) and the result-

6

Figure 5. Visualization of the loss and probing landscape. The left
column corresponds to the non-local view with α = 1. The right
column depicts the shared view, containing both the local (α = 0)
and the non-local solution (α = 1). The first row displays the loss
landscape and the second one shows the probing landscape. Con-
tours lines represent ||θ||2, orthogonal projections are in App. C.3.

ing trained student θ∗
S(1), i.e., we set v1 = θS(1) − θT and
v2 = θ∗
S(1) − θT . As a second choice, we refer to a shared
view as the plane defined by the random teacher θT , the
trained student starting from a fresh initialization θ∗
S(1) and
the trained student θ∗
S(0) initialized closely to the teacher,
i.e., we set v1 = θ∗
S(0) − θT and v2 = θ∗
S(1) − θT . Note
that α is not exactly zero but around 10−10.

We show the results in Fig. 5, where the left and right
columns represent the non-local and the shared view re-
spectively, while the first and the second row display loss
and probing landscapes respectively. Let us focus on the
non-local view first. Clearly, for α = 1 the converged stu-
dent θ∗
S(1) ends up in a qualitatively different minimum
than the teacher, i.e., the two points are separated by a
significant loss barrier. This is expected as the student is
initialized far away from the teacher. Further, we see that
the probing landscape is largely unaffected by moving from
the initialization θS(0) to the solution θ∗
S(0), confirming
our empirical observation in Fig. 3 that far way initialized
students only improve slightly. The shared view reveals
more structure. We see that although it was initialized very
closely to the teacher, the student θ∗
S(0) moved consider-
ably. While the loss barrier is lower as in the case of θ∗
S(1),
it is still very apparent that θ∗
S(0) settled for a different,
local minimum that coincides with a region of high probing
accuracy. This is surprising as the teacher itself is the global
loss minimum. For more visualizations, including the loss
landscape for the encoder, we refer to App. C.3.

0.000.250.500.751.00losslandscapeθTθS(1)θ∗S(1)non-localview9090120120150150180210240270300θTθ∗S(0)θ∗S(1)sharedview901001001101101201201301301401400.00.51.00.000.250.500.751.00probinglandscapeθTθS(1)θ∗S(1)90901201201501501802102402703000.00.51.0θTθ∗S(0)θ∗S(1)9010010011011012012013013014014010−710−610−510−410−30.10.20.30.40.5Random Teachers are Good Teachers

Figure 6. Illustration of the lottery ticket hypothesis and iterative
magnitude-based pruning.

Asymmetric valleys. A striking structure in the loss land-
scape of the shared view is the very pronounced asymmetric
valley around the teacher θT . While there is a very steep
increase in loss towards the left of the view (dark blue), the
loss increases only gradually in the opposite direction (light
turquoise) and quickly decreases into the local minimum
of the converged student θ∗
S(0). Surprisingly, this direction
orthogonal to the cliff identifies a region of high accuracy in
the probing landscape. A fact remarkably in line with this
situation is proven by He et al. (2019). They show that be-
ing on the flatter side of an asymmetric valley (i.e., towards
θ∗
S(0)) provably leads to better generalization compared to
lying in the valley itself (i.e., θT ). Initializing the student
closely to the teacher seems to capitalize on that fact and
leads to systematically better generalization. Still, it remains
unclear why such an asymmetric valley is only encountered
close to the teacher and not for initializations with α = 1.
We leave a more in-depth analysis of this phenomenon for
future work.

6. Connection to Supervised Optimization

Lottery Tickets. A way to assess the structure present in
neural networks is through sparse network discovery, i.e.,
the lottery ticket hypothesis. The lottery ticket hypothesis
by Frankle & Carbin (2018) posits the following: Any large
network possesses a sparse subnetwork that can be trained
as fast and which achieves or surpasses the test error of
the original network. They prove this using the power of
hindsight and discover such sparse networks through the
following iterative pruning strategy:

1. Fix an initialization θ(0) ∼ INIT and train a network
to convergence in a supervised fashion, leading to θ∗.

2. Prune the parameters based on some criterion, leading
to a binary mask m and pruned parameters m ⊙ θ∗.

3. Prune the initialized network m ⊙ θ(0) and re-train it.

The above procedure is repeated for a fixed number of

7

Figure 7. Illustration of stability of SGD and linear mode-
connectivity. Blue contour lines indicate a basin of low test loss,
πi denote different batch orderings for SGD.

times r, and in every iteration, a fraction k ∈ [0, 1] of
the weights is pruned, leading to an overall pruning rate
of pr = (cid:80)r−1
i=0 (1 − k)i × k percentage of weights. We
illustrate the algorithm in Fig. 6. The choice of pruning
technique is flexible, but in the common variant iterative
magnitude pruning (IMP), the globally smallest weights are
pruned. The above recipe turns out to work very well for
MLPs and smaller convolutional networks, and indeed very
sparse solutions can be discovered without any deterioration
in terms of training time or test accuracy (Frankle & Carbin,
2018). However, for more realistic architectures such as
ResNets, the picture changes and subnetworks can only be
identified if the employed learning rate is low enough. Sur-
prisingly, Frankle et al. (2019) find that subnetworks in such
architectures develop very early in training and thus add
the following modification to the above strategy: Instead of
rewinding back to the initialization θ(0) and applying the
pruning there, another checkpoint θ(l) early in training is
used and m ⊙ θ(l) is re-trained instead of m ⊙ θ(0).
Frankle et al. (2019) demonstrate that checkpoints as early
as 1 epoch can suffice to identify lottery tickets, even at
standard learning rates. Interestingly, Frankle et al. (2019)
further show that the point in time l where lottery tickets
can be found coincides with the time where SGD becomes
stable to different batch orderings π, i.e., different runs of
SGD with distinct batch orderings but the same initialization
θ(l) end up in the same linear basin. This property is also
called linear mode connectivity; we provide an illustration
in Fig. 7. Notice that in general, linear mode-connectivity
does not hold, i.e., two SGD runs from the same initializa-
tion end up in two disconnected basins (Frankle et al., 2019;
Garipov et al., 2018).

IMP from the Student. A natural question that emerges
now is whether rewinding to a student checkpoint θ∗
S, ob-
tained through random teacher distillation, already devel-
oped sparse structures in the form of lottery tickets. We com-

Random Teachers are Good Teachers

Figure 8. Test accuracy as a function of sparsity for different ini-
tialization and rewinding strategies. Fresh initializations θS are
not robust to IMP with rewinding to initialization (l = 0), this
only emerges with rewinding to l ≥ 1. Student checkpoints θ∗S are
always robust to IMP even with rewinding to l = 0. One epoch
corresponds to 196 steps. Aggregation is done over 5 checkpoints.

pare the robustness of our student checkpoints θ∗
S with ran-
dom initialization at different rewinding points θ(l), closely
following the setup in Frankle et al. (2019). We display
the results in Fig. 8, where we plot test performance on CI-
FAR10 as a function of the sparsity level. We use a ResNet18
and iterative magnitude pruning, reducing the network by a
fraction of 0.2 every round. We compare against rewinding
to supervised checkpoints θ(l) for l ∈ {0, 1, 2, 5} where l is
measured in number of epochs.
We observe that rewinding to random initialization (l = 0),
as shown in Frankle & Carbin (2018); Frankle et al. (2019),
incurs strong losses in terms of test accuracy at all pruning
levels and thus θS does not constitute a lottery ticket. The
distilled student θ∗
S, on the other hand, contains a lottery
ticket, as it remains very robust to strong degrees of pruning.
In fact, θ∗
S shows similar behavior to the networks rewound
to epoch 1 and 2 in supervised training. This suggests that
random teacher distillation imitates some of the learning
dynamics in the first epochs of supervised optimization. We
stress here that no label information was required for sparse
subnetworks to develop. This aligns with results in (Frankle
et al., 2020), showing that auxiliary tasks such as rotation
prediction can lead to lottery tickets. However, this is no
surprise, as Anagnostidis et al. (2022) show that the data-
informed bias of augmentations can already lead to strong
forms of learning. We believe our result is more powerful
since random teacher distillation relies solely on implicit
regularization in SGD and does not require a task at all.

Linear Mode Connectivity.
In light of the observation
regarding the stability of SGD in Frankle et al. (2019), we
verify whether a similar stability property holds for the stu-
dent checkpoint θ∗
S. To that end, we train several runs of
SGD in a supervised fashion with initialization θ∗
S on differ-

8

Figure 9. Test error when interpolating between networks that were
trained from the same initialization. Left: Networks initialized at
the teacher location, i.e., random initialization. Right: Networks
initialized at the converged student θ∗S(0). Aggregation is done
over 3 initializations and 5 different data orderings πi.

ent batch orderings π1, . . . , πb and study the test accuracies
occurring along linear paths between different solutions θ∗
πi
for i = 1, . . . , b, i.e.

θπi−→πj (γ) := γθ∗
πi

+ (1 − γ)θ∗
πj

.

and θ∗
πj

If the test accuracy along the path does not significantly
worsen, we call θ∗
linearly mode-connected. We
πi
contrast the results with the interpolation curves for SGD
runs started from the original, random initialization θS. We
display the interpolation curves in Fig. 9, where we used
three ResNet18 student checkpoints and finetuned each in
five SGD runs with different seeds on CIFAR10. We observe
that, indeed, the resulting parameters θ∗
all lie in approxi-
πi
mately the same linear basin. However, the networks trained
from the random initialization face a significantly larger
barrier. This confirms that random teacher distillation con-
verges towards parameterizations θ∗
S, which are different
from those at initialization θS. In particular, such θ∗
S would
only appear later in supervised optimization when SGD is al-
ready more stable to noise. Ultimately, it shows that random
teacher distillation obeys similar dynamics as supervised
optimization and can navigate toward linear basins of the
supervised loss landscape.

7. Discussion and Conclusion

In this work, we examined the teacher-student setting to
disentangle its implicit regularization from other very com-
mon components such as dark knowledge in trained teachers
or data augmentations in self-supervised learning. Surpris-
ingly, students learned strong structures even from random
teachers in the absence of data augmentation. We studied
the quality of the students and observed that (1) probing
accuracies significantly improve over the teacher, (2) fea-
tures are data-dependent and transferable across tasks, and
(3) student checkpoints develop sparse subnetworks at the
border of linear basins without training on a supervised task.

100.051.226.213.46.93.51.80.90.5weightsremaining(%)0.900.910.920.930.940.95testaccuracyinitθ∗S,rewindto‘=0initθS,rewindto‘=0initθS,rewindto‘=1initθS,rewindto‘=2initθS,rewindto‘=5θ∗πiθ∗πj0.10.20.30.40.5testerrorﬁnetuneθSθ∗πiθ∗πjﬁnetuneθ∗SRandom Teachers are Good Teachers

The success of teacher-student frameworks such as knowl-
edge distillation and non-contrastive learning can thus at
least partially be attributed to the regularizing nature of
the learning dynamics. These label-independent dynamics
allow the student to mimic the early phase of supervised
training by navigating the supervised loss landscape without
label information. The simple and minimal nature of our
setting makes it an ideal test bed for better understanding
this early phase of learning. We hope that future theoretical
work can build upon our simplified framework.

Acknowledgements

We thank Sidak Pal Singh for his valuable insights and
interesting discussions on various aspects of the topic.

References

Allen-Zhu, Z. and Li, Y. Towards Understanding En-
semble, Knowledge Distillation and Self-Distillation in
In 11th International Conference on
Deep Learning.
Learning Representations (ICLR), 2 2020. URL https:
//arxiv.org/abs/2012.09816.

Anagnostidis, S., Bachmann, G., Noci, L., and Hofmann,
T. The Curious Case of Benign Memorization. In 11th
International Conference on Learning Representations
(ICLR), 2022. doi: 10.48550/arxiv.2210.14019. URL
https://arxiv.org/abs/2210.14019.

Assran, M., Caron, M., Misra, I., Bojanowski, P., Bordes,
F., Vincent, P., Joulin, A., Rabbat, M., and Ballas, N.
Masked Siamese Networks for Label-Efficient Learning.
In European Conference on Computer Vision (ECCV),
2022. doi: 10.48550/arxiv.2204.07141. URL https:
//arxiv.org/abs/2204.07141.

Ba, J. and Caruana, R. Do deep nets really need to
In 28th Conference on Neural Information
be deep?
Processing Systems (NeurIPS), 2013. URL https:
//arxiv.org/abs/1312.6184.

Bardes, A., Ponce,

J., and LeCun, Y.

VICReg:
Variance-Invariance-Covariance Regularization for Self-
In 10th International Confer-
Supervised Learning.
ence on Learning Representations (ICLR), 2021. doi:
10.48550/arxiv.2105.04906. URL https://arxiv.
org/abs/2105.04906.

Benzing, F., Schug, S., Ch, S., Meier, R., Von Oswald, J.,
Ch, V., Akram, Y., Zucchet, N., Aitchison, L., Steger,
A., and Ch, S. E. Random initialisations performing
above chance and how to find them. ArXiv, 9 2022. URL
https://arxiv.org/abs/2209.07509.

Beyer, L., Zhai, X., Royer, A., Markeeva, L., Anil, R., and
Kolesnikov, A. Knowledge distillation: A good teacher

9

is patient and consistent. In IEEE/CVF Conference on
Computer Vision and Pattern Recognition (CVPR), 2022.
URL https://arxiv.org/abs/2106.05237.

Bucilˇa, C., Caruana, R., and Niculescu-Mizil, A. Model
In ACM International Conference on
compression.
Knowledge Discovery and Data Mining (SIGKDD),
2006. URL https://dl.acm.org/doi/abs/10.
1145/1150402.1150464.

Caron, M., Touvron, H., Misra, I., J´egou, H., Mairal, J.,
Bojanowski, P., and Joulin, A. Emerging Properties in
Self-Supervised Vision Transformers. In IEEE/CVF Inter-
national Conference on Computer Vision (ICCV), 2021.
URL https://arxiv.org/abs/2104.14294.

Chen, T., Kornblith, S., Norouzi, M., and Hinton, G. A Sim-
ple Framework for Contrastive Learning of Visual Repre-
sentations. In 37th International Conference on Machine
Learning (ICML), 2020.
ISBN 9781713821120. doi:
10.48550/arxiv.2002.05709. URL https://arxiv.
org/abs/2002.05709.

Chen, X. and He, K. Exploring Simple Siamese Repre-
sentation Learning. In IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition, 2021.
ISBN
9781665445092. doi: 10.48550/arxiv.2011.10566. URL
https://arxiv.org/abs/2011.10566.

Coates, A., Ng, A., and Lee, H.

An analysis of
single-layer networks in unsupervised feature learn-
In 14th International Conference on Artificial
ing.
Intelligence and Statistics (AISTATS). PMLR, 2011.
URL https://proceedings.mlr.press/v15/
coates11a.html.

Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. BERT:
Pre-training of deep bidirectional transformers for lan-
guage understanding. In Conference of the North Amer-
ican Chapter of the Association for Computational Lin-
guistics: Human Language Technologies, 2019. doi:
10.18653/v1/N19-1423. URL https://arxiv.org/
abs/1810.04805.

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn,
D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer,
M., Heigold, G., Gelly, S., et al. An image is worth
16x16 words: Transformers for image recognition at scale.
In 9th International Conference on Learning Represen-
tations (ICLR), 2020. URL https://arxiv.org/
abs/2010.11929.

Draxler, F., Veschgini, K., Salmhofer, M., and Hamprecht,
F. A. Essentially No Barriers in Neural Network Energy
Landscape. In 35th International Conference on Machine
Learning (ICML), 2018. ISBN 9781510867963. URL
https://arxiv.org/abs/1803.00885.

Random Teachers are Good Teachers

Frankle, J. and Carbin, M. The Lottery Ticket Hypothe-
sis: Finding Sparse, Trainable Neural Networks. In 7th
International Conference on Learning Representations
(ICLR), 2018. doi: 10.48550/arxiv.1803.03635. URL
https://arxiv.org/abs/1803.03635.

Frankle, J., Dziugaite, G. K., Roy, D. M., and Carbin, M.
Linear Mode Connectivity and the Lottery Ticket Hy-
pothesis. In 37th International Conference on Machine
Learning (ICML), 2019. URL https://arxiv.org/
abs/1912.05671.

Frankle, J., Schwab, D. J., and Morcos, A. S. The Early
In 8th Interna-
Phase of Neural Network Training.
tional Conference on Learning Representations (ICLR),
2020. doi: 10.48550/arxiv.2002.10365. URL https:
//arxiv.org/abs/2002.10365.

Furlanello, T., Lipton, Z. C., Tschannen, M., Itti, L., and
Anandkumar, A. Born Again Neural Networks. 35th
International Conference on Machine Learning (ICML),
2018. doi: 10.48550/arxiv.1805.04770. URL https:
//arxiv.org/abs/1805.04770.

Garipov, T., Izmailov, P., Podoprikhin, D., Vetrov, D., and
Wilson, A. G. Loss Surfaces, Mode Connectivity, and
Fast Ensembling of DNNs. In 32nd Conference on Neural
Information Processing Systems (NeurIPS), 2018. doi:
10.48550/arxiv.1802.10026. URL https://arxiv.
org/abs/1802.10026.

Grill, J. B., Strub, F., Altch´e, F., Tallec, C., Richemond,
P. H., Buchatskaya, E., Doersch, C., Pires, B. A., Guo,
Z. D., Azar, M. G., Piot, B., Kavukcuoglu, K., Munos,
R., and Valko, M. Bootstrap your own latent: A new
approach to self-supervised Learning. In 34th Conference
on Neural Information Processing Systems (NeurIPS),
2020. doi: 10.48550/arxiv.2006.07733. URL https:
//arxiv.org/abs/2006.07733.

He, H., Huang, G., and Yuan, Y. Asymmetric Val-
In 33rd
leys: Beyond Sharp and Flat Local Minima.
Conference on Neural Information Processing Systems
(NeurIPS), 2019. URL https://arxiv.org/abs/
1902.00744.

He, K., Zhang, X., Ren, S., and Sun, J. Delving deep
into rectifiers: Surpassing human-level performance on
imagenet classification. In IEEE International Confer-
ence on Computer Vision (ICCV), 2015. URL https:
//arxiv.org/abs/1502.01852.

He, K., Zhang, X., Ren, S., and Sun, J. Deep residual
learning for image recognition. In IEEE Conference on
Computer Vision and Pattern Recognition (CVPR), 2016.
URL https://arxiv.org/abs/1512.03385.

10

Hinton, G., Vinyals, O., and Dean, J. Distilling the
Knowledge in a Neural Network. ArXiv, 2015. doi:
10.48550/arxiv.1503.02531. URL https://arxiv.
org/abs/1503.02531.

Ioffe, S. and Szegedy, C. Batch normalization: Accelerating
deep network training by reducing internal covariate shift.
In 32nd International Conference on Machine Learning
(ICML), 2015. URL https://proceedings.mlr.
press/v37/ioffe15.html.

Izmailov, P., Vikram, S., Hoffman, M. D., and Wilson, A. G.
What Are Bayesian Neural Network Posteriors Really
Like? 2021.

Jacot, A., Gabriel, F., and Hongler, C. Neural tangent
kernel: Convergence and generalization in neural net-
In 32nd Conference on Neural Information
works.
Processing Systems (NeurIPS), 2018. URL https:
//arxiv.org/abs/1806.07572.

Ji, G. and Zhu, Z. Knowledge distillation in wide neu-
ral networks: Risk bound, data efficiency and imper-
In 34th Conference on Neural Informa-
fect teacher.
tion Processing Systems (NeurIPS), 2020. URL https:
//arxiv.org/abs/2010.10090.

Kingma, D. P. and Ba, J. Adam: A method for stochas-
In 3rd International Conference on
tic optimization.
Learning Representations (ICLR), 2014. URL https:
//arxiv.org/abs/1412.6980.

Krizhevsky, A. and Hinton, G.
features

from tiny images.

layers of
report, University of Toronto, 2009.
cal
https://www.cs.toronto.edu/˜kriz/
learning-features-2009-TR.pdf.

Learning multiple
Techni-
URL

Le, Y. and Yang, X. Tiny imagenet visual recognition chal-
lenge. Technical report, Stanford University, 2015. URL
http://vision.stanford.edu/teaching/
cs231n/reports/2015/pdfs/yle_project.
pdf.

Li, H., Xu, Z., Taylor, G., Studer, C., and Goldstein, T.
Visualizing the loss landscape of neural nets. In 32nd
Conference on Neural Information Processing Systems
(NeurIPS), 2018. URL https://arxiv.org/abs/
1712.09913.

Mobahi, H., Farajtabar, M., and Bartlett, P. L. Self-
Distillation Amplifies Regularization in Hilbert Space. In
34th Conference on Neural Information Processing Sys-
tems (NeurIPS), 2020. doi: 10.48550/arxiv.2002.05715.
URL https://arxiv.org/abs/2002.05715.

Random Teachers are Good Teachers

Wang, Y., Li, H., Chau, L.-p., and Kot, A. C. Embrac-
ing the dark knowledge: Domain generalization using
In 29th ACM In-
regularized knowledge distillation.
ternational Conference on Multimedia, 2021. doi: 10.
1145/3474085.3475434. URL https://doi.org/
10.1145/3474085.3475434.

Xu, K., Park, D. H., Yi, C., and Sutton, C.

Interpret-
ing deep classifier by visual distillation of dark knowl-
edge. ArXiv, 2018. URL https://arxiv.org/
abs/1803.04042.

Yang, C., Xie, L., Su, C., and Yuille, A. L. Snapshot
distillation: Teacher-student optimization in one gener-
In IEEE/CVF Conference on Computer Vision
ation.
and Pattern Recognition (CVPR), 2018. URL https:
//arxiv.org/abs/1812.00123.

Yim, J., Joo, D., Bae, J., and Kim, J. A gift from
knowledge distillation: Fast optimization, network min-
In IEEE Conference
imization and transfer learning.
on Computer Vision and Pattern Recognition (CVPR),
2017. URL https://ieeexplore.ieee.org/
document/8100237.

Yuan, L., Tay, F. E., Li, G., Wang, T., and Feng, J. Re-
visiting knowledge distillation via label smoothing reg-
In IEEE/CVF Conference on Computer
ularization.
Vision and Pattern Recognition (CVPR), 2020. URL
https://arxiv.org/abs/1909.11723.

Zaidi, S., Berariu, T., Kim, H., Bornschein, J., Clopath, C.,
Teh, Y. W., and Pascanu, R. When Does Re-initialization
Work? Understanding Deep Learning Through Empirical
Falsification (NeurIPS Workshop), 6 2022. URL https:
//arxiv.org/abs/2206.10011.

Zbontar, J., Jing, L., Misra, I., LeCun, Y., and Deny, S.
Barlow Twins: Self-Supervised Learning via Redundancy
Reduction. In 38th International Conference on Machine
Learning (ICML), 2021. doi: 10.48550/arxiv.2103.03230.
URL https://arxiv.org/abs/2103.03230.

Zhang, C., Zhang, K., Zhang, C., Pham, T. X., Yoo,
C. D., and Kweon, I. S. How Does SimSiam Avoid
Collapse Without Negative Samples? A Unified Un-
derstanding with Self-supervised Contrastive Learning.
In 10th International Conference on Learning Represen-
tations (ICLR), 2022. URL https://arxiv.org/
abs/2203.16262.

Nakkiran, P., Neyshabur, B., and Sedghi, H. The deep
bootstrap framework: Good online learners are good of-
fline generalizers. In 9th International Conference on
Learning Representations (ICLR), 2021. URL https:
//openreview.net/forum?id=guetrIHLFGI.

Phuong, M. and Lampert, C. Towards understanding
In 36th International Confer-
knowledge distillation.
ence on Machine Learning (ICML), 2019. URL https:
//arxiv.org/abs/2105.13093.

Polino, A., Pascanu, R., and Alistarh, D. Model compres-
In 6th Interna-
sion via distillation and quantization.
tional Conference on Learning Representations (ICLR),
2018. URL https://openreview.net/forum?
id=S1XolQbRW.

Safran, I. and Shamir, O. Spurious Local Minima are Com-
mon in Two-Layer ReLU Neural Networks. 35th Inter-
national Conference on Machine Learning (ICML), 2017.
URL https://arxiv.org/abs/1712.08968.

Schroff, F., Kalenichenko, D., and Philbin, J. Facenet: A
unified embedding for face recognition and clustering.
In IEEE Conference on Computer Vision and Pattern
Recognition (CVPR), 2015. doi: 10.1109/CVPR.2015.
7298682. URL https://arxiv.org/abs/1503.
03832.

Simonyan, K. and Zisserman, A. Very deep convolu-
In
tional networks for large-scale image recognition.
3rd International Conference on Learning Representa-
tions (ICLR), 2014. URL https://arxiv.org/
abs/1409.1556.

Stanton, S., Izmailov, P., Kirichenko, P., Alemi, A. A., and
Wilson, A. G. Does Knowledge Distillation Really Work?
In 35th Conference on Neural Information Processing
Systems (NeurIPS), 2021. ISBN 9781713845393. URL
https://arxiv.org/abs/2106.05945.

Tian, Y., Chen, X., and Ganguli, S. Understanding self-
supervised Learning Dynamics without Contrastive Pairs.
In 38th International Conference on Machine Learning
(ICML), 2021. doi: 10.48550/arxiv.2102.06810. URL
https://arxiv.org/abs/2102.06810.

van den Oord, A., Li, Y., and Vinyals, O. Representation
learning with contrastive predictive coding. ArXiv, 2018.
URL https://arxiv.org/abs/1807.03748.

Wang, X., Chen, X., Du, S. S., and Tian, Y. Towards de-
mystifying representation learning with non-contrastive
self-supervision. ArXiv, 2022. URL https://arxiv.
org/abs/2110.04947.

11

A. The Algorithm

Random Teachers are Good Teachers

Distillation from a random teacher has two important details. The outputs are very high-dimensional, 216-d. And a special
component, the l2-bottleneck, is hidden in the architecture of the projection head just before the softmax. It linearly maps a
feature vector to a low-dimensional space, normalizes it, and computes the dot product with a normalized weight matrix, i.e.

x → ˜V T W T x + b
||W T x + b||2

with || ˜V:,i||2 = 1

for x ∈ Rn, W ∈ Rn×k, b ∈ Rk, ˜V ∈ Rk×m. This architecture is heavily inspired by DINO (Caron et al., 2021). Let us
summarize the method in pseudo-code:

1 e n c o d e r , head , w n l a y e r = ResNet ( 5 1 2 ) , MLP( 2 0 4 8 , 2 0 4 8 , 2 5 6 ) , L i n e a r ( 216 )

3 s t u d e n t = i n i t i a l i z e ( e n c o d e r , head , w n l a y e r )

5

7

9

11

13

15

17

19

21

23

25

t e a c h e r = copy ( s t u d e n t ) # i n i t i a l i z e w i t h same p a r a m e t e r s
f o r x , y i n r e p e a t ( d a t a , n e p o c h s ) :
# a p p l y w e i g h t − n o r m a l i z a t i o n
n o r m a l i z e d w e i g h t
t = n o r m a l i z e ( t e a c h e r . w n l a y e r . w e i g h t )
n o r m a l i z e d w e i g h t s = n o r m a l i z e ( s t u d e n t . w n l a y e r . w e i g h t )

t a r g e t

# p r e p a r e
x t = t e a c h e r . h e a d ( t e a c h e r . e n c o d e r ( x ) )
x t = n o r m a l i z e ( x t )
x t = d o t ( n o r m a l i z e d w e i g h t
t a r g e t = s o f t m a x ( x t )

t , x t )

# p r e p a r e p r e d i c t i o n
x s = s t u d e n t . h e a d ( s t u d e n t . e n c o d e r ( x ) )
x s = n o r m a l i z e ( x s )
x s = d o t ( n o r m a l i z e d w e i g h t s , x s )
p r e d i c t i o n = s o f t m a x ( x s )

# compute l o s s , b a c k p r o p a g a t e and u p d a t e
l o s s = sum ( t a r g e t * − l o g ( p r e d i c t i o n ) ) # c r o s s − e n t r o p y
l o s s . b a c k w a r d ( )
o p t i m i z e r . s t e p ( s t u d e n t ) # u p d a t e o n l y s t u d e n t

Figure 10. Comparing different output dimensions m of the projection head. Large m = 216 are not crucial for feature learning, but there
is phase transition at the bottleneck dimension m = 28 = 256 Linear probing on CIFAR10. Left: ResNet18 (red). Right: VGG11 (green).

12

0255075100epoch0.10.20.30.40.5probingaccuracy0255075100epoch2022242628210212214216mResNet18VGG11Random Teachers are Good Teachers

B. Ablating the Projector

B.1. Ablating Normalization Layers

If the teacher is used in evaluation mode, then one possible source of asymmetry is introduced by batch normalization layers.
But is the effect caused by this batch-dependent signal? Or does the batch dependency amplify the mechanism? In Fig. 11
we compare different types of normalization layers and no normalization (Identity). We observe that although BN stabilizes
training, the effect also occurs with batch-independent normalization. Further, networks without normalization reach similar
performance but take longer to converge.

Figure 11. Comparing different types of normalization layers on CIFAR10. Left: ResNet18. Right: VGG11.

B.2. Ablating the L2-Bottleneck

The l2-Bottleneck is a complex layer with many unexplained design choices. We compare different combinations of
weight-normalization (wn), linear layer (lin), and feature normalization (fn) for the first and second part of the bottleneck in
Figures 12 for a ResNet18 and a VGG11 respectively. While the default setup is clearly the most performant, removing
feature normalization is more destructive than removing weight normalization. In particular, only one linear layer followed
by a feature normalization still exhibits a similar trend and does not break down.

Figure 12. Ablating components of the l2-bottleneck on CIFAR10. Left: ResNet18. Right: VGG11.

13

020406080100epoch0.10.20.30.40.5probing accuracyresnet18020406080100epochvgg11BatchNormInstanceNormGroupNorm8LayerNormIdentity0255075100epoch0.10.20.30.40.5probing accuracy0255075100epochwn1 lin1 fn1 wn2 lin2 fn2C. Additional Results

Random Teachers are Good Teachers

We present additional experimental results that serve to better understand the regularization properties of self-distillation
with random teachers.

C.1. K-NN probing

A different probing choice, instead of learning a linear layer on top of the extracted embeddings, is to perform K-NN
classification on the features. We apply K-nearest-neighbour classification with the number of neighbors set to K = 20, as
commonly done in practice. As in Table 1 in the main text, we present results under K-NN evaluation in Table 3. Also, as in
Table 2, we evaluate using K-NN probing the transferability of the learned embeddings from TinyImageNet in Table 4.

DATASET

MODEL

TEACHER

STUDENT

INPUT

CIFAR10

CIFAR100

STL10

TinyImageNet

ResNet18
VGG11

ResNet18
VGG11

ResNet18
VGG11

ResNet18
VGG11

37.65
44.92

13.77
18.10

31.71
36.92

4.59
5.98

44.67
51.32

20.22
23.53

37.41
43.58

7.11
9.23

33.61

14.87

28.94

3.44

Table 3. K-NN probing accuracies (in percentage) of the representations for various datasets for teacher, student, and raw pixel inputs.

DATASET

MODEL

TEACHER

STUDENT

CIFAR10

CIFAR100

STL10

ResNet18
VGG11

ResNet18
VGG11

ResNet18
VGG11

37.65
44.92

13.77
18.10

31.71
36.92

44.45
51.48

19.48
23.95

38.86
42.26

Table 4. K-NN probing accuracies (in percentage) of the representations for various datasets for teacher and student when transferred
from TinyImageNet.

14

C.2. Architectures

Random Teachers are Good Teachers

For our experiments in the main text, we used the very common VGG11 and ResNet18 architectures. Here, we report results
for different types of architectures to provide a better picture of the relevance of architectural inductive biases. In particular,
we compare with the Vision Transformer (ViT) (Dosovitskiy et al., 2020) (patch size 8 for 32 × 32 images of CIFAR10) and
find that the effect of representation learning is still present, albeit less pronounced. More generally, we observe that with
less inductive bias, the linear probing accuracy diminishes but never breaks down.

MODEL

#PARAMS

TEACHER

STUDENT

NONE (INPUT)

0

VGG11
VGG13
VGG16
VGG19

ResNet20*
ResNet56*
ResNet18
ResNet34
ResNet50

ViT-Tiny
ViT-Small
ViT-Medium
ViT-Base

9′220′480
9′404′992
14′714′688
20′024′384

271′824
855′120
11′168′832
21′276′992
23′500′352

594′048
2′072′832
3′550′208
7′684′608

39.02

36.55
34.73
33.08
30.84

28.68
14.05
35.50
28.18
19.69

32.93
38.57
41.09
41.71

39.02

51.98
49.26
46.35
43.90

36.62
27.92
46.02
41.04
27.53

35.76
41.68
43.13
44.38

Table 5. Linear probing accuracies (in percentage) of the representations for various architectures for teacher, student, and flattened inputs
on CIFAR10. ResNet20* and ResNet56* are the smaller CIFAR-variants from He et al. (2016). The students outperform their teachers in
all cases.

15

C.3. Loss landscapes

Random Teachers are Good Teachers

The parameter plane visualized in Fig. 5 is defined by interpolation between three parameterizations, thus, distances and
angles are not preserved. In the following Fig. 13, we orthogonalize the basis of the parameter plane to achieve a distance
and angle-preserving visualization. We note that both converged solutions of the students θ∗
S(1) stay comparably
close to their initializations. Further, we provide a zoomed crop of the asymmetric valley around the teacher θST in Fig. 14.

S(0) and θ∗

Figure 13. Orthogonal projection of the loss landscape in the parameter plane.

Figure 14. Higher resolution crop of the global optimum around the teacher.

16

−60−40−200204060losslandscapeθTθS(1)θ∗S(1)non-localview90100110110120120130130140140150150150150θTθ∗S(0)θ∗S(1)sharedview90100110110120120130130140140150150150150−100−50050100−60−40−200204060probinglandscapeθTθS(1)θ∗S(1)90100110110120120130130140140150150150150−100−50050100θTθ∗S(0)θ∗S(1)9010011011012012013013014014015015015015010−710−610−510−410−30.10.20.30.40.5−90−85−80−75−70−65−60−40−35−30−25−20−15−10θTθ∗S(0)10410811211612012410−710−610−510−410−3Random Teachers are Good Teachers

The same visualization technique allows plotting the KL divergence between embeddings produced by the teacher and other
parametrization in the plane. While in Fig,13, the basin of the local solution matches with the area of increased probing
accuracy, such a correlation is not visible if one only considers the encoder.

Figure 15. Orthogonal projection of the embedding KL divergence landscape in the parameter plane.

Figure 16. Higher resolution crop of the global optimum around the teacher.

17

−30−20−100102030losslandscapeθTθS(1)θ∗S(1)non-localview48546060666672727878848490909090θTθ∗S(0)θ∗S(1)sharedview485460606666727278788484849090−60−40−200204060−30−20−100102030probinglandscapeθTθS(1)θ∗S(1)48546060666672727878848490909090−60−40−200204060θTθ∗S(0)θ∗S(1)48546060666672727878848484909010−610−510−410−310−210−11000.10.20.30.40.5−60−55−50−45−40−35−30−20.0−17.5−15.0−12.5−10.0−7.5−5.0θTθ∗S(0)576063666972757810−610−410−2100D. Optimization Metrics

Random Teachers are Good Teachers

To convince ourselves that independently initialized students (α = 1) are more difficult to optimize, we provide an overview
of the KL-Divergence and distance from initialization for all α ∈ [0, 1] in Fig. 17. We observe that, indeed, for students
initialized far away from their teacher, the loss cannot be reduced as efficiently. This coincides with worse probing
performance. Note, however, that even the students with α = 1 are able to outperform their teachers.

Figure 17. Optimization metrics for locality parameter α on CIFAR10. Left: ResNet18. Middle: VGG11. Right: Summary.

18

D.1. Restarting

Random Teachers are Good Teachers

An evident idea would be to restart the random teacher distillation procedure in some way or another. We considered several
approaches, such as reintroducing the exponential moving average of the teacher, but were not successful. In Fig. 18, we
show the most straightforward approach, where the student is reused as a new teacher, and a second round of distillation is
performed. The gradient dynamics around the restarted student seem much more stable, and the optimization procedure
does not even begin.

Figure 18. Restarting random teacher distillation on CIFAR10 with ResNet18 and VGG11. Left: First round of distillation. Right: Second
round of distillation

19

012345kldivergence×10−5kldivergenceResNet18VGG11010203040distancefrominitdistancefrominit050100epoch0.200.250.300.350.400.450.50probingaccuracy05101520epochprobingaccuracyE. Experimental Details

Random Teachers are Good Teachers

Our main goal is to demystify the properties of distillation in a simplistic setting, removing a series of ‘tricks’ used in
practice. For clarity reasons, we here present a comprehensive comparison with the popular framework of DINO (Caron
et al., 2021).

E.1. Architecture

Configuration
Encoder

Projection Head

L2-Bottleneck(in, mid, out)

ResNet18&VGG1 from torchvision, without fc or classification layers (embedding ∈ R512)
(ResNet18 adjusted stem for CIFAR: conv from 7x7 to 3x3, remove maxpool)
3-Layer MLP: 512 → 2048 → 2048 → l2-bottleneck(256) → 216
(GELU activation, no batchnorms, init: trunc normal with σ = 0.02, biases=0)
for x ∈ Rin, W ∈ Rin×mid, b ∈ Rmid, ˜V ∈ Rmid×out
1. linear to bottleneck: z = W T x + b ∈ Rmid
2. feature normalization: ˜z = z/||z||2
3. weightnormalized linear: y = ˜V T ˜z ∈ Rout, with || ˜V:,i||2 = 1

⇒ f ˜V ,W (x) = ˜V T W T x+b
||W T x+b||2

with || ˜V:,i||2 = 1

E.2. Data

Configuration
Augmentations
Training batchsize
Evaluation batchsize

DINO default
Random Teacher
Multicrop (2 × 2242 + 10 × 962) + SimCLR-like None (1 × 322)
64 per GPU
128 per GPU

256
256

E.3. DINO Hyperparameters

Configuration
Teacher update
Teacher BN update BN in train mode
Teacher centering

DINO default
ema with momentum 0.996 cos→ 1

Teacher sharpening
Student sharpening
Loss function

track statistics with momentum 0.9
temperature 0.04 (paper: 0.04 lin→ 0.07)
temperature 0.1
opposite-crop cross-entropy

Random Teacher
no updates
BN in eval mode
not applied

temperature 1
temperature 1
single-crop cross-entropy

E.4. Random Teacher Training

Configuration
Optimizer

Learning rate

Weight decay
Gradient Clipping
Freezing of last layer

DINO default
AdamW
0 lin→ 0.0005 cos→ 1e-6 schedule
0.04 lin→ 0.4 schedule
to norm 3
during first epoch

Random Teacher
AdamW

0.001 (torch default)

not applied
not applied
not applied

E.5. IMP Training

Configuration
Training Epochs
Optimizer

Learning rate
Weight decay
Augmentations

Lottery Ticket Hypothesis (Frankle et al., 2020) Random Teacher
160
SGD (momentum 0.9)
80 epochs

160
SGD (momentum 0.9)
80 epochs

40 epochs

→ 0.01

MultiStep: 0.1
0.0001
Random horizontal flip & padded crop (4px)

→ 0.001

→ 0.01

MultiStep: 0.1
0.0001
Random horizontal flip & padded crop (4px)

→ 0.001

40 epochs

20

