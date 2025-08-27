Jasper and Stella: distillation of SOTA embedding models

Dun Zhang1, Jiacheng Li1∗, Ziyang Zeng1,2, Fulong Wang1
1NovaSearch Team
2Beijing University of Posts and Telecommunications

infgrad@163.com

jcli.nlp@gmail.com

ziyang1060@bupt.edu.cn

wangfl1989@163.com

5
2
0
2

n
a
J

3
2

]

R

I
.
s
c
[

2
v
8
4
0
9
1
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

Abstract

A crucial component in many deep learning
applications, such as Frequently Asked Ques-
tions (FAQ) and Retrieval-Augmented Gener-
In this pro-
ation (RAG), is dense retrieval.
cess, embedding models transform raw text
into numerical vectors. However, the embed-
ding models that currently excel on text embed-
ding benchmarks, like the Massive Text Embed-
ding Benchmark (MTEB), often have numer-
ous parameters and high vector dimensionality.
This poses challenges for their application in
real-world scenarios. To address this issue, we
propose a novel multi-stage distillation frame-
work that enables a smaller student embedding
model to distill multiple larger teacher embed-
ding models through three carefully designed
losses. Meanwhile, we utilize Matryoshka Rep-
resentation Learning (MRL) to reduce the vec-
tor dimensionality of the student embedding
model effectively. Our student model named
Jasper with 2 billion parameters, built upon the
Stella embedding model, obtained the No.3 po-
sition on the MTEB leaderboard (as of Decem-
ber 24, 2024), achieving average 71.54 score
across 56 datasets. We have released the model
and data on the Hugging Face Hub 1 2, and
the training codes are available in this project
repository 3.

1

Introduction

With the rapid development of natural language pro-
cessing technologies, text embedding models play
a crucial role in text representation (Kashyap et al.,
2024), information retrieval (Zhao et al., 2024a),
and text generation tasks (Gao et al., 2023). By
mapping words, sentences, or documents into a

∗∗Dun Zhang and Jiacheng Li make equal contributions to

this work.

1https://huggingface.co/infgrad/jaspe

r_en_vision_language_v1

2https://huggingface.co/datasets/infg

rad/jasper_text_distill_dataset

high-dimensional continuous space, these models
bring similar texts closer together in their vector
representations, thereby not only enhancing the
manipulability of textual data but also significantly
improving the performance of various downstream
tasks (Agarwal et al., 2024; Wang et al., 2024; Zhou
et al., 2024).

However, embedding models that demonstrate
excellent performance on the METB leaderboard4
(Muennighoff et al., 2023) usually contain a large
number of parameters and high vector dimensions.
For instance, both NV-Embed-v2 (Lee et al., 2024;
Moreira et al., 2024) and bge-en-icl (Xiao et al.,
2023; Li et al., 2024) have 7 billion parameters and
4096-dimensional vector representations. These
characteristics lead to slow inference speeds and
high storage costs, posing a significant challenge
to their direct practical application.

To address the aforementioned challenges, we
propose a novel multi-stage knowledge distillation
framework for embedding models. Knowledge dis-
tillation is widely recognized for enhancing the
effectiveness of dense retrieval training (Hofstätter
et al., 2021; Lin et al., 2021). In our framework,
we introduce three carefully designed loss func-
tions to distill knowledge from the teacher model
to the student model. These loss functions shift
from a specific constraint to a broader constraint.
The first, cosine loss, calculates the absolute dif-
ference in text representations between the student
and teacher models. The pointwise signal derived
from a single text is straightforward, yet its lim-
ited optimization direction tends to readily lead to
overfitting on the training data. Thus, we introduce
the similarity loss, which measures the semantic
discrepancies between the student and teacher mod-
els from a text-pair perspective. Additionally, we
design the relative similarity distillation loss to fur-
ther leverage relative ranking information. This

3https://github.com/NLPJCL/RAG-Retriev

4https://huggingface.co/spaces/mteb/l

al

eaderboard

ensures that the student model learns the teacher’s
ranking preferences across all potential positive
and negative text pairs within the batch, thereby
improving the robustness of embedding learning.

To further improve the performance of the stu-
dent model, we utilize multiple powerful large em-
bedding models as teachers. Specifically, we con-
catenate the vectors produced by all teacher models
to create the final ground truth, which inevitably
leads to an increase in the student model’s vector
dimension. To address this issue, we adopt a Ma-
tryoshka Representation Learning (MRL)-based
training method (Kusupati et al., 2024) to effec-
tively compress the student model’s vector rep-
resentation. Additionally, to develop the multi-
modal retrieval capability of our student model,
we integrate a vision encoder and introduce a self-
distillation mechanism to align the visual embed-
dings with the textual embeddings. In terms of the
overall training process, we employ a 4-stage dis-
tillation approach to progressively transfer knowl-
edge from the teacher models to the student model.
Each stage focuses on specific aspects, combining
three loss functions and fine-tuning different pa-
rameters of the student model to ensure a smooth
and effective distillation process.

Experimental results on the MTEB leaderboard
demonstrate that our student model named Jasper
with 2 billion (2B) parameters, primarily built upon
the foundation of the Stella embedding model, de-
livers excellent performance (average 71.54 score
across 56 datasets) comparable to other embedding
models with 7 billion (7B) parameters, and sig-
nificantly outperforms models with fewer than 2B
parameters.

The main contributions of this paper can be sum-

marized as follows:

(1) We propose a novel multi-stage distillation
framework, which enables a smaller student
embedding model to effectively distill knowl-
edge from multiple larger teacher embedding
models through three carefully designed loss
functions.

(2) Our 2B Jasper model obtained the No.3 posi-
tion on the MTEB leaderboard (as of Decem-
ber 24, 2024), producing results comparable
to other top-ranked 7B embedding models and
significantly outperforming other models with
less than 2B parameters.

2 Methods

2.1 Definitions

For a more comprehensive introduction of our
model and distillation framework, we make the
following definitions:

• Student Model: The text embedding model
that is the subject of training, tasked with
learning to produce effective vector represen-
tations.

• Teacher Model: The state-of-the-art (SOTA)
embedding model serving as a teacher, guid-
ing the student model in generating effective
vectors. Notably, the teacher model will not
be trained.

• sx: The normalized vector representation of a

text x produced by the student model.

• tx: The vector representation of the same
text x, first normalized, then concatenated,
and normalized again, produced by multiple
teacher models.

• SX : A matrix of normalized vector represen-
tations for a batch of text X produced by the
student model.

• TX : A corresponding matrix of vector rep-
resentations for the same batch of text X,
first normalized, then concatenated, and subse-
quently normalized again, generated by multi-
ple teacher models.

2.2 Model Architecture

Our student model architecture follows the sim-
ple and standard design of combining a language
model with a vision encoder. As shown in Figure 1,
it consists of four components:

1. A encoder-based language model that gener-
ates text embeddings through mean pooling.

2. A vision encoder that independently maps im-

ages into vision token embeddings.

3. A pooler that maps vision token embed-
dings to the same dimension as the language
model’s input textual embeddings, while re-
ducing the length of visual token sequences.

4. Several fully connected (FC) layers that
project the embeddings to a specific dimen-
sion for the final output.

The Lcosine is designed to minimize the angular
difference between student and teacher vectors in
the high-dimensional space, with the aim of align-
ing their absolute text representations. However,
the Lcosine value generally does not converge to
zero, suggesting a persistent angular discrepancy
between the student and the teachers. Meanwhile,
the pointwise signal derived from a single text has
a limited optimization direction, which can easily
lead to overfitting on the training data.

Lsim = M SE(SX ST

X , TX T T

X ))

(2)

To complement the limitations of Lcosine, we in-
troduce the second loss function, similarity loss, as
defined in (2), which models the semantic matching
differences between the student and teacher mod-
els from a text-pair perspective. This loss function
ensures a relatively consistent judgment of simi-
larity between the student model and the teacher
models, without enforcing an absolute fit between
the student model and the teacher model.

Lresim =

1
N

(cid:88)

M AX(0,

ti·tj >tm·tn
sm · sn − si · sj + margin)

(3)

To further leverage relative comparison signals,
inspired by CoSENT loss7, we propose the third
loss function, relative similarity distillation loss, as
defined in (3). For each batch of text data, we em-
ploy teacher models to automatically generate soft
labels for all text pairs, thereby identifying poten-
tial positive and negative samples. Subsequently,
the student model is trained to ensure that the simi-
larity between positive pairs exceeds that between
negative pairs, with the margin hyperparameter
controlling the degree of this difference.
If the
batch size is m, the total number of text pairs (i.e.,
N ) is given by C2
C2
m

.

L = λ1Lcosine + λ2Lsim + λ3Lresim (4)

The final loss L is a weighted sum of the afore-
mentioned three loss functions. where λ1,λ2, and
λ3 are hyperparameters. The biggest advantage
of distillation vectors is that we do not need any
supervised data. Without considering resource con-
straints, we can use trillions of unsupervised texts

7https://spaces.ac.cn/archives/8847

Figure 1: The model architecture of Jasper model.

2.3

Stage 1&2: Distillation from Multiple
Teachers

In the first two stages of distillation, we use a fully
connected layer to map the vectors of the student
model onto the dimensions of the teacher mod-
els. Specifically, we employ NV-Embed-v25 and
stella_en_1.5B_v56 as teacher models, which have
vector dimensions of 4096 and 8192, respectively.
After the mapping process, the student model’s
vector dimension is adjusted to 12288, equal to the
combined vector dimensions of two teacher models
(4096 + 8192).

The objective of the first two stages is to enable
the student model to effectively learn text represen-
tations from multiple teacher models by aligning
its output vectors with the corresponding teacher
vectors. To achieve this goal, we carefully design
three loss functions that progress from a specific
to a broader perspective. The first loss function is
cosine loss, which is formulated as follows:

Lcosine =

(cid:88)

x

1 − sx · tx.

(1)

5https://huggingface.co/nvidia/NV-Emb

ed-v2

6https://huggingface.co/dunzhang/stel

la_en_1.5B_v5

for distillation training to achieve extreme perfor-
mance for a given model size.

Notably, the main difference between stage 1
and stage 2 lies in the trained parameters. In stage
1, only the fully connected layer (FC1) is trained,
whereas in stage 2, both the fully connected layer
(FC1) and the last three encoder layers of the stu-
dent model are trained.

2.4 Stage 3: Dimension Reduction

In the first two stages, the student model is trained
by learning from the teacher models. Specifically,
we concatenate the vectors produced by the two
teacher models, resulting in a student model vector
with a dimensionality of 12,288 (4,096 + 8,192),
which is impractically large.
Inspired by MRL
(Kusupati et al., 2024), we introduce three addi-
tional, independent fully connected layers (FC2,
FC3, and FC4) to generate low-dimensionality vec-
tors, each achieving a different level of dimension
reduction. For instance, by incorporating the fully
connected layer FC3 with a shape of (15368, 512),
we obtain a more manageable 512-dimensional vec-
tor space.

For the three FC layers, since the dimensions of
the reduced vectors do not align with those of the
concatenated teacher vector, the Lcosine is omitted
and only the Lsim and Lresim are utilized. To en-
sure the accuracy of the vectors generated from
the FC1 layer (i.e., the 12288-dimensional vec-
tors), they continue to be trained using all three
loss functions. During this stage, all parameters of
the student model are trained.

In addition to the previously mentioned dimen-
sion reduction method, we present a potentially
promising approach to self-distillation, where the
aligned vectors from an earlier stage of the student
model’s training serve as teacher vectors. Specifi-
cally, we propose to utilize the 12288-dimensional
vectors output from the FC1 layer to serve as teach-
ers for the shorter vectors generated by the other
three FC layers. This approach provides a unique
advantage by enabling the reduction of the dimen-
sionality of any embedding model, utilizing only
unsupervised data and the model itself. Given
that this paper primarily focuses on introducing
the training methods of the Stella and Jasper mod-
els, we did not conduct experiments to evaluate the
specific merits of this proposed approach.

8This refers to the dimensionality of the encoder layer’s

hidden state.

2.5

Stage 4: Unlock Multimodal Potential

In stage 4, we leverage image-caption pairs as the
training dataset, focusing exclusively on training
the visual encoder while keeping the other compo-
nents frozen. The training process is based on self-
distillation, where the caption’s vector representa-
tion serves as the teacher vector, and the image’s
vector representation acts as the student vector. All
fully connected layers introduced in previous stages
are employed to generate multiple pairs of student
and teacher vectors. For each pair, we calculate
three losses, which are then averaged to obtain the
final loss.

It is important to note that this stage achieves
only a preliminary alignment between the text and
image modalities, leaving significant room for im-
provement. In future work, we aim to further ex-
plore and refine the modality alignment process.

3 Experiments

3.1

Implementation details

Our model named Jasper is initialized from
stella_en_1.5B_v5 and google/siglip-so400m-
patch14-384 (Zhai et al., 2023; Alabdulmohsin
et al., 2024). stella_en_1.5B_v5 and NV-Embed-v2
are our teacher models. The total number of
parameters in our Jasper model is 1.9B (stella
1.5B parameters and siglip 400M parameters). For
hyperparameters, we set λ1 = 10, λ2 = 200, λ3 =
20, margin = 0.015.

In all four stages, the model is trained using 8 ×
RTX A6000 GPUs, with a maximum input length
of 512 tokens, mixed precision training (BF16),
DeepSpeed ZERO-stage-2, and the AdamW opti-
mizer. During stage 1 (distillation training), the
batch size is set to 128, the learning rate is 1e-4
per step, and the model checkpoint at step 4000 is
selected as the final model. In the case of stage 2
(also distillation training), the batch size remains
128, the learning rate drops to 8e-5 per step, and
the final model is the checkpoint at step 7000. For
stage 3 (dimension reduction training), the batch
size is again 128, the learning rate is adjusted to 7e-
5 per step, and the checkpoint at step 2200 serves
as the final model. Lastly, in stage 4 (multimodal
training), the batch size is reduced to 90, the learn-
ing rate returns to 1e-4 per step, and the final model
is chosen from the checkpoint at step 3500.

Model

Model Size

Average(56 datasets)

Classification

Clustering

PairClassification

Reranking

Retrieval

STS

Summarization

NV-Embed-v2
bge-en-icl
Stella_en_1.5B_v5
SFR-Embedding-2_R
gte-Qwen2-1.5B-instruct
voyage-lite-02-instruct
Jasper (our model)

7851M
7111M
1543M
7111M
1776M
1220M
1543M+400M

72.31
71.67
71.19
70.31
67.16
67.13
71.54

90.37
88.95
87.63
89.05
82.47
79.25
88.49

58.46
57.89
57.69
56.17
48.75
52.42
58.04

88.67
88.14
88.07
88.07
87.51
86.87
88.07

60.65
59.86
61.21
60.14
59.98
58.24
60.91

62.65
62.16
61.01
60.18
58.29
56.60
61.33

84.31
84.24
84.51
81.26
82.73
85.79
84.67

30.7
30.77
31.49
30.71
31.17
31.01
31.42

Table 1: MTEB Results as of December 24, 2024. We use the original model names on the leaderboard for clarity.

3.2 Datasets

In stage 1, stage 2 and stage 3, we use fineweb-edu
(Lozhkov et al., 2024) as our main text training
dataset, which accounts for 80% of the full text
data. The remaining 20% of the text data comes
from sentence-transformers/embedding-training-
data9.
The reason we choose the sentence-
transformers/embedding-training-data is that the
majority of the fineweb-edu data consists of pas-
sages. However, in addition to passages, we also
require questions to enhance the diversity of our
training data. The total amount of text training data
is 8 million.

For the documents in our dataset, we perform

the following actions:

1. We randomly select 30% of the documents
and divide them into short texts, each consist-
ing of 1 to 10 sentences.

2. We randomly select 0.08% of the text and

shuffle the words within it.

In stage 4, we use the caption data of
BAAI/Infinity-MM (Gu et al., 2024) as our vision
training data.

3.3 Results

We evaluate the proposed Jasper and Stella models
on the full MTEB benchmark, which encompasses
15 retrieval datasets, 4 reranking datasets, 12 clas-
sification datasets, 11 clustering datasets, 3 pair
classification datasets, 10 semantic textual similar-
ity datasets, and 1 summarization dataset.

Table 1 presents the average score of our Jasper
model across the overall performance and seven
subcategory tasks of the METB benchmark. We
compare our model with other frontier models on
the MTEB leaderboard, as well as those with fewer
than 2B parameters. Experimental results demon-
strate that our Jasper model significantly outper-
forms other models with fewer than 2B parameters.

9https://huggingface.co/datasets/sent
ence-transformers/embedding-training-dat
a

Furthermore, despite having only 2B parameters,
our model produces results that are comparable to
those of models with 7B parameters.

4 Discussion

4.1

Instruction Robustness

Instruction-based embedding models require an in-
struction to be prepended to a query or passage dur-
ing text encoding. Currently, many state-of-the-art
text embedding models use instructions to prompt
the model and obtain better embeddings. Similar
to the usage of large language models (Zhao et al.,
2024b), different tasks necessitate different instruc-
tions, which is both logical and intuitive. Therefore,
the ability to understand instructions is crucial for
these text embedding models.

Jasper is also an instruction-based embedding
model. To demonstrate the impact of different
prompts on the Jasper model, we conducted a sim-
ple experiment. Specifically, we evaluated Jasper
on some short evaluation tasks using similar in-
structions generated by GPT-4o. Table 2 lists all
the original and modified instructions. Based on
the results shown in Table 3, we conclude that our
Jasper model is robust to instructions and can accu-
rately understand different instructions.

4.2 Possible Improvements for Vision

Encoding

Due to time and resource constraints, we were only
able to equip the Jasper model with a basic image
encoding capability. Initially, stage 4 was envi-
sioned as a fundamental visual-language alignment
training phase, with a potential stage 5 involving
contrastive learning utilizing a Visual Question An-
swering (VQA) dataset. Additionally, we observed
oscillatory behavior in our loss function during
stage 4. Overall, there is considerable room for
enhancement in the multimodal training.

5 Conclusion

In this paper, we present the distillation-based train-
ing procedure for the Jasper model. We have

Original Instruction

Synonym of Original Instruction

Classify the sentiment expressed in the given movie review text from the IMDB dataset
Identify the topic or theme of StackExchange posts based on the titles
Given a news summary, retrieve other semantically similar summaries
Retrieve duplicate questions from StackOverflow forum
Given a title of a scientific paper, retrieve the titles of other relevant papers
Classify the sentiment of a given tweet as either positive, negative, or neutral
Given a claim, find documents that refute the claim
Given a question, retrieve relevant documents that best answer the question
Retrieve tweets that are semantically similar to the given tweet
Retrieve semantically similar text.
Identify the main category of Medrxiv papers based on the titles
Retrieve duplicate questions from AskUbuntu forum
Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question
Identify the main category of Biorxiv papers based on the titles and abstracts
Given a financial question, retrieve user replies that best answer the question
Given a online banking query, find the corresponding intents
Identify the topic or theme of the given news articles
Classify the emotion expressed in the given Twitter message into one of the six emotions: anger, fear, joy, love, sadness, and surprise
Given a user utterance as query, find the user intents
Identify the main category of Biorxiv papers based on the titles
Classify the given Amazon review into its appropriate rating category
Given a scientific claim, retrieve documents that support or refute the claim
Identify the topic or theme of StackExchange posts based on the given paragraphs
Given a scientific paper title, retrieve paper abstracts that are cited by the given paper
Classify the given comments as either toxic or not toxic
Classify the intent domain of the given utterance in task-oriented conversation
Retrieve duplicate questions from Sprint forum
Given a user utterance as query, find the user scenarios
Classify the intent of the given utterance in task-oriented conversation
Classify a given Amazon customer review text as either counterfactual or not-counterfactual
Identify the main category of Medrxiv papers based on the titles and abstracts
Given a query on COVID-19, retrieve documents that answer the query

Determine the sentiment conveyed in the provided movie review text from the IMDB dataset.
Determine the subject or theme of StackExchange posts based on the titles.
Given a news summary, find other summaries with similar meanings.
Find duplicate questions on the StackOverflow forum.
Given the title of a scientific paper, find the titles of other related papers.
Determine the sentiment of a given tweet as positive, negative, or neutral.
Given a claim, locate documents that contradict the claim.
Given a question, find relevant documents that best answer it.
Find tweets that have similar meanings to the given tweet.
Find text with similar meanings.
Determine the primary category of Medrxiv papers based on the titles.
Find duplicate questions on the AskUbuntu forum.
Given a question, find detailed question descriptions from Stackexchange that are duplicates.
Determine the primary category of Biorxiv papers based on the titles and abstracts.
Given a financial question, find user replies that best answer it.
Given an online banking query, identify the corresponding intents.
Determine the subject or theme of the given news articles.
Determine the emotion expressed in the given Twitter message as one of six emotions: anger, fear, joy, love, sadness, and surprise.
Given a user utterance as a query, identify the user intents.
Determine the primary category of Biorxiv papers based on the titles.
Classify the given Amazon review into its appropriate rating category.
Given a scientific claim, find documents that support or contradict the claim.
Determine the subject or theme of StackExchange posts based on the given paragraphs.
Given a scientific paper title, find paper abstracts that are cited by the given paper.
Classify the given comments as toxic or non-toxic.
Determine the intent domain of the given utterance in task-oriented conversation.
Find duplicate questions on the Sprint forum.
Given a user utterance as a query, identify the user scenarios.
Determine the intent of the given utterance in task-oriented conversation.
Classify a given Amazon customer review text as counterfactual or non-counterfactual.
Determine the primary category of Medrxiv papers based on the titles and abstracts.
Given a query on COVID-19, find documents that answer the query.

Table 2: Original instructions and corresponding synonyms.

Task Type

Task Name

Original Score

Score with Modified Instructions

Classification
Classification
Classification
Classification
Classification
Classification
Classification
Classification
Classification
Classification
Classification
Clustering
Clustering
Clustering
Clustering
Clustering
Clustering
Clustering
PairClassification
PairClassification
PairClassification
Reranking
Reranking
Reranking
Retrieval
Retrieval
Retrieval
Retrieval
Retrieval
Retrieval
Retrieval
Retrieval
Retrieval
Retrieval
Retrieval
Retrieval
Retrieval
Retrieval
Retrieval
Retrieval
Retrieval
Retrieval
STS
STS
STS
STS
STS
STS
STS
STS
STS
STS
Summarization

Average Score

MTOPDomainClassification
AmazonCounterfactualClassification
TweetSentimentExtractionClassification
EmotionClassification
MassiveIntentClassification
AmazonReviewsClassification
MassiveScenarioClassification
Banking77Classification
ImdbClassification
ToxicConversationsClassification
MTOPIntentClassification
MedrxivClusteringS2S
StackExchangeClusteringP2P
StackExchangeClustering
TwentyNewsgroupsClustering
MedrxivClusteringP2P
BiorxivClusteringS2S
BiorxivClusteringP2P
TwitterURLCorpus
SprintDuplicateQuestions
TwitterSemEval2015
StackOverflowDupQuestions
SciDocsRR
AskUbuntuDupQuestions
CQADupstackMathematicaRetrieval
CQADupstackStatsRetrieval
CQADupstackTexRetrieval
SCIDOCS
CQADupstackEnglishRetrieval
ArguAna
TRECCOVID
CQADupstackUnixRetrieval
CQADupstackGamingRetrieval
CQADupstackGisRetrieval
CQADupstackWordpressRetrieval
FiQA2018
SciFact
CQADupstackPhysicsRetrieval
NFCorpus
CQADupstackProgrammersRetrieval
CQADupstackAndroidRetrieval
CQADupstackWebmastersRetrieval
BIOSSES
STS13
STS12
STSBenchmark
STS15
STS14
STS16
STS22
SICK-R
STS17
SummEval

0.992
0.958
0.773
0.877
0.853
0.629
0.912
0.873
0.971
0.913
0.915
0.448
0.494
0.800
0.630
0.470
0.476
0.520
0.877
0.964
0.803
0.546
0.891
0.674
0.369
0.413
0.362
0.247
0.543
0.653
0.865
0.482
0.632
0.444
0.388
0.601
0.805
0.549
0.431
0.505
0.571
0.464
0.848
0.897
0.803
0.888
0.902
0.853
0.864
0.672
0.822
0.911
0.313

0.686

0.992
0.957
0.776
0.859
0.854
0.630
0.912
0.875
0.971
0.910
0.912
0.448
0.492
0.795
0.625
0.468
0.475
0.518
0.877
0.964
0.801
0.548
0.890
0.676
0.370
0.413
0.362
0.247
0.543
0.652
0.866
0.482
0.633
0.448
0.386
0.601
0.805
0.548
0.431
0.505
0.571
0.464
0.854
0.888
0.804
0.886
0.900
0.851
0.869
0.748
0.823
0.908
0.314

0.687

Table 3: MTEB Results on different instructions.

designed three loss functions to distill multiple
large teacher embedding models into a student em-
bedding model from diverse perspectives. Subse-
quently, we utilized a MRL-based training method
to reduce the vector dimensionality of the student
model. Experimental results on the MTEB demon-
strate that our Jasper model achieves state-of-the-
art performance at the 2B parameter scale and ex-
hibits comparable results to other top-ranked em-
bedding models with 7B parameters. Future work
will further explore the alignment between multiple
modalities.

References

Prabhat Agarwal, Minhazul Islam SK, Nikil Pancha,
Kurchi Subhra Hazra, Jiajing Xu, and Chuck Rosen-
berg. 2024. Omnisearchsage: Multi-task multi-entity
embeddings for pinterest search. In Companion Pro-
ceedings of the ACM on Web Conference 2024, WWW
2024, Singapore, Singapore, May 13-17, 2024, pages
121–130. ACM.

Ibrahim Alabdulmohsin, Xiaohua Zhai, Alexander
Kolesnikov, and Lucas Beyer. 2024. Getting vit in
shape: Scaling laws for compute-optimal model de-
sign.

Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Qianyu Guo,
Meng Wang, and Haofen Wang. 2023. Retrieval-
augmented generation for large language models: A
survey. CoRR, abs/2312.10997.

Shuhao Gu, Jialing Zhang, Siyuan Zhou, Kevin Yu,
Zhaohu Xing, Liangdong Wang, Zhou Cao, Jintao
Jia, Zhuoyi Zhang, Yixuan Wang, Zhenchong Hu,
Bo-Wen Zhang, Jijie Li, Dong Liang, Yingli Zhao,
Yulong Ao, Yaoqi Liu, Fangxiang Feng, and Guang
Liu. 2024. Infinity-mm: Scaling multimodal perfor-
mance with large-scale and high-quality instruction
data.

Sebastian Hofstätter, Sheng-Chieh Lin, Jheng-Hong
Yang, Jimmy Lin, and Allan Hanbury. 2021. Effi-
ciently teaching an effective dense retriever with bal-
anced topic aware sampling. In SIGIR ’21: The 44th
International ACM SIGIR Conference on Research
and Development in Information Retrieval, Virtual
Event, Canada, July 11-15, 2021, pages 113–122.
ACM.

Abhinav Ramesh Kashyap, Thanh-Tung Nguyen, Vik-
tor Schlegel, Stefan Winkler, See-Kiong Ng, and
Soujanya Poria. 2024. A comprehensive survey of
sentence representations: From the BERT epoch to
the CHATGPT era and beyond. In Proceedings of
the 18th Conference of the European Chapter of the
Association for Computational Linguistics, EACL

Wayne Xin Zhao, Jing Liu, Ruiyang Ren, and Ji-Rong
Wen. 2024a. Dense text retrieval based on pretrained
language models: A survey. ACM Trans. Inf. Syst.,
42(4):89:1–89:60.

Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang,
Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen
Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen
Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang,
Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu,
Peiyu Liu, Jian-Yun Nie, and Ji-Rong Wen. 2024b.
A survey of large language models.

Junjie Zhou, Zheng Liu, Shitao Xiao, Bo Zhao, and
Yongping Xiong. 2024. VISTA: visualized text em-
bedding for universal multi-modal retrieval. In Pro-
ceedings of the 62nd Annual Meeting of the Associa-
tion for Computational Linguistics (Volume 1: Long
Papers), ACL 2024, Bangkok, Thailand, August 11-
16, 2024, pages 3185–3200. Association for Compu-
tational Linguistics.

2024 - Volume 1: Long Papers, St. Julian’s, Malta,
March 17-22, 2024, pages 1738–1751. Association
for Computational Linguistics.

Aditya Kusupati, Gantavya Bhatt, Aniket Rege,
Matthew Wallingford, Aditya Sinha, Vivek Ramanu-
jan, William Howard-Snyder, Kaifeng Chen, Sham
Kakade, Prateek Jain, and Ali Farhadi. 2024. Ma-
tryoshka representation learning.

Chankyu Lee, Rajarshi Roy, Mengyao Xu, Jonathan
Raiman, Mohammad Shoeybi, Bryan Catanzaro, and
Wei Ping. 2024. Nv-embed: Improved techniques for
training llms as generalist embedding models. arXiv
preprint arXiv:2405.17428.

Chaofan Li, MingHao Qin, Shitao Xiao, Jianlyu Chen,
Kun Luo, Yingxia Shao, Defu Lian, and Zheng Liu.
2024. Making text embedders few-shot learners.

Sheng-Chieh Lin, Jheng-Hong Yang, and Jimmy Lin.
2021. In-batch negatives for knowledge distillation
with tightly-coupled teachers for dense retrieval. In
Proceedings of the 6th Workshop on Representation
Learning for NLP, RepL4NLP@ACL-IJCNLP 2021,
Online, August 6, 2021, pages 163–173. Association
for Computational Linguistics.

Anton Lozhkov, Loubna Ben Allal, Leandro von Werra,
and Thomas Wolf. 2024. Fineweb-edu: the finest
collection of educational content.

Gabriel de Souza P Moreira, Radek Osmulski, Mengyao
Xu, Ronay Ak, Benedikt Schifferer, and Even
Oldridge. 2024. Nv-retriever: Improving text em-
bedding models with effective hard-negative mining.
arXiv preprint arXiv:2407.15831.

Niklas Muennighoff, Nouamane Tazi, Loïc Magne, and
Nils Reimers. 2023. MTEB: massive text embedding
benchmark. In Proceedings of the 17th Conference of
the European Chapter of the Association for Compu-
tational Linguistics, EACL 2023, Dubrovnik, Croatia,
May 2-6, 2023, pages 2006–2029. Association for
Computational Linguistics.

Xiaohua Wang, Zhenghua Wang, Xuan Gao, Feiran
Zhang, Yixin Wu, Zhibo Xu, Tianyuan Shi,
Zhengyuan Wang, Shizheng Li, Qi Qian, Ruicheng
Yin, Changze Lv, Xiaoqing Zheng, and Xuanjing
Searching for best practices in
Huang. 2024.
retrieval-augmented generation. In Proceedings of
the 2024 Conference on Empirical Methods in Natu-
ral Language Processing, EMNLP 2024, Miami, FL,
USA, November 12-16, 2024, pages 17716–17736.
Association for Computational Linguistics.

Shitao Xiao, Zheng Liu, Peitian Zhang, and Niklas
Muennighoff. 2023. C-pack: Packaged resources
to advance general chinese embedding.

Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov,
and Lucas Beyer. 2023. Sigmoid loss for language
image pre-training.

