Forget What You Know about LLMs Evaluations - LLMs are Like a
Chameleon

Nurit Cohen-Inger1, Yehonatan Elisha2, Bracha Shapira1, Lior Rokach1, Seffi Cohen1

1Ben Gurion University

2Tel Aviv University

5
2
0
2

b
e
F
1
1

]
L
C
.
s
c
[

1
v
5
4
4
7
0
.
2
0
5
2
:
v
i
X
r
a

Abstract

Large language models (LLMs) often appear
to excel on public benchmarks, but these high
scores may mask an overreliance on dataset-
specific surface cues rather than true language
understanding. We introduce the Chameleon
Benchmark Overfit Detector (C-BOD), a
meta-evaluation framework that systematically
distorts benchmark prompts via a parametric
transformation and detects overfitting of LLMs.
By rephrasing inputs while preserving their
semantic content and labels, C-BOD exposes
whether a model’s performance is driven by
memorized patterns. Evaluated on the MMLU
benchmark using 26 leading LLMs, our method
reveals an average performance degradation of
2.15% under modest perturbations, with 20 out
of 26 models exhibiting statistically significant
differences. Notably, models with higher base-
line accuracy exhibit larger performance dif-
ferences under perturbation, and larger LLMs
tend to be more sensitive to rephrasings indi-
cating that both cases may overrely on fixed
prompt patterns. In contrast, the Llama family
and models with lower baseline accuracy show
insignificant degradation, suggesting reduced
dependency on superficial cues. Moreover, C-
BOD’s dataset- and model-agnostic design al-
lows easy integration into training pipelines to
promote more robust language understanding.
Our findings challenge the community to look
beyond leaderboard scores and prioritize re-
silience and generalization in LLM evaluation.

1

Introduction

Large Language Models (LLMs) have achieved
impressive results on a wide range of NLP tasks
(Chang et al., 2024). Consequently, hundreds of
benchmarks have been established to track progress
and evaluate model capabilities (Lu et al., 2024;
Liang et al., 2022). However, the rapid prolifera-
tion of LLMs and the frequent use of public leader-
boards raise concerns about the robustness of these
evaluation practices (Castillo-Bolado et al., 2024).
Specifically, as benchmark data becomes more
widely recognized, models may learn to exploit sur-
face patterns or spurious correlations, rather than

exhibit genuine language understanding. This is-
sue can lead to deceptively high scores that do not
reflect true progress.

In this paper, we examine whether LLMs rely
excessively on benchmark-specific cues potentially
overfitting to the patterns inherent in widely pub-
lished evaluation benchmarks and explore system-
atic methods to detect and mitigate this behavior.
In other words, are LLMs prone to overfitting on
popular benchmarks, and what underlying factors
contribute to this phenomenon?

To answer

this question, we

introduce
Chameleon Benchmark Overfit Detector (C-BOD),
a framework that reveals how heavily a model
depends on the exact wording or structure of a test
set. By introducing controlled textual distortions to
benchmark prompts at varying intensities (defined
by a distortion parameter µ), as demonstrated in
Figure 1, our method reveals whether high per-
formance is built on superficial patterns. Notably,
our framework requires only the evaluation set,
without accessing the model’s training data or
architecture. Unlike conventional leaderboards
that solely track performance, our meta-evaluation
framework acts as a safeguard ensuring that high
scores do not stem from superficial memorization
of benchmark cues.

Our Contributions:

1. Robust Overfitting Detection with Statistical
Significance. Our framework computes the per-
formance difference ∆µ between original and
perturbed prompts and confirms its statistical
significance, ensuring that observed differences
indeed indicate overfitting rather than chance
variations.

2. New Findings For LLM Community Our ex-
tensive analysis reveals new trends regarding
how LLMs function with respect to their num-
ber of parameters and baseline performance.

3. Extensive Empirical Validation. We apply
our method to multiple LLM families of var-
ious architectures and parameter sizes. Even

NER, and summarization. Another recent resource,
JUDGE-BENCH (Bavaresco et al., 2024), com-
prises 20 NLP datasets that assess models against
human judgments. We focus on MMLU because
of its widespread adoption and comprehensive do-
main coverage (Wang et al., 2024), which makes
it particularly effective for exposing overfitting to
canonical prompt structures.

While these benchmarks have been critical for
comparing new models’ versions, recent studies
warn that publicly released evaluation sets can be-
come less reliable over time due to overexposure
and memorization (Yu et al., 2024; Chang et al.,
2024). In some cases, LLMs learn superficial pat-
terns specific to well-known datasets, boosting per-
formance without reflecting genuine semantic or
conceptual understanding. (Kiela et al., 2021) fur-
ther emphasize the need for continuously refresh-
ing benchmarks to ensure real progress in language
understanding. For example, OpenAI’s GPT mod-
els have shown steady improvement on MMLU:
GPT-3 achieved approximately 43% accuracy in
2020 (Brown et al., 2020), rising to nearly 70%
with GPT-3.5 in 20221, and reaching 86% with
GPT-4 in 2023 2.

Memorization in LLMs has been widely stud-
ied (Kiyomaru et al., 2024; Biderman et al., 2024),
with larger models especially prone to retaining
training data verbatim (Carlini et al., 2022). This
phenomenon can inflate performance metrics while
obscuring genuine model capabilities. Moreover,
several works highlight training-set contamina-
tion, where test samples appear exactly or as near-
duplicates in the training data, as another crucial
form of overfitting (Deng et al., 2023; Yao et al.,
2024), leading to overly optimistic performance
estimates (Yang et al., 2023).

2.2 Gap in Current Work

Researchers have introduced various methods to
detect or mitigate training contamination: Finding
the N-gram overlap (e.g., 13-grams or 50-character
matches) between training and test data (Brown
et al., 2020; OpenAI, 2023), though it can miss se-
mantically equivalent rephrasing. Embedding simi-
larity search (Reimers, 2019) that uses transformer-
based embeddings to identify semantically close
training-test pairs (Lee et al., 2023). Decoding
Matching probes the model by providing partial
test prompts and measuring how likely it is to com-
plete them exactly (Li, 2023) or completing miss-
ing words (Deng et al., 2023). A recent study pre-

1https://cdn.openai.com/papers/

GPT-4-Technical-Report.pdf

2https://cdn.openai.com/papers/

GPT-4-Technical-Report.pdf

Figure 1: An example demonstrating the C-BOD
method. The original question (top) is perturbed (bot-
tom) while preserving the semantic meaning and cor-
rect answer options. The model correctly answers the
original question but fails on the perturbed version, sug-
gesting potential overfitting. Changes in the perturbed
question are highlighted in bold.

modest textual distortions cause significant per-
formance differences in most models, provid-
ing strong empirical evidence that overfitting is
widespread.

4. Publicly Available Benchmarks and Code.
We release rephrased versions of the widely
used MMLU evaluation set under different
distortion levels (µ). These resources enable
the community to adopt more robust, surface-
invariant tests for reliable LLM assessment with
our method reproducible code.

5. Blueprint for Iterative Overfit Mitigation. Be-
yond detection, these µ-based rephrasings can
be integrated into model training or fine-tuning
pipelines. Regularly exposing models to di-
verse prompt variations helps reduce reliance
on benchmark-specific phrasing, thus promot-
ing more generalizable language understanding.

2 Related Work

2.1 Benchmark Evaluation and Overfitting

LLMs have achieved impressive results on many
benchmarks. This success has driven the develop-
ment of comprehensive evaluation suites such as
BIG-Bench (Srivastava et al., 2022) and HELM
(Liang et al., 2022). MMLU benchmark set
(Hendrycks et al., 2020) evaluates question answer-
ing across 57 subjects—including STEM, humani-
ties, and social sciences, while (Zhang et al., 2024a)
introduced 25 enterprise-focused datasets covering
domains like finance, legal, cybersecurity, and cli-
mate sustainability for tasks such as classification,

sented an overfit detection of editing knowledge to
a LLM (Zhang et al., 2025).

Although these studies have focused on detect-
ing training data contamination or focusing on ad-
ditional knowledge, they lack with addressing a
critical issue: overfitting to benchmark-specific ar-
tifacts. In many cases, LLMs may never see the
test data during training yet still learn to rely on
superficial cues unique to a benchmark’s canonical
format. Existing techniques such as n-gram overlap
and embedding similarity fail to capture this subtle
form of overfitting. In contrast, our approach ex-
plicitly quantifies the dependence of a model’s per-
formance on the precise phrasing and structure of
evaluation prompts, thereby filling this gap in cur-
rent evaluation methodologies. By systematically
applying a controllable distortion parameter to eval-
uation prompts, without requiring additional train-
ing or access to training data, our method shows
how performance metrics degrade under textual
perturbations, providing a robust means of diagnos-
ing and mitigating broader overfitting behavior.

3 Method

Let D denote a benchmark dataset with N sam-
ples, and E a LLM to be evaluated with respect
to a given performance function M. Our goal is
to detect whether E exhibits overfitting to D. Fig-
ure 2 provides an overview of our proposed method,
Chameleon Benchmark Overfit Detector (C-BOD).
C-BOD employs a rephrasing transformation to
generate a perturbed dataset from D, evaluates on
both the original and perturbed datasets, and ap-
plies a statistical test to assess whether performance
discrepancies indicate overfitting. The following
subsections detail each component of C-BOD.

3.1 C-BOD rephrased dataset generation

To systematically introduce textual variations, C-
BOD utilizes a rephrasing tool, denoted as T ,
which uses as a distortion operator to generate a
perturbed dataset Dµ from D. This operator is
parameterized by µ (temperature), which controls
the extent of textual modification, ranging from
low (e.g., 0.1 for minimal changes like synonym
substitution) to moderate (e.g., 1.0 for rewording
and sentence fragment reordering) and high (e.g.,
1.5 for aggressive modifications such as question
reformulation). We define:

Tµ : X → X ′

Given a prompt xi, the distortion operator pro-
i = Tµ(xi). The per-

duces a perturbed prompt x′
turbed dataset is then constructed as:

Figure 2: High-level pipeline of our parametric ap-
proach. The original dataset D is passed through the
distortion operator Tµ to form Dµ. Both sets are evalu-
ated by a LLM, and differences in performance are used
to quantify overfitting.

i, yi

Dµ = (cid:8) (cid:0)x′
Although each pair (x′

(cid:1) (cid:12)
(cid:12) (xi, yi) ∈ D(cid:9)
i, yi) in the perutbed
dataset remains semantically equivalent to (xi, yi)
in the original dataset, the textual variations intro-
duced by Tµ can disrupt purely memorized map-
pings from surface patterns to correct labels. This
step presented in Lines 5-6 of Algorithm 1.

3.2 Evaluating the Impact of Distortion
To assess the impact of distortion, we evaluate E
using a performance function, M. This function
evaluates E based on a given ground truth or prompt
yi, considering two versions of an input: the orig-
inal xi ∈ D and the perturbed version x′
i ∈ Dµ,
where i denotes the index of a sample in the dataset.
Specifically, M is a boolean function that takes as

input the model E and two data pairs, (xi, yi) and
(x′
i, yi), and returns whether the model performs
better on the original input than on the perturbed
one. The function is formulated as follows:

result with b > c indicates a genuine performance
difference due to the transformation, suggesting ev-
idence of overfitting. This step presented in Lines
10-19 of Algorithm 1.

M(E, (xi, yi), (x′

i, yi) =




1,

if P (E, xi, yi)
> P (E, (x′

i, yi),



0, otherwise.

Algorithm 1 Chameleon Benchmark Overfit De-
tector
Require:

where P (E, x, y) represents the performance
score of model E on input x with reference to
ground truth y. This formulation is designed to
be generalizable across different evaluation metrics
and natural language understanding (NLU) tasks.
The performance difference between the original
set and the perturbed set is then calculated as:

∆µ, b =

N
(cid:88)

i=0

M(E, (xi, yi), (x′

i, yi))

(1)

The performance difference between the perturbed
set and the original set is then calculated as:

c =

N
(cid:88)

i=0

M(E, (x′

i, yi), (xi, yi))

(2)

A large positive ∆µ indicates a significant per-
formance decline due to textual perturbations, sug-
gesting that E may be overly reliant on surface-
level patterns rather than exhibiting robust gener-
alization. Notably, this approach remains metric-
agnostic, making it applicable to a wide range of
evaluation measures. This step presented in Lines
7-8 of Algorithm 1.

3.3 Statistical Validation

To assess the statistical significance of performance
differences, we employ McNemar’s test (McNe-
mar, 1947), which is specifically designed for
paired data. This test evaluates whether the dis-
crepancies between two related sets of classifica-
tion outcomes, correct and incorrect predictions,
are significant. In our context, McNemar’s test
is well-suited for comparing each pair of samples
(xi, yi) ∈ D and (x′
i, yi) ∈ Dµ, we record whether
E classifies them correctly and aggregate into b
(original is better) and c (perturbed is better) as pre-
sented in Equation 1, Equation 2. The McNemar
statistic is then calculated as:

χ2 =

(b − c)2
b + c

(3)

We derive a p-value from the chi-squared dis-
tribution (with df=1, i.e., one degree of freedom),
rejecting the null hypothesis if p < α. A significant

D: Original benchmark dataset of size N ,
E: LLM,
µ: Distortion parameter,
Tµ: Transformation operator,
M: Performance function (returns 1 if the first
input is better, 0 otherwise),
α: Significance level.
1: C-BOD Computation:
2: b, c ← 0
3: Dµ ← {}
4: for each xi ∈ D do
5:

x′
i ← Tµ(xi)
Dµ ← Dµ ∪ x′
i
b ← b + M(E, (xi, yi), (x′
c ← c + M(E, (x′

i, yi))
i, yi), (xi, yi))

6:

7:

8:
9: end for
(b − c)2
10: χ2 ←
b + c
11: p ← p-value(χ2, df = 1)
12: if p < α then
13:

if b > c then

14:

15:

16:

Overf it_F lag ← T rue

else

Overf it_F lag ← F alse

end if

17:
18: else
19:
20: end if
21: return Overf it_F lag

Overf it_F lag ← F alse

4 Experimental Setting

In this section, we describe the experimental setup
used to evaluate our overfitting detection frame-
work. We detail the benchmark dataset, the pro-
cedure for generating perturbed inputs, the LLMs
under evaluation, implementation specifics, and the
evaluation metrics.

4.1 Dataset and Rephrasing Process

experiments use

Our
the MMLU bench-
mark (Hendrycks et al., 2020), which comprises
multiple-choice questions spanning 57 subjects.
The broad coverage and public availability of
MMLU make it an ideal candidate for assessing

general knowledge and the degree to which LLMs
overfit canonical prompt formats. The MMLU
dataset
is distributed under the MIT License,
which allows for free use, modification, and
distribution as long as the original copyright notice
and license terms are maintained. We generate a
perturbed versions of the original dataset to probe
overfitting, following the methodology described
in Section 3. We used DeepSeek 3 to create the
transformed version of each question. We generate
the perturbed dataset using µ = 1.0 (the default
temperature parameter).

These perturbations include synonym substitu-
tions, sentence reordering, and the insertion of dis-
tractor phrases, while preserving the original se-
mantic meaning and correct answers. Automated
formatting checks and manual audits (performed
on approximately 10% of the samples) ensure that
the integrity of the questions is maintained. For
example, an original question:“The coronal suture
joins the?” is rephrased as:“Which bones does the
coronal suture connect?”. The perturbed dataset,
denoted by Dµ, is released alongside our code for
reproducibility 3.

4.2 Models Under Evaluation

Table 1 provides an overview of the LLMs evalu-
ated in our experiments. Our study covers a diverse
set of architectures and parameter scales ranging
from 1B to 236B parameters. This broad selection
enables an in-depth analysis of how both architec-
tural choices and model scale affect robustness to
prompt perturbations.

4.3

Implementation Details

All experiments were executed under standardized
conditions to ensure reproducibility and fair com-
parisons:
(1) Inference Environment: Most models were
accessed via the HuggingFace transformers li-
brary using RTX 6000 GPU. DeepSeek 236B
model was evaluated using the official API.
(2) Dataset Rephrasing Prompt: We instruct the
rephrasing tool using the following prompt to
generate an alternative version of each ques-
tion while preserving its original meaning and
correct answer: “Rephrase the following ques-
tion without changing its context or the correct
answer: {question}”

(3) Query Prompt: For every query, we con-
struct a standardized input by prepending a
fixed instruction to the original MMLU ques-
tion. Importantly, the multiple-choice options
remain identical between the original and the

3https://github.com/SeffiCohen/CBOD

Table 1: Overview of the evaluated LLMs. Models are
grouped by family, model version, and the number of
parameters (in billions).

Family

Version

Params

Qwen

Llama 3

Gemma

Qwen2.5 1.5B (Yang et al., 2024)
Qwen2.5 3B
Qwen2.5 7B
Qwen2.5 32B
Qwen2.5 72B

Llama 3.2 1B (Dubey et al., 2024)
Llama 3.2 3B
Llama 3.1 8B

Gemma 2 2B (Team et al., 2024)
Gemma 7B
Gemma 27B

Phi

Phi 3.5 4B (Abdin et al., 2024)
Phi 4 15B

DeepSeek DeepSeek 7B (Bi et al., 2024)
DeepSeek V2 16B
DeepSeek 236B

Yi

Others

Yi 6B (Young et al., 2024)
Yi 9B

Apollo2 7B (Zhu et al., 2024b)
Aquila 7B (Zhang et al., 2024b)
Bloomz 7B (Zhu, 2023)
Falcon 7B (Almazrouei et al., 2023)
Starling 7B (Zhu et al., 2024a)
Jetmoe 8B (Shen et al., 2024)
GLM 4 9B (GLM et al., 2024)
Mistral 8B (Jiang et al., 2023)

1.5
3
7
32
72

1
3
8

2
7
27

4
15

7
16
236

6
9

7
7
7
7
7
8
9
8

rephrased forms. The fixed instruction is:
“Select the best answer from the given options.
Respond with only the letter corresponding to
the correct choice.
Question: {question}”

4.4 Evaluation Metrics

We assess model performance by comparing the
original dataset, D, with its perturbed counterpart,
D1.0, using the following metrics:

Correct Predictions and Accuracy: For each
dataset, we report the number of correct answers
and the corresponding accuracy, defined as

Accuracy =

#Correct Predictions
#Total Samples

.

Absolute and Percentage Performance Differ-
ence: The absolute difference in the number of
correct answers between D and D1.0 is denoted by
∆1.0; we also report the relative difference. Statis-
tical Significance: McNemar’s test is applied on
the paired predictions to determine whether the per-
formance gap is statistically significant (p < 0.05)

4.5 Reproducibility

C-BOD source code and datasets, including scripts
for data pre-processing, perturbation generation,
perturbed datasets, model evaluation, and statistical
analysis, are publicly available4. This ensures that
our experiments can be independently replicated
and verified.

5 Results

5.1 Overall Performance

As shown in Table 2, most models (20 out of 26)
exhibit a noticeable drop in performance on the
rephrased test set compared to the original, reinforc-
ing our motivation that these LLMs overfit to the
standard MMLU format. Notably, the Llama 1B,
Llama 3B models maintain relatively stable accu-
racy, suggesting they are less susceptible to overfit-
ting. We also observed that Falcon 7B, DeepSeek
7B, Qwen 2.5 3B and Jetmoe 8B show statistically
insignificant differences, likely due to their lower
baseline accuracy. McNemar’s test confirms that
the performance declines observed in most mod-
els are statistically significant. Notably, no model
shows a significant improvement when inputs are
rephrased. This indicates that the C-BOD method
reliably uncovers model vulnerabilities rather than
occasionally yielding unintentional performance
gains.

Across all evaluated models, the average drop in
accuracy was 2.15%, and when considering only
the models with statistically significant differences,
this drop increased to 2.72%.

5.2 Relationship Between Model Size and

Overfit Detection

Figure 3 illustrates the scatter plot of the percent-
age performance difference versus the number of
parameters, with a red dashed line representing the
logarithmic fit (∆1.0 = 0.6318 · ln(cid:0)# Params(cid:1) +
0.7920). The significant log-linear relationship in-
dicates that the performance difference increases
with model size in a logarithmic fashion, suggest-
ing diminishing returns as the number of parame-
ters grows.

Figure 4 plots the percentage performance dif-
ference (∆1.0) for µ = 1.0 against the number of
model parameters, with separate plots for models
below and above 10B parameters (different x-axis
scales). The data reveals a positive trend: larger
models tend to exhibit greater performance degra-
dation under textual perturbations. For example,
models in the Gemma family show a progressive in-
crease in ∆1.0 with higher parameter counts, while

4https://github.com/SeffiCohen/CBOD

Figure 3: Scatter plot of the performance difference
(∆1.0) versus the number of model parameters (log
scale). A logarithmic trendline is shown. Different
colors represent different model families, highlighting
how scaling affects robustness to perturbations.

Llama models maintain low ∆1.0 values across
scales. The dotted trend line further highlights this
relationship.

5.3 Relationship Between Model Accuracy

and Overfit Detection

Figure 5 examines the relationship between base-
line accuracy on the original prompts and the cor-
responding percentage difference in performance
when evaluated on rephrased inputs. The plot
clearly indicates that models with higher original
accuracy tend to experience larger declines when
exposed to prompt perturbations. For example, a
model achieving over 80% accuracy on the origi-
nal set shows one of the largest ∆1.0 values, while
models with lower baseline accuracy exhibit only
minor, often statistically insignificant, differences.
This observation highlights a paradox in current
LLM evaluation: models that perform exception-
ally well on standard benchmarks may be capital-
izing on dataset-specific cues rather than demon-
strating robust language understanding. The posi-
tive correlation between original accuracy and ∆1.0
underscores the need to carefully interpret high
benchmark scores, as they might mask underlying
vulnerabilities to prompt variations.

These findings underscore the importance of
evaluating LLMs under varied prompt formula-
tions to ensure that improvements in benchmark
performance reflect genuine advances in language
understanding rather than overfitting.

6 Discussion

6.1 Why Do LLMs Overfit?

Table 3 highlights cases where LLMs answer
the original questions correctly but fail on the
rephrased versions. The failures suggest potential

Table 2: Comparison of LLM performance on the original and perturbed MMLU datasets (µ = 1.0). The table
shows the number of correct answers on each dataset, accuracy, the absolute and percentage performance difference
(∆1.0), statistical significance (p < 0.05), and whether the model performed better on the original or perturbed
dataset. Models are sorted by parameter count (ascending).

Model
Name

Model
Family

Llama
Gemma
Qwen
Llama
Phi
Qwen
Yi
Qwen
Gemma
Apollo2
Aquila
Bloomz
Falcon
Starling
DeepSeek
Llama
Mistral
Jetmoe
GLM
Yi
Phi

Llama 3.2 1B
Gemma 2 2B
Qwen 2.5 3B
Llama 3.2 3B
Phi 3.5 4B
Qwen 2.5 1.5B
Yi 1.5 6B
Qwen 2.5 7B
Gemma 7B
Apollo2 7B
Aquila 7B
Bloomz 7B
Falcon 7B
Starling 7B
DeepSeek 7B
Llama 3.1 8B
Mistral 8B
Jetmoe 8B
GLM 4 9B
Yi 9B
Phi 4 15B
DeepSeek V2 16B DeepSeek
Gemma 27B
Qwen 2.5 32B
Qwen 2.5 72B
DeepSeek 236B

Gemma
Qwen
Qwen
DeepSeek

Par
(B)

1
2
3
3
4
5
6
7
7
7
7
7
7
7
7
8
8
8
9
9
15
16
27
32
72
236

D
Correct

D
Accuracy

D1.0
Correct

D1.0
Accuracy

#
∆1.0

%
∆1.0

Sig.

3799
6656
5836
7940
9547
5137
8899
6990
8173
9547
4586
6251
3733
8364
6049
6290
6928
6162
9674
9345
10776
7466
10300
11262
11456
10648

26.98
47.28
41.45
56.40
67.81
36.49
63.21
49.65
58.05
67.81
32.57
44.40
26.51
59.41
42.96
44.68
49.21
43.77
68.71
66.38
76.54
53.03
73.16
80.00
81.37
75.63

3802
6552
5739
7989
9325
4986
8525
6810
8050
9146
4475
6133
3718
8179
6077
6169
6691
6132
9346
9180
10530
7358
9903
10816
11081
10292

27.00
46.54
40.76
56.74
66.23
35.41
60.55
48.37
57.18
64.96
31.78
43.56
26.41
58.09
43.16
43.82
47.52
43.55
66.38
65.20
74.79
52.26
70.34
76.82
78.71
73.10

-3
104
97
-49
222
151
374
180
123
401
111
118
15
185
-28
121
237
30
328
165
246
108
397
446
375
356

-0.08 No
Yes
1.56
No
1.66
-0.62 No
Yes
2.33
Yes
2.94
Yes
4.20
Yes
2.58
Yes
1.50
Yes
4.20
Yes
2.42
Yes
1.89
No
0.40
2.21
Yes
-0.46 No
Yes
1.92
Yes
3.42
No
0.49
Yes
3.39
Yes
1.77
Yes
2.28
Yes
1.45
Yes
3.85
Yes
3.96
Yes
3.27
Yes
3.34

Better
Perf.

Not Sig
Original
Not Sig
Not Sig
Original
Original
Original
Original
Original
Original
Original
Original
Not Sig
Original
Not Sig
Original
Original
Not Sig
Original
Original
Original
Original
Original
Original
Original
Original

Figure 4: Performance difference (∆1.0) versus the number of model parameters for models with (a) fewer than and
(b) more than 10 billion parameters. A trendline is shown, and different colors represent different model families,
illustrating how model scale within each family relates to the impact of perturbations.

overfitting, where models overly rely on surface-
level cues, memorized patterns, or specific termi-
nologies. Overfitting in this context occurs because
the model tends to associate certain question for-
mats or keywords directly with answers instead of
generalizing underlying concepts. Common root
causes include shifts in terminology, subtle changes
in phrasing that alter the semantic scope, and depen-
dence on memorized patterns from training data.

6.2 Forget What You Know About LLM

Evaluation

Ideally, LLMs should exhibit resilience when faced
with variations in prompt wording and structure. In
other words, robust LLMs are expected to main-
tain their performance regardless of how a question
is phrased, thereby reflecting true language under-
standing rather than mere memorization. However,
our experiments reveal a contrary trend: models

Table 3: Examples of how rephrasing affects LLM performance, illustrating potential overfitting to specific phrasing
in the original MMLU dataset. The table shows original and rephrased questions, along with an explanation of why
the model’s prediction changed. The examples are from Qwen2.5 (32B parameters).

Subject
Professional
Law

Original Question
“If the defendant is pros-
ecuted for the man’s mur-
der, he will most likely be
found...”

the

Rephrased Question
is
defendant
“If
charged with the man’s
murder, what is the most
probable outcome?”

Moral Dis-
putes

College
Chemistry

“Of the following social
problems that could result
from a genetic supermarket,
which does Singer think is
the least serious?”
“Which of the following
statements is not a rea-
son why tetramethylsilane
is used as a 1H chemical
shift reference?”

“Which of the following so-
cial issues arising from a
genetic supermarket does
Singer consider to be the
least concerning?”
“Which of the following
statements does not ex-
plain why tetramethylsi-
lane is used as a reference
for 1H chemical shifts?”

World Reli-
gions

“When did the first Jaina
temples appear?.”

“At what point in time
were the initial Jaina tem-
ples established?”

Why the Model Was Wrong?
In legal contexts, terms like “prosecuted” and
“found guilty/not guilty” are tied to specific legal
standards. The rephrased question is more open-
ended, leading the model to discuss outcomes
like plea bargaining instead of focusing on the
legal verdict.
The word “problems” was changed to “issues,”
altering the model’s interpretation. “Issues” can
broaden the context of "problems", causing the
model to incorrectly interpret which concerns
are least serious.
The model may have overfit to the structure of
the original question, particularly the phrase “is
not a reason why,” as it directly signals the cor-
rect retrieval path. The rephrased version, with
slight syntactic adjustments disrupts this memo-
rization, leading to incorrect retrieval.
The rephrased question shifts key terms (“When”
to “At what point in time”), obscuring historical
framing. The LLM fails to map this modified
phrasing to the original temporal context.

7 Conclusion

In this paper, we introduced a novel approach for
detecting overfit to benchmarks datasets in LLMs
by applying parametric transformations to these
datasets. Our method revealed that many mod-
els rely heavily on surface features of public test
sets, leading to significant performance drops when
these features are altered. This finding underscores
a critical insight: what appears to be robust perfor-
mance may, in fact, be largely driven by memoriza-
tion rather than true generalization.

We demonstrated the effectiveness of our ap-
proach across multiple LLM families. Notably,
larger models tend to exhibit more pronounced per-
formance declines under perturbation, while cer-
tain models (such as Llama) show greater stability.
These observations suggest that training strategies
and architectural choices play a significant role
in mitigating overfitting, prompting a necessary re-
thinking of how we evaluate and benchmark LLMs.

By providing a practical, dataset-agnostic frame-
work, our work equips the community with a pow-
erful tool to uncover overfitting and to drive the
development of benchmarks that better capture gen-
uine generalization. Incorporating these parametric
transformations into the evaluation process not only
exposes hidden vulnerabilities in current LLMs but
also suggests a way for the creation of more re-
silient models that can adapt to the evolving chal-
lenges of language tasks.

Figure 5: Scatter plot showing ∆1.0 for µ = 1.0 against
the original accuracy of the model. Models within the
same family are marked with the same color.

that score highly on standard benchmarks often
display heightened sensitivity to even minor alter-
ations in prompt formulation. This behavior sug-
gests that such models have implicitly overfitted to
the specific linguistic patterns and structures char-
acteristic of these benchmarks. As a result, when
these surface-level cues are modified, performance
declines, a phenomenon that underscores the para-
dox between high benchmark accuracy and genuine
generalization.

Agnosticism to Benchmark Set. Although we
used MMLU as a demonstration, our approach is
inherently dataset-agnostic. It can be applied to any
benchmark by simply adapting the performance
metric used to compare the original samples with
their rephrased counterparts.

8 Limitations

While C-BOD serves as a promising framework
for detecting overfitting in LLMs and has success-
fully identified overfitting in most evaluated mod-
els, it remains subject to several limitations. First,
our approach primarily targets textual rephrasings
that preserve semantic content. Consequently, it
may overlook deeper forms of overfitting, such
as factual inaccuracies or logical inconsistencies,
which may require more specialized probing tech-
niques. Moreover, incorporating µ-based transfor-
mations into the training or fine-tuning loop can
Itera-
significantly increase computational cost.
tively rephrasing large datasets and retraining with
multiple µ values imposes a heavy resource burden,
which may not be feasible for LLMs or under re-
stricted computational budgets. Future work should
investigate more lightweight or partial-integration
strategies. In summary, while C-BOD provides
an effective means of detecting surface-level over-
fitting, further advancements are necessary to en-
hance its efficiency, scalability, and ability to cap-
ture more nuanced forms of model overfitting.

Acknowledgements

We used ChatGPT-4o for editing the language and
refining the presentation of the text in this paper.
The authors affirm that all research content and
ideas are their own, and they take full responsibility
for the final submitted manuscript.

References

Marah Abdin, Jyoti Aneja, Hany Awadalla, Ahmed
Awadallah, Ammar Ahmad Awan, Nguyen Bach,
Amit Bahree, Arash Bakhtiari, Jianmin Bao, Harkirat
Behl, et al. 2024. Phi-3 technical report: A highly ca-
pable language model locally on your phone. arXiv
preprint arXiv:2404.14219.

Ebtesam Almazrouei, Hamza Alobeidli, Abdulaziz Al-
shamsi, Alessandro Cappelli, Ruxandra Cojocaru,
Mérouane Debbah, Étienne Goffinet, Daniel Hess-
low, Julien Launay, Quentin Malartic, et al. 2023.
The falcon series of open language models. arXiv
preprint arXiv:2311.16867.

Anna Bavaresco, Raffaella Bernardi, Leonardo Berto-
lazzi, Desmond Elliott, Raquel Fernández, Albert
Gatt, Esam Ghaleb, Mario Giulianelli, Michael A.
Hanna, Alexander Koller, André F. T. Martins,
Philipp Mondorf, Vera Neplenbroek, Sandro Pezzelle,
Barbara Plank, David Schlangen, Alessandro Suglia,
Aditya Surikuchi, Ece Takmaz, and Alberto Testoni.
2024. 6. llms instead of human judges? a large scale
empirical study across 20 nlp evaluation tasks.

Xiao Bi, Deli Chen, Guanting Chen, Shanhuang Chen,
Damai Dai, Chengqi Deng, Honghui Ding, Kai Dong,

Qiushi Du, Zhe Fu, et al. 2024. Deepseek llm: Scal-
ing open-source language models with longtermism.
arXiv preprint arXiv:2401.02954.

Stella Biderman, Usvsn Prashanth, Lintang Sutawika,
Hailey Schoelkopf, Quentin Anthony, Shivanshu
Purohit, and Edward Raff. 2024. Emergent and pre-
dictable memorization in large language models. Ad-
vances in Neural Information Processing Systems,
36.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie
Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind
Neelakantan, Pranav Shyam, Girish Sastry, Amanda
Askell, et al. 2020. Language models are few-shot
learners. Advances in neural information processing
systems, 33:1877–1901.

Nicholas Carlini, Daphne Ippolito, Matthew Jagielski,
Katherine Lee, Florian Tramer, and Chiyuan Zhang.
2022. Quantifying memorization across neural lan-
guage models. arXiv preprint arXiv:2202.07646.

David Castillo-Bolado, Joseph Davidson, Finlay Gray,
and Marek Rosa. 2024. Beyond prompts: Dynamic
conversational benchmarking of large language mod-
els. arXiv preprint arXiv:2409.20222.

Yupeng Chang, Xu Wang, Jindong Wang, Yuan Wu,
Linyi Yang, Kaijie Zhu, Hao Chen, Xiaoyuan Yi,
Cunxiang Wang, Yidong Wang, et al. 2024. A sur-
vey on evaluation of large language models. ACM
Transactions on Intelligent Systems and Technology,
15(3):1–45.

Chunyuan Deng, Yilun Zhao, Xiangru Tang, Mark Ger-
stein, and Arman Cohan. 2023. Investigating data
contamination in modern benchmarks for large lan-
guage models. arXiv preprint arXiv:2311.09783.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey,
Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman,
Akhil Mathur, Alan Schelten, Amy Yang, Angela
Fan, et al. 2024. The llama 3 herd of models. arXiv
preprint arXiv:2407.21783.

Team GLM, Aohan Zeng, Bin Xu, Bowen Wang, Chen-
hui Zhang, Da Yin, Dan Zhang, Diego Rojas, Guanyu
Feng, Hanlin Zhao, et al. 2024. Chatglm: A family
of large language models from glm-130b to glm-4 all
tools. arXiv preprint arXiv:2406.12793.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou,
Mantas Mazeika, Dawn Song, and Jacob Steinhardt.
2020. Measuring massive multitask language under-
standing. arXiv preprint arXiv:2009.03300.

Albert Q Jiang, Alexandre Sablayrolles, Arthur Men-
sch, Chris Bamford, Devendra Singh Chaplot, Diego
de las Casas, Florian Bressand, Gianna Lengyel, Guil-
laume Lample, Lucile Saulnier, et al. 2023. Mistral
7b. arXiv preprint arXiv:2310.06825.

Douwe Kiela, Max Bartolo, Yixin Nie, Divyansh
Kaushik, Atticus Geiger, Zhengxuan Wu, Bertie Vid-
gen, Grusha Prasad, Amanpreet Singh, Pratik Ring-
shia, et al. 2021. Dynabench: Rethinking benchmark-
ing in nlp. arXiv preprint arXiv:2104.14337.

Hirokazu Kiyomaru, Issa Sugiura, Daisuke Kawahara,
and Sadao Kurohashi. 2024. A comprehensive anal-
ysis of memorization in large language models. In
Proceedings of the 17th International Natural Lan-
guage Generation Conference, pages 584–596.

Ariel N Lee, Cole J Hunter, and Nataniel Ruiz. 2023.
Platypus: Quick, cheap, and powerful refinement of
llms. arXiv preprint arXiv:2308.07317.

Yucheng Li. 2023.

Estimating contamination via
perplexity: Quantifying memorisation in language
model evaluation. arXiv preprint arXiv:2309.10677.

Percy Liang, Rishi Bommasani, Tony Lee, Dimitris
Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian
Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Ku-
mar, et al. 2022. Holistic evaluation of language
models. arXiv preprint arXiv:2211.09110.

Yuting Lu, Chao Sun, Yuchao Yan, Hegong Zhu, Dong-
dong Song, Qing Peng, Li Yu, Xiaozheng Wang, Jian
Jiang, and Xiaolong Ye. 2024. A comprehensive sur-
vey of datasets for large language model evaluation.
In 2024 5th Information Communication Technolo-
gies Conference (ICTC), pages 330–336. IEEE.

Quinn McNemar. 1947. Note on the sampling error
of the difference between correlated proportions or
percentages. Psychometrika, 12(2):153–157.

R OpenAI. 2023.

Gpt-4 technical report. arxiv

2303.08774. View in Article, 2(5).

N Reimers. 2019. Sentence-bert: Sentence embed-
dings using siamese bert-networks. arXiv preprint
arXiv:1908.10084.

Yikang Shen, Zhen Guo, Tianle Cai, and Zengyi Qin.
2024. Jetmoe: Reaching llama2 performance with
0.1 m dollars. arXiv preprint arXiv:2404.07413.

Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao,
Abu Awal Md Shoeb, Abubakar Abid, Adam Fisch,
Adam R Brown, Adam Santoro, Aditya Gupta,
Adrià Garriga-Alonso, et al. 2022. Beyond the
imitation game: Quantifying and extrapolating the
arXiv preprint
capabilities of language models.
arXiv:2206.04615.

Gemma Team, Thomas Mesnard, Cassidy Hardin,
Robert Dadashi, Surya Bhupatiraju, Shreya Pathak,
Laurent Sifre, Morgane Rivière, Mihir Sanjay Kale,
Juliette Love, et al. 2024. Gemma: Open models
based on gemini research and technology. arXiv
preprint arXiv:2403.08295.

Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni,
Abhranil Chandra, Shiguang Guo, Weiming Ren,
Aaran Arulraj, Xuan He, Ziyan Jiang, et al. 2024.
Mmlu-pro: A more robust and challenging multi-task
language understanding benchmark. arXiv preprint
arXiv:2406.01574.

An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui,
Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu,
Fei Huang, Haoran Wei, et al. 2024. Qwen2. 5 tech-
nical report. arXiv preprint arXiv:2412.15115.

Shuo Yang, Wei-Lin Chiang, Lianmin Zheng, Joseph E
Gonzalez, and Ion Stoica. 2023.
Rethinking
benchmark and contamination for language mod-
arXiv preprint
els with rephrased samples.
arXiv:2311.04850.

Feng Yao, Yufan Zhuang, Zihao Sun, Sunan Xu, Ani-
mesh Kumar, and Jingbo Shang. 2024. Data contam-
ination can cross language barriers. arXiv preprint
arXiv:2406.13236.

Alex Young, Bei Chen, Chao Li, Chengen Huang,
Ge Zhang, Guanwei Zhang, Heng Li, Jiangcheng
Zhu, Jianqun Chen, Jing Chang, et al. 2024. Yi:
Open foundation models by 01. ai. arXiv preprint
arXiv:2403.04652.

Yuan Yu, Lili Zhao, Kai Zhang, G.Y. Zheng, and Meng-
han Liu. 2024. 1. do llms overcome shortcut learn-
ing? an evaluation of shortcut challenges in large
language models.

Bing Zhang, Mikio Takeuchi, Ryo Kawahara, Shubhi
Asthana, M. Shamim Hossain, Ge Ren, Kate Soule,
and Yada Zhu. 2024a. 1. enterprise benchmarks for
large language model evaluation.

Bo-Wen Zhang, Liangdong Wang, Jijie Li, Shuhao Gu,
Xinya Wu, Zhengduo Zhang, Boyan Gao, Yulong
Ao, and Guang Liu. 2024b. Aquila2 technical report.
arXiv preprint arXiv:2408.07410.

Mengqi Zhang, Xiaotian Ye, Qiang Liu, Shu Wu,
Pengjie Ren, and Zhumin Chen. 2025. Uncover-
ing overfitting in large language model editing. In
The Thirteenth International Conference on Learning
Representations.

Banghua Zhu, Evan Frick, Tianhao Wu, Hanlin Zhu,
Karthik Ganesan, Wei-Lin Chiang, Jian Zhang, and
Jiantao Jiao. 2024a. Starling-7b: Improving helpful-
ness and harmlessness with rlaif. In First Conference
on Language Modeling.

Guan-Zhi Zhu. 2023. Enhancing news headline genera-
tion with bloomz model and domain-specific knowl-
edge. Master’s thesis, National Yang Ming Chiao
Tung University.

Hanqing Zhu, Zhenyu Zhang, Wenyan Cong, Xi Liu,
Sem Park, Vikas Chandra, Bo Long, David Z Pan,
Zhangyang Wang, and Jinwon Lee. 2024b. Apollo:
Sgd-like memory, adamw-level performance. arXiv
preprint arXiv:2412.05270.

