Learn Beyond The Answer: Training Language Models
with Reflection for Mathematical Reasoning

Zhihan Zhang(cid:0)1†, Tao Ge2, Zhenwen Liang1†, Wenhao Yu2,
Dian Yu2, Mengzhao Jia1†, Dong Yu2, Meng Jiang1
2Tencent AI Lab, Seattle
1University of Notre Dame

zzhang23@nd.edu

4
2
0
2

t
c
O
5

]
L
C
.
s
c
[

3
v
0
5
0
2
1
.
6
0
4
2
:
v
i
X
r
a

Abstract

Supervised fine-tuning enhances the problem-
solving abilities of language models across var-
ious mathematical reasoning tasks. To maxi-
mize such benefits, existing research focuses
on broadening the training set with various data
augmentation techniques, which is effective for
standard single-round question-answering set-
tings. Our work introduces a novel technique
aimed at cultivating a deeper understanding of
the training problems at hand, enhancing perfor-
mance not only in standard settings but also in
more complex scenarios that require reflective
thinking. Specifically, we propose reflective
augmentation, a method that embeds prob-
lem reflection into each training instance. It
trains the model to consider alternative perspec-
tives and engage with abstractions and analo-
gies, thereby fostering a thorough comprehen-
sion through reflective reasoning. Extensive
experiments validate the achievement of our
aim, underscoring the unique advantages of our
method and its complementary nature relative
to existing augmentation techniques.1

1

Introduction

The ability to engage in step-by-step reasoning is
pivotal for language models (LMs) to solve mathe-
matical problems (Wei et al., 2022; Kojima et al.,
2022). Supervised fine-tuning, particularly on data
with detailed reasoning paths, effectively advances
the problem-solving performance of LMs (Fu et al.,
2023; Yue et al., 2023). To enlarge such benefits,
most previous efforts focus on creating additional
instances to augment model training (Luo et al.,
2023a; Yu et al., 2024; Mitra et al., 2024; Li et al.,
2024a). While these data expansion approaches
allow LMs to handle a broader range of math prob-
lems by increasing the diversity of training data,

† This work was done when Zhihan, Zhenwen, and

Mengzhao were interns at Tencent AI Lab, Seattle.

1Code and data are available at https://github.com/

ytyz1307zzh/RefAug.

stacking more training instances does not necessar-
ily lead to a deeper understanding of each prob-
lem. Moreover, the scope of resulting models is
confined to single-round question-answering (QA)
settings that primarily require basic forward rea-
soning skills. Consequently, these methods provide
limited benefits for more complex reflective rea-
soning scenarios that involve reviewing past steps
for further reasoning, such as addressing follow-up
questions, correcting errors, or leveraging external
feedback (Liang et al., 2024; Wang et al., 2024a).
Similarly, the strategy in human learning is not
always to practice an increasing number of prob-
lems (Rohrer and Taylor, 2006). Instead of merely
memorizing superficial solutions to more problems,
it can be more advantageous to gain a deep under-
standing of the existing problems (Semerci, 2005).
Reflection, therefore, becomes an essential accom-
paniment to practice. Stacey et al. (1982) define
reflection as “to review thoughtfully, consider alter-
natives and follow extensions”, which encourages
learners to contemplate their previous actions to
engage in deeper reasoning, thereby fostering re-
flective thinking capabilities (Kagan et al., 1964;
Anderson and Fincham, 2014).

Inspired by such human cognition, we propose a
novel training strategy for LMs that integrates re-
flection into each math problem. Unlike traditional
data expansion methods which operate on the in-
stance dimension by adding more training exam-
ples (see Figures 1b & 1c), our approach targets a
complementary direction, i.e., the sequence dimen-
sion of the training data. We introduce reflective
augmentation (RefAug), which appends a reflec-
tive section to the original answer of each training
instance, advancing model learning beyond mere
answer generation (see Figure 1d). Such a design
not only strengthens the model’s understanding of
the associated knowledge and methodologies in
training problems, but also maintains the inference
efficiency as the model ceases generation before

Figure 1: Question augmentation creates new questions based on existing ones. Answer augmentation re-samples
answers for each problem to increase diversity. Both methods expand the size of the training set. Reflective aug-
mentation appends the original answer with a reflective section, which is complementary to traditional approaches.
Corresponding training sequences are shown in an (input, output) format, where augmented parts are in red.

Figure 2: The model that learned the standard solution does not fully understand when and how to apply substitution
when facing a different scenario. In contrast, the model trained with reflection on the substitution technique gains a
deeper understanding of its principles, patterns, and its flexible application in new contexts.

decoding the reflective section during inference.
Following the definition by Stacey et al. (1982),
these reflective sections include two components:
alternative and follow-up reasoning. For example,
Figure 2 shows a scenario where the model strug-
gles to apply the substitution technique in a differ-
ent context if only rigidly transferring the pattern
from the standard solution. In contrast, training the
model to reflect on an equivalent substitution ex-
pression followed by devising a more challenging
equation facilitates a deeper understanding of the
principles and variations of the technique, thereby
enabling flexible adaptation in new contexts.

Extensive experimentation on diverse math rea-
soning tasks reveals multiple benefits of RefAug:
(1) It boosts the problem-solving performance of
LMs in the standard single-round QA settings,
yielding a +7.2 accuracy gain over direct fine-

tuning. (2) It remarkably enhances the LMs’ perfor-
mance in multiple reflective math reasoning scenar-
ios, where traditional data expansion methods fall
short. (3) Its benefits are complementary to those
of existing data expansion techniques, allowing
for seamless integration that leads to even greater
performance improvements.

2 Related Work

2.1 Data Augmentation for Math Reasoning

Due to the scarcity (Li et al., 2024a) and quality
issues (Fan et al., 2024) of human-annotated data,
data augmentation is a prevalent strategy in math
reasoning tasks. Most research focused on creat-
ing additional training instances, typically using
advanced LMs to minimize human effort. This in-
clude question augmentation which generates new
questions from existing ones (Yu et al., 2024; Tang

QuestionAnswerQuestionAnswerAug.QuestionAug.AnswerQuestionAnswerAug.AnswerQuestionQuestionAnswerReflection𝑄!,𝐴!𝑄!,𝐴!	&	𝑄!",𝐴!"𝑄!,𝐴!	&	𝑄!,𝐴!"𝑄!,[𝐴!;𝑅!](a) Standard data(b) Question augmentation(c) Answer augmentation(d) Reflective augmentationQuestionTrainTestAnswer(EarlyStop)TrainingQuestion:Solvetheequation𝑥−99𝑥−101=8.Let𝑡=𝑥−100.Thentheequationbecomes𝑡−1(𝑡+1)=8⇒𝑡!−1=8.Therefore,𝑡=±3, and accordingly we get 𝑥=97 or 103.Let𝑡=𝑥−100.…[omitted,sameasstandardtraining]…we get 𝑥=97 or 103.(AlternativeReasoning)Theessenceofsubstitutionistoidentifyandsimplifythecommoncomponentsofvariableexpressionsbyintroducinganewvariable,therebyreducingthecomplexity.Let’srevisittheoriginalequation.Expressions𝑥−99and𝑥−101shareasimilarform:alargeconstantoffsetfrom𝑥.Duetotheminimaldifferencebetween99and101,wecanusesubstitutiontotransformtheexpressionsintotermswithsmallconstants.Therefore,analternativeapproachistosubstitute𝑡=𝑥−99,which transforms the equation into𝑡𝑡−2=8⇒𝑡!−2𝑡−8=0.Thiscanbeeasilyfactorizedinto𝑡−4𝑡+2=0.Hence,𝑡=4or−2,leadingtothesameresults𝑥=97 or 103.Thisalternativeapproachisequallyeffectiveasitalsosimplifiestheequationbysubstituting𝑥andreducingthescaleoftheoffsetterms.(Follow-upReasoning)Extendingtheideaofsubstitution,considertheequation𝑥𝑥+1(𝑥+2)𝑥+3=360.Wenoticethat𝑥𝑥+3=𝑥!+3𝑥,and𝑥+1𝑥+2=𝑥!+3𝑥+2.Therefore,tosimplifytheexpression,wesetthecommonterm𝑥!+3𝑥as𝑡,whichtransformstheequationinto𝑡𝑡+2=360⇒𝑡!+2𝑡−360=0⇒𝑡=−20or18.If𝑡=−20,then𝑥!+3𝑥+20=0.Here,thediscriminantΔ=−71<0,resultinginnorealsolutionsfor𝑥.If𝑡=18,then𝑥!+3𝑥−18=0,so𝑥=3or−6.Thisscenarioreiteratestheimportanceofidentifyingcommoncomponentsof𝑥tostreamlinetheequationthroughsubstitution.TestQuestion:Theunknownnumber𝑥satisfies(7𝑥+1)(9𝑥+1)=61.Find𝑥.Let𝑡=8𝑥+1.Theequationbecomes𝑡−𝑥+1(𝑡+𝑥+1)=61.Next,we…???Ifweset𝑡=7𝑥+1,theoriginalequationturnsinto𝑡(𝑡+2𝑥+1)=61.Thenwecanget…???Let’susesubstitutiontosolvetheequation.Tocreateacommonstructureforeasiersubstitution,wemultiplytheexpressionstounifythecoefficientof𝑥:(7𝑥+1)×9and(9𝑥+1)×7,yielding63𝑥+963𝑥+7=61×63.Observingtheequation,wenoticethatbyletting𝑡=63𝑥+8,itcanberewrittenas𝑡+1𝑡−1=(62+1)(62−1),whichsimplifiesto𝑡!=62!⇒𝑡=±62.Therefore,𝑥="#or𝑥=−$%&.(Earlystopping:noreflectionneededduringinference)InferenceAfterReflectiveAugmentationTrainingAfterStandardTrainingTrainingwithReflectiveAugmentationStandardTrainingFigure 3: Relationship between the original instance and the reflective section. Either abstraction or analogy is
annotated for each instance. Core ideas are shown but textual explanations (like those in Figure 2) are omitted.

et al., 2024; Li et al., 2024a; Liu et al., 2024; Huang
et al., 2024b), and answer augmentation which re-
samples the answer for each question (Yuan et al.,
2023; Li et al., 2023; Yu et al., 2024). Others also
explored answer refinement, aiming to insert addi-
tional reasoning details (Anonymous, 2024) or to
restructure answers for clearer reasoning paths (Fan
et al., 2024). Not only is reflective augmentation
complementary to existing approaches, but it also
exhibits unique advantages in reflective reasoning
scenarios, as we will show in §4.

Another branch of research augmented code
snippets within problem solutions, which trans-
forms text reasoning into code generation (Wang
et al., 2023a; Gou et al., 2024; Lu et al., 2024). This
method is effective for math problems but is typi-
cally considered a separate track since it uses exter-
nal tools (i.e., the code interpreter). Beyond super-
vised fine-tuning, some works augmented data for
further preference optimization (Pang et al., 2024;
Yuan et al., 2024), whereas we leave exploring re-
flective data in preference tuning for future work.

2.2 Reflection in LMs

Previous applications of reflection in LMs primar-
ily focused on enabling LMs to rectify their own
responses during inference (i.e., self-reflect). Some
works equipped the LM with external feedback,
such as code execution or expert critiques (Shinn
et al., 2023; Chen et al., 2024). Others prompted
LMs to use only internal knowledge to correct
answers (Madaan et al., 2023; Li et al., 2024b),
though the effectiveness of this approach is under
debate (Huang et al., 2024a). Some specific tasks
(e.g., math word problems) permit reverse verifi-
cation, where the generated answer is used to re-
derive the question to confirm its correctness (Weng
et al., 2023; Wu et al., 2024). These works demon-
strate that reflection is a common aspect of lan-
guage processing. However, RefAug explores aug-
menting reflective data for better training instead
of answer refinement during inference. Unifying
these approaches is a promising future study.

3 Approach

RefAug extends each training sequence with a re-
flective section that encourages the LM to reflect
on its initial reasoning process to engage in further
math reasoning. Figure 1 contrasts RefAug with
traditional augmentation methods, and its detailed
implementation is elaborated below.

Reflection Types
Following the definition by
Stacey et al. (1982) to “review thoughtfully, con-
sider alternatives and follow extensions”, we con-
sider two types of reflection in composing the re-
flective section: alternative reasoning and follow-
up reasoning.

Alternative reasoning involves thinking about
the problem from different perspectives (Kagan
et al., 1964; Wetzstein and Hacker, 2004). There-
fore, besides the initial solution, we annotate an
alternative approach that also effectively solves
the problem. This helps the model master related
methodologies and develop critical thinking skills.
Follow-up reasoning associates the initial solu-
tion to a broader class of problems (Silver, 1994;
Lim et al., 2020). To fit various contexts, we con-
sider two options: abstraction and analogy. Ab-
straction refers to creating a generalized form of the
original problem, thereby encouraging the model
to reduce dependency on specific numerical values.
Analogy challenges the model in applying method-
ologies of solving the original problem to a more
complex situation. Learning to design follow-up
scenarios enables the model to understand the as-
sociated math concepts and principles better and
apply them flexibly in new contexts. The relation-
ship between the initial instance and components
of the reflective section is illustrated in Figure 3.

Data Annotation
Following a common ap-
proach (Li et al., 2023; Yu et al., 2024; Li et al.,
2024a), we employ an expert LM, GPT-4-turbo, to
annotate the reflective sections for high-quality rea-
soning paths and minimal human effort2. This en-

2We also tried LLaMA-3-70B for data annotation in § 4.4.6

QuestionAnswerFind the maximum of Original ProblemInitial SolutionComplete the square: . The maximum is at the vertex ReflectionAlternative ReasoningFind the derivative . Solve :  is the maximum.Find the maximum of Follow-up: AbstractionFind all extrema of Follow-up: Analogyortails reviewing the original problem and solution to
generate a section consisting of the aforementioned
two types of reflective reasoning. We prompt the
expert model to choose between abstraction and
analogy in follow-up reasoning based on the prob-
lem context. Figure 2 shows an annotated example
with alternative reasoning and follow-up analogy,
and the full annotation prompt is in Appendix E.
The manual inspection and quality analysis of GPT-
annotated data are detailed in Appendix A.4.
Training & Inference During training, given a
math question as input, we include the reflective
section in the output immediately following the
initial answer, starting with a Reflection: prefix.
Thus, the training objective is to learn P([a; r]|q),
where [; ] denotes sequence concatenation. Loss is
calculated on tokens from both the initial answer
and the reflective section. The format of the whole
training sequence is detailed in Appendix D.

During inference, the generation early stops
upon delivering the answer to the input question
and ignores the reflective section, as shown in Fig-
ures 1-2. This is achieved by using Reflection:
as a termination string during model generation.

4 Experiments

We test RefAug in a variety of mathematical tasks
that cover both standard single-round QA and re-
flective reasoning scenarios. We mainly evaluate
two aspects: the influence of RefAug on LMs’
math reasoning abilities and its interaction with
existing augmentation techniques. Besides, we
extend our approach to code generation tasks and
perform comprehensive analyses.

4.1 Standard Math Reasoning

4.1.1 Settings
Standard math reasoning tasks follow a single-
round QA format. Following a popular approach,
we use the training sets of GSM8k (Cobbe et al.,
2021) and MATH (Hendrycks et al., 2021b). We ad-
ditionally include out-of-distribution test sets from
MAWPS (Koncel-Kedziorski et al., 2016), Mathe-
matics (Davies et al., 2021), SVAMP (Patel et al.,
2021), plus the math subsets of MMLU (Hendrycks
et al., 2021a) and SAT (Zhong et al., 2023). We
mainly experiment with two LMs known for su-
perior reasoning performance: Mistral-7B (Jiang
et al., 2023a) and Gemma-7B (Mesnard et al.,
2024), and have also tested LLaMA-3-8B (Meta,

but its performance lags behind GPT-4-turbo.

2024) in Appendix A.1. Models are trained for
3 epochs with batch size 128. The learning rate
peaks at 1e-5 with a 3% warmup period followed
by linear decay. Greedy decoding is applied dur-
ing inference. Additional details of datasets and
training settings are in Appendix B.1.

4.1.2 Existing Training Methods
• Standard Fine-tuning (Figure 1a): Utilizes orig-
inal problem solutions from GSM8k and MATH,
each containing a chain-of-thought reasoning pro-
cess before reaching the final prediction.

• Question Augmentation (Q-Aug, Figure 1b):
Involves training on both original and GPT-
augmented questions. We adopt the augmenta-
tion prompt from Li et al. (2024a), detailed in
Appendix C. We also explore Q-Aug + RefAug
by applying RefAug to all questions after Q-Aug,
and Q-Aug×2 by adding a second augmentation
round to further expand the dataset.

• Answer Augmentation (A-Aug, Figure 1c): Re-
samples the solution for each problem using
GPT-4-turbo, following the approach of Yu et al.
(2024). We also explore its combination with Q-
Aug (A-Aug + Q-Aug), RefAug (A-Aug + Re-
fAug), and another round of A-Aug (A-Aug×2).
• MetaMath Augmentation: MetaMath (Yu et al.,
2024) creates a training set of 400K instances
using various augmentation techniques. Due
to budget constraints, we examine the follow-
ing subsets: (1) A uniformly sampled 40K sub-
set (MetaMath40k), which we augment with
RefAug to compare against an 80K sample
(MetaMath80k); (2) The entire 400K dataset,
of which 40K instances are augmented with Re-
fAug (MetaMath400k+RefAug40k), to compete
with the public MetaMath checkpoint; (3) A one-
epoch continual training (CT) from the public
checkpoint on the same dataset as (2).

The augmentation prompt for Q-Aug and A-Aug,
along with the sampling strategy on MetaMath can
be found in Appendix C.

4.1.3 Results
Table 1 lists the QA accuracy of fine-tuned LMs.
We summarize several findings on RefAug:

Enhancement in Single-Round Math Reason-
ing: RefAug boosts model performance across
both in-distribution and out-of-distribution tasks,
outscoring the direct fine-tuning approach by +7.2
across two base LMs. As the reflective section is
not utilized during inference, this advancement un-

Model

Training Data

In-Distribution
GSM MATH Mathematics MAWPS SVAMP MMLU-Math SAT-Math

Out-Of-Distribution

Avg.

GPT-4-turbo
GPT-3.5-turbo

-
-

94.62
74.68

62.92
44.36

79.70
64.70

97.71
94.27

93.50
82.40

Closed-Source Models

Mistral

Gemma

Mistral

Gemma

Mistral

Gemma

Mistral

Standard
Standard + RefAug

Standard
Standard + RefAug

Q-Aug
Q-Aug×2
Q-Aug + RefAug

Q-Aug
Q-Aug×2
Q-Aug + RefAug

A-Aug
A-Aug×2
A-Aug + Q-Aug
A-Aug + RefAug

A-Aug
A-Aug×2
A-Aug + RefAug

MetaMath40k
MetaMath80k
MetaMath40k + RefAug40k

MetaMath400k*
MetaMath400k + RefAug40k

MetaMath400k (CT)
MetaMath400k + RefAug40k (CT)

Standard Training Data

13.96
17.36

17.06
23.04

14.80
19.40

19.80
26.70

Question Augmentation Data

18.06
21.26
21.66

21.98
24.42
26.38

18.00
20.90
20.50

23.90
23.50
28.70

Answer Augmentation Data

23.08
27.12
24.32
29.40

28.78
31.14
33.60

23.90
28.30
26.90
31.20

33.10
33.30
38.20

MetaMath Augmentation Data

20.96
23.54
26.60

28.42
32.50

28.72
30.12

20.30
23.20
27.00

33.00
34.50

32.70
36.20

56.25
60.05

60.05
64.59

56.03
59.14
63.00

61.11
63.68
68.61

66.19
67.93
69.67
72.93

68.31
70.66
74.15

68.46
69.29
73.84

77.48
78.70

78.39
78.92

73.07
80.25

76.81
85.64

79.99
80.84
81.78

81.78
82.12
85.39

81.10
83.26
81.82
84.41

83.05
85.22
85.68

85.09
86.75
87.68

90.10
91.59

90.87
91.46

53.50
59.30

57.10
64.70

59.10
61.50
60.20

59.70
59.50
66.00

62.20
66.50
61.20
71.50

65.10
69.70
69.10

66.50
68.60
75.30

79.10
77.90

78.90
79.90

75.46
61.70

37.68
43.63

39.32
46.61

38.19
40.86
42.20

40.45
42.71
48.05

37.78
42.61
38.50
47.74

46.51
47.13
52.26

38.09
41.17
44.15

48.77
49.69

49.08
49.69

90.45
77.27

84.91
71.34

31.82
48.64

42.73
55.00

36.16
46.82
50.91

48.18
48.18
51.82

40.91
45.91
46.82
60.45

61.36
54.55
64.09

42.73
43.64
53.18

55.00
59.09

55.91
57.27

40.15
46.95

44.70
52.33

43.65
47.33
48.61

48.16
49.16
53.56

47.88
51.66
49.90
56.80

55.17
55.96
59.58

48.88
50.88
55.39

58.84
60.57

59.22
60.51

Table 1: Accuracy on single-round math reasoning tasks. * The public checkpoint released by Yu et al. (2024).

derscores RefAug’s role in enhancing model learn-
ing, which strengthens math problem-solving capa-
bilities without providing additional context.

Complementary Benefits with Existing Meth-
ods: While data expansion methods (Q-Aug, A-
Aug, and MetaMath) have improved model per-
formance, combining RefAug with them leads to
further substantial gains, improving overall accu-
racy by +6.1 on average. This demonstrates that
RefAug still holds value on high-quality data3 and
is complementary to data expansion strategies. Fur-
thermore, such synergistic benefits outpace the di-
minishing returns seen with repeated dataset expan-
sions: these three methods bring +6.8 improvement
initially but only +2.3 in the second round. This
disparity indicates that expanding data does not
always yield proportionate gains, whereas the bal-
ance of practicing new problems and reflecting on
existing ones maximizes the learning effect.

Effectiveness on Large Datasets: Even when

3In Appendix A.3, we show that GPT-written solutions are
of higher quality than those original ones in GSM and MATH.

only 10% of the full-sized MetaMath dataset in-
cludes the reflective section, the resulting model
surpasses the public MetaMath checkpoint by ~2
points. This confirms RefAug’s efficacy on larger
scales of data. Additionally, the MetaMath model
barely benefits from continual training on its orig-
inal QA data, suggesting a good memorization of
these math problems. Nevertheless, RefAug still
manages to elevate its performance, indicating that
the model has not fully internalized the dataset’s
knowledge and RefAug effectively deepens the
model’s understanding of these problems.

4.2 Reflective Math Reasoning

4.2.1 Tasks

Many realistic math applications require models to
reflect on previous predictions and perform further
reasoning. We employ three tasks of this kind: the
follow-up QA (FQA) and error correction (EC)
tasks of MathChat (Liang et al., 2024), and the
math subset of MINT (Wang et al., 2024a). FQA
involves solving two subsequent questions linked

Training Data

Standard
Standard + RefAug

Q-Aug
Q-Aug×2
Q-Aug + RefAug

A-Aug
A-Aug×2
A-Aug + Q-Aug
A-Aug + RefAug

MetaMath
MetaMath×2
MetaMath + RefAug

MathChat-FQA
2nd
1st

3rd

MathChat-EC

k = 1

k = 2

MINT-Math
k = 4
k = 3

k = 5

∆

56.25
60.05

56.03
59.14
63.00

66.19
67.93
69.67
72.93

68.46
69.29
73.84

25.72
35.36

30.65
32.70
42.19

34.29
36.57
37.86
44.92

37.48
38.92
43.93

15.25
27.54

21.02
22.99
34.37

23.60
28.00
27.31
36.19

24.89
26.10
34.98

50.68
72.99

65.48
63.51
76.48

72.08
71.93
69.58
80.20

61.15
60.09
79.51

20.88
22.34

21.98
27.11
26.74

23.08
25.64
23.44
28.94

22.34
21.61
27.47

24.91
33.70

27.47
32.60
37.36

30.77
31.87
31.87
42.12

27.84
25.64
36.63

27.47
37.00

30.04
35.16
41.03

33.70
33.33
35.16
46.15

31.50
26.74
39.93

28.57
38.10

31.87
36.26
42.86

35.16
34.80
37.36
47.28

32.23
27.47
40.66

28.94
39.56

32.60
37.73
43.22

35.53
34.80
38.10
47.99

33.70
27.84
41.03

8.06
17.22

10.62
10.62
16.48

12.45
9.16
14.66
19.05

11.36
6.23
13.56

Table 2: Accuracy on reflective math reasoning tasks. Each question in MathChat-FQA has two subsequent
questions (2nd and 3rd turns), and the accuracy of each turn is calculated separately. MINT evaluates whether the
model solves the math problem within k interaction turns with the feedback from GPT-4, and we use the difference
(∆) between k = 5 and k = 1 to indicate the model’s ability in leveraging external feedback.

Model

GPT-4-turbo
GPT-3.5-turbo

Data

FQA

2nd

3rd

EC Avg.

-
-

77.67 73.03 83.09 77.93
55.26 45.59 75.90 58.92

MAmmoTH
MetaMath
WizardMath
InternLM2-Math
DeepSeek-Math
Mistral+A-Aug+RefAug
Gemma+A-Aug+RefAug

184K 32.16 19.31 54.15 35.21
395K 43.98 32.16 56.30 44.15
112K* 44.81 36.86 68.22 49.96
~2M 40.20 28.64 72.70 47.18
776K 48.19 35.70 74.34 52.74
30K 44.92 36.19 80.20 53.77
30K 47.80 38.54 81.11 55.82

Table 3: MathChat results compared with other open-
source 7B math models. Baseline scores are from Liang
et al. (2024). The best scores are bolded and the second
bests are underlined. GPT models are listed as a refer-
ence for state-of-the-art performance. *Including both
supervised fine-tuning and reinforcement learning data.

to each initial query, forming a three-round interac-
tion. EC deliberately writes an erroneous solution
to test the model’s error identification and correc-
tion abilities. MINT evaluates the model’s ability
to leverage external language feedback to improve
its reasoning process through up to k turns of inter-
action. More task details are in Appendix B.2.

4.2.2 Results

Results on reflective math reasoning tasks are dis-
played in Tables 2-3 for Mistral and Table 13 for
Gemma. We summarize the key findings below.

Challenges for Data Expansion Methods: De-
spite improving single-round QA performance,
methods like Q-Aug, A-Aug, and MetaMath fall
short in enhancing LMs’ reflective reasoning abil-
ities. For instance, these methods hurt Mistral’s
error correction performance. Moreover, a second

round of augmentation yields minimal or negative
gains across key metrics on reflective reasoning:
+2.5 in FQA-3rd, -1.1 in EC, -0.5 in MINTk=5, and
-4.2 in MINT∆. This indicates that initial augmen-
tation benefits are mainly due to the improved an-
swer quality from GPT annotation3 rather than an
actual increase in reflective reasoning skills, which
echos the findings of Liang et al. (2024) that con-
ventional training approaches overly focus on the
single-round QA setting and neglect many other
important mathematical scenarios.

Superiority of RefAug in Enhancing Reflec-
tive Reasoning: RefAug significantly enhances
the model’s reflective reasoning performance, with
gains of +12.3 in FQA-3rd, +22.3 in EC, +10.6 in
MINTk=5, and +9.2 in MINT∆, far exceeding the
corresponding improvements of +7.9, +15.5, +5.0,
and +3.4 brought by three data expansion meth-
ods on average. An effective solution, however,
is to combine RefAug with these methods, which
yields substantial improvements over them, e.g.,
+12 on FQA-3rd and +10.1 on MINTk=5. These
results highlight RefAug’s exceptional capability
to improve LMs’ reflective math reasoning, which
complements the disregard of existing augmenta-
tion methods on this dimension.

Comparison with Existing Open-Source Mod-
els: Our RefAug-enhanced models excel in the
reflective reasoning scenarios of MathChat with
just 30K training instances, surpassing many open-
source models trained on larger math datasets or
with reinforcement learning. This further supports
RefAug’s effectiveness in cultivating LMs’ reflec-
tive reasoning skills in solving math problems.

Data

GSM MATH Mathematics MAWPS SVAMP MMLU-Math SAT-Math Avg.

Standard

56.25
+ Alternative Reasoning 59.51
56.25
+ Follow-up Reasoning
60.05
+ RefAug

13.96
16.42
16.82
17.36

14.80
17.90
18.80
19.40

73.07
79.57
77.10
80.25

53.50
58.30
58.50
59.30

37.68
39.63
38.09
43.63

31.82
44.09
44.05
48.64

40.15
45.06
44.23
46.95

Table 4: Accuracy on standard math reasoning tasks when varying the components of the reflective section.

Model

HE HE+ MBPP MBPP+ Avg.

CodeLlama-std
CodeLlama-RefAug

Mistral-std
Mistral-RefAug

StarCoder2-std
StarCoder2-RefAug

53.7 50.6
57.9 53.0

38.4 35.4
50.0 45.1

54.3 49.4
56.7 50.6

67.1 59.8
DeepSeekCoder-std
DeepSeekCoder-RefAug 67.1 62.2

62.9
65.4

53.1
56.4

62.7
66.7

75.4
76.7

51.6
52.4

40.1
46.4

51.4
51.6

60.4
63.2

54.7
57.2

41.7
49.5

54.4
56.4

65.7
67.3

Table 5: Pass@1 on code generation, scored by EvalPlus.
-std denotes training with the standard QA setting.

Based on findings from §4.1 and §4.2, we con-
clude the benefits of RefAug on math reasoning
as: Not only does it enhance LMs’ basic problem-
solving skills but also advances their reflective
reasoning abilities, making it a valuable comple-
ment to existing augmentation techniques.

Figure 4: Average accuracy on 7 standard math reason-
ing tasks when different proportions of data are aug-
mented with reflective sections (remaining data are in
the standard QA form).

ability for LMs to possess.

4.4 Analysis

In this section, we dive deeper into additional as-
pects of RefAug. Results are tested on Mistral.

4.3 Code Generation

4.4.1 Ablation Study

Besides math reasoning, we extend the application
of RefAug to code generation. In this task, a query
instructs the model to craft a code snippet that ful-
fills a specific functionality, which also requires
a step-by-step logical flow. We use HumanEval
(Chen et al., 2021) and MBPP (Austin et al., 2021)
as the evaluation benchmarks, along with their plus
versions provided by EvalPlus (Liu et al., 2023).
Training is conducted using the Python subset of
Magicoder-OSS-Instruct (Wei et al., 2023), which
includes 38K QA instances. Considering the ab-
stractive nature of code, we annotate problem analo-
gies as the follow-up section of RefAug.

The outcomes are summarized in Table 5, cover-
ing four different base LMs: CodeLLaMA (Rozière
et al., 2023), Mistral, StarCoder2 (Lozhkov et al.,
2024), and DeepSeekCoder (Guo et al., 2024). The
results demonstrate that RefAug consistently ele-
vates the LMs’ proficiency in following instructions
to generate accurate, reasonable code, as evidenced
by an average improvement of +3.5 in Pass@1
across the evaluated benchmarks. These results
indicate that RefAug is able to enhance LMs’ capa-
bilities in solving code problems, which reaffirms
from another scenario that reflection is an essential

To further assess the efficacy of the reflective sec-
tion, we conduct an ablation study on its two com-
ponents: alternative and follow-up reasoning. Ac-
cording to Table 4, incorporating any single reflec-
tive component to the original data significantly en-
hances model performance by an average of +4.5
points. This suggests that the original solutions
lack sufficient information for the model to fully
grasp the math reasoning skills, which is consistent
with the findings of Anonymous (2024). Combin-
ing both reflective components further enhances
the model’s comprehension of associated concepts
and methodologies, improving the performance by
+2.3 points over using any single one.

4.4.2 The Amount of RefAug Data

We explore the impact of varying the quantity of
reflection-augmented instances in the whole train-
ing set. As depicted by Figure 4, the model’s
overall performance continually improves as more
instances are augmented with reflective sections.
When the model is trained through reflecting on
all instances, the model maximizes its grasp of the
training data and reaches the best performance, un-
derscoring the scalability of RefAug’s benefits.

01/81/41/21Portion of Data with RefAug384042444648Avg. Accuracy40.1543.0743.4045.3246.95Data

GSM MATH Mathematics MAWPS SVAMP MMLU SAT Avg. FQA-2nd FQA-3rd EC

A-Aug
66.19
+RefAug-front 72.78
72.93
+RefAug

23.08
27.34
29.40

23.90
28.30
31.20

81.10
84.62
84.41

62.20
70.30
71.50

37.78
47.23
47.74

40.91 47.88
56.82 55.34
60.45 56.80

34.29
30.96
44.92

23.60
20.64
36.19

72.08
68.29
80.20

Table 6: Comparison between RefAug and prepending the reflective section to the answer (RefAug-front).

Data

Standard

GSM MATH Mathematics MAWPS SVAMP MMLU-Math SAT-Math

Avg.

+ RefAug #1
+ RefAug #2
+ RefAug #3
+ RefAug (Avg.) 61.18±1.1 17.16±0.2

56.25
60.05
62.70
60.80

13.96
17.36
17.26
16.86

14.80
19.40
19.20
18.60
19.07±0.3

73.07
80.25
82.16
80.29

53.50
59.30
60.40
59.70

80.90±0.9 59.80±0.4

37.68
43.63
42.51
42.92
43.02±0.5

31.82
48.64
44.55
45.45
46.21±1.7

40.15
46.95
46.97
46.37
46.76±0.3

Table 7: We sample the reflective sections three times using the same annotation prompt in Figure 8, and train a
separate Mistral model using each batch of the augmented data (labeled as #1~#3). The last row lists the average
scores of three runs as well as their standard deviation.

Training Reasoning Calculation Total

Standard
RefAug

424
374(-50)

287
264(-23)

577
527

Table 8: Error analysis on GSM8k test set. The reduc-
tion of errors is denoted in gray parentheses.

4.4.3 RefAug vs. Chain-of-Thought

For a deeper understanding of the reflective sec-
tion, we experiment with positioning it before the
original solution, i.e., modeling P([r; a]|q). This
arrangement can be regarded as augmenting the
chain-of-thought (CoT, Wei et al., 2022) for solv-
ing the original problem. According to Table 6,
since the reflective section contains relevant rea-
soning steps to the original problem, integrating it
into CoT yields similar improvements as RefAug
on single-round QA. However, such setup hurts
performance in reflective math reasoning, which
supports the original design of RefAug in devel-
oping reflective reasoning skills and reaffirms that
reflective reasoning demands distinct capabili-
ties from standard forward reasoning. Besides,
augmenting CoT increases the token count required
for predicting the final answer, thereby reducing
inference efficiency (see Appendix A.5 for details).

4.4.4 Error Analysis

We analyze how the model’s math capabilities has
been enhanced through the lens of an error analysis.
Following Li et al. (2024a), we classify errors in
GSM8k into calculation errors and reasoning er-
rors. Calculation errors include incorrect identifica-
tion of arithmetic relationships or wrong numerical
computations. Reasoning errors include mistakes
pertaining to the reasoning logic, e.g., incoherent
reasoning steps, misunderstandings of the problem,

etc. Using the gold reasoning paths from GSM8k
test data as a benchmark, we employ GPT-4 to
determine whether solutions contain calculation
errors, reasoning errors, or both. As shown in Ta-
ble 8, the improvement mostly comes from the
reduction of reasoning errors. This supports the
hypothesis that training with reflection enhances
the model’s problem-solving accuracy by deepen-
ing its grasp of underlying math reasoning skills.

4.4.5 Stability of RefAug Data Annotation
To verify the stability of the improvements and to
avoid bias from cherry-picking augmented data,
we sampled reflective sections three times using
GPT-4-turbo with the same prompt in Figure 8.
Each batch of augmented data is used to train a
separate model. As shown in Table 7, the perfor-
mance gains are consistent across all augmentation
samples, with a minimal standard deviation of 0.3
in overall accuracy. These results confirm that re-
flective practices aid in model learning and that
the observed improvements are not due to the
variability of data sampling.

4.4.6 Data Annotation with Open-Source

Models

Besides using GPT to annotate RefAug data, we
explore whether state-of-the-art open-source mod-
els can also serve as data annotators. We employ
LLaMA-3-70B-Instruct (Meta, 2024) for data an-
notation using the same prompt shown in Figure 8,
and train a Mistral-7B model based on this data.
According to results in Table 9, RefAug data anno-
tated by LLaMA-3 yields a similar improvement in
Mistral’s performance on standard math reasoning
tasks. However, the reflective reasoning capability
of the resulting model falls short of its counterpart

Data

GSM MATH Mathematics MAWPS SVAMP MMLU SAT Avg. FQA-2nd FQA-3rd EC

Standard

56.25
+ RefAug (GPT)
60.05
+ RefAug (LLaMA) 62.02

13.96
17.36
17.00

14.80
19.40
17.80

73.07
80.25
80.29

53.50
59.30
61.60

37.68
43.63
39.43

31.82 40.15
48.64 46.95
44.55 46.10

25.72
35.36
32.63

15.25
27.54
23.90

50.68
72.99
50.00

Table 9: Training Mistral-7B with data where reflection sections are annotated by GPT-4-turbo or LLaMA-3-70B-
Instruct. Data annotated by LLaMA-3 yields similar improvements in standard math reasoning tasks, but fails to
match GPT-annotated data in enhancing Mistral’s reflective reasoning capabilities.

trained with GPT-annotated data. This suggests
that developing models with advanced reflective
math reasoning skills demands higher quality
data, compared to what is typically required for
standard forward reasoning in single-round QA.

4.4.7 Data Contamination Analysis

To prevent the augmented data from contaminating
the test sets, we check the n-gram overlap between
the augmented reflective sections and the gold solu-
tions within the test sets of GSM8k and MATH. Fol-
lowing a common approach (Huang et al., 2024b;
Liu et al., 2024), we utilize the test script provided
by Azerbayev et al. (2023) and conduct a 20-gram
check for questions and a 30-gram check for solu-
tions. According to the results in Table 10, RefAug
does not contaminate any test instances in GSM8k.
In the MATH dataset, there is a pre-existing con-
tamination issue: 228 questions and 167 solutions
in the test set are already contaminated by the orig-
inal training set. On the other hand, our RefAug
data overlaps with only 5 instances in the test set,
and these 5 instances were already contaminated
by the training set. In other words, RefAug does
not introduce new contamination to both test sets.
In summary, there is minimal contamination risk
associated with RefAug in our experiments.

In addition to the above perspectives, further
analyses of RefAug’s impact on model efficiency
are presented in Appendix A.5.

5 Conclusion

This paper proposed reflective augmentation (Re-
fAug) for math reasoning, a method that incor-
porates reflection into training problems and is
complementary to existing data augmentation ap-
proaches. We proved the efficacy of RefAug in
not only enhancing LMs’ basic problem-solving
skills on single-round math problems but also in
cultivating their capabilities to solve more complex
reflective reasoning tasks. We further verified the
effectiveness of RefAug in code generation tasks
and its scalability, along with ablation studies and

Dataset

Source

Target

Overlap

GSM8k

MATH

Train Question Test Question
Test Answer
Train Answer
Test Answer
RefAug

Train Question Test Question
Test Answer
Train Answer
Test Answer
RefAug

1
0
0

228
167
5*

Table 10: The contamination check on GSM8k and
MATH: the number of instances from the test set (target)
sharing n-gram overlaps with the training data (source).
We use n = 20 for questions and n = 30 for answers.
* The 5 test instances that overlap with the augmented
reflective sections were already contaminated by the
original MATH training set.

analyses of the methodological choices, such as the
impact of data sequencing and the stability of the
annotation process.

Limitations

Some previous data augmentation studies in math
reasoning created millions of data instances with
OpenAI’s GPT models (Li et al., 2024a; Tang et al.,
2024; Huang et al., 2024b). While testing our
method at a similar scale would be valuable, budget
constraints limit our ability to do so. For instance,
our augmentation data for MetaMath is capped at
40K instances. In §4.4.6, we note that LLaMA-
3-70B shows some promising performance in an-
notating RefAug data for math reasoning tasks,
though its capabilities have not fully matched those
of GPT-4 yet. We anticipate that the develop-
ment of stronger open-source models will reduce
researchers’ dependence on paid services of propri-
etary models.

Acknowledgements

We would like to thank Hongming Zhang (Tencent
AI Lab) for his valuable suggestions on experi-
mental design and paper writing. We also thank
Fangkai Jiao (Nanyang Technological University)
and Zhenyu Wu (Xi’an Jiaotong University) for
their suggestions that help shape our idea.

References

Shengnan An, Zexiong Ma, Zeqi Lin, Nanning Zheng,
Jian-Guang Lou, and Weizhu Chen. 2023. Learning
from mistakes makes LLM better reasoner. Arxiv
preprint, 2310.20689.

John R Anderson and Jon M Fincham. 2014. Extend-
ing problem-solving procedures through reflection.
Cognitive psychology.

Anonymous. 2024. Enrichmath: Enriching idea and so-
lution elicit mathematical reasoning in large language
models. OpenReview.net.

Jacob Austin, Augustus Odena, Maxwell I. Nye,
Maarten Bosma, Henryk Michalewski, David Dohan,
Ellen Jiang, Carrie J. Cai, Michael Terry, Quoc V. Le,
and Charles Sutton. 2021. Program synthesis with
large language models. Arxiv preprint, 2108.07732.

Zhangir Azerbayev, Hailey Schoelkopf, Keiran Paster,
Marco Dos Santos, Stephen McAleer, Albert Q.
Jiang, Jia Deng, Stella Biderman, and Sean Welleck.
2023. Llemma: An open language model for mathe-
matics. Arxiv preprint, 2310.10631.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan,
Henrique Pondé de Oliveira Pinto, Jared Kaplan, Har-
rison Edwards, Yuri Burda, Nicholas Joseph, Greg
Brockman, Alex Ray, Raul Puri, Gretchen Krueger,
Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela
Mishkin, Brooke Chan, Scott Gray, Nick Ryder,
Mikhail Pavlov, and et al. 2021. Evaluating large
language models trained on code. Arxiv preprint,
2107.03374.

Wenhu Chen, Ming Yin, Max Ku, Pan Lu, Yixin Wan,
Xueguang Ma, Jianyu Xu, Xinyi Wang, and Tony
Xia. 2023. Theoremqa: A theorem-driven question
answering dataset. In EMNLP 2023.

Xinyun Chen, Maxwell Lin, Nathanael Schärli, and
Denny Zhou. 2024. Teaching large language models
to self-debug. In ICLR 2024.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian,
Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias
Plappert, Jerry Tworek, Jacob Hilton, Reiichiro
Nakano, Christopher Hesse, and John Schulman.
2021. Training verifiers to solve math word prob-
lems. Arxiv preprint, 2110.14168.

Tri Dao. 2023.

Flashattention-2: Faster attention
with better parallelism and work partitioning. Arxiv
preprint, 2307.08691.

Alex Davies, Petar Velickovic, Lars Buesing, Sam
Blackwell, Daniel Zheng, Nenad Tomasev, Richard
Tanburn, Peter W. Battaglia, Charles Blundell, An-
drás Juhász, Marc Lackenby, Geordie Williamson,
Demis Hassabis, and Pushmeet Kohli. 2021. Advanc-
ing mathematics by guiding human intuition with AI.
Nature.

Run-Ze Fan, Xuefeng Li, Haoyang Zou, Junlong Li,
Shwai He, Ethan Chern, Jiewen Hu, and Pengfei
Liu. 2024. Reformatted alignment. Arxiv preprint,
2402.12219.

Yao Fu, Hao Peng, Litu Ou, Ashish Sabharwal, and
Tushar Khot. 2023. Specializing smaller language
models towards multi-step reasoning. In ICML 2023.

Zhibin Gou, Zhihong Shao, Yeyun Gong, Yelong Shen,
Yujiu Yang, Minlie Huang, Nan Duan, and Weizhu
Chen. 2024. Tora: A tool-integrated reasoning agent
for mathematical problem solving. In ICLR 2024.

Daya Guo, Qihao Zhu, Dejian Yang, Zhenda Xie, Kai
Dong, Wentao Zhang, Guanting Chen, Xiao Bi,
Y. Wu, Y. K. Li, Fuli Luo, Yingfei Xiong, and Wen-
feng Liang. 2024. Deepseek-coder: When the large
language model meets programming - the rise of code
intelligence. Arxiv preprint, 2401.14196.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou,
Mantas Mazeika, Dawn Song, and Jacob Steinhardt.
2021a. Measuring massive multitask language under-
standing. In ICLR 2021.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul
Arora, Steven Basart, Eric Tang, Dawn Song, and
Jacob Steinhardt. 2021b. Measuring mathematical
problem solving with the MATH dataset. In NeurIPS
Datasets and Benchmarks 2021.

Jie Huang, Xinyun Chen,

Swaroop Mishra,
Huaixiu Steven Zheng, Adams Wei Yu, Xiny-
ing Song, and Denny Zhou. 2024a. Large language
models cannot self-correct reasoning yet. In ICLR
2024.

Yiming Huang, Xiao Liu, Yeyun Gong, Zhibin Gou,
Yelong Shen, Nan Duan, and Weizhu Chen. 2024b.
Key-point-driven data synthesis with its enhance-
ment on mathematical reasoning. Arxiv preprint,
2403.02333.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Men-
sch, Chris Bamford, Devendra Singh Chaplot, Diego
de Las Casas, Florian Bressand, Gianna Lengyel,
Guillaume Lample, Lucile Saulnier, Lélio Re-
nard Lavaud, Marie-Anne Lachaux, Pierre Stock,
Teven Le Scao, Thibaut Lavril, Thomas Wang, Timo-
thée Lacroix, and William El Sayed. 2023a. Mistral
7b. Arxiv preprint, 2310.06825.

Weisen Jiang, Han Shi, Longhui Yu, Zhengying Liu,
Yu Zhang, Zhenguo Li, and James T. Kwok. 2023b.
Forward-backward reasoning in large language mod-
els for verification. Arxiv preprint, 2308.07758.

Jerome Kagan, Bernice L Rosman, Deborah Day,
Joseph Albert, and William Phillips. 1964. Informa-
tion processing in the child: Significance of analytic
and reflective attitudes. Psychological Monographs:
General and Applied.

Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yu-
taka Matsuo, and Yusuke Iwasawa. 2022. Large lan-
guage models are zero-shot reasoners. In NeurIPS
2022.

Rik Koncel-Kedziorski, Subhro Roy, Aida Amini, Nate
Kushman, and Hannaneh Hajishirzi. 2016. MAWPS:
A math word problem repository. In NAACL-HLT
2016.

Chen Li, Weiqi Wang, Jingcheng Hu, Yixuan Wei, Nan-
ning Zheng, Han Hu, Zheng Zhang, and Houwen
Peng. 2024a. Common 7b language models already
possess strong math capabilities. Arxiv preprint,
2403.04706.

Chengpeng Li, Zheng Yuan, Hongyi Yuan, Guanting
Dong, Keming Lu, Jiancan Wu, Chuanqi Tan, Xiang
Wang, and Chang Zhou. 2023. Query and response
augmentation cannot help out-of-domain math rea-
soning generalization. Arxiv preprint, 2310.05506.

Yanhong Li, Chenghao Yang, and Allyson Ettinger.
2024b. When hindsight is not 20/20: Testing lim-
its on reflective thinking in large language models.
Arxiv preprint.

Zhenwen Liang, Dian Yu, Wenhao Yu, Wenlin Yao, Zhi-
han Zhang, Xiangliang Zhang, and Dong Yu. 2024.
Mathchat: Benchmarking mathematical reasoning
and instruction following in multi-turn interactions.
Arxiv preprint, 2405.19444.

Woong Lim, Ji-Eun Lee, Kersti Tyson, Hee-Jeong Kim,
and Jihye Kim. 2020. An integral part of facilitating
mathematical discussions: Follow-up questioning.
International Journal of Science and Mathematics
Education.

Haoxiong Liu, Yifan Zhang, Yifan Luo, and An-
drew Chi-Chih Yao. 2024. Augmenting math word
problems via iterative question composing. Arxiv
preprint, 2401.09003.

Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, and Ling-
ming Zhang. 2023. Is your code generated by chatgpt
really correct? rigorous evaluation of large language
models for code generation. In NeurIPS 2023.

Anton Lozhkov, Raymond Li, Loubna Ben Allal, Fed-
erico Cassano, Joel Lamy-Poirier, Nouamane Tazi,
Ao Tang, Dmytro Pykhtar, Jiawei Liu, Yuxiang Wei,
Tianyang Liu, Max Tian, Denis Kocetkov, Arthur
Zucker, Younes Belkada, Zijian Wang, Qian Liu,
Dmitry Abulkhanov, Indraneil Paul, Zhuang Li, Wen-
Ding Li, Megan Risdal, and et al. 2024. Starcoder 2
and the stack v2: The next generation. Arxiv preprint,
2402.19173.

Zimu Lu, Aojun Zhou, Houxing Ren, Ke Wang,
Weikang Shi, Junting Pan, Mingjie Zhan, and Hong-
sheng Li. 2024. Mathgenie: Generating synthetic
data with question back-translation for enhancing
mathematical reasoning of llms. Arxiv preprint,
2402.16352.

Haipeng Luo, Qingfeng Sun, Can Xu, Pu Zhao, Jian-
guang Lou, Chongyang Tao, Xiubo Geng, Qingwei
Lin, Shifeng Chen, and Dongmei Zhang. 2023a. Wiz-
ardmath: Empowering mathematical reasoning for
large language models via reinforced evol-instruct.
Arxiv preprint, 2308.09583.

Ziyang Luo, Can Xu, Pu Zhao, Qingfeng Sun, Xiubo
Geng, Wenxiang Hu, Chongyang Tao, Jing Ma, Qing-
wei Lin, and Daxin Jiang. 2023b. Wizardcoder:
Empowering code large language models with evol-
instruct. Arxiv preprint, 2306.08568.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler
Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon,
Nouha Dziri, Shrimai Prabhumoye, Yiming Yang,
Shashank Gupta, Bodhisattwa Prasad Majumder,
Katherine Hermann, Sean Welleck, Amir Yazdan-
bakhsh, and Peter Clark. 2023. Self-refine: Iterative
refinement with self-feedback. In NeurIPS 2023.

Thomas Mesnard, Cassidy Hardin, Robert Dadashi,
Surya Bhupatiraju, Shreya Pathak, Laurent Sifre,
Morgane Rivière, Mihir Sanjay Kale, Juliette Love,
Pouya Tafti, Léonard Hussenot, Aakanksha Chowdh-
ery, Adam Roberts, Aditya Barua, Alex Botev, Alex
Castro-Ros, Ambrose Slone, Amélie Héliou, Andrea
Tacchetti, Anna Bulanova, Antonia Paterson, Beth
Tsai, Bobak Shahriari, and et al. 2024. Gemma:
Open models based on gemini research and technol-
ogy. Arxiv preprint, 2403.08295.

Meta. 2024. Introducing meta llama 3: The most capa-

ble openly available llm to date. Blog.

Arindam Mitra, Hamed Khanpour, Corby Rosset, and
Ahmed Awadallah. 2024. Orca-math: Unlocking
the potential of slms in grade school math. Arxiv
preprint, 2402.14830.

Richard Yuanzhe Pang, Weizhe Yuan, Kyunghyun Cho,
He He, Sainbayar Sukhbaatar, and Jason Weston.
2024. Iterative reasoning preference optimization.
Arxiv preprint, 2404.19733.

Arkil Patel, Satwik Bhattamishra, and Navin Goyal.
2021. Are NLP models really able to solve simple
math word problems? In NAACL-HLT 2021.

Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase,
and Yuxiong He. 2020. Zero: memory optimizations
In SC
toward training trillion parameter models.
2020.

Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase,
and Yuxiong He. 2020. Deepspeed: System opti-
mizations enable training deep learning models with
over 100 billion parameters. In KDD 2020.

Doug Rohrer and Kelli Taylor. 2006. The effects of
overlearning and distributed practise on the reten-
tion of mathematics knowledge. Applied Cognitive
Psychology: The Official Journal of the Society for
Applied Research in Memory and Cognition.

Yixuan Weng, Minjun Zhu, Fei Xia, Bin Li, Shizhu He,
Shengping Liu, Bin Sun, Kang Liu, and Jun Zhao.
2023. Large language models are better reasoners
with self-verification. In Findings of EMNLP 2023.

Annekatrin Wetzstein and Winfried Hacker. 2004. Re-
flective verbalization improves solutions—the effects
of question-based reflection in design problem solv-
ing. Applied Cognitive Psychology.

Zhenyu Wu, Qingkai Zeng, Zhihan Zhang, Zhaoxuan
Tan, Chao Shen, and Meng Jiang. 2024. Large lan-
guage models can self-correct with minimal effort.
Arxiv preprint, 2405.14092.

Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu,
Zhengying Liu, Yu Zhang, James T. Kwok, Zhenguo
Li, Adrian Weller, and Weiyang Liu. 2024. Meta-
math: Bootstrap your own mathematical questions
for large language models. In ICLR 2024.

Lifan Yuan, Ganqu Cui, Hanbin Wang, Ning Ding,
Xingyao Wang, Jia Deng, Boji Shan, Huimin Chen,
Ruobing Xie, Yankai Lin, Zhenghao Liu, Bowen
Zhou, Hao Peng, Zhiyuan Liu, and Maosong Sun.
2024. Advancing LLM reasoning generalists with
preference trees. Arxiv preprint, 2404.02078.

Zheng Yuan, Hongyi Yuan, Chengpeng Li, Guanting
Dong, Chuanqi Tan, and Chang Zhou. 2023. Scaling
relationship on learning mathematical reasoning with
large language models. Arxiv preprint, 2308.01825.

Xiang Yue, Xingwei Qu, Ge Zhang, Yao Fu, Wen-
hao Huang, Huan Sun, Yu Su, and Wenhu Chen.
2023. Mammoth: Building math generalist models
through hybrid instruction tuning. Arxiv preprint,
2309.05653.

Wanjun Zhong, Ruixiang Cui, Yiduo Guo, Yaobo Liang,
Shuai Lu, Yanlin Wang, Amin Saied, Weizhu Chen,
and Nan Duan. 2023. Agieval: A human-centric
benchmark for evaluating foundation models. Arxiv
preprint, 2304.06364.

Baptiste Rozière, Jonas Gehring, Fabian Gloeckle, Sten
Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi,
Jingyu Liu, Tal Remez, Jérémy Rapin, Artyom
Kozhevnikov, Ivan Evtimov, Joanna Bitton, Man-
ish Bhatt, Cristian Canton-Ferrer, Aaron Grattafiori,
Wenhan Xiong, Alexandre Défossez, Jade Copet,
Faisal Azhar, Hugo Touvron, Louis Martin, Nico-
las Usunier, Thomas Scialom, and Gabriel Synnaeve.
2023. Code llama: Open foundation models for code.
Arxiv preprint, 2308.12950.

Nuriye Semerci. 2005. The effects of problem-based
learning on the academic achievement of students in
development and learning. International Journal of
Educational Reform.

Noah Shinn, Federico Cassano, Ashwin Gopinath,
Karthik Narasimhan, and Shunyu Yao. 2023. Re-
flexion: language agents with verbal reinforcement
learning. In NeurIPS 2023.

Edward A Silver. 1994. On mathematical problem pos-

ing. For the learning of mathematics.

Kaye Stacey, L Burton, and J Mason. 1982. Thinking

mathematically. Addison-Wesley.

Zhengyang Tang, Xingxing Zhang, Benyou Wang, and
Furu Wei. 2024. Mathscale: Scaling instruction
tuning for mathematical reasoning. Arxiv preprint,
2403.02884.

Ke Wang, Houxing Ren, Aojun Zhou, Zimu Lu, Sichun
Luo, Weikang Shi, Renrui Zhang, Linqi Song,
Mingjie Zhan, and Hongsheng Li. 2023a. Mathcoder:
Seamless code integration in llms for enhanced math-
ematical reasoning. Arxiv preprint, 2310.03731.

Xingyao Wang, Zihan Wang, Jiateng Liu, Yangyi Chen,
Lifan Yuan, Hao Peng, and Heng Ji. 2024a. MINT:
evaluating llms in multi-turn interaction with tools
and language feedback. In ICLR 2024.

Yejie Wang, Keqing He, Guanting Dong, Pei Wang, Wei-
hao Zeng, Muxi Diao, Yutao Mou, Mengdi Zhang,
Jingang Wang, Xunliang Cai, and Weiran Xu. 2024b.
Dolphcoder: Echo-locating code large language mod-
els with diverse and multi-objective instruction tun-
ing. Arxiv preprint, 2402.09136.

Yizhong Wang, Hamish Ivison, Pradeep Dasigi, Jack
Hessel, Tushar Khot, Khyathi Chandu, David Wad-
den, Kelsey MacMillan, Noah A. Smith, Iz Beltagy,
and Hannaneh Hajishirzi. 2023b. How far can camels
go? exploring the state of instruction tuning on open
resources. In NeurIPS 2023.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten
Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le,
and Denny Zhou. 2022. Chain-of-thought prompt-
ing elicits reasoning in large language models. In
NeurIPS 2022.

Yuxiang Wei, Zhe Wang, Jiawei Liu, Yifeng Ding, and
Lingming Zhang. 2023. Magicoder: Source code is
all you need. Arxiv preprint, 2312.02120.

Data

Standard

+ RefAug

GSM MATH Mathematics MAWPS SVAMP MMLU-Math SAT-Math Avg.

GPT-Written Solutions 71.72
75.74

+ RefAug

64.59
67.10

19.86
22.08

28.04
31.64

20.20
25.60

32.90
32.00

81.35
83.64

85.26
87.38

66.00
69.40

73.20
75.80

45.59
48.97

47.84
51.75

47.73
55.00

55.00
69.09

49.33
53.11

56.28
60.49

Table 11: Results on LLaMA-3-8B. We test integrating RefAug with (1) the original training data, and (2) the data
where answers are re-written by GPT-4-turbo (see Appendix A.3 for GPT answer re-writing).

Data

GSM MATH Mathematics MAWPS SVAMP MMLU-Math SAT-Math Avg.

Original Solutions
56.25
GPT-4-turbo Solutions 65.73
71.80

+ RefAug

13.96
23.10
26.12

14.80
23.90
29.50

73.07
81.14
82.84

53.50
68.80
70.80

37.68
40.25
44.76

31.82
41.36
57.73

40.15
49.18
54.79

Table 12: Comparison between using synthetic solutions written by GPT-4-turbo and using the originally annotated
ones in GSM8k and MATH training sets, as well as applying RefAug on the synthetic solutions. Solutions written
by GPT-4-turbo are of much higher quality than the original ones.

A Additional Experiments

Training Data

MathChat-FQA
2nd

3rd

1st

MathChat-EC

In this section, we present more experimental re-
sults in addition to those in §4.

A.1 Results on LLaMA-3

In addition to training Mistral-7B and Gemma-7B
with RefAug, we also test LLaMA-3-8B (Meta,
2024) on the RefAug data. According to the results
in Table 11, RefAug enhances the math reason-
ing capabilities of LLaMA-3 as well, no matter if
integrating with the original solutions or with solu-
tions re-written by GPT-4-turbo. This again shows
the generalizability of the RefAug method, which
leads to consistent improvements across various
base models.

A.2 Gemma on Reflective Math Reasoning

Besides evaluating Mistral-based models on reflec-
tive reasoning tasks (shown in Table 2, we report
scores on our Gemma-based models as well. As
shown in Table 13, the performance trends for
Gemma models align with those observed on
Mistral models. RefAug demonstrates a clear ad-
vantage over traditional augmentation methods in
enhancing reflective math reasoning capabilities of
LMs. For instance, RefAug outscores both Q-Aug
and A-Aug in the third round of follow-up QA and
in the accuracy of error correction. Furthermore,
as shown in Table 3, a combination of A-Aug and
RefAug data results in the best-performing model
on the reflective reasoning scenarios of MathChat,
outperforming many open-source models that are
trained on substantially larger math datasets.

Standard
Standard + RefAug

Q-Aug
Q-Aug×2
Q-Aug + RefAug

A-Aug
A-Aug×2
A-Aug + RefAug

60.05
64.59

61.11
63.68
68.61

68.31
70.66
74.15

30.05
40.44

34.67
34.45
42.64

41.05
42.79
47.80

20.56
33.16

26.25
26.40
34.22

29.59
32.25
38.54

61.99
77.47

67.68
70.41
79.97

73.98
77.39
81.11

Table 13: Results of Gemma on reflective math reason-
ing tasks. The general trend is similar to that of Mistral
(Table 2).

A.3 Quality of GPT-Written Answers

In Table 1, we find that answer augmentation sig-
nificantly enhances performance. It improves the
overall accuracy by +9.1 over the use of original
training data, when averaged across Mistral and
Gemma models. This surpasses the improvement
of +7.2 on average seen with RefAug over the orig-
inal data. A deeper analysis reveals that the rea-
soning paths generated by GPT-4-turbo are of
significantly higher quality than those originally
provided in the GSM8k and MATH datasets.
As demonstrated in Table 12, merely replacing the
original solutions with those generated by GPT-4-
turbo increased the accuracy from 40.15 to 49.18
on Mistral. However, RefAug does not receive such
benefits as it does not alter the original reasoning
paths during augmentation. Given the complemen-
tary nature of these two augmentation methods,
their combination further improves the model ac-
curacy to 54.79. This echoes the synergistic per-

Dataset Alternative Follow-up

Training

Train Tokens Test Tokens

GSM8K
MATH

96%
76%

96%
72%

Table 14: The percentage of error-free RefAug annota-
tions by GPT-4-turbo, including the alternative reason-
ing section and the follow-up reasoning section.

Training

Data

Time

Standard
Q-Aug / A-Aug
RefAug

15K
60 min
30K 123 min
90 min
15K

Table 15: The impact of various augmentation methods
on dataset size and training time. These stats are tested
on 8×A100 GPUs.

formance advantage achieved by A-Aug+RefAug
over both A-Aug and A-Aug×2 in Table 1.

A.4 Quality of GPT-annotated Reflective

Sections

We analyze the correctness of GPT-annotated re-
flective sections by manually reviewing 50 samples
(25 from GSM8K, 25 from MATH) in the training
set. The results, as shown in Table 14, indicate that
generating reflective sections is generally easier for
GPT than solving entirely new problems. This is
due to the fact that we provide both the original
problem and solution during RefAug annotation.
Consequently, the correctness of the annotated re-
flective sections is generally satisfactory.

Verification of LM-generated data is a com-
mon challenge in data augmentation. We did not
dive deep into answer verification in this paper
for two reasons: (1) Common methods like self-
consistency voting or LM-based validation are or-
thogonal to our study’s focus on different augmen-
tation types. (2) Studies have indicated that data
verification often does not lead to significant perfor-
mance gains, and noisy answers could help training
as well (Yu et al., 2024; Tang et al., 2024; Li et al.,
2024a). This is because such answers often include
many correct reasoning steps before making an
error, and filtering them trades data diversity for
correctness.

A.5 Training and Inference Efficiency

For a deeper understanding of RefAug, we analyze
its impact on the efficiency of model training and
inference. To begin with, according to Table 15,
while RefAug does introduce additional time over-

Standard
GPT Solutions
RefAug-front
RefAug

171.4
358.3
910.1
892.3

185.5
423.5
980.5
219.1

Table 16: The resulting sequence lengths of each aug-
mentation method during training and testing.

head during model training, this increase is less
significant than that caused by Q-Aug or A-Aug
which doubles the optimization steps due to dataset
expansion. Additionally, although RefAug results
in longer sequence lengths in training instances,
it does not impair inference efficiency, as shown
by the average number of tokens generated in Ta-
ble 16. This is due to the early stopping feature that
eliminates the need to generate reflective sections
during inference. Overall, the efficiency impact
brought by RefAug is minimal.

B Detailed Task Settings

In this section, we detail the datasets, training
hyper-parameters, and evaluation settings of each
task used in our experiments. We list the size of all
datasets in Table 17.

B.1 Standard Math Reasoning

Datasets
In standard math reasoning, we fol-
low a common approach (Wang et al., 2023a; Yu
et al., 2024; Li et al., 2024a) to adopt the train-
ing data from GSM8k (Cobbe et al., 2021) and
MATH (Hendrycks et al., 2021b) as they are paired
with human-labeled reasoning paths. For evalua-
tion, we employ a comprehensive suite of bench-
marks that span a wide range of mathematical top-
ics. Specifically, GSM8k, SVAMP (Patel et al.,
2021), and MAWPS (Koncel-Kedziorski et al.,
2016) focus mainly on arithmetic math word prob-
lems, while datasets such as MATH, Mathemat-
ics (Davies et al., 2021), MMLU (Hendrycks et al.,
2021a), and SAT (Zhong et al., 2023) encompass a
broader scope including algebra, geometry, number
theory, probability, and formal logic. By difficulty
levels, they cover elementary (MAWPS, SVAMP),
middle school (GSM8K, SAT), and more advanced
levels (Mathematics, MATH, MMLU), providing
an exhaustive assessment of the mathematical ca-
pabilities of language models.

Training Settings During model training, we
first tune the hyper-parameters using the original

Dataset

Train Test

GSM8k (Cobbe et al., 2021)
MATH (Hendrycks et al., 2021b)
Mathematics (Davies et al., 2021)
MAWPS (Koncel-Kedziorski et al., 2016)
SVAMP (Patel et al., 2021)
MMLU-Math (Hendrycks et al., 2021a)
SAT-Math (Zhong et al., 2023)

MathChat-FQA (Liang et al., 2024)
MathChat-EC (Liang et al., 2024)
MINT-Math (Wang et al., 2024a)

Magicoder (Wei et al., 2023)
HumanEval (Chen et al., 2021)
MBPP (Austin et al., 2021)

7473
7500
-
-
-
-
-

-
-
-

38284
-
-

1319
5000
1000
2354
1000
974
220

1319
1319
273

-
164
399

Table 17: Statistics of all datasets used in our training
and evaluation.

data under the standard fine-tuning recipe. then,
these settings remain fixed across all models to
avoid extensive hyper-parameter tuning for each
variant. This approach is common in studies com-
paring models fine-tuned on varied datasets (Yuan
et al., 2023; Li et al., 2023; An et al., 2023). Specif-
ically, we train models for 3 epochs with a batch
size of 128. The learning rate starts at 1e-5, includ-
ing a warmup for the initial 3% of steps, and then
linearly decreases to 20% of its initial value by the
end of training. Training sequences are truncated
to 4096 tokens. To speed up training, our model
utilize bfloat16 precision and are supported by
FlashAttention-2 (Dao, 2023), DeepSpeed (Rasley
et al., 2020), and ZeRO-3 optimization (Rajbhan-
dari et al., 2020). For training on the full set of
MetaMath, we follow the original authors’ rec-
ommendation4 to lower the learning rate to 2e-6,
and for continued training on the public MetaMath
checkpoint, we use a reduced learning rate of 1e-6
to be more consistent with its initial fine-tuning.

Evaluation To facilitate answer extraction during
evaluation, we append The answer is XXX. to the
reasoning path of each training instance so that the
final predicted answer is explicitly stated. We adopt
the evaluation script from Yue et al. (2023) that first
extracts the predicted answer and then checks for
an exact match with the ground-truth. Exceptions
are MMLU and SAT which use multiple-choice
formats instead of numerical answers. Since our
training data does not contain multiple-choice ques-
tions, the model may predict the content of an op-

4https://huggingface.co/meta-math/

MetaMath-Mistral-7B

tion rather than its letter identifier. Thus, on these
datasets, we leverage GPT-3.5-turbo to match the
predicted content to the appropriate option before
computing accuracy.

B.2 Reflective Math Reasoning

Reflective math reasoning encompasses scenarios
where models must consider previously provided
answers to engage in further reasoning. However,
benchmarks that adequately capture this dynamic
are scarce in the existing literature. Utilizing the
currently available resources, we evaluate our mod-
els on three tasks: follow-up QA, error correction,
and feedback utilization.

The follow-up QA (FQA) task is assessed using
the MathChat dataset (Liang et al., 2024). Each
test instance consists of three turns of questions.
The first turn uses the original GSM8k test set, and
subsequent turns contain follow-up questions based
on earlier turns. These follow-ups often require a
deeper understanding of the problem, such as per-
forming subsequent calculations based on previous
answers or introducing new constraints to the origi-
nal question. The solutions generated by the model
for each turn are incorporated into the input for the
next turn, creating a multi-turn interaction. The
accuracy of each turn is evaluated separately.

The error correction (EC) task, also sourced
from the MathChat dataset and derived from the
GSM8k test set, pairs each question with an inten-
tionally incorrect answer. The model is then tasked
with identifying and correcting errors in the reason-
ing process. Accuracy is determined by comparing
the model’s corrected answer to the ground truth.
For both tasks from MathChat, we follow the
approach of Liang et al. (2024) to concatenate all
previous turns into the instruction part of the input
sequence. For example, in the third round of FQA,
the model decodes P(a3|[q1; a1; q2; a2; q3]); In EC,
it decodes P(a|[q; awrong; f ]), where f is binary
feedback indicating that awrong is incorrect.

The MINT (Wang et al., 2024a) benchmark eval-
uates the ability of LMs to leverage natural lan-
guage feedback to improve their predictions. We
utilize the math subset from the original bench-
mark, which includes 273 carefully selected in-
stances from four datasets: 48 from GSM8k, 100
from MATH, 76 from MMLU, and 49 from Theo-
remQA (Chen et al., 2023). We adhere to the same
evaluation protocols as the original paper except
that we omit the code execution step as our math
models are based on text reasoning. At each in-

teraction turn, the model proposes a solution, and
we collect binary feedback on answer correctness
along with natural language feedback from an ex-
pert (i.e., GPT-4). This feedback is then provided to
the model in the subsequent turn of prediction. The
model have at most k = 5 chances to propose so-
lutions, and the accuracy of each turn is calculated
independently. We also measure the improvement
in accuracy (∆) from the first to the fifth turn to
assess the model’s efficacy in leveraging feedback.

B.3 Code Generation

HumanEval (Chen et al., 2021) and MBPP
(Austin et al., 2021) are the most popular bench-
marks for evaluating code generation capabilities
of LMs (Luo et al., 2023b; Wang et al., 2024b).
Each test instance within these benchmarks in-
cludes a natural language prompt, based on which
LMs generate a corresponding code snippet. The
correctness of the code is verified using test cases.
Additionally, EvalPlus (Liu et al., 2023) has de-
veloped enhanced versions of these benchmarks
(HumanEval+ / MBPP+) that include more com-
prehensive test cases for a more rigorous evaluation.
Therefore, we utilize the evaluation suite provided
by EvalPlus on these benchmarks, where MBPP is
reduced to 399 instances for quality control.

For the training dataset, we use the OSS-
Instruct dataset collected by Magicoder (Wei et al.,
2023), which consists of synthetic instruction-code
pairs generated from random code snippets sourced
from GitHub. Since HumanEval and MBPP focus
on Python code, we extracted the Python subset
from OSS-Instruct to reduce annotation costs, re-
sulting in a total of 38K training instances. Given
the abstractive nature of code generation, we opt
for analogy annotations in the follow-up reasoning
part of RefAug.

We adhere to the training settings outlined in
the Magicoder paper for our experiments. Models
are trained over two epochs with a batch size of
512. The learning rate is initiated at 5e-5, with 15
warm-up steps followed by a linear decay. Greedy
decoding is employed during inference.

C Baseline Implementation

In this section, we detail our implementation of
the major baseline methods that we compare with
in the main paper, including question augmenta-
tion (Q-Aug), answer augmentation (A-Aug), and
MetaMath augmentation.

C.1 Question Augmentation

A single round of Q-Aug enerates a new question
from each existing question in the training set, ef-
fectively doubling the dataset (illustrated in Fig-
ure 1b). Both the augmented question and its so-
lution are annotated by GPT-4-turbo. During the
annotation, we employ a temperature of 0.7 and a
top_p of 1.0 to ensure the diversity of math reason-
ing paths for both Q-Aug and A-Aug. we largely
follow the question generation prompt from Li et al.
(2024a) with minor adjustments. The detailed an-
notation prompt is provided in Figure 6.

C.2 Answer Augmentation

A single round of A-Aug involves re-sampling a
solution for each math problem in the training set.
The new solution, paired with the original ques-
tion, forms a new training instance (illustrated in
Figure 1c). Consistent with other methods, the aug-
mented solution is generated by GPT-4-turbo. If
the sampled solution diverges from the gold answer,
it is discarded and re-sampled; And if a correct an-
swer is not produced after five attempts, we retain
the last sampled solution. Following the methodol-
ogy described by Yu et al. (2024), the prompt for
A-Aug simply instructs the model to solve an arbi-
trary math problem, which is detailed in Figure 7.

C.3 MetaMath

MetaMath (Yu et al., 2024) introduces a compre-
hensive suite of augmentation methods tailored for
math reasoning tasks, which has received much
attention. This suite includes answer augmentation,
question rephrasing, and two backward reasoning
augmentation techniques: self-verification (Weng
et al., 2023) and FOBAR (Jiang et al., 2023b). Each
method is sampled for multiple rounds to generate
a large set of 400K training data. Please refer to Yu
et al. (2024) for more details on these methods.

When creating the MetaMath40k subset for our
experiments in §4.1, we randomly select one in-
stance from each of the four augmentation tech-
niques for every seed math question, which we
believe is the most uniform sampling strategy. For
the MetaMath80k subset, we add one more in-
stance from each technique for every seed ques-
tion. The initially sampled 40K instances are fur-
ther equipped with RefAug to be included in the
full-dataset training (MetaMath400k+RefAug40k).

Figure 5: Prompt used for training the model. Text
in gray are placeholders and will be replaced by the
corresponding sections in the training instance.

D Training Prompt

The prompt we use to build training sequences
is shown in Figure 5. The format mainly fol-
lows Wang et al. (2023b), and the reflection sec-
tion is appended to the original answer as the
output. Loss is only calculated to tokens after
<|assistant|>.

E RefAug Annotation Prompt

The prompt we use for annotating reflective sec-
tions are detailed in Figure 8, which includes a de-
scription of the general principles of reflective rea-
soning and two in-context examples. We use tem-
perature=0.7 and top_p=1.0 when sampling with
GPT-4-turbo.

F License of Artifacts

We note that the collection of RefAug data, if anno-
tated by an external model, should comply with its
terms of use. For example, using GPT-generated
data is subject to the terms of use of OpenAI ser-
vices5, and using LLaMA-generated data is subject
to Meta’s LLaMA license agreement6.

5https://openai.com/policies/terms-of-use/
6https://llama.meta.com/llama3/license/

TrainingPrompt<|system|>Below is an instruction that describes a task. Follow the instruction to complete the request.<|user|>{Question}<|assistant|>{Answer}Reflection:{Reflection}Figure 6: Prompt for question augmentation, adopted from Li et al. (2024a). The only difference is that we combine
question generation and solution annotation into a single prompt to save costs.

Figure 7: Prompt for answer augmentation, which is basically an in-context learning prompt for solving a given
math problem. Two in-context examples come from MATH and GSM8k training sets, respectively.

Question Augmentation PromptPlease act as a professional math teacher. Your goal is to create high quality math problems to help students learn math. Youwill be given a math question. Please generate a similar but new question according to the Given Question.You have four principles to do this.# Ensure the new question only asks for one thing, be reasonable, be based on the Given Question, and have a definite answer.For example, DO NOT ask, "what is the amount of A, B and C?".# Ensure the new question is in line with common sense of life. For example, the amount someone has or pays must be a positive number, and the number of people must be an integer.# Ensure your student can answer the new question without the given question. If you want to use some numbers, conditions or background in the given question, please restate them to ensure no information is omitted in your new question.# Ensure your created question is solvable. Write the solution to it after the question.Given Question: $$QUESTION$$Now write a new question and its solution. The question must begin with "New Question:" and the solution must begin with "Solution to the New Question:". The solution must end with "The answer is XXX" where XXX should be the final answer to the question.Answer Augmentation PromptYour task is to solve a math word problem. You should solve the problem step by step. At the end of your solution, write the final answer in the form of "The answer is X". Here are two examples:## Example 1Question:Let 𝐹!=(0,1)	and 𝐹"=(4,1). Then the set of points 𝑃such that 𝑃𝐹!+𝑃𝐹"=6	form an ellipse. The equation of this ellipse can be written as ($%&)!(!+()%*)!+!=1. Find ℎ+𝑘+𝑎+𝑏.Solution:We have that 2𝑎=6, so 𝑎=3.  The distance between the foci is 2𝑐=4, so 𝑐=2.Hence, 𝑏=𝑎"−𝑐"=5. The center of the ellipse is the midpoint of 𝐹!𝐹", which is (2,1).	Thus, the equation of the ellipse is ($%")!,!+()%!)!(-)!=1. Hence, ℎ+𝑘+𝑎+𝑏=2+1+3+5=6+5.	The answer is 6+5.## Example 2Question:Each bird eats 12 beetles per day, each snake eats 3 birds per day, and each jaguar eats 5 snakes per day. If there are 6 jaguars in a forest, how many beetles are eaten each day?Solution:First find the total number of snakes eaten: 5 snakes/jaguar ×6 jaguars =30 snakes. Then find the total number of birds eaten per day: 30 snakes ×3 birds/snake =90 snakes. Then multiply the number of snakes by the number of beetles per snake to find the total number of beetles eaten per day: 90 snakes ×12 beetles/snake =1080 beetles. The answer is 1080.Now solve the following problem. The solution must end with "The answer is XXX" where XXX should be the final answer to the question.Question:$$QUESTION$$Solution:Figure 8: Prompt for annotating the reflective section. The prompt first explains the contents to annotate within
the reflective section, and then presents two in-context examples for demonstration. GPT-4-turbo is employed for
annotation.

DataAnnotationPromptYou are a professional math teacher, and your goal is to teach your student to learn a given math problem. Now that your studenthas successfully solved the original problem,in order tomake the student thoroughly understand the involved knowledge and problem-solving methodology, your task is to write a reflection section that go through the problem-solving process and provide additional insights. The reflection section should include the following components:1. Alternative Reasoning: Present an alternative approach to solve the original problem. This alternative approach should be distinct from the originalsolution and still lead to the correct answer. While writing the alternative reasoning approach, consider explaining the principle of the methodology used in the original solution, how the alternative approach differs from the original method, and why it leads to the same correct answer.2. Follow-up Reasoning: Associate the solution to a broader class of problems. You can either create a general form of the original problem to encouragethe student to reduce reliance on specific values (e.g., use letters or variables to replace specific numbers in the original problem), or apply the concepts and methodologies from the original problem to a more challenging situation. Please do not just replace the original numbers in the question with new numbers, because that isessentiallythe same problem. The follow-up problem must also be solvable, and you need to provide the solution for it. Besides, please explain briefly how the new scenario associates with the original problem.Example 1:Original Problem:Youngsville had a population of 684 people. The town had a growth spurt and the population increased by 25% then they witnessed that 40% of the population moved away. What is the current population?Solution to the Original Problem:The town had 684 people, and then had a 25% growth spurt, so the population increased by 684×0.25=171	people. This increase brought the population to 684+171=855	people. 40% of the population moved away, so 855×0.40=342	people moved away. The new population is 855−342=513	people. The answer is 513.Alternative Reasoning:The key to solve the problem is to understand the concept of relative increase and decrease percentages. Increasing by 𝑎%means the population grows to (100+𝑎)%	of the original, while decreasing by 𝑏%means the population reduces to (100−𝑏)%	based on the increased population. Therefore, this is essentially a problem of consecutive multiplication: multiply the initial total population by the percentage of change twice.Therefore, an alternative calculation involves deriving a single effective percentage change of the whole process. A 25% increase is equivalent to multiplying by 1.25, and a 40% decrease is equivalent to multiplying by 0.60. Combining these two changes, the effective percentage change is 1.25×0.60=0.75, which corresponds to a 25% decrease from the original population. Therefore, the current population is 684×0.75=513. The alternative approach leads to the same result because the associative property of multiplication: (684×1.25)×0.60=684×(1.25×0.60)=684×0.75=513.Follow-up reasoning:Let‘s think of a more general scenario. Suppose a town has a population of 𝑃people. The population increases by 𝑎percent, then 𝑏percent of the population moves away, and we would like to know the final population. In this context, the first increase corresponds to multiplying by (1	+	𝑎/100),	and the subsequent decrease corresponds to multiplying by (1	−	𝑏/100).	Sothe total population change is(1	+	𝑎/100)(1	−	𝑏/100).	Therefore, the final population is 𝑃(1	+	𝑎/100)(1	−	𝑏/100).	This abstract problem allows us to apply the same principles of relative percentage changes to calculate the final populationbased on the initial population and the two percentage changes. This generalization helps to understand the problem conceptually and apply it to various scenarios.Example 2:Original Problem:Solve the equation (𝑥−99)(𝑥−101)=8.Solution to the Original Problem:Let t=x-100. Then the equation becomes (𝑡−1)(𝑡+1)=8, which transforms into 𝑡!−1=8. Therefore, 𝑡=3or 𝑡=−3, and accordingly we get 𝑥=97	or 𝑥=103. The answer is 97or 103.Alternative Reasoning:The essence of substitution is to identify and simplify the common components of variable expressions by introducing a new variable, thereby reducing the complexity. Let's revisit the original equation. Expressions 𝑥−99and 𝑥−101share a similar form: a large constant offset from 𝑥. Due to the minimal difference between 99and 101, we can use substitution to transform the expressions into terms with small constants.Therefore, an alternative approach is to substitute 𝑡=𝑥−99, which transforms the equation into 𝑡(𝑡−2)=8⇒𝑡!−2𝑡−8=0. This can be easily factorized into (𝑡−4)(𝑡+2)=0. Hence, 𝑡=4	or 𝑡=−2, leading to the same results 𝑥=97	or 𝑥=103. This alternative approach is equally effective as it also simplifies the equation by substituting 𝑥and reducing the scale of the offset terms.Follow-up Reasoning:Extending the idea of substitution, consider the equation 𝑥(𝑥+1)(𝑥+2)(𝑥+3)=360. We notice that 𝑥(𝑥+3)=𝑥^2+3𝑥, and (𝑥+1)(𝑥+2)=𝑥!+3𝑥+2. Therefore, to simplify the expression, we set the common term	𝑥!+3𝑥	as 𝑡, which transforms the equation into 𝑡(𝑡+2)=360⇒𝑡!+2𝑡−360=0⇒	𝑡=−20	or 𝑡=18. If 𝑡=−20, then 𝑥!+3𝑥+20=0. Here, the discriminant Δ=−71<0, resulting in no real solutions for 𝑥. If 𝑡=18, then 𝑥!+3𝑥−18=0, so 𝑥=3	or 𝑥=−6. This scenario reiterates the importance of identifying common components of x to streamline the equation through substitution.Now write a reflection section for the following case based on the examples above. Make sure to use "Alternative Reasoning:" and"Follow-up Reasoning:" to separate the two components.Original Problem:$$QUESTION$$Solution to the Original Problem:$$RESPONSE$$