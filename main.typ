#import "@preview/tracl:0.8.1": *
#import "@preview/pergamon:0.7.0": *
#import "@preview/booktabs:0.0.4": *
#show: booktabs-default-table-style
#import "macros.typ": *
#show "https://": ""
#set math.equation(numbering: "(1)")
#show table.cell: text.with(size: 9pt)
#set footnote.entry(clearance: 0.4em)

#let ANONYMOUS = false
#show: acl.with(
  title: [
    Optimal Allocation of Evaluation Effort
    // Optimal Allocation of Human Evaluation Effort
    // Allocating Human Evaluation Effort via Bandits
    // Human Evaluation as Best-Arm Identification

    #text(
      fill: red,
      weight: "medium",
      size: 11pt,
    )[this is only a preliminary investigation, framing and experiments; bound to change]
  ],
  anonymous: ANONYMOUS,
  authors: hide(make-authors(
    name: "authors",
    affiliation: [
      affiliations\ #email("vzouhar@ethz.ch")
    ],
  )),
)

#set enum(indent: 0pt)
#set list(indent: 0pt)
#if ANONYMOUS {
  TODO = body => none
}

#abstract[
  Human evaluation of machine translation remains the gold standard but suffers from prohibitive costs and poor scalability.
  Traditional evaluation protocols waste effort by exhaustively evaluating all models on the entire benchmark, a safe but inefficient approach when the primary objective is to identify top-tier models.
  We formalize human evaluation as a best-arm identification in a multi-armed bandit setup with correlated arms.
  By sampling adaptively based on the intermediate model rankings collected so far during the evaluation, we can focus the annotation budget on the most competitive models.
  We prove the optimality of the proposed algorithms.
  Dynamic evaluation improves the discrimination between top-performing models, increases the statistical power, and enables large-scale evaluations of many models without having to apriori guess and remove low-performing models.
  #if ANONYMOUS { footnote[Anonymized code submitted, available for camera-ready.] } else {
    footnote[https://github.com/zouharvi/evaluation-bandit]
  }
]

// #let paragraph = (body) => [
//   #box()[===== #body]
// ]

// #v(10pt)
= Introduction

Human evaluation is essential for guiding research and deployment decisions, yet it remains unsustainable due to the massive budgets required for annually obtaining exhaustive rankings #citep("graham-etal-2013-crowd", "kocmi-etal-2025-findings").
In the absence of human supervision, researchers must rely on automatic metrics which are often only modestly correlated with quality, susceptible to reward-hacking, and less accurate for state-of-the-art models #citep("lavie-etal-2025-findings").
Various modifications have been proposed to improve assessment quality #citep("graham-etal-2013-continuous", "freitag-etal-2021-experts", "kocmi-etal-2024-error"), or selecting informative evaluation items #citep("subset2evaluate", "proietti-etal-2025-estimating").
Unfortunately, these efforts typically operate within a static framework where all models are evaluated on all items.
This means that to be able to find out which of the top 3 models is truly the best, we still have to evaluate all models on more evaluation items.

#figure(
  {
    set text(weight: "bold", size: 9pt)
    h(1fr)
    box(width: 45%)[Traditional: Uniform]
    h(1fr)
    box(width: 40%)[Proposed: Dynamic]
  }
    + v(-3pt)
    + image("img/intro_figure.svg")
    + v(-2pt),
  caption: [In two approaches we evaluate the same number of items but in dynamic evaluation we automatically focus on top-performing models. This informs a better choice of close-matching state-of-the-art models.],
  // Shaded areas correspond to confidence intervals.
  placement: top,
)
<fig:intro>

We frame this as a search problem and introduce a dynamic evaluation policy that treats human annotators as a finite resource allocated via a multi-armed bandit framework.
Through weighted sampling, we automatically concentrate evaluation effort on top-performing models, which is ultimately more useful in determining the best model.
As illustrated in @fig:intro, this contrasts with standard uniform allocation: by discarding poor models early, we reserve the annotation budget for resolving uncertainty among the top tier.

This approach can be shown to be mathematically optimal with respect to two evaluation goals in cases when the order of evaluation items is fixed.
If this assumption does not hold, modifications to the estimator of model performance still make the algorithms applicable.
In this work we:
- formalize the problem as a constrained best-arm identification (@sec:dynamic_evaluation).
- propose algorithms (@sec:algorithms) which we theoretically prove to be optimal for two evaluation objectives: ranking and quality data collection.
// We consider cases with both simple mean-based and advanced estimators, which makes our bandit-based  algorithms compatible with evaluation item selection strategies (@sec:algorithms).
- incorporate competition modelling which compensates for biased evaluation items.
- simulate dynamic evaluation on past data for translation benchmarking in WMT (@sec:experiments), demonstrating that dynamic allocation yields superior rankings of top models compared to static baselines.
This unlocks efficient large-scale evaluations with many competing models.


= Related Work
<sec:related_work>

// For bandit specifically, #citep("kreutzer2018neuralmachinetranslationimproved", "kreutzer2021banditsdontfollowrules", "sokolov2016banditstructuredpredictionlearning", "wmt_bandit_learning_task_2017") have applied it to translation, though the goal was to improve the translation models, rather than to evaluate them.


One of the goals of human evaluation is to be able to rank models.
Typically, this is bottlenecked by human ability to accurately assess output quality, or the economy and logistics of evaluation campaigns.
See @sec:related_work_extended for an extended related work discussion.

#paragraph[Matchmaking and model evaluation.]
Pairwise comparisons allow for fast human evaluations.
Mathematical models, such as ELO or TrueSkill, then infer global model rankings from noisy pairwise outcomes #citep("elo1978rating", "herbrich2006trueskill", "minka2018trueskill").
Recent work adapts this paradigm to language models: Chatbot Arena aggregates open-ended pairwise judgments via ELO #citep("zheng2023judging"), while K-Sort Arena employs bandit-style exploration to improve tournament efficiency #citep("Li_2025_CVPR").

These settings assume that matches are independent due to free-form inputs.
In contrast, we wish to evaluate models based on a specific dataset.
Standard approaches aim to best estimate the full model ranking #citep("sakaguchi-van-durme-2018-efficient", "balkır2026confidentrankingsfeweritems").
Our objective is not to rank every model, but to rapidly discard weak candidates and concentrate the budget on resolving uncertainty among top models.
Additionally, we consider the practical restrictions on evaluating as many common items in the dataset as possible.

#paragraph[Evaluation item selection.]
A separate line of work, distinct from model allocation, optimizes the set of items to annotate.
Subset selection methods approximate full-benchmark rankings using a small budget of difficult or otherwise informative items #citep("tinyBenchmarks", "subset2evaluate", "proietti-etal-2025-estimating").
These strategies typically assume a uniform allocation of effort across models and optimize _item_ choice.
However, we show that smart evaluation item selection approaches are compatible with selecting which models to evaluate.
// Our framework fundamentally addresses the orthogonal problem of _model_ allocation.
// We thus first treat the item queue as a given input and dynamically schedule which models to evaluate on it.


= Which Models and Items to Evaluate?
<sec:dynamic_evaluation>

In this section we formalize the problem of dynamic evaluation together with the optimization goals (dubbed "objective" for disambiguation).

== Problem (general)
<sec:problem_statement>

We first describe a general form of the dynamic evaluation problem and then later constrain it, which we use in the rest of the work.

#paragraph[Problem (general).]
Let $cal(M)$ be a set of models, $cal(X)$ a set of atomic evaluation items, and budget $B in bb(R)^+$.
Given the output of model $m$ on item $x$, we have its evaluation $r_(x, m) in [0, 1]$ and also the cost of this evaluation, $c_x in bb(R)^+$.
At each step $1 <= i <= B$ we select a particular item $x in cal(X)$ and model combination $m in cal(M)$, which is stored $R_i = R_(i-1) union \{ chevron.l x, m chevron.r \}, R_0 = \{\}$.
During the selection at step $i$, we have access to previous results $R_(i-1)$.

// Our goal is to make such a sequence of selections for evaluation to maximize $"Objective"(R^dagger)$ such that $sum_(x, m in R^dagger) c_x <= B$.

Formally, we define the Allocation policy $pi$ that maps from history to the next selection:
$
      pi & = union.big_(i in bb(N)_0) (cal(X) times cal(M))^i arrow cal(X) times cal(M) \
  R_i^pi & = R_(i-1)^pi union { pi(R_(i-1)^pi) }
$

We find the last selection step within budget:
$
  #h(-10pt) T_pi &= max { 1 <= t <= |cal(X)| #h(1pt) mid(|) #h(-1pt) limits(sum)_(chevron.l x, m chevron.r in R_t^pi)#h(-10pt) c_x <= B }
$

Ultimately, we wish to obtain the policy that optimizes the objective of the final evaluation result under the budget.
$
  pi^* & = "argmax"_pi "Objective"(R_(T_pi)^pi)
$


#paragraph[Objective of policy $bold(pi)$.]
Given two active learning policies $pi_A$ and $pi_B$, how do we know which one is better?
We offer two complementary perspectives that instantiate $"Objective"(R^pi_(T_pi))$.
For convenience we define $r_(*,m)$ to be the set of results $\{r_(x,m) | x in cal(X), chevron.l x, m chevron.r in R \}$ and $"rank"_m$ and $"avg"_m$ to be the rank and average of a model with respect to a particular timestep:

The simplest objective is the correlation between model ranking given $R_B$ against "true model ranking" given by the full $cal(X) times cal(M)$.
Common ranking correlations equally weight ranking of top-performing models and poor models, which is misaligned with our goal to discriminate between top models.
For this reason, we turn to *weighted Kendall's $tau$* #citep("vigna2015weightedtau") in which exchanges of high weight are more influential than exchanges of low weight.
the weighting function is arbitrary such as the reciprocal of the rank, $1/"rank"_m$ #box(v(-3pt) + image("img/microplot_weight_pow1.svg", height: 1em), height: 0.5 * 1em), which assigns higher weights to top-ranking models, or a step-function that considers only top 3 models #box(v(-3pt) + image("img/microplot_weight_top3.svg", height: 1em), height: 0.5 * 1em), only top 1 model #box(v(-3pt) + image("img/microplot_weight_top1.svg", height: 1em), height: 0.5 * 1em), or a more skewed version of the harmonic sequence, $1/"rank"_m^2$ #box(v(-3pt) + image("img/microplot_weight_pow3.svg", height: 1em), height: 0.5 * 1em), which we choose.

Second, human evaluation data frequently serves as a training signal for reward models or direct preference optimization.
The information content of an annotated comparison is not uniform: distinguishing between top-tier models yields high signal, whereas comparisons involving weak models are often trivial and contribute little to the reward model's discriminative power at the frontier.
Concentrating the budget on the best models therefore maximizes the informativeness of the collected dataset for future alignment.
We operationalize this as *evaluation focus*, the amount of evaluations per model.
We scale this amount with a logarithm such that this objective can not be optimized by simply allocating all evaluations to the top model.
Again, we use $1/"rank"$ for weighting:
// proportional to a normalized weighing function.
// , which allows to easily normalize the computation by $B$ (perfect allocation to top model).
// $
// #h(-0.5em)
//  (Z dot sum_(m in cal(M)) 1/"rank"_m dot |\{ x#h(1pt)|#h(1pt)chevron.l x, m chevron.r in R_B^pi \}|)/B
// $

$
  #h(-0.5em)
  // "e.f."(m) = 1/"rank"_m dot log |\{ x#h(1pt)|#h(1pt)chevron.l x, m chevron.r in R_B^pi \}| \
  sum_(m in cal(M)) 1/"rank"_m dot log |\{ x#h(1pt)|#h(1pt)chevron.l x, m chevron.r in R_B^pi \}|
$
<eq:evaluation_focus>

In @sec:experiments_extended we discuss other objectives, such as average payoff, statistical power and common ranking correlations.

// == Constrained Problem Statement
// <sec:problem_statement_constrained>


#v(5mm)
#h(-1.1em)
We now discuss a set of assumptions which simplify the problem statement.

#heading(level: 2, numbering: none)[Assumption 1: Not repeating annotations]

In the current statement it is possible to select the same evaluation item for the same model multiple times.
This improves the estimate at a particular evaluation item due to the reduced annotation noise.
However, this is suboptimal for estimating the model mean due to the wasted information in covariance of the duplicate items.

#lemma("thm:non-repeating", [Non-repeating annotations], [
  The variance of the sample mean estimator $hat(mu)_m$ of a model $m in cal(M)$ is strictly minimized when the selection policy $pi$ selects distinct items.
])

// See @proof-thm:non-repeating.
Empirically with real annotations this has also been shown by #citet("riley2025reannotation") and we can thus safely avoid duplicate model-item selections.

#heading(level: 2, numbering: none)[Assumption 2: Maximizing item overlap]

Evaluation noise stems not only from annotator disagreement but from the variance in difficulty across evaluation items.
When comparing two models on different sets of items, we introduce a confounder.
For evaluation practitioners, it is more beneficial to have models evaluated on the same set items so that they can be more directly compared.
While we still wish to allow for the better models to be evaluated more often, we wish to prevent models having the same number of evaluations but on different items.

Formally, let $R_m$ be the set of evaluation items proposed by policy $pi$ for model $m$.
Then, we require that $ #h(-5mm)
forall m, m' in cal(M): R_m subset.eq R_m' or R_m' subset.eq R_m $
// (item difficulty + model quality)
Importantly, under an additive error model, maximizing overlap minimizes the variance of the estimator for the difference in model quality.
#lemma("thm:maximize-overlap", [Maximizing item overlap], [
  // #TODO[minimizes variance of $Delta$ between models]
  Let $hat(Delta)_(m, m')$ be the estimator for the quality difference between two models $m$ and $m'$, given evaluation sets $R_m$ and $R_m'$ of fixed sizes.
  Under an additive random effect model where item difficulties are independent and identically distributed, the variance $"Var"(hat(Delta)_(m, m'))$ is strictly minimized when the intersection $|R_m inter R_m'|$ is maximized.
  // This condition is uniquely satisfied for all pairs if the evaluation sets form a nested sequence, i.e., $X_a subset.eq X_b$ or $X_b subset.eq X_a$.
])

// See @proof-thm:maximize-overlap.
Practically, if we maximize the overlap between the two models, we can use paired statistical tests in post-hoc comparison rather than independent tests.
#TODO[here also list existing works that fulfill this asusmption. perhaps mini review?]

As a corollary, if further for all models $m, m': |R_m minus R_m'| lt.eq 1$ then the problem statement is equivalent to having a fixed item ordering and the policy $pi$ only chooses which model to evaluate next.
Effectively this reduces the dimensionality of the bandit action space from $|cal(X)|times |cal(M)|$ to $|cal(M)|$.
We accept this simplification and practically we order $cal(X)$ based on some ordering $prec$.



#heading(level: 2, numbering: none)[Assumption 3: Unbiased item ordering]

Having a fixed evaluation item ordering exposes a critical vulnerability.
For example, if item are ordered based on difficulty, such as difficult items first, simple average estimators become biased.
A model evaluated on fewer items is thus more at a disadvantage as it has been exposed only to difficult items. To retain the validity of simple averaging, the fixed ordering must be statistically independent of item difficulty.

Formally, let the fixed evaluation queue be the sequence $x_1 prec x_2 prec dots prec x_(|cal(X)|)$.
We assume that the expected score of a model $m$ is invariant to the position $i$ in the queue:
$ bb(E)[r_(x_i, m)] = mu_m $

#lemma("thm:unbiased-ordering", [Unbiased item ordering], [
  If the item difficulty is strictly monotonic with respect to the item ordering (i.e., items get progressively harder or easier), then the simple average estimator $hat(mu)_m$ is a biased estimator of true quality $q_m$ when comparing models with different budget allocations.
])
<thm:unbiased_item_ordering>

// See @proof-thm:unbiased-ordering.
This assumption is necessary only because we rely on simple averages ($hat(mu)_m$) for efficiency and interpretability.
By default this makes it incompatible with smart evaluation item selection #citep("proietti-etal-2025-estimating", "subset2evaluate").
Later, we relax this constraint by adopting estimators that explicitly model item difficulty, allowing for arbitrary item orderings.

#heading(level: 2, numbering: none)[Assumption 4: Equal evaluation cost]

Evaluating a very long text, for example, takes much longer and is therefore more expensive than evaluating a short text.
However, features like text length or number of expected errors is often correlated with the evaluation cost #citep("zouhar-etal-2025-ai").
When using a simple average estimator, it might be biased if we prioritize models whose next evaluation item is e.g. cheap.
We therefore assume $ forall x in cal(X): c_x = 1 $

Later we also relax this assumption by incorporating cost $c_x$ into item ordering, where the objective becomes prioritizing items with highest information gain per unit cost rather than per item.
// I.e. we don't let the bandit make use of costs because they can only see decide on a horizon of 1

== Problem (updated)


By accepting Assumptions 1 and 2 the policy turns into predicting solely which next model to evaluate (which is then evaluated on the next item it has not been evaluated on).
$
      #h(-1em)pi & : scripts(union.big)_(i in bb(N)_0) (cal(X) times cal(M))^i arrow cal(M) \
       #h(-1em)x & = min_prec \{ x#h(1pt)|#h(1pt)x in cal(X), chevron.l x, pi(R_(i-1)^pi) chevron.r in.not R_(i-1) \} \
  #h(-1em)R_i^pi & = R_(i-1)^pi union { chevron.l x, pi(R_(i-1)^pi) chevron.r }
$
By later relaxing Assumptions 3 and 4 we also define a new ordering $prec$ on $cal(X)$, potentially based on the evaluation item cost, which determines the evaluation queue.



#figure(
  line(length: 100%)
    + v(-10pt)
    + algo2(
      parameters: ($M$, $B$).map(x => text(size: 9pt, x)),
      title: [
        #set text(size: 9pt)
        *Input*: models $M$, budget $B$ \
        *Output*: evaluation selection $R$ \
        #v(0pt) #h(-1.25em)
        *SuccessiveReject*
      ],
    )[
      $h = (2 B) / (|cal(M)|^2 + |cal(M)| - 2)$ \
      $R = emptyset$ \
      while $|cal(M)| > 1$: #i \
      repeat $h$: for $m in cal(M)$: #i \
      $x arrow.l min_prec \{ x#h(1pt)|#h(1pt)x in cal(X), chevron.l x, m chevron.r in.not R \}$ \
      $R arrow.l R union \{ chevron.l x, m chevron.r \}$ #d \
      $m_"worst" = "argmin"_(m in cal(M)) sum_(chevron.l x, m chevron.r in R) r_(x, m)$ \
      $cal(M) arrow.l cal(M) without \{ m_"worst" \}$ #d \
      return $R$
    ]
    + v(-10pt)
    + line(length: 100%)
    + v(-5pt),
  placement: top,
  kind: "Algorithm",
  supplement: "Algorithm",
  caption: [Allocation policy in phases based on successive rejection of the worst model. The $h$ is computed so that $B =& sum_(i=0)^(|cal(M)|-2) (|cal(M)|-i) dot h$.],
)
<alg:successive_rejects>

#figure(
  line(length: 100%)
    + v(-10pt)
    + algo2(
      parameters: ($cal(M)$, $B$, $C$).map(x => text(size: 9pt, x)),
      title: [
        #set text(size: 9pt)
        *Input*: models $cal(M)$, budget $B$, cold start phase $C$ \
        *Hyperparameter*: cold start $C$, weighter $F_(R)#h(1pt):#h(1pt)cal(M)#h(1pt)arrow#h(1pt)bb(R)^+$#h(-10pt) \
        *Output*: evaluation results $R$ \
        #v(0pt) #h(-1.25em)
        *WeightedSampling*
      ],
    )[
      $R arrow.l emptyset$ \
      for $m in cal(M)$: #i \
      repeat $C$: #i \
      $x arrow.l min_prec \{ x#h(1pt)|#h(1pt)x in cal(X), chevron.l x, m chevron.r in.not R \}$ \
      $R arrow.l R union \{ chevron.l x, m chevron.r \}$ #d#d \
      while $|R| < B$: #i \
      $m arrow.l "sample" m#h(1pt)in#h(1pt)cal(M) "with prob." prop omega_w$ \
      // $m arrow.l "sample" m in cal(M) "with probability" prop (omega_w) / (sum_(m in cal(M)) F_R (m))$ \
      $x arrow.l min_prec \{ x#h(1pt)|#h(1pt)x in cal(X), chevron.l x, m chevron.r in.not R \}$ \
      $R arrow.l R union \{ chevron.l x, m chevron.r \}$ #d \
      return $R$
    ]
    + h(0em)
    + box(width: 90%)[
      #set text(size: 9pt)
      #set align(left)
      #set math.equation(numbering: none)
      #set par(leading: 0.4em)
      $
        epsilon"-Greedy" & omega_w = 1"-"epsilon "if" "rank"_m"="1 "else" epsilon\/(|cal(M)|) \
              "Bolzmann" & omega_w = exp("avg"_m\/tau) #h(-10pt) \
                  "Rank" & omega_w = 1 \/ "rank"_m \
        // "Rank-sqrt" &omega_w = sqrt(1 \/ "rank"_m) \
                   "UCB" & omega_w = "avg"_m + gamma dot sqrt(ln(|R|) \/ |R_m|)
      $
    ]
    + line(length: 100%)
    + v(-5pt),
  placement: top,
  kind: "Algorithm",
  supplement: "Algorithm",
  caption: [Allocation policy with weighted sampling and four weighting functions. By default we use $C=5$, $tau=1$, $epsilon=50%$, and $gamma = 100 dot sqrt(2)$.],
)
<alg:weighted_sampling>

// #figure(
//   line(length: 100%) + v(-10pt) +
//   algo2(
//     parameters: ($cal(M)$, $B$, $C$).map(x => text(size: 9pt, x)),
//     title: [
//       #set text(size: 9pt)
//       *Input*: models $cal(M)$, budget $B$ \
//       *Hyperparameter*: cold start $C$, top-$k$, exploration $gamma$ \
//       *Output*: evaluation results $R$ \
//       #v(0pt) #h(-1.25em)
//       *UpperConfidenceBound*
//     ],
//   )[
//     $R arrow.l emptyset$ \
//     for $m in cal(M)$: #i \
//       repeat $C$: #i \
//         $x arrow.l min_prec \{ x#h(1pt)|#h(1pt)x in cal(X), chevron.l x, m chevron.r in.not R \}$ \
//         $R arrow.l R union \{ chevron.l x, m chevron.r \}$ #d#d \
//     while $|R| < B$: #i \
//        $cal(M)_"topk" arrow.l "top-"k_(m in cal(M)) "avg"_m + gamma dot sqrt(ln(|R|) / (|R_m|))$ \
//        for $m in cal(M)_"topk"$: #i \
//         $x arrow.l min_prec \{ x#h(1pt)|#h(1pt)x in cal(X), chevron.l x, m chevron.r in.not R \}$ \
//         $R arrow.l R union \{ chevron.l x, m chevron.r \}$ #d#d \
//     return $R$
//   ] + v(-10pt) + line(length: 100%) + v(-5pt),
//   placement: auto,
//   kind: "Algorithm",
//   supplement: "Algorithm",
//   caption: [Allocation policy based on Upper Confidence Bound with top-$k$ models. By default we use $C"="5$, $gamma"="100 dot sqrt(2)$ and $k"="1$.]
// )
// <alg:ucb>

= Algorithms
<sec:algorithms>

// We consider multiple strategies for dynamic model selection, ranging from a uniform baseline to active learning policies.
We frame this optimization as a best-arm identification problem in multi-armed bandits, where the objective is to identify the optimal model with high probability rather than maximizing cumulative reward #citep("audibert2010best").
This shift focuses the policy on pure exploration and reducing uncertainty around top contenders given a finite budget.

See @sec:algorithms_extended for other intuitive though underperforming algorithms.

#paragraph[Baseline] (@alg:baseline).
The simplest baseline just draws randomly from $cal(M)$.
However, even in practice a more reasonable baseline is used in which every model is evaluated on an identical number of items, $B/(∣cal(M)∣)$.
While it provides an unbiased estimate of the global ranking, it wastes budget on clearly inferior models that do not require further discrimination.

== Multi-armed Bandits

We formalize the evaluation policy $pi$ as a best-arm identification problem #citep("audibert2010best"), specifcially with correlated arms #citep("Gupta_2021").
Each model $m in cal(M)$ constitutes an arm; pulling arm $m$ reveals its score on the next available item $x$ in the evaluation item queue.
Unlike the classic regret minimization objective, which seeks to maximize the cumulative reward, our setting is one of pure exploration.
We seek to allocate the fixed budget $B$ to minimize the uncertainty regarding the identity of the optimal model $m^*$.

#paragraph[Successive Rejects] (@alg:successive_rejects).
Grounded in the framework of best-arm identification, this policy operates in phases, discarding the worst-performing model at each step to concentrate the budget on top-ranking candidates.
We appeal to #citet("audibert2010best") for theoretical guarantees on identification of the optimal model.
Similar methods exist with different scheduling, such as successive halving #citep("jamieson2016non").
Methods which discard models from a competition harbor the practical risk of annotation drift: as the phases progress, annotators are exposed to an increasingly high-quality pool of models, which may cause them to recalibrate their internal scoring metrics and assign harsher scores.

#paragraph[Weighted Sampling] (@alg:weighted_sampling).
This policy continuously samples across the entire model space.
We use four version with different sampling probability strategies.
The first is $epsilon$-greedy sampling, which evaluates the top model and occasionally (in $epsilon$ cases), the other models.
The second is Boltzmann distribution over the current score averages (proportional to $exp("avg"_m \/tau)$), a standard exploration strategy in reinforcement learning #citep("sutton2018reinforcement").
The third and fourth are based on the inverse of the model rank, which draws on selection mechanisms from evolutionary computation #citep("baker1985adaptive") and is not sensitive to the scale.
// We include the square root to later show optimality.
This approach affords the greatest flexibility in focusing evaluation on top-tier models, though it necessitates a cold-start phase ($C$ items) to stabilize initial estimates.
Lastly, upper confidence bound is a standard algorithm that balances the exploration-exploitation trade-off by selecting models that maximize the Upper Confidence Bound score #citep("auer2002finite").
The score consists of the empirical mean estimate of the model average (exploitation) and a confidence term based on the number of times the model has been evaluated (exploration).
Crucially, the exploration term ensures that even models with initially low scores are revisited as the uncertainty about their true average remains high relative to the growing total budget.
// While the standard upper confidence bound algorithm selects a single arm, we adapt it to select top-$k$ models to align more with the task of ranking top-scoring models better.

// #TODO[
//   - LUCB or TopTwo?
//   - The lemma might not hold for LUCB and UCB
// ]

Weighted sampling has an advantage over successive rejects or successive halving: when running weighted sampling we can dynamically continue even with changing budget.
As opposed to successive rejects, non-UCB weighted sampling is consistent; i.e. with enough samples, the method will produce the correct model ranking ($tau = 1$).

// https://en.wikipedia.org/wiki/Consistency_(statistics)
#lemma("thm:consistency-weighted-sampling", [Consistency of weighted sampling])[
  Let $cal(M)$ be a finite set of models and Assumption 3 holds.
  Let $pi$ be a sampling policy such that for any step $t$, the selection probability for every model $m$ satisfies $P_t(m) >= zeta$ for some fixed constant $zeta > 0$.
  Then, the estimated ranking converges to the true ranking as $B arrow infinity$.
]
#v(-0.5em)
// See @proof-thm:consistency-weighted-sampling.
Furthermore, our weighted sampling algorithm (Rank-based) is the optimal allocation algorithms for evaluation focus and weighted sampling under a budgetary constraint, respectively.

#theorem("thm:optimality-evaluation-focus", [Optimality of weighted sampling for evaluation focus])[
  Sampling according to importance weights $omega_m$ is the optimal allocation strategy for maximizing the logarithmic evaluation focus objective among mean-based estimators.
  // Let $omega_w$ be the importance weight of model $m$ with which evaluation focus is computed.
  Then, sampling according to $omega_w$ is optimal among mean-based estimators.
]
#v(-0.5em)
// See @proof-thm:optimality-evaluation-focus.
In our for evaluation focus (@eq:evaluation_focus), we use $1/"rank"_m$ which corresponds to rank-based weighted sampling.

#theorem(
  "thm:optimality-weighted-tau",
  [Optimality of weighted sampling for weighted $tau$],
)[
  Assume the estimated score $hat(mu)_m$ follows an independent Gaussian distribution $cal(N)(mu_m, sigma^2/(|R_m|))$.
  Sampling according to $omega_m$ is the strictly optimal allocation strategy for minimizing the weighted (according to $omega_m^2$) ranking uncertainty loss among mean-based estimators.

  // Let $tau_omega$ be a weighted ranking metric where the importance of correctly ranking model $m$ is $omega_m$.
  // Then, sampling according to $sqrt(omega_w)$ is optimal among mean-based estimators.
  // (e.g., $1/"rank"_m$).
  // Under the approximation that pairwise error probabilities are linear in the sum of variances,
  // The sampling policy that maximizes $tau_omega$ subject to a fixed total budget $B$ is given by:
  // $ N_m prop sqrt(omega_m) $
]
#v(-0.5em)
// See @proof-thm:optimality-weighted-tau.
For the weighted $tau$ (@sec:problem_statement), we use $1/"rank"_m^2$ which corresponds to optimliaty of rank-based weighted sampling that samples according to $1 / "rank"_m$.


== Unconstrained Item Ordering
<sec:unconstrained_ordering>

To lift Assumptions 3 and 4 which mandate random evaluation item ordering, we have to adapt a less simplistic estimator of model performance.
This is similar to Item Response Theory or EASL #citep("balkır2026confidentrankingsfeweritems", "sakaguchi-van-durme-2018-efficient") which jointly model performance and item difficulty.


#paragraph[Predicting scores.]
Given a list of previous results, we estimate $r_(x,m)$ even for $chevron.l x, m chevron.r in.not R$.
We do so by assuming that the observed score is given by:
$ r_(x,m) = q_m + d_x + epsilon_(x,m) $
where $q_m$ is the true latent quality of model $m$, $d_x tilde cal(N)(0, sigma_d^2)$ is the inherent difficulty of item $x$, and $epsilon_(x,m) tilde cal(N)(0, sigma_epsilon^2)$ is independent observation noise.
We fit the parameters $bold(q)$ and $bold(d)$ such that to follow the distribution and minimize loss based on existing observations $R$:
$
  cal(L) = ||r_(x, m) - (hat(q)_m + hat(d)_x)||^2
$
Then, whenever we compute $"avg"_m$ or $"rank"_m$ we do so based on the predictions.

#paragraph[Smart item ordering.]
By removing Assumptions 3 and 4, we can now arbitrarily modify the evaluation item queue.
This process is orthogonal to the bandit-based search.
Formally, each item has an assigned $"Utility"(x)$, which is oftentimes its expected item difficulty, discriminability, diversity, or variance.
// , or metric uncertainty , "zouhar2025earlyexit"
#citep("proietti-etal-2025-estimating", "subset2evaluate").
We use this utility to order evaluation items $cal(X)$, so $x prec x' arrow.double.l.r "Utility"(x) < "Utility"(x')$.

We make the item ordering cost-aware by considering the $"Utility"(x)$ per unit of cost:
$"Utility"(x)/c_x$.
In general, cost-aware evaluation item selection is NP-complete and can only be approximated #citep("subset2evaluate").
However, in our case all items cost much less than the total budget ($max c_x << B$), which makes the above density-based heuristic asymptotically optimal #citep("dantzig1957discrete").


#import "img/simulation.typ": table-content-ordered, table-content-random
#figure(
  table(
    columns: 3,
    column-gutter: 0pt,
    row-gutter: (1pt, -3pt),
    align: (left + horizon, left + horizon, center + horizon, center + horizon),
    ..table-content-random,
  )
    + v(-5pt),
  placement: top,
  caption: [
    Selection with unbiased items.
    Normalized area under curve (higher is better) for weighted ranking ($tau_omega$), and evaluation focus (e.f.).
  ],
)
<tab:simulation_random>

= Experiments
<sec:experiments>

#TODO[there's currently a bug in the simulation code. the oracle should be 1 for evaluation focus at least and much higher for weighted tau]

We evaluate the proposed dynamic policies by simulating annotation campaigns using historical data from WMT General Translation shared tasks #citep("kocmi-etal-2023-findings", "kocmi-etal-2024-findings", "kocmi-etal-2025-findings").
This simulation-based approach allows us to compare the policies against a "gold standard" ranking derived from the full exhaustive dataset while controlling the annotation budget.

/*
import subset2evaluate.utils
data0 = subset2evaluate.utils.load_data_wmt_all()
data0 = {k:v for k, v in data0.items() if k[0] in {"wmt25", "wmt24", "wmt23"}}
print(len(data0))
import statistics
print(statistics.mean([len(v) for v in data0.values()]))
print(statistics.mean([len(v[0]["scores"]) for v in data0.values()]))



import subset2evaluate.utils
data1 = subset2evaluate.utils.load_data_wmt_all(require_human=False)
data1 = {k:v for k, v in data1.items() if k[0] in {"wmt25", "wmt24", "wmt23"}}
print(len(data1))
import statistics
print(statistics.mean([len(v) for v in data1.values()]))
print(statistics.mean([len(v[0]["scores"]) for v in data1.values()]))

import translation_bandit.utils
data2 = translation_bandit.utils.load_data_bymetrics()
print(len(data2))
import statistics
print(len({k.split("_")[-1] for k in data2.keys()}))
print(statistics.mean([len(v) for v in data2.values()]))
print(statistics.median([len(v[0]["scores"]) for v in data2.values()]))
*/


#paragraph[Setup.]
We use the data collected by the WMT General Translation Shared Task between 2023 and 2025 #citep("kocmi-etal-2023-findings", "kocmi-etal-2024-findings", "kocmi-etal-2025-findings") across 37 campaigns (language pair, year), with $|cal(X)|$=570 items and $|cal(M)|$=15 models on average, with human evaluation (290k points total) being done with DA, ESA and MQM protocols #citep("kocmi-etal-2024-error", "freitag-etal-2021-experts") that can be put on $[0, 1]$.
// We present results based on these data in @sec:experiments.
// and expand it to include automatic metrics in @sec:results_automatic.

// #TODO[
Due to the simulation on existing data, we stop sampling evaluation items for a particular model $m in cal(M)$ once it has been evaluated on all items $x in cal(X)$.


For easy comparison of algorithms, we define a summary statistic as normalized area under curve:
// which spans from 0% to 100%:
$
  #h(-1.5em)
  // dot|cal(X) times cal(M)|)
  // 1/script(90%-10%) dot
  (integral_(0)^(B) "Objective"(R_(b)) #h(0.5em) d"b")/B
$
// For $p$-value objective we substract the result from 100% to align its directionality with the rest.


#figure(
  {
    [Uniform]
    v(-10pt)
    image("img/trace_plot_random.svg")
    [Sampling rank-based]
    v(-10pt)
    image("img/trace_plot_sampling.svg")
  },
  placement: top,
  scope: "parent",
  caption: [Timeline of evaluation based on WMT25 Czech#{ sym.arrow }German campaign. Uniform balances focus equally across all models while rank-based sampling priorities top-performing models, which leads to fewer mistakes in ranking top models.],
)
<fig:trace_plot>



#figure(
  image("img/simulation_legend.svg") + image("img/simulation.svg"),
  placement: top,
  scope: "parent",
  caption: [Dynamic tournament evaluation of translation models based on existing WMT data, averaged across all languages and 100 seeds. Lightly shaded areas correspond to 95% t-distribution confidence interval. Arrows indicate if higher is better.],
)
<fig:simulation>

// == Results on Human Judgments
// <sec:results_human>

#paragraph[Case-study.]
We first show an illustrative run of a simulated evaluation on WMT25 Czech$arrow$German campaign done with uniform selection and rank-based sampling in @fig:trace_plot.
Under the uniform policy, the rankings for mid- to top-performing models are volatile, which persists well into the later evaluation steps due to the budget being diluted across inferior models.
Conversely, the rank-based sampling policy focuses on top-performing models which localizes the ranking volatility to the worse-performing models.
The histograms on the right show that each model is allocated the same number of evaluations under the uniform policy while it follows $1/"rank"$ for rank-based sampling.
In this case, the final model ranking is also more misaligned from the ground truth.

#paragraph[Varying budget.]
In @fig:simulation we show the results across all campaigns and varying budget.
In terms of budget efficiency, the uniform baseline remains the safest and best option for standard Kendall's $tau$ as it treats all rank swaps equally.
Interestingly, it becomes worse with slightly higher budget which is because most of the close matches happen between top-performing models (#box(v(-3pt) + image("img/microplot_model_avg.svg", height: 1em), height: 0.5 * 1em), @fig:trace_plot).
At this point, the other policies have allocated enough evaluations to distinguish between the worse models but still have the benefit of having focused more on the top models.

For weighted $tau$​, which prioritizes the top of the leaderboard, rank-based sampling and successive rejects outperform the baseline.
Minimizing statistical ambiguity successfully lowers the average p-value but fails to improve ranking correlations.
It also presents a confounder: model comparisons with by-chance lower $p$-value are not evaluated which prevents them from raising their $p$-value.
This is why the average $p$-value starts going up again even though we are adding more evaluation points, because the only available evaluations are of the ones previously skipped with by-chance low $p$-value.
For evaluation focus, the ambiguity reduction also fails, as it can spend its evaluation budget on any high-variance model regardless of its rank.
On the other hand, all stochastic sampling policies excel in this regard.

#paragraph[Aggregate results.]
Lastly, in @tab:simulation_random we show the aggregate area under the curve results, including existing item selection-based methods.
This confirms the previous trends of rank-based sampling being a good alternative to uniform sampling, excelling primarily in the evaluation focus objective.

On the other hand, existing item-selection methods that prioritize diversity or difficulty yield only negligible occasional gains over the uniform baseline.
This can be partly explained by the more constrained problem setup and also the WMT 2025 general shared task using diversity and difficulty item selection to create the final evaluation set $cal(X)$.

#figure(
  table(
    columns: 6,
    column-gutter: (-6pt, -1pt, 0pt),
    row-gutter: (-3pt, 1pt, -3pt),
    align: (
      left + horizon,
      left + horizon,
      left + horizon,
      center + horizon,
      center + horizon,
      center + horizon,
      center + horizon,
    ),
    ..table-content-ordered,
  )
    + v(-5pt),
  placement: top,
  caption: [
    Selection with biased evaluation items (most difficult or most diverse).
    Normalized area under curve (higher is better) for weighted ranking ($tau_omega$), and evaluation focus (e.f.).
  ]
    + TODO[linear estimators],
)
<tab:simulation_ordered>

#paragraph[Smart item ordering with non-constant cost.]
We now use smart item ordering, which requires lifting Assumption 3 and 4 (@sec:problem_statement).
Specifically, we order the evaluation items based on average automatics metric scores (Metric-X, #citen("juraska-etal-2023-metricx", "juraska-etal-2024-metricx", "juraska-etal-2025-metricx")), and also normalize by the evaluation cost estimated through the annotation time.
The results are shown in @tab:simulation_ordered.
// Unbiased item ordering
As per @thm:unbiased-ordering mean-based estimator of model performance break the algorithms, despie improving the uniform sampling performance, as shown by #citet("subset2evaluate").
By applying the estimators from @sec:unconstrained_ordering, the bandit algorithms are also able to benefit from the smart item ordering.

// == Results on Automatic Judgments
// <sec:results_automatic>
// #TODO[scale up using a generative process fitted on humans?]
// Despite the data scale, the number of competing models in the WMT data is always limited.
// This is because the shared task already pre-selects top-performing models based on automatic metrics to compete in the human evaluation.
// This introduces selection bias favoring models that overfit the automatic metrics.
// So far, the judgments of quality $r_(x,m)$ were based on human annotations, which are based on the limited model selection.
// To simulate competition at scale, we proxy ground truth with WMT metrics shared task submissions #citep("freitag-etal-2023-results", "freitag-etal-2024-llms", "lavie-etal-2025-findings") and treat the automatic metric scores, inclusive their biased, as the gold truth.
// This leads to 113 unique metrics across 1639 campaigns (language pair, year, metric), with average of $|cal(X)|$=970 evaluation items and $|cal(M)|$=20 models, leading to 27M points total.
// #TODO[figure etc]

= Conclusion

Standard uniform evaluation protocols inefficiently allocate effort to inferior models, which prohibits large-scale evaluation.
We formalize human evaluation as best-arm identification within a multi-armed bandit framework with correlated arms and prove the optimality of proposed algorithms under certain conditions.
In contrast, our dynamic allocation policies, such as rank-based or UCB, trade off global model ranking precision for ranking precision among top models.
Such models also undergo more evaluations, which helps in high-quality data for reward models.

#paragraph[Recommendations.]
We provide specific recommendations to practitioners conducting large-scale evaluations:
- Establish evaluation goals: are we interested in all models equally or the top ones?
- Use weighted sampling with $1/"rank"_m$: this is a simple yet effective method that can explicitly optimize towards the particular evaluation goal. For translation and multilingual tasks, this is implemented in Pearmut #citep("zouhar2026pearmuthumanevaluationtranslation") human annotation interface.
- Do not use evaluation with statistical ambiguity reduction which overfits on the existing data and provides biased final evaluation results.

#paragraph[Future work.]
So far we assumed that evaluation items are atomic and independent.
A natural extension is the formulation of evaluation as a _contextual_ multi-armed bandit problem, where the policy has access to the item or model features to make better estimates of expected scores.

Furthermore, dynamic allocation risks annotator drift and subjectivity.
The annotator distribution can change between the beginning to later stages, especially for long-running campaigns.
Future works should thus model the annotators and correct the bias.

#heading(numbering: none)[Limitations]
// We explicitly trade global ranking for resolution in top-tier ranking.
// If global ranking (measured e.g. by standard Kendall's $tau$) is desired, this needs to be encoded in the weighting function ($w_m$).
// Alternatively, use only methods for smart evaluation subset selection with uniform sampling.

Our experimnents, due to their scale, are simulated on existing evaluation data.
This creates a natural cap on how many times we can evaluate a particular model (see @fig:trace_plot).
This obscures potential behaviors in scenarios where the item pool $cal(X)$ is vastly larger than the budget $B$, which prevents full convergence of the mean estimator.

We also assume bounded scores based on Gaussian distribution.
While a very common assumption corresponding to modern evaluation protocols, such as human scoring on a scale 0% to 100% or Likert, it makes our theoretical findings inapplicable to cases where the result can be e.g. unbounded reward or loss in financial modelling.

#{
  // pagebreak()
  add-bib-resource(read("bibliography.bib"))
  print-acl-bibliography()
}

#show: appendix
#set page(columns: 1)

= Proofs
<sec:proofs>

#proof("thm:non-repeating", [
  Let $sigma^2$ be the population variance of the scores over $cal(X)$. Let $r_1, ..., r_B$ be the sequence of score random variables for model $m$ under budget $B$.
  The variance of the estimator sum is:
  $
    "Var"(sum_(i=1)^B r_i) = sum_(i=1)^B "Var"(r_i) + sum_(i != j) "Cov"(r_i, r_j)
  $

  - Case 1 (with replacement):
    Draws are independent, so $"Cov"(r_i, r_j) = 0$.
    $ "Var"(hat(mu)_"with") = 1/B^2 (B sigma^2) = sigma^2/B $

  - Case 2 (without replacement):
    Draws are dependent.
    The sum of the scores of the full finite population without replacement is a constant, so its variance is zero.
    $
      0 & = "Var"(scripts(sum)_(k=1)^(|cal(X)|) r_k) \
        & = |cal(X)| sigma^2 + |cal(X)|(|cal(X)|-1) "Cov"(r_i, r_j)
    $
    Solving for the covariance:
    $ "Cov"(r_i, r_j) = -sigma^2/(|cal(X)|-1) $
    Substituting this into the variance of the sample sum of size $B$:
    $
      "Var"(sum_(i=1)^B r_i) & = B sigma^2 + B(B-1) (-sigma^2/(|cal(X)|-1)) \
                             & = B sigma^2 (1 - (B-1)/(|cal(X)|-1))
    $
    Finally, for the mean $hat(mu) = 1/B sum r_i$:
    $
      "Var"(hat(mu)_"without") & = 1/B^2 "Var"(sum_(i=1)^B r_i) \
                               & = sigma^2/B (1 - (B-1)/(|cal(X)|-1))
    $

  Since $(B-1)/(|cal(X)|-1) > 0$ for $B > 1$, the variance is strictly reduced:
  $ "Var"(hat(mu)_"without") < "Var"(hat(mu)_"with") $
  #h(1fr) $qed$
])



#proof("thm:maximize-overlap")[
  We model the score $r_(x,m)$ of a model $m$ on item $x$ using a standard additive error decomposition:
  $ r_(x,m) = q_m + d_x + epsilon_(x,m) $
  where $q_m$ is the true latent quality of model $m$, $d_x tilde cal(N)(0, sigma_d^2)$ is the inherent difficulty of item $x$, and $epsilon_(x,m) tilde cal(N)(0, sigma_epsilon^2)$ is independent observation noise.

  Our objective is to estimate the quality difference $Delta_(m, m') = q_m - q_m'$. The unbiased estimator $hat(Delta)$ is the difference of the empirical means over the respective evaluation sets $R_m$ and $R_m'$:
  $
    hat(Delta) & = hat(mu)_m - hat(mu)_m' \
               & = 1/(|R_m|) sum_(x in R_m) r_(x, m) - 1/(|R_m'|) sum_(x in R_m') r_(x, m')
  $

  The variance of this estimator expands to:
  $
    "Var"(hat(Delta)) = "Var"(hat(mu)_m) + "Var"(hat(mu)_m') - 2 "Cov"(hat(mu)_m, hat(mu)_m')
  $

  Assuming $epsilon$ terms are independent across models and items, the covariance term arises solely from the shared item difficulty components $d_x$ present in the intersection $R_m inter R_m'$:
  $
    "Cov"(hat(mu)_m, hat(mu)_m') & = "Cov"(1/(|R_m|) sum_(x in R_m) d_x, 1/(|R_m'|) sum_(x in R_m') d_x) \
                                 & = 1/(|R_m| |R_m'|) sum_(i in R_m) sum_(j in R_m') "Cov"(d_i, d_j) \
                                 & = 1/(|R_m| |R_m'|) sum_(x in R_m inter R_m') "Var"(d_x) \
                                 & = (|R_m inter R_m'|)/(|R_m| |R_m'|) sigma_d^2
  $

  To strictly minimize $"Var"(hat(Delta))$, we must maximize the covariance term. For any fixed budget sizes $|R_m|$ and $|R_m'|$, this term is maximized when the intersection $|R_m inter R_m'|$ is maximized. This condition is met if and only if the evaluation sets satisfy the subset property (i.e., $R_m subset.eq R_m'$ or $R_m' subset.eq R_m$).
  #h(1fr) $qed$
]



#proof("thm:unbiased-ordering")[
  Let the score be $r_(x_i, m) = q_m + d_(x_i)$, where $d_(x_i)$ is the difficulty of the $i$-th item.
  The simple average estimator for model $m$ evaluated on the first $|R_m|$ items is:
  $
    bb(E)[hat(mu)_m] & = bb(E)[1/(|R_m|) sum_(k=1)^(|R_m|) (q_m + d_(x_i))] \
                     & = q_m + 1/(|R_m|) sum_(k=1)^(|R_m|) bb(E)[d_(x_i)]
  $

  Let $dash(d)_n = 1/n sum_(k=1)^n bb(E)[d_(x_i)]$ be the average expected difficulty of the prefix of length $n$.
  Now consider two models $m$ and $m'$ with true qualities $q_m = q_m' = q$, but unequal budget allocations $|R_m| < |R_m|'$. The expected difference is:
  $
    bb(E)[hat(mu)_a - hat(mu)_b] & = (q + dash(d)_(|R_m|)) - (q + dash(d)_(|R_m|')) \
                                 & = dash(d)_(|R_m|) - dash(d)_(|R_m|')
  $

  If the ordering is biased such that difficulty increases with index (monotonic increasing $bb(E)[d_(x_i)]$), then the prefix average $dash(d)_n$ is strictly increasing in $n$.
  Consequently, $dash(d)_(|R_m|) < dash(d)_(|R_m|')$, leading to a non-zero bias term where the model with fewer evaluations ($a$) appears superior solely due to the queue ordering.
  $ bb(E)[hat(mu)_m - hat(mu)_m'] eq.not 0 $
  // Thus, simple averages are invalid for ranking unless $dash(d)_n$ is constant for all $n$, which requires Assumption 3.
  #h(1fr) $qed$
]



// #proof("thm:consistency-weighted-sampling")[
// // https://en.wikipedia.org/wiki/Borel%E2%80%93Cantelli_lemma
// Since $P[pi(R) = m] >= epsilon$, each model $m$ is sampled infinitely often (second Borel-Cantelli lemma).
// Consequently, by the strong law of large numbers, the sample mean $hat(mu)_(m, B)$ converges to the true quality $q_m$.
// Given distinct true qualities, there exists a finite $B$ after which all pairwise orderings are correct.
// #h(1fr) $qed$
// ]

#proof("thm:consistency-weighted-sampling")[
  We first compute the minimum and maximum for all samplers in the weighted sampling.
  We make use of the fact that the average is based on observations that are in $[0, 1]$ and so is also bounded:
  - $epsilon$-Greedy: $1-epsilon >= omega_w >= epsilon / (|cal(M)|)$ for fixed $epsilon$
  - Bolzmann: $exp(1/tau) >= omega_w >= 1$ for fixed $tau$
  - Rank-based: $1 >= omega_w >= 1/(|cal(M)|)$
  // - UCB: $1 + gamma ??? >= omega_w >= 0 + gamma ???$ for fixed $gamma$
  The minimum probability of sampling a particular model is at least $(min_m omega_w) / (|cal(M)| dot max omega_w)$.
  Because $omega_w$ is bounded, the minimum probability is also lower-bounded.
  We exclude UCB because the only way for UCB probability to tend to zero is for a model to already be pulled infinite amount of times.

  #paragraph[]
  Let $|R_(m, B)|$ denote the number of times model $m$ has been evaluated under budget $B$.
  Since the selection probability for every model is bounded below by $zeta > 0$, the second Borel-Cantelli lemma guarantees that every model is sampled infinitely often:
  $
    lim_(B arrow infinity) |R_(m, B)| = infinity, quad forall m in cal(M)
  $

  Let $hat(mu)_(m, B)$ be the simple average estimator for model $m$ after $B$ steps.
  By the Strong Law of Large Numbers, since $|R_(m, B)| arrow infinity$, the estimator converges to the true quality:
  $
    hat(mu)_(m, B) arrow.long q_m
  $

  Let $delta = min_(m, m' : q_m != q_m') |q_m - q_m'|$ be the minimum separation between two distinct true model qualities.
  For the ranking $hat(R)_B$ to differ from $R^*$, there must exist at least one pair $m, m'$ such that their estimated order is incorrect.
  This requires the estimation error for at least one model to exceed $delta/2$.
  Since $hat(mu)_(m, B)$ converges to $q_m$, there exists a finite budget $B^dagger$ such that for all $B > B^dagger$, $|hat(mu)_(m, B) - q_m| < delta/2$ for all $m$, implying $hat(R)_B = R^*$.
  #h(1fr) $qed$
]


#noindent
We now introduce @thm:asymptotic-convergence which is used for proving @thm:optimality-evaluation-focus and @thm:optimality-weighted-tau.

#lemma("thm:asymptotic-convergence", [Asymptotic allocation convergence])[
  Let $pi$ be a sampling policy where at each step $t$, a model $m$ is selected with probability $P_t(m)$ proportional to its estimated importance weight $hat(omega)_(m,t)$.
  Assuming the policy guarantees infinite exploration (i.e., $P_t(m) > epsilon_t > 0$), then the asymptotic allocation converges to the true importance weights:
  $ lim_(B -> infinity) (|R_(m, B)|) / B = (omega_m) / (sum_k omega_k) $
]

#proof("thm:asymptotic-convergence", show_lemma: false)[
  Because the policy ensures every model is sampled infinitely often as $B -> infinity$, by the Strong Law of Large Numbers, the empirical mean estimator $hat(mu)_(m,t)$ converges to the true mean: $hat(mu)_(m,t) arrow mu_m$.
  Because $omega_m$ is based on $hat(mu)_(m,t)$, similarly $hat(omega)_m arrow omega_w$ and also the probability of sampling $P_(t(m)) = (omega_m) / (sum_k omega_k) arrow = (sum_k omega_k) = P^*(m)$.

  The number of samples $|R_(m, B)|$ is the sum of Bernoulli trials with probabilities converging to $P^*(m)$. By the limit theorems for sums of independent random variables, the averaged allocation converges to the sampling probability:
  $ lim_(B -> infinity) (|R_(m, B)|) / B = P^*(m) = (omega_m) / (sum_k omega_k) $
  #h(1fr) $qed$
]


#proof("thm:optimality-evaluation-focus")[
  We first find the maximum of $sum_(m in cal(M)) omega_m ln(|R_m|)$ subject to the constraint $sum |R_m| = B$ using Lagrange multiplier $lambda$:
  $
    cal(L)(R, lambda) = sum_(m in cal(M)) omega_m ln(|R_m|) - lambda (sum_(m in cal(M)) |R_m| - B)
  $
  We find the stationary points with respect to $|R_m|$ by setting the function to zero:
  $
    (partial cal(L)) / (partial |R_m|) = (omega_m) / (|R_m|) - lambda = 0 quad arrow.double quad |R_m| = (omega_m) / lambda
  $
  <eq:optimality-evaluation-focus-stationary>

  Sum of the counts also needs to hold to satisfy the budget $B$:
  $
    B = sum_m |R_m| = sum_m (omega_m) / lambda quad arrow.double quad lambda = (sum_m omega_m) / B
  $

  We substitute $lambda$ back into the stationarity @eq:optimality-evaluation-focus-stationary:
  $ |R_m|^* = (omega_m) / ((sum_k omega_k) / B) = B dot (omega_m) / (sum_k omega_k) $

  The optimal number of samples $|R_m|^*$ is linearly proportional to the importance weight $omega_m$.
  From @thm:asymptotic-convergence, we established that a policy sampling proportional to $omega_m$ converges exactly to this allocation ratio.
  Therefore, the rank-based sampling policy is optimal for this objective.
  #h(1fr) $qed$
]

#proof("thm:optimality-weighted-tau")[
  Let the true quality of model $m in cal(M)$ be  $mu_m$.
  Let $hat(mu)_m$ be the estimator for model $m$ after $|R_m|$ samples.
  Let $omega_m^2$ be the importance weight of correctly ranking model $m$.
  We assume the estimated score follows an independent Gaussian distribution:
  $ hat(mu)_m tilde cal(N)(mu_m, sigma^2 / (|R_m|)) $

  We first define a differentiable loss function that proxies the discrete ranking metric $tau_omega$.
  The probability of incorrectly ranking two models $i$ and $j$ (where $mu_i > mu_j$) is the probability that their estimated difference $hat(mu)_i - hat(mu)_j$ is negative.
  Since $hat(mu)$ are independent Gaussians, the difference $hat(mu)_i - hat(mu)_j$ is also Gaussian:
  $
    hat(mu)_i - hat(mu)_j tilde cal(N)(mu_i - mu_j, sigma^2/ (|R_i|) + sigma^2/ (|R_j|))
  $
  The probability of ranking error is determined by the tail of this distribution:
  $
    P(hat(mu)_i < hat(mu)_j) = Phi(- (mu_i - mu_j) / sqrt(sigma^2/(|R_i|) + sigma^2/(|R_j|)))
  $

  Minimizing the ranking error is analytically intractable due to the $Phi$ function and pairwise terms. However, the error probability is strictly monotonic with the variance of the estimators.
  Therefore, to maximize the precision of the ranking for model $m$ weighted by $omega_m^2$, we minimize the weighted cumulative variance of the estimators.
  We define the objective loss function as:
  $
    sum_(m in cal(M)) omega_m^2 dot "Var"(hat(mu)_m) = sum_(m in cal(M)) (omega_m^2 sigma^2) / (|R_m|)
  $
  <eq:optimality-weighted-tau-loss>

  We now show that minimizing @eq:optimality-weighted-tau-loss under a fixed budget $B$ requires sampling each model $m$ proportional to $omega_m$.
  Again, we turn to Lagrange multiplier $lambda$:
  $
    sum_(m in cal(M)) (omega_m^2 sigma^2) / (|R_m|) + lambda (sum_(m in cal(M)) |R_m| - B)
  $

  We take the partial derivative with respect to the sample count $|R_m|$ and set it to zero to find the stationary points.
  $
    & (partial quad sum_(m in cal(M)) (omega_m^2 sigma^2) / (|R_m|) + lambda (sum_(m in cal(M)) |R_m| - B)) / (partial |R_m|) = - (omega_m^2 sigma^2) / (|R_m|^2) + lambda = 0
  $
  Rearrranging the terms yields:
  $
    (omega_m^2 sigma^2) / (|R_m|^2) = lambda
    quad arrow.double quad
    |R_m| = (sigma) / sqrt(lambda) omega_m
  $

  Sum of the counts also needs to hold to satisfy the budget $B$:
  $
    B & = sum_(m in cal(M)) |R_m| \
      & = sum_(m in cal(M)) (sigma / sqrt(lambda)) omega_m \
      & = (sigma / sqrt(lambda)) sum_(m in cal(M)) omega_m
  $
  $
    #h(-82pt)
    arrow.double
    (sigma / sqrt(lambda)) = B / (sum_k sqrt(omega_k))
  $

  We substitute this constant back into the expression for $|R_m|$:
  $
    |R_m|^* & = ( B / (sum_k omega_k) ) omega_m \
            & = B dot (omega_m) / (sum_(k in cal(M)) omega_k)
  $

  The optimal number of samples $|R_m|^*$ is then proportional to the square root of the importance weight $omega_m$, which is attained asymptotically by Rank-sqrt weighted sampler according to @thm:asymptotic-convergence.
  #h(1fr) $qed$

]



#figure(
  line(length: 100%)
    + v(-10pt)
    + algo2(
      parameters: ($M$, $B$).map(x => text(size: 9pt, x)),
      title: [
        #set text(size: 9pt)
        *Input*: models $M$, budget $B$ \
        *Output*: evaluation selection $R$ \
        #v(0pt) #h(-1.25em)
        *Uniform*
      ],
    )[
      $R = emptyset$ \
      for $m in cal(M)$: #i \
      repeat $B/(|cal(M)|)$: #i \
      $x arrow.l min_prec \{ x#h(1pt)|#h(1pt)x in cal(X), chevron.l x, m chevron.r in.not R \}$ \
      $R arrow.l R union \{ chevron.l x, m chevron.r \}$ #d \
      return $R$
    ]
    + v(-10pt)
    + line(length: 100%)
    + v(-5pt),
  placement: auto,
  kind: "Algorithm",
  supplement: "Algorithm",
  caption: [Baseline evaluation selection. All models are evaluated on the same set of items.],
)
<alg:baseline>

#figure(
  line(length: 100%)
    + v(-10pt)
    + algo2(
      parameters: ($cal(M)$, $B$, $C$).map(x => text(size: 9pt, x)),
      title: [
        #set text(size: 9pt)
        *Input*: models $cal(M)$, budget $B$ \
        *Hyperparameter*: cold start $C$, weight $lambda$ \
        *Output*: evaluation results $R$ \
        #v(0pt) #h(-1.25em)
        *AmbiguityReduction*
      ],
    )[
      $R arrow.l emptyset$ \
      for $m in cal(M)$: #i \
      repeat $C$: #i \
      $x arrow.l min \{ x#h(1pt)|#h(1pt)x in cal(X), chevron.l x, m chevron.r in.not R \}$ \
      $R arrow.l R union \{ chevron.l x, m chevron.r \}$ #d#d \
      while $|R| < B$: #i \
      $k_1 arrow.l cal(M) "ranking by" "CI"(r_(*,m))$ \
      $k_2 arrow.l cal(M) "ranking by" sum_(m' in cal(M)) "ttest"(r_(*, m'), r_(*, m))$ \
      $m^* arrow.l "argmin"_(m in cal(M)) lambda dot k_1 + k_2$ \
      $x arrow.l min \{ x#h(1pt)|#h(1pt)x in cal(X), chevron.l x, m^* chevron.r in.not R \}$ \
      $R arrow.l R union \{ chevron.l x, m^* chevron.r \}$ #d \
      return $R$
    ]
    + v(-10pt)
    + line(length: 100%)
    + v(-5pt),
  // placement: top,
  kind: "Algorithm",
  supplement: "Algorithm",
  caption: [Allocation policy minimizing pointwise ambiguity (CI width) and pairwise ambiguity (neighbor p-value).],
)
<alg:ambiguity>

= Algorithms (extended)
<sec:algorithms_extended>

#TODO[intro]


#paragraph[Statistical Ambiguity] (@alg:ambiguity).
A good evaluation policy must navigate a trade-off between two distinct sources of ambiguity: variance of a model's score distribution #citep("cohn1996active"), and pairwise indiscriminability which presents itself as low statistical power between adjacent models in the partial ranking #citep("card-etal-2020-little").
Concretely, this strategy operationalizes the to-be-reduced "ambiguity" as the weighted ($lambda$) sum of its confidence interval width and the $p$-values separating it from its neighbors.
The reason to consider both at the same time is that minimizing the variance alone is not directly aligned with any of the policy utilities (@sec:problem_statement), or can lead to overfocusing on a single high-variance model.
Similarly, reducing pairwise ambiguity could lead to overfocusing on two models that are naturally very close to each other.
We therefore minimize their combined ranks, in order to match their scales.

#paragraph[Successive Halving] (@alg:successive_halving).
Successive Halving is a bracket-based elimination algorithm. It operates in rounds, where in each round the budget is equally distributed among the surviving models. After evaluation, the models are ranked by their empirical mean performance, and the bottom half is discarded. This process repeats until a single model remains or the budget is exhausted.

#figure(
  line(length: 100%)
    + v(-10pt)
    + algo2(
      parameters: ($cal(M)$, $B$).map(x => text(size: 9pt, x)),
      title: [
        #set text(size: 9pt)
        *Input*: models $cal(M)$, budget $B$ \
        *Output*: evaluation results $R$ \
        #v(0pt) #h(-1.25em)
        *SuccessiveHalving*
      ],
    )[
      $R arrow.l emptyset$ \
      $n_"rounds" arrow.l ceil(log_2 |cal(M)|)$ \
      while $|cal(M)| > 1$ and $n_"rounds" > 0$: #i \
      $B_"round" arrow.l (B - |R|) / n_"rounds"$ \
      $k arrow.l floor(B_"round" / |cal(M)|)$ \
      for $m in cal(M)$: #i \
      repeat $k$: #i \
      $x arrow.l min_prec \{ x#h(1pt)|#h(1pt)x in cal(X), chevron.l x, m chevron.r in.not R \}$ \
      $R arrow.l R union \{ chevron.l x, m chevron.r \}$ #d#d#d \
      $cal(M) arrow.l "top-"ceil(|cal(M)|/2) "by avg"_m$ \
      $n_"rounds" arrow.l n_"rounds" - 1$ #d \
      return $R$
    ]
    + v(-10pt)
    + line(length: 100%)
    + v(-5pt),
  kind: "Algorithm",
  supplement: "Algorithm",
  caption: [Successive halving algorithm which eliminates half of the models in each round.],
)
<alg:successive_halving>


#paragraph[P-value Rejects] (@alg:pvalue_rejects).
P-value Rejects is a dynamic elimination strategy that continuously monitors the pairwise statistical significance between models. It proceeds in a round-robin fashion, evaluating all active models. After each round, it compares the worst-performing model with the next-worst model using a statistical test (e.g., t-test). If the p-value is below a significance threshold (e.g., 0.05), the worst model is discarded. This allows for early pruning of clearly inferior models without waiting for fixed phases.

#figure(
  line(length: 100%)
    + v(-10pt)
    + algo2(
      parameters: ($cal(M)$, $B$, $alpha$).map(x => text(size: 9pt, x)),
      title: [
        #set text(size: 9pt)
        *Input*: models $cal(M)$, budget $B$, threshold $alpha$ \
        *Output*: evaluation results $R$ \
        #v(0pt) #h(-1.25em)
        *PValueRejects*
      ],
    )[
      $R arrow.l emptyset$ \
      while $|R| < B$ and $|cal(M)| > 1$: #i \
      for $m in cal(M)$: #i \
      $x arrow.l min_prec \{ x#h(1pt)|#h(1pt)x in cal(X), chevron.l x, m chevron.r in.not R \}$ \
      $R arrow.l R union \{ chevron.l x, m chevron.r \}$ #d \
      Sort $cal(M)$ by $"avg"_m$ \
      $m_"worst", m_"next" arrow.l cal(M)[0], cal(M)[1]$ \
      $p arrow.l "ttest"(m_"worst", m_"next")$ \
      if $p < alpha$: #i \
      $cal(M) arrow.l cal(M) without \{m_"worst"\}$ #d#d \
      return $R$
    ]
    + v(-10pt)
    + line(length: 100%)
    + v(-5pt),
  kind: "Algorithm",
  supplement: "Algorithm",
  caption: [Dynamic elimination based on pairwise p-value significance.],
)
<alg:pvalue_rejects>


#paragraph[Thompson Sampling] (@alg:thompson_sampling).
Thompson Sampling is a Bayesian approach to the exploration-exploitation trade-off. It maintains a posterior distribution for the mean capability of each model. In each step, it samples a potential mean from each model's posterior (approximated as Gaussian) and selects the model (or top-k models) with the highest sampled value for the next evaluation. This naturally balances exploring models with high uncertainty and exploiting models with high estimated performance.

#figure(
  line(length: 100%)
    + v(-10pt)
    + algo2(
      parameters: ($cal(M)$, $B$, $C$).map(x => text(size: 9pt, x)),
      title: [
        #set text(size: 9pt)
        *Input*: models $cal(M)$, budget $B$, cold start $C$ \
        *Output*: evaluation results $R$ \
        #v(0pt) #h(-1.25em)
        *ThompsonSampling*
      ],
    )[
      $R arrow.l emptyset$ \
      for $m in cal(M)$: #i \
      repeat $C$: #i \
      $x arrow.l min_prec \{ x#h(1pt)|#h(1pt)x in cal(X), chevron.l x, m chevron.r in.not R \}$ \
      $R arrow.l R union \{ chevron.l x, m chevron.r \}$ #d#d \
      while $|R| < B$: #i \
      for $m in cal(M)$: #i \
      $hat(mu)_m, hat(sigma)_m^2 arrow.l "mean, var of" m$ \
      $theta_m tilde cal(N)(hat(mu)_m, hat(sigma)_m^2 / |R_m|)$ #d \
      $cal(M)_"next" arrow.l "top-"k "by" theta_m$ \
      for $m in cal(M)_"next"$: #i \
      $x arrow.l min_prec \{ x#h(1pt)|#h(1pt)x in cal(X), chevron.l x, m chevron.r in.not R \}$ \
      $R arrow.l R union \{ chevron.l x, m chevron.r \}$ #d#d \
      return $R$
    ]
    + v(-10pt)
    + line(length: 100%)
    + v(-5pt),
  kind: "Algorithm",
  supplement: "Algorithm",
  caption: [Thompson sampling for best arm identification.],
)
<alg:thompson_sampling>


= Experiments (extended)
<sec:experiments_extended>

#TODO[discuss why average payoff is a bad metric that is at odds with our evaluation goal of difficult examples and wouldn't work]

#paragraph[Objective of policy $bold(pi)$ (extended).]
The simplest objective is the correlation between model ranking given $R_B$ against "true model ranking" given by the full $cal(X) times cal(M)$.
We choose *standard Kendall's $tau$* variant b for simplicity.
Many evaluations also care about the statistical power of the results and maximizing the number of statistical clusters.
We compute the *average $p$-value* between neighboring ranked models using two-sided t-test for related samples (intersection on items that were evaluated by both models).

#TODO[extended results]




= Related Works (extended)
<sec:related_work_extended>

This section extends the related work in @sec:related_work and provides a more in-depth overview of methods related to dynamic evaluation.

#paragraph[Selection guarantees.]
Prior to multi-armed bandits, #citet("bechhofer1954single") defines _indifference zone selection_, which requires that the mean of the selected distribution ("arm") is at most $delta$ (indifference zone) lower than the true maximum, with a pre-specified probability threshold.
This needs to be done with the smallest possible number of samples.
Similarly, #citet("Gupta01051965") turns this problem into selecting a set of distributions that are guaranteed with a probability threhold to contain the best one, again with the smallest possible number of samples.

These findings were applied to optimal computing budget allocation #citep("Chen2000") to determine the best distribution.
The key finding is that the number of samples is proportial to the variance of the particular distribution and inversely proportional to the gap between the best alternative and this distribution.
Distributions with high variance and proximity to the best alternative are thus prioritized.


#paragraph[Multi-armed bandits.]
Modern machine learning has reframed evaluation as a "best arm identification" problem.
#citet("audibert2010best", "pmlr-v28-karnin13", "jamieson2016non") introduce algorithms for fixed-budget best arm identification, such as successive rejects or successive halving.
LUCB (Lower Upper Confidence Bound) #citep("kalyanakrishnan2012pac") refines this by sampling the most ambiguous pair: the current best arm and the strongest contender.
This algorithm is also formally shown to be good for selecting multiple top models.

Standard algorithms assume that the distribution behind each arm is static and not related to the number of pulls.
In our case, however, $i$-th pull for model $m$ is correlated with $i$-th pull for model $m'$.
This is known as correlated arms and #citet("Gupta_2021") use pseudo-rewards to adapt the above algorithms.
Similarly, we use the competition models to estimate the missing scores.

// In evaluation contexts #citet("ye-etal-2021-assessing") use low-rank matrix factorization to infer missing scores.

#TODO[correlated arms]

#paragraph[Adaptive testing.]
Item response theory (IRT, #citen("lord1968statistical")) provides a foundation for determining expected student success on a particular exam item by modelling the item difficulty, discriminability, and feasability.
Commonly the most discriminative items are selected to be part of a testset #citep("rodriguez-etal-2021-evaluation").
TrueSkill and ELO #citep("elo1978rating", "herbrich2006trueskill", "minka2018trueskill"), while methodologically different, provide a similar output by modelling the likelihood of winning a match.

Having a model that preditcs success is useful in determining who should have a match with whom and on what evaluation items.
This principle guides many recent works #citep("sakaguchi-etal-2014-efficient", "chiang2024chatbot", "balkır2026confidentrankingsfeweritems") which prioritize matches and evaluation items with the highest uncertainty.
