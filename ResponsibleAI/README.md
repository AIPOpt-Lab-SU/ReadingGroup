

# Privacy & fairness related papers




# Differentially Private Latent Diffusion Models [View](https://arxiv.org/abs/2305.15759)

Diffusion models (DMs) are widely used for generating high-quality image datasets.
However, since they operate directly in the high-dimensional pixel space, optimization of DMs is computationally expensive, requiring long training times. This
contributes to large amounts of noise being injected into the differentially private
learning process, due to the composability property of differential privacy. To
address this challenge, we propose training Latent Diffusion Models (LDMs) with
differential privacy. LDMs use powerful pre-trained autoencoders to reduce the
high-dimensional pixel space to a much lower-dimensional latent space, making
training DMs more efficient and fast. Unlike [Ghalebikesabi et al., 2023] that pretrains DMs with public data then fine-tunes them with private data, we fine-tune
only the attention modules of LDMs at varying layers with privacy-sensitive data,
reducing the number of trainable parameters by approximately 96% compared to
fine-tuning the entire DM. We test our algorithm on several public-private data
pairs, such as ImageNet as public data and CIFAR10 and CelebA as private data,
and SVHN as public data and MNIST as private data. Our approach provides a
promising direction for training more powerful, yet training-efficient differentially
private DMs that can produce high-quality synthetic images


# Privacy Auditing with One (1) Training Run [View](https://arxiv.org/abs/2305.08846)

We propose a scheme for auditing differentially private machine learning systems
with a single training run. This exploits the parallelism of being able to add or remove
multiple training examples independently. We analyze this using the connection between differential privacy and statistical generalization, which avoids the cost of group
privacy. Our auditing scheme requires minimal assumptions about the algorithm and
can be applied in the black-box or white-box setting

# Privacy Loss of Noisy Stochastic Gradient Descent Might Converge Even for Non-Convex Losses [View](https://arxiv.org/abs/2305.09903)

The Noisy-SGD algorithm is widely used for privately training machine learning models. Traditional
privacy analyses of this algorithm assume that the internal state is publicly revealed, resulting in privacy
loss bounds that increase indefinitely with the number of iterations. However, recent findings have shown
that if the internal state remains hidden, then the privacy loss might remain bounded. Nevertheless, this
remarkable result heavily relies on the assumption of (strong) convexity of the loss function. It remains
an important open problem to further relax this condition while proving similar convergent upper bounds
on the privacy loss. In this work, we address this problem for DP-SGD, a popular variant of Noisy-SGD
that incorporates gradient clipping to limit the impact of individual samples on the training process. Our
findings demonstrate that the privacy loss of projected DP-SGD converges exponentially fast, without
requiring convexity or smoothness assumptions on the loss function. In addition, we analyze the privacy
loss of regularized (unprojected) DP-SGD. To obtain these results, we directly analyze the hockey-stick
divergence between coupled stochastic processes by relying on non-linear data processing inequalities.

# Selective Pre-training for Private Fine-tuning [View](https://arxiv.org/abs/2305.13865)

Suppose we want to train text prediction models in email clients or word processors. The models must preserve the
privacy of user data and adhere to a specific fixed size to meet memory and inference time requirements. We introduce
a generic framework to solve this problem. Specifically, we are given a public dataset Dpub and a private dataset Dpriv
corresponding to a downstream task T. How should we pre-train a fixed-size model M on Dpub and fine-tune it on
Dpriv such that performance of M with respect to T is maximized and M satisfies differential privacy with respect to
Dpriv? We show that pre-training on a subset of dataset Dpub that brings the public distribution closer to the private
distribution is a crucial ingredient to maximize the transfer learning abilities of M after pre-training, especially in the
regimes where model sizes are relatively small. Besides performance improvements, our framework also shows that
with careful pre-training and private fine-tuning, smaller models can match the performance of much larger models,
highlighting the promise of differentially private training as a tool for model compression and efficiency

# Personalized DP-SGD using Sampling Mechanisms [View](https://arxiv.org/abs/2305.15165)

Personalized privacy becomes critical in deep learning for Trustworthy AI. While
Differentially Private Stochastic Gradient Descent (DP-SGD) is widely used in
deep learning methods supporting privacy, it provides the same level of privacy
to all individuals, which may lead to overprotection and low utility. In practice,
different users may require different privacy levels, and the model can be improved
by using more information about the users with lower privacy requirements. There
are also recent works on differential privacy of individuals when using DP-SGD,
but they are mostly about individual privacy accounting and do not focus on satisfying different privacy levels. We thus extend DP-SGD to support a recent privacy
notion called (Φ, ∆)-Personalized Differential Privacy ((Φ, ∆)-PDP), which extends an existing PDP concept called Φ-PDP. Our algorithm uses a multi-round
personalized sampling mechanism and embeds it within the DP-SGD iterations.
Experiments on real datasets show that our algorithm outperforms DP-SGD and
simple combinations of DP-SGD with existing PDP mechanisms in terms of model
performance and efficiency due to its embedded sampling mechanism.

# Can Copyright be Reduced to Privacy? [View](https://arxiv.org/abs/2305.14822)
There is an increasing concern that generative AI models may produce outputs that are
remarkably similar to the copyrighted input content on which they are trained. This worry has
escalated as the quality and complexity of generative models have immensely improved, and
the availability of large datasets containing copyrighted material has increased. Researchers
are actively exploring strategies to mitigate the risk of producing infringing samples, and a
recent line of work suggests to employ techniques such as differential privacy and other forms
of algorithmic stability to safeguard copyrighted content.
In this work, we examine the question whether algorithmic stability techniques such as
differential privacy are suitable to ensure the responsible use of generative models without inadvertently violating copyright laws. We argue that there are fundamental differences between
privacy and copyright that should not be overlooked. In particular we highlight that although
algorithmic stability may be perceived as a practical tool to detect copying, it does not necessarily equate to copyright protection. Therefore, if it is adopted as standard for copyright
infringement, it may undermine copyright law intended purposes.

# Flocks of Stochastic Parrots: Differentially Private Prompt Learning for Large Language Models [View](https://arxiv.org/abs/2305.15594)

Large language models (LLMs) are excellent in-context learners. However, the
sensitivity of data contained in prompts raises privacy concerns. Our work first
shows that these concerns are valid: we instantiate a simple but highly effective
membership inference attack against the data used to prompt LLMs. To address
this vulnerability, one could forego prompting and resort to fine-tuning LLMs with
known algorithms for private gradient descent. However, this comes at the expense
of the practicality and efficiency offered by prompting. Therefore, we propose
to privately learn to prompt. We first show that soft prompts can be obtained
privately through gradient descent on downstream data. However, this is not the
case for discrete prompts. Thus, we orchestrate a noisy vote among an ensemble of
LLMs presented with different prompts, i.e., a flock of stochastic parrots. The vote
privately transfers the flock’s knowledge into a single public prompt. We show that
LLMs prompted with our private algorithms closely match the non-private baselines.
For example, using GPT3 as the base model, we achieve a downstream accuracy
of 92.7% on the sst2 dataset with (ε = 0.147, δ = 10−6
)-differential privacy vs.
95.2% for the non-private baseline. Through our experiments, we also show that
our prompt-based approach is easily deployed with existing commercial APIs.

# Learning with Impartiality to Walk on the Pareto Frontier of Fairness, Privacy, and Utility [View](https://arxiv.org/abs/2302.09183)
Deploying machine learning (ML) models often requires both fairness and privacy guarantees. Both of these objectives present unique
trade-offs with the utility (e.g., accuracy) of the model. However, the mutual interactions between fairness, privacy, and utility are
less well-understood. As a result, often only one objective is optimized, while the others are tuned as hyper-parameters. Because
they implicitly prioritize certain objectives, such designs bias the model in pernicious, undetectable ways. To address this, we adopt
impartiality as a principle: design of ML pipelines should not favor one objective over another. We propose impartially-specified models,
which provide us with accurate Pareto frontiers that show the inherent trade-offs between the objectives. Extending two canonical
ML frameworks for privacy-preserving learning, we provide two methods (FairDP-SGD and FairPATE) to train impartially-specified
models and recover the Pareto frontier. Through theoretical privacy analysis and a comprehensive empirical study, we provide an
answer to the question of where fairness mitigation should be integrated within a privacy-aware ML pipeline.

# Differentially Private Attention Computation [View](https://arxiv.org/abs/2305.04701)

Large language models (LLMs) have had a profound impact on numerous aspects of daily
life including natural language processing, content generation, research methodologies and so
on. However, one crucial issue concerning the inference results of large language models is
security and privacy. In many scenarios, the results generated by LLMs could possibly leak
many confidential or copyright information. A recent beautiful and breakthrough work [Vyas,
Kakade and Barak 2023] focus on such privacy issue of the LLMs from theoretical perspective.
It is well-known that computing the attention matrix is one of the major task during the LLMs
computation. Thus, how to give a provable privately guarantees of computing the attention
matrix is an important research direction.
Previous work [Alman and Song 2023, Brand, Song and Zhou 2023] have proposed provable
tight result for fast computation of attention without considering privacy concerns. One natural
mathematical formulation to quantity the privacy in theoretical computer science graduate
school textbook is differential privacy. Inspired by [Vyas, Kakade and Barak 2023], in this
work, we provide a provable result for showing how to differentially private approximate the
attention matrix.
From technique perspective, our result replies on a pioneering work in the area of differential
privacy by [Alabi, Kothari, Tankala, Venkat and Zhang 2022].

# FARA: Future-aware Ranking Algorithm for Fairness Optimization [View](https://arxiv.org/abs/2305.16637)
Ranking systems are the key components of modern Information Retrieval (IR) applications, such as search engines and recommender systems. Besides the ranking relevance to users, the exposure fairness to item providers has also been considered an important factor in ranking optimization. M any fair ranking Algorithms have been proposed to jointly optimize both ranking relevance and fairness. However, we find that most existing fair ranking methods adopt greedy algorithms that only optimize rankings for the next immediate session or request. As shown in this paper, such a myopic paradigm could limit the upper bound of ranking optimization and lead to suboptimal performance in the long term. To this end, we propose FARA, a novel Future-Aware Ranking Algorithm for ranking relevance and fairness optimization.Instead of greedily optimizing rankings for the next immediate session, FARA plans ahead by jointly optimizing multiple ranklists together and saving them for future sessions. Particularly, FARA first uses the Taylor expansion to investigate how future ranklist s will influence the overall fairness of the system. Then , based on the analysis of the Taylor expansion, FARA adopts a two-phase optimization algorithm where we first solve an optimal future exposure planning problem and then construct the optimal ranklists according to the optimal future exposure planning. Theoretically, we show that FARA is optimal for ranking relevance and fairness joint optimization. Empirically, our extensive experiments on three semi-synthesized datasets show that FARA is efficient, effective,and can deliver significantly better ranking performance compared to state-of-the-art fair ranking methods.


# Constitutional AI: Harmlessness from AI Feedback [View](https://arxiv.org/abs/2212.08073)
As AI systems become more capable, we would like to enlist their help to supervise other AIs. We experiment with methods for training a harmless AI assistant through self-improvement, without any human labels identifying harmful outputs. The only human oversight is provided through a list of rules or principles, and so we refer to the method as 'Constitutional AI'. The process involves both a supervised learning and a reinforcement learning phase. In the supervised phase we sample from an initial model, then generate self-critiques and revisions, and then finetune the original model on revised responses. In the RL phase, we sample from the finetuned model, use a model to evaluate which of the two samples is better, and then train a preference model from this dataset of AI preferences. We then train with RL using the preference model as the reward signal, i.e. we use 'RL from AI Feedback' (RLAIF). As a result we are able to train a harmless but non-evasive AI assistant that engages with harmful queries by explaining its objections to them. Both the SL and RL methods can leverage chain-of-thought style reasoning to improve the human-judged performance and transparency of AI decision making. These methods make it possible to control AI behavior more precisely and with far fewer human labels.


# Can Foundation Models Help Us Achieve Perfect Secrecy? [View](https://arxiv.org/abs/2205.13722) 
A key promise of machine learning is the ability to assist users with personal tasks. Because the personal context required to make accurate predictions is often sensitive, we require systems that protect privacy. A gold standard privacy-preserving system will satisfy perfect secrecy, meaning that interactions with the system provably reveal no private information. However, privacy and quality appear to be in tension in existing systems for personal tasks. Neural models typically require copious amounts of training to perform well, while individual users typically hold a limited scale of data, so federated learning (FL) systems propose to learn from the aggregate data of multiple users. FL does not provide perfect secrecy, but rather practitioners apply statistical notions of privacy -- i.e., the probability of learning private information about a user should be reasonably low. The strength of the privacy guarantee is governed by privacy parameters. Numerous privacy attacks have been demonstrated on FL systems and it can be challenging to reason about the appropriate privacy parameters for a privacy-sensitive use case. Therefore our work proposes a simple baseline for FL, which both provides the stronger perfect secrecy guarantee and does not require setting any privacy parameters. We initiate the study of when and where an emerging tool in ML -- the in-context learning abilities of recent pretrained models -- can be an effective baseline alongside FL. We find in-context learning is competitive with strong FL baselines on 6 of 7 popular benchmarks from the privacy literature and a real-world case study, which is disjoint from the pretraining data. [code](https://github.com/simran-arora/focus) 

# Training Private Models That Know What They Don't Know [View](https://arxiv.org/abs/2305.18393)
Training reliable deep learning models which avoid making overconfident but incorrect predictions is a longstanding challenge. This challenge is further exacerbated when learning has to be differentially private: protection provided to sensitive data comes at the price of injecting additional randomness into the learning process. In this work, we conduct a thorough empirical investigation of selective classifiers -- that can abstain when they are unsure -- under a differential privacy constraint. We find that several popular selective prediction approaches are ineffective in a differentially private setting as they increase the risk of privacy leakage. At the same time, we identify that a recent approach that only uses checkpoints produced by an off-the-shelf private learning algorithm stands out as particularly suitable under DP. Further, we show that differential privacy does not just harm utility but also degrades selective classification performance. To analyze this effect across privacy levels, we propose a novel evaluation mechanism which isolate selective prediction performance across model utility levels. Our experimental results show that recovering the performance level attainable by non-private models is possible but comes at a considerable coverage cost as the privacy budget decreases.

# Unleashing the Power of Randomization in Auditing Differentially Private ML [View](https://arxiv.org/abs/2305.18447)
We present a rigorous methodology for auditing differentially private machine learning algorithms by adding multiple carefully designed examples called canaries. We take a first principles approach based on three key components. First, we introduce Lifted Differential Privacy (LiDP) that expands the definition of differential privacy to handle randomized datasets. This gives us the freedom to design randomized canaries. Second, we audit LiDP by trying to distinguish between the model trained with K canaries versus K−1 canaries in the dataset, leaving one canary out. By drawing the canaries i.i.d., LiDP can leverage the symmetry in the design and reuse each privately trained model to run multiple statistical tests, one for each canary. Third, we introduce novel confidence intervals that take advantage of the multiple test statistics by adapting to the empirical higher-order correlations. Together, this new recipe demonstrates significant improvements in sample complexity, both theoretically and empirically, using synthetic and real data. Further, recent advances in designing stronger canaries can be readily incorporated into the new framework.

# What Can We Learn from Unlearnable Datasets? [View](https://arxiv.org/pdf/2305.19254.pdf)
In an era of widespread web scraping, unlearnable dataset methods have the potential to protect data privacy by preventing deep neural networks from generalizing.
But in addition to a number of practical limitations that make their use unlikely,
we make a number of findings that call into question their ability to safeguard data.
First, it is widely believed that neural networks trained on unlearnable datasets
only learn shortcuts, simpler rules that are not useful for generalization. In contrast,
we find that networks actually can learn useful features that can be reweighed for
high test performance, suggesting that image privacy is not preserved. Unlearnable
datasets are also believed to induce learning shortcuts through linear separability
of added perturbations. We provide a counterexample, demonstrating that linear
separability of perturbations is not a necessary condition. To emphasize why linearly separable perturbations should not be relied upon, we propose an orthogonal
projection attack which allows learning from unlearnable datasets published in
ICML 2021 and ICLR 2023. Our proposed attack is significantly less complex
than recently proposed techniques.

# Tree-Ring Watermarks: Fingerprints for Diffusion Images that are Invisible and Robust [View](https://arxiv.org/pdf/2305.20030.pdf)
Watermarking the outputs of generative models is a crucial technique for tracing
copyright and preventing potential harm from AI-generated content. In this paper,
we introduce a novel technique called Tree-Ring Watermarking that robustly fingerprints diffusion model outputs. Unlike existing methods that perform post-hoc
modifications to images after sampling, Tree-Ring Watermarking subtly influences
the entire sampling process, resulting in a model fingerprint that is invisible to
humans. The watermark embeds a pattern into the initial noise vector used for
sampling. These patterns are structured in Fourier space so that they are invariant to convolutions, crops, dilations, flips, and rotations. After image generation,
the watermark signal is detected by inverting the diffusion process to retrieve the
noise vector, which is then checked for the embedded signal. We demonstrate
that this technique can be easily applied to arbitrary diffusion models, including text-conditioned Stable Diffusion, as a plug-in with negligible loss in FID.
Our watermark is semantically hidden in the image space and is far more robust
than watermarking alternatives that are currently deployed [Code](https://github.com/YuxinWenRick/tree-ring-watermark)

# Training Private Models That Know What They Don’t Know [View](https://arxiv.org/pdf/2305.18393.pdf)
Training reliable deep learning models which avoid making overconfident but incorrect predictions is
a longstanding challenge. This challenge is further exacerbated when learning has to be differentially
private: protection provided to sensitive data comes at the price of injecting additional randomness
into the learning process. In this work, we conduct a thorough empirical investigation of selective
classifiers—that can abstain when they are unsure—under a differential privacy constraint. We find
that several popular selective prediction approaches are ineffective in a differentially private setting as
they increase the risk of privacy leakage. At the same time, we identify that a recent approach that
only uses checkpoints produced by an off-the-shelf private learning algorithm stands out as particularly
suitable under DP. Further, we show that differential privacy does not just harm utility but also
degrades selective classification performance. To analyze this effect across privacy levels, we propose
a novel evaluation mechanism which isolate selective prediction performance across model utility
levels. Our experimental results show that recovering the performance level attainable by non-private
models is possible but comes at a considerable coverage cost as the privacy budget decreases.

