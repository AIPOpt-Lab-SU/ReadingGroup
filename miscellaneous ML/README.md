

# Nonspecific ML focused papers 

# Neural Functional Transformers [View](https://arxiv.org/abs/2305.13546)
The recent success of neural networks as implicit representation of data has driven
growing interest in neural functionals: models that can process other neural networks
as input by operating directly over their weight spaces. Nevertheless, constructing
expressive and efficient neural functional architectures that can handle
high-dimensional weight-space objects remains challenging. This paper uses the
attention mechanism to define a novel set of permutation equivariant weight-space
layers and composes them into deep equivariant models called neural functional
Transformers (NFTs). NFTs respect weight-space permutation symmetries while
incorporating the advantages of attention, which have exhibited remarkable success
across multiple domains. In experiments processing the weights of feedforward
MLPs and CNNs, we find that NFTs match or exceed the performance of prior
weight-space methods. We also leverage NFTs to develop INR2ARRAY, a novel
method for computing permutation invariant latent representations from the weights
of implicit neural representations (INRs). Our proposed method improves INR
classification accuracy by up to +17% over existing methods.

# Permutation Equivariant Neural Functionals [View](https://arxiv.org/abs/2302.14040)
This work studies the design of neural networks that can process the weights or
gradients of other neural networks, which we refer to as neural functional networks
(NFNs). Despite a wide range of potential applications, including learned optimization,
processing implicit neural representations, network editing, and policy
evaluation, there are few unifying principles for designing effective architectures
that process the weights of other networks. We approach the design of neural
functionals through the lens of symmetry, in particular by focusing on the permutation
symmetries that arise in the weights of deep feedforward networks because
hidden layer neurons have no inherent order. We introduce a framework for building
permutation equivariant neural functionals, whose architectures encode these
symmetries as an inductive bias. The key building blocks of this framework are
NF-Layers (neural functional layers) that we constrain to be permutation equivariant
through an appropriate parameter sharing scheme. In our experiments, we
find that permutation equivariant neural functionals are effective on a diverse set of
tasks that require processing the weights of MLPs and CNNs, such as predicting
classifier generalization, producing “winning ticket” sparsity masks for initializations,
and editing the weights of implicit neural representations (INRs). In addition,
we provide code for our models and experiments.

# Visualizing the Loss Landscape of Neural Nets [View](https://arxiv.org/abs/1712.09913)
Neural network training relies on our ability to find “good” minimizers of highly
non-convex loss functions. It is well-known that certain network architecture
designs (e.g., skip connections) produce loss functions that train easier, and wellchosen training parameters (batch size, learning rate, optimizer) produce minimizers that generalize better. However, the reasons for these differences, and their
effect on the underlying loss landscape, are not well understood. In this paper, we
explore the structure of neural loss functions, and the effect of loss landscapes on
generalization, using a range of visualization methods. First, we introduce a simple
“filter normalization” method that helps us visualize loss function curvature and
make meaningful side-by-side comparisons between loss functions. Then, using
a variety of visualizations, we explore how network architecture affects the loss
landscape, and how training parameters affect the shape of minimizers.

# On progressive sharpening, flat minima and generalisation [View](https://arxiv.org/abs/2305.14683)
We present a new approach to understanding the relationship between loss curvature and generalisation in deep learning. Specifically, we use existing empirical analyses of the spectrum of deep network loss Hessians to ground an ansatz tying together the loss Hessian and the input-output Jacobian of a deep neural network. We then prove a series of theoretical results which quantify the degree to which the input-output Jacobian of a model approximates its Lipschitz norm over a data distribution, and deduce a novel generalisation bound in terms of the empirical Jacobian. We use our ansatz, together with our theoretical results, to give a new account of the recently observed progressive sharpening phenomenon, as well as the generalisation properties of flat minima. Experimental evidence is provided to validate our claims.

# Deep reinforcement learning from human preferences [View](https://arxiv.org/abs/1706.03741)
For sophisticated reinforcement learning (RL) systems to interact usefully with
real-world environments, we need to communicate complex goals to these systems.
In this work, we explore goals defined in terms of (non-expert) human preferences
between pairs of trajectory segments. We show that this approach can effectively
solve complex RL tasks without access to the reward function, including Atari
games and simulated robot locomotion, while providing feedback on less than
1% of our agent’s interactions with the environment. This reduces the cost of
human oversight far enough that it can be practically applied to state-of-the-art
RL systems. To demonstrate the flexibility of our approach, we show that we can
successfully train complex novel behaviors with about an hour of human time.
These behaviors and environments are considerably more complex than any which
have been previously learned from human feedback.

# Surgical Fine-Tuning Improves Adaptation To Distribution Shifts [View](https://arxiv.org/abs/2210.11466)

A common approach to transfer learning under distribution shift is to fine-tune the
last few layers of a pre-trained model, preserving learned features while also adapting to the new task. This paper shows that in such settings, selectively fine-tuning
a subset of layers (which we term surgical fine-tuning) matches or outperforms
commonly used fine-tuning approaches. Moreover, the type of distribution shift influences which subset is more effective to tune: for example, for image corruptions,
fine-tuning only the first few layers works best. We validate our findings systematically across seven real-world data tasks spanning three types of distribution shifts.
Theoretically, we prove that for two-layer neural networks in an idealized setting,
first-layer tuning can outperform fine-tuning all layers. Intuitively, fine-tuning
more parameters on a small target dataset can cause information learned during
pre-training to be forgotten, and the relevant information depends on the type of
shift.

# Guiding continuous operator learning through Physics-based boundary constraints [View](https://arxiv.org/abs/2212.07477)

Boundary conditions (BCs) are important groups of physics-enforced constraints that are necessary for solutions of Partial Differential Equations (PDEs) to satisfy at specific spatial locations. These constraints carry important physical meaning, and guarantee the existence and the uniqueness of the PDE solution. Current neural-network based approaches that aim to solve PDEs rely only on training data to help the model learn BCs implicitly. There is no guarantee of BC satisfaction by these models during evaluation. In this work, we propose Boundary enforcing Operator Network (BOON) that enables the BC satisfaction of neural operators by making structural changes to the operator kernel. We provide our refinement procedure, and demonstrate the satisfaction of physics-based BCs, e.g. Dirichlet, Neumann, and periodic by the solutions obtained by BOON. Numerical experiments based on multiple PDEs with a wide variety of applications indicate that the proposed approach ensures satisfaction of BCs, and leads to more accurate solutions over the entire domain. The proposed correction method exhibits a (2X-20X) improvement over a given operator model in relative L2 error (0.000084 relative L2 error for Burgers' equation).

# Learning Physical Models that Can Respect Conservation Laws [View](https://arxiv.org/abs/2302.11002)

Recent work in scientific machine learning (SciML) has focused on incorporating partial differential equation (PDE) information into the learning process. Much of this work has focused on relatively ``easy'' PDE operators (e.g., elliptic and parabolic), with less emphasis on relatively ``hard'' PDE operators (e.g., hyperbolic). Within numerical PDEs, the latter problem class requires control of a type of volume element or conservation constraint, which is known to be challenging. Delivering on the promise of SciML requires seamlessly incorporating both types of problems into the learning process. To address this issue, we propose ProbConserv, a framework for incorporating conservation constraints into a generic SciML architecture. To do so, ProbConserv combines the integral form of a conservation law with a Bayesian update. We provide a detailed analysis of ProbConserv on learning with the Generalized Porous Medium Equation (GPME), a widely-applicable parameterized family of PDEs that illustrates the qualitative properties of both easier and harder PDEs. ProbConserv is effective for easy GPME variants, performing well with state-of-the-art competitors; and for harder GPME variants it outperforms other approaches that do not guarantee volume conservation. ProbConserv seamlessly enforces physical conservation constraints, maintains probabilistic uncertainty quantification (UQ), and deals well with shocks and heteroscedasticities. In each case, it achieves superior predictive performance on downstream tasks.

# Sharpness-Aware Minimization Leads to Low-Rank Features [View](https://arxiv.org/abs/2305.16292)
Sharpness-aware minimization (SAM) is a recently proposed method that minimizes the sharpness of the training loss of a neural network. While its generalization improvement is well-known and is the primary motivation, we uncover an additional intriguing effect of SAM: reduction of the feature rank which happens at different layers of a neural network. We show that this low-rank effect occurs very broadly: for different architectures such as fully-connected networks, convolutional networks, vision transformers and for different objectives such as regression, classification, language-image contrastive training. To better understand this phenomenon, we provide a mechanistic understanding of how low-rank features arise in a simple two-layer network. We observe that a significant number of activations gets entirely pruned by SAM which directly contributes to the rank reduction. We confirm this effect theoretically and check that it can also occur in deep networks, although the overall rank reduction mechanism can be more complex, especially for deep networks with pre-activation skip connections and self-attention layers.

# SGD with large step sizes learns sparse features [View](https://arxiv.org/abs/2210.05337)
We showcase important features of the dynamics of the Stochastic Gradient Descent (SGD)
in the training of neural networks. We present empirical observations that commonly used
large step sizes (i) lead the iterates to jump from one side of a valley to the other causing
loss stabilization, and (ii) this stabilization induces a hidden stochastic dynamics orthogonal
to the bouncing directions that biases it implicitly toward simple predictors. Furthermore, we
show empirically that the longer large step sizes keep SGD high in the loss landscape valleys,
the better the implicit regularization can operate and find sparse representations. Notably,
no explicit regularization is used so that the regularization effect comes solely from the SGD
training dynamics influenced by the step size schedule. Therefore, these observations unveil
how, through the step size schedules, both gradient and noise drive together the SGD dynamics
through the loss landscape of neural networks. We justify these findings theoretically through the
study of simple neural network models as well as qualitative arguments inspired from stochastic
processes. Finally, this analysis allows to shed a new light on some common practice and
observed phenomena when training neural networks

# The Curse of Recursion: Training on Generated Data Makes Models Forget [View](https://arxiv.org/abs/2305.17493)

Stable Diffusion revolutionised image creation from descriptive text. GPT-2, GPT-3(.5) and GPT-4 demonstrated astonishing performance across a variety of language tasks. ChatGPT introduced such language models to the general public. It is now clear that large language models (LLMs) are here to stay, and will bring about drastic change in the whole ecosystem of online text and images. In this paper we consider what the future might hold. What will happen to GPT-{n} once LLMs contribute much of the language found online? We find that use of model-generated content in training causes irreversible defects in the resulting models, where tails of the original content distribution disappear. We refer to this effect as Model Collapse and show that it can occur in Variational Autoencoders, Gaussian Mixture Models and LLMs. We build theoretical intuition behind the phenomenon and portray its ubiquity amongst all learned generative models. We demonstrate that it has to be taken seriously if we are to sustain the benefits of training from large-scale data scraped from the web. Indeed, the value of data collected about genuine human interactions with systems will be increasingly valuable in the presence of content generated by LLMs in data crawled from the Internet.

# Bounding and Counting Linear Regions of Deep Neural Networks [View](http://proceedings.mlr.press/v80/serra18b/serra18b.pdf)

We investigate the complexity of deep neural
networks (DNN) that represent piecewise linear
(PWL) functions. In particular, we study the number of linear regions, i.e. pieces, that a PWL function represented by a DNN can attain, both theoretically and empirically. We present (i) tighter
upper and lower bounds for the maximum number
of linear regions on rectifier networks, which are
exact for inputs of dimension one; (ii) a first upper
bound for multi-layer maxout networks; and (iii)
a first method to perform exact enumeration or
counting of the number of regions by modeling
the DNN with a mixed-integer linear formulation.
These bounds come from leveraging the dimension of the space defining each linear region. The
results also indicate that a deep rectifier network
can only have more linear regions than every shallow counterpart with same number of neurons if
that number exceeds the dimension of the input.

#  Learning Transformer Programs [View](https://arxiv.org/pdf/2306.01128.pdf)
Recent research in mechanistic interpretability has attempted to reverse-engineer
Transformer models by carefully inspecting network weights and activations. However, these approaches require considerable manual effort and still fall short of
providing complete, faithful descriptions of the underlying algorithms. In this
work, we introduce a procedure for training Transformers that are mechanistically
interpretable by design. We build on RASP [Weiss et al., 2021], a programming
language that can be compiled into Transformer weights. Instead of compiling
human-written programs into Transformers, we design a modified Transformer
that can be trained using gradient-based optimization and then be automatically
converted into a discrete, human-readable program. We refer to these models as
Transformer Programs. To validate our approach, we learn Transformer Programs
for a variety of problems, including an in-context learning task, a suite of algorithmic problems (e.g. sorting, recognizing Dyck-languages), and NLP tasks including
named entity recognition and text classification. The Transformer Programs can
automatically find reasonable solutions, performing on par with standard Transformers of comparable size; and, more importantly, they are easy to interpret. To
demonstrate these advantages, we convert Transformers into Python programs
and use off-the-shelf code analysis tools to debug model errors and identify the
“circuits” used to solve different sub-problems. We hope that Transformer Programs
open a new path toward the goal of intrinsically interpretable machine learning

