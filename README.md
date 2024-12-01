# Robovac Neural Network Project

What does it take to build the AI for a robovac using Neural Networks for each of the components?

I have a cheap brand of robovac at home and the mistakes it routinely makes are a constant frustration. As
an opportunity to experiment with different forms of NN and different learning algorithms, this project attempts
to build the entire control system for a robovac out of NN models.

Keep in mind that this is being carried out as a learning exercise. In a real-world commercial product,
some parts of the system would be better implemented using known algorithms that are more accurate and reliable
than neural networks.


## Components

The development of this project has a number of stages. Each stage focuses on a different component of the
overall solution, with the final solution combining and orchestrating these components in a larger system.

Development is presently towards the end of development of the first stage.

Stages are as follows, see referenced notebooks for detailed design and outcomes.

1. SLAM
   * See: [SLAM.ipynb](SLAM.ipynb)
   * The agent needs to perform Simultaneous Localisation and Mapping (SLAM) while it explores
     and navigates within the environment.
   * This component is passive w.r.t. navigation in the environment. It depends on the other components
     for the generation of motor commands.
   * Uses supervised learning and a modified U-Net

2. Planning
   * The agent needs to plan a route around the environment that most efficiently covers the floor area during
     vacuuming.
   * This also includes choosing the most appropriate operation mode depending on the agent's current state.
     The agent should explore a new environment, clean a known environment, and return to the charging station
     when the floor is clean or its battery is low. If floor cleaning was recently interrupted, it should resume
     where it left off without repeating areas already cleaned.
   * This component depends on accurate mapping output from the SLAM component.
   * Model architecture yet to be decided. Likely either supervised learning with a CNN, or RL with a high-level
     policy network.

3. Motor Policy
   * The agent needs to generate fine-grained motor control signals in order to follow its plan.
   * This is the low-level view of action control that works in conjunction with the Planning component as the
     high-level view of action control. The high-level view operates against course-grained mapping information
     and generates course-grained plans while the low-level view translates that course-grained plans into
     fine-grained real-world motion.
   * Uses RL (reinforcement learning) and a policy network.

A number of additional stages of development could also be carried out which modify or enhance what's build so far:

* Online Learning
    * All of the above stages use simulated data for training and the agent is unlikely to do well once applied
      in the real-world. One solution is for the agent to "learn on the job", by enabling online learning as
      fine-tuning.
    * The basic metaphor is that a robovac is usually used in only one environment once bought, and thus it is safe
      for it to over-train on that environment. If it is to be later used in a new environment, it can simply be
      "factory reset" to its original pre-trained state.
    * Taking heavy inspiration from the ideas of _Adversarial Networks_ and _Predictive Coding_, I believe it is
      possible to produce a stable self-learning strategy, where the robovac can continue to fine-tune its models
      without further external input.

* Embedded Computing
    * The models created above are initially optimised for ease and simplicity of development. Unfortunately
      these are likely too big to fit within the limited memory capacity of an actual robovac, and too slow
      to work in real-time without a GPU.
    * Techniques to makes the models smaller need to be employed to make them fit onto a real robovac.

There are a number of reasons for choosing a modularised solution over an end-to-end one: 1) it is easier
to conceptualise the individual components, whereas the right architecture for a complete end-to-end system
is much harder to identify; 2) it is faster to develop and test smaller and simpler individual components; 3)
this gives the opportunity to experiment with different forms of network and learning algorithm; 4) In order to
implement online fine-tuning, I need key outputs from each component to feed into the loss function.


## Experiments

Each development stage goes through many experiments. They are available under the various `experiments-xxx` folders:
* [experiments-slam](experiments-slam/README.md)


## Acknowledgments

The DeepLearning.AI
[Deep Learning Specialization](https://www.deeplearning.ai/courses/deep-learning-specialization/) course by Andrew Ng
has had a profound influence on the development of this work. Dr. Ng's advise for troubleshooting models and on
where to focus improvement effort is an ongoing guide. The original UNet that the SLAM model is based on was heavily
inspired by the example presented within the course.
