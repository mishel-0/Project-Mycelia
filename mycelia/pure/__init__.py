"""Pure Mycelia v0.1: non-neural symbiotic rule learning.

No neural networks, pretrained models or LLMs. Agents induce word meanings
and compositional rules from demonstrations, share verified discoveries on a
message bus, check each other with a critic, and keep provenance-tracked
memory in SQLite. Experiments compare the network against single-learner,
voting and memorising baselines under equal compute budgets.
"""
