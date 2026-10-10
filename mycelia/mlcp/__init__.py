"""MLCP: Mycelial Latent Communication Protocol (node-to-node, v0.1).

Nodes summarise their state as a small vector, quantize it into learned
codebook tokens, and send them in a checksummed binary envelope. A receiver
decodes and fuses messages; an English decoder translates them for humans,
sentence by sentence, only from message fields.
"""
VERSION = 1
