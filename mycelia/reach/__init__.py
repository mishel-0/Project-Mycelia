"""Mycelia Reach: internet channels for Mycelia, modelled on Agent-Reach
(https://github.com/Panniantong/Agent-Reach, MIT): each channel declares
which sources it handles, checks whether a backend really works, and reads
through ordered fallback backends.

Deliberately excluded from Agent-Reach's feature set: browser-cookie
extraction and logged-in platforms (personal-account access), and Exa search
(an external AI service, which Pure Mycelia does not use). Everything read
is untrusted data, logged with provenance, and never executed.
"""
