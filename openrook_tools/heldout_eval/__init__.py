"""Held-out evaluation: intents are stored, wording is written fresh at run time.

The package never contains example sentences. A writer model produces the
phrasings in several languages for each abstract intent; they are sent to a
running OpenRook server and scored from its reported perception and decision.
"""

__version__ = "1"
