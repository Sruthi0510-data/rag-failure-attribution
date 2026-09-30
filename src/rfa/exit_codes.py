"""Exit codes as an interface, because CI reads them and the README publishes them.

The pair that matters is OK against FINDING. Both mean the tool ran. A step that treats every
non-zero code as a crash reports a real finding as a broken tool, and the fix a team reaches
for is to stop running the tool.
"""

OK = 0
"""The evaluation ran and no failure was misattributed by recall@k."""

FINDING = 3
"""The evaluation ran and recall@k misattributes at least one failure. Not an error."""

USAGE = 2
"""The tool could not answer: bad arguments, unreadable corpus, an annotation off its document."""

UNDERPOWERED = 4
"""The evaluation set is too small to support the confidence the caller requested."""

NAMES = {OK: "OK", USAGE: "USAGE", FINDING: "FINDING", UNDERPOWERED: "UNDERPOWERED"}
