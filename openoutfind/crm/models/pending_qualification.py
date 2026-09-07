from django.db import models


class PendingQualification(models.Model):
    """The one lead handed to a calling agent for a verdict, if any.

    `find --agent-qualify` stops the job right before the call that would otherwise go
    to `AI_MODEL`, and the process exits — there is nobody left in memory to remember
    which candidate that was. This row is what a second, separate invocation reads to
    resume exactly that candidate rather than asking the qualifier's balance-driven
    selection to choose again, which could legitimately land on someone else now that
    nothing about the pool has changed except this process no longer holding it.

    At most one row ever exists: the underlying loop asks one candidate at a time,
    same as the `AI_MODEL`-keyed path does, so there is never a queue to manage or an
    id to thread back through the CLI.
    """

    lead = models.OneToOneField("outfind_crm.Lead", on_delete=models.CASCADE)
    created = models.DateTimeField(auto_now_add=True)
