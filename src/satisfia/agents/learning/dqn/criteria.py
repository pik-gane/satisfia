from satisfia.util.interval_tensor import IntervalTensor

from torch import Tensor, no_grad
from typing import List, Dict

def action_losses( loss_names: List[str],
                   criteria: Dict[str, Tensor],
                   state_aspirations: IntervalTensor,
                   action_aspirations: IntervalTensor,
                   estimated_action_probabilities: Tensor ) -> Tensor:
    
    ...

@no_grad()
def complete_criteria(criteria: Dict[str, Tensor]):
    if "maxAdmissibleQ" in criteria and "maxAdmissibleV" not in criteria:
        criteria["maxAdmissibleV"] = criteria["maxAdmissibleQ"].max(-1).values

    if "minAdmissibleQ" in criteria and "minAdmissibleV" not in criteria:
        criteria["minAdmissibleV"] = criteria["minAdmissibleQ"].min(-1).values