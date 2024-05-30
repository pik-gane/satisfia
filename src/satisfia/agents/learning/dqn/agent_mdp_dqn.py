from satisfia.agents.makeMDPAgentSatisfia import AspirationAgent
from satisfia.agents.learning.dqn.criteria import action_losses
from satisfia.util.interval_tensor import IntervalTensor, relative_position

import torch
from torch import tensor, Tensor, zeros_like, full_like, allclose, inference_mode
from torch.distributions.categorical import Categorical
from torch.nn import Module
from functools import cache
from dataclasses import dataclass
from numbers import Number
from typing import List, Dict, Any, Callable

@inference_mode()
def state_aspirations(criteria: Dict[str, Tensor], aspirations: IntervalTensor) -> IntervalTensor:
    state_feasibility_intervals  = IntervalTensor( criteria["minAdmissibleV"],
                                                   criteria["maxAdmissibleV"] )
    return aspirations.clip_to(state_feasibility_intervals)

@inference_mode()
def action_aspirations( criteria: Dict[str, Tensor],
                        state_aspirations: IntervalTensor ) -> IntervalTensor:

    action_feasibility_intervals = IntervalTensor( criteria["minAdmissibleQ"],
                                                   criteria["maxAdmissibleQ"] )
    action_aspirations = state_aspirations.unsqueeze(-1)
    action_aspirations = \
        (action_aspirations - action_aspirations.lower + action_feasibility_intervals.lower).where(
            action_aspirations.lower < action_feasibility_intervals.lower,
            action_aspirations
        )
    action_aspirations = \
        (action_aspirations - action_aspirations.upper + action_feasibility_intervals.lower).where(
            action_aspirations.upper > action_feasibility_intervals.upper,
            action_aspirations
        )
    action_aspirations = action_aspirations.clip_to(action_feasibility_intervals)
    return action_aspirations

@inference_mode()
def combined_action_losses( params: Dict[str, Any],
                            criteria: Dict[str, Tensor],
                            state_aspirations: IntervalTensor,
                            action_aspirations: IntervalTensor,
                            estimated_action_probabilities: Tensor ) -> Tensor:
    
    LOSS_COEFFICIENT_PREFIX = "lossCoeff4"
    loss_coefficients = { param_name[len(LOSS_COEFFICIENT_PREFIX):]: param_value
                          for param_name, param_value in params.items()
                          if param_name.startswith(LOSS_COEFFICIENT_PREFIX) }
    
    action_losses_ = action_losses( list(loss_coefficients.keys()),
                                    criteria,
                                    state_aspirations,
                                    action_aspirations,
                                    estimated_action_probabilities )

    combined_action_losses = zeros_like(action_aspirations.lower)
    for loss_name, coefficient in loss_coefficients.items():
        if coefficient == 0:
            continue
        coecombined_action_losess += coefficient * action_losses_[loss_name]
    
    return combined_action_losses

@inference_mode()
def action_propensities( params: Dict[str, Any],
                         criteria: Dict[str, Tensor],
                         state_aspirations: IntervalTensor,
                         action_aspirations: IntervalTensor ) -> Tensor:

    num_actions = action_aspirations.lower.size(-1)
    losses = combined_action_losses( params,
                                     criteria,
                                     state_aspirations,
                                     action_aspirations,
                                     estimated_action_probabilities = 1 / num_actions )
    
    # replaced 1e-100 by 1e-10 because with tocrh's precision, 1e-100 == 0
    return (-(losses - losses.min(-1, keepdim=True).values)).exp().maximum(tensor(1e-10))

@inference_mode()
def local_policy( params: Dict[str, Any],
                  criteria: Dict[str, Tensor],
                  aspirations: IntervalTensor ) -> Categorical:
    
    # TensorInterval[batch]
    state_aspirations_   = state_aspirations(criteria, aspirations)

    # TensorInterval[batch, action]
    action_aspirations_  = action_aspirations(criteria, state_aspirations_)
    
    # Tensor[batch, action]
    action_propensities_ = action_propensities( params,
                                                criteria,
                                                state_aspirations=state_aspirations_,
                                                action_aspirations=action_aspirations_ )
    
    # TensorInterval[batch, action]
    action_probabilities = action_propensities_ / action_propensities_.sum(-1, keepdim=True)
    
    # Tensor[batch, action] of bools
    action_aspiration_midpoint_sides = \
        action_aspirations_.midpoint() > state_aspirations_.midpoint().unsqueeze(-1)
    
    # Tensor[batch, action_candidate] of bools
    # action_aspiration_midpoints_close_to_state_aspiration_midpoints = \
    #     (action_aspirations_.midpoint() - state_aspirations_.midpoint().unsqueeze(-1)).abs() <= 1e-5
    
    # Tensor[batch, first_action_candidate, second_action_candidate] of bools
    action_aspiration_midpoints_on_same_side =    action_aspiration_midpoint_sides.unsqueeze(-1) \
                                               == action_aspiration_midpoint_sides.unsqueeze(-2)
    
    # action_aspiration_midpoints_on_same_side &= \
    #     action_aspiration_midpoints_close_to_state_aspiration_midpoints.logical_not().unsqueeze(-1)
    # action_aspiration_midpoints_on_same_side &= \
    #     action_aspiration_midpoints_close_to_state_aspiration_midpoints.logical_not().unsqueeze(-2)

    # Tensor[batch, first_action_candidate]
    first_action_candidate_probabilities = action_probabilities
    
    # Tensor[batch, first_action_candidate, second_action_candidate] 
    second_action_candidate_probabilities_conditional_on_first_action_candidate = \
        action_probabilities.unsqueeze(-1).where(
            action_aspiration_midpoints_on_same_side.logical_not(),
            full_like(action_probabilities.unsqueeze(-1), 1e-10)
        )
    second_action_candidate_probabilities_conditional_on_first_action_candidate /= \
        second_action_candidate_probabilities_conditional_on_first_action_candidate.sum(-1, keepdim=True)
    
    # Tensor[batch, first_action_candidate, second_action_candidate]
    action_pair_probabilities = \
          first_action_candidate_probabilities.unsqueeze(-1) \
        * second_action_candidate_probabilities_conditional_on_first_action_candidate
    
    # Tensor[batch, first_action_candidate, second_action_candidate]
    action_candidate_mixture_probabilities = relative_position(
        action_aspirations_.midpoint().unsqueeze(-2),
        state_aspirations_.midpoint().unsqueeze(-1).unsqueeze(-1),
        action_aspirations_.midpoint().unsqueeze(-1)
    ).clip(0, 1)
    
    # Tensor[batch, action]
    action_probabilities = \
          (action_pair_probabilities *      action_candidate_mixture_probabilities ).sum(-1) \
        + (action_pair_probabilities * (1 - action_candidate_mixture_probabilities)).sum(-2)
    
    return Categorical(probs=action_probabilities, validate_args=True)

class AgentMDPDQN(AspirationAgent):
    def __init__(self, params: Dict[str, Any],
                       model: Callable[[Tensor], Dict[str, Tensor]],
                       num_actions: int,
                       device: str = "cpu" ):
        
        super().__init__(params)

        self.model = model
        self.num_actions = num_actions
        self.device = device

    @cache
    def modelOutput(self, state, aspiration) -> Dict[str, Tensor]:
        state = tensor(list(state), dtype=torch.float, device=self.device).unsqueeze(0)
        aspiration_low, aspiration_high = aspiration
        aspiration = IntervalTensor( tensor([aspiration_low],  dtype=torch.float, device=self.device),
                                     tensor([aspiration_high], dtype=torch.float, device=self.device) )
        output = self.model(state, aspiration, noisy=False)
        return {key: value.squeeze(0) for key, value in output.items()}

    def maxAdmissibleQ(self, state, action):
        return self.modelOutput(state, aspiration=(0, 0))["maxAdmissibleQ"][action].item()
    
    def minAdmissibleQ(self, state, action):
        return self.modelOutput(state, aspiration=(0, 0))["minAdmissibleQ"][action].item()

    def Q(self, state, action, action_aspiration):
        return self.modelOutput(state, action_aspiration)["Q"][action].item()

    def possible_actions(self, state=None):
        return list(range(self.num_actions))
    

    def disorderingPotential_action(self, state, action):
        raise NotImplemented()
    
    def agencyChange_action(self, state, action):
        raise NotImplemented()

    
    def LRAdev_action(self, state, action, aleph4action, myopic=False):
        raise NotImplemented()
    
    def behaviorEntropy_action(self, state, actionProbability, action, aleph4action):
        raise NotImplemented()
    
    def behaviorKLdiv_action(self, state, actionProbability, action, aleph4action):
        raise NotImplemented()
    
    def trajectoryEntropy_action(self, state, actionProbability, action, aleph4action):
        raise NotImplemented()
    
    def stateDistance_action(self, state, action, aleph4action):
        raise NotImplemented()
    
    def causation_action(self, state, action, aleph4action):
        raise NotImplemented()
    
    def causationPotential_action(self, state, action, aleph4action):
        raise NotImplemented()
    
    def otherLoss_action(self, state, action, aleph4action):
        raise NotImplemented()


    def Q2(self, state, action, aleph4action):
        raise NotImplemented()
    
    def Q3(self, state, action, aleph4action):
        raise NotImplemented()
    
    def Q4(self, state, action, aleph4action):
        raise NotImplemented()
    
    def Q5(self, state, action, aleph4action):
        raise NotImplemented()
    
    def Q6(self, state, action, aleph4action):
        raise NotImplemented()

    def Q_ones(self, state, action, aleph4action):
        raise NotImplemented()
    
    def Q_DeltaSquare(self, state, action, aleph4action):
        raise NotImplemented()

    def ETerminalState_action(self, state, action, aleph4action, policy="actual"):
        raise NotImplemented()
    
    def ETerminalState2_action(self, state, action, aleph4action, policy="actual"):
        raise NotImplemented()
