from satisfia.agents.learning.dqn.agent_mdp_dqn import local_policy
from satisfia.agents.learning.dqn.criteria import complete_criteria
from satisfia.agents.learning.dqn.config import DQNConfig
from satisfia.agents.learning.dqn.replay_buffer import ReplayBufferSample

from torch import Tensor, argmax, argmin, max, min, where, zeros_like, no_grad, linspace
from typing import Dict, Callable

@no_grad()
def bellman_formula( replay_buffer_sample: ReplayBufferSample,
                     q_network:            Callable[[Tensor], Dict[str, Tensor]],
                     target_network:       Callable[[Tensor], Dict[str, Tensor]],
                     predicted_criteria:   Dict[str, Tensor],
                     cfg:                  DQNConfig ) -> Dict[str, Tensor]:
    """
    Compute the Bellman formula for the given replay buffer sample.
    :param replay_buffer_sample: The sample from the replay buffer.
    :param q_network: The Q-network which is used for double Q-learning.
    :param target_network: The target network which is being trained
    :param predicted_criteria: NOTE: I believe this is unused
    :param cfg: The configuration of the DQN agent.
    """

    criteria = dict()
    support = linspace(cfg.support_min, cfg.support_max, cfg.num_atoms, device=cfg.device)
    criterion_names = [ criterion
                       for criterion, coefficient in cfg.criterion_coefficients_for_loss.items()
                       if coefficient != 0 ]
    


    next_criteria = target_network( replay_buffer_sample.next_observations,
                                    replay_buffer_sample.next_aspirations,
                                    noisy=False,
                                    distribution=cfg.distributional )
    complete_criteria(next_criteria) #TODO: Can we get rid of these? 
    
    if cfg.double_q_learning:
        next_criteria_from_q_network = q_network( replay_buffer_sample.next_observations,
                                                  replay_buffer_sample.aspirations,
                                                  noisy=False,
                                                   distribution=cfg.distributional )

        # Calculate current state probabilities
        log_next_criteria_from_q_network = q_network( replay_buffer_sample.observations,
                                        replay_buffer_sample.aspirations,
                                        noisy=False,
                                        distribution=cfg.distributional,
                                        log=True )
        log_next_criteria_from_q_network_a = log_next_criteria_from_q_network
        complete_criteria(log_next_criteria_from_q_network)

        for criteria in log_next_criteria_from_q_network.keys():
            log_next_criteria_from_q_network_a[criteria] = log_next_criteria_from_q_network[criteria][range(cfg.batch_size), replay_buffer_sample.actions]


    if any( criterion_name in criterion_names
            for criterion_name in ["Q", "Q2", "Q3", "Q4", "Q5", "Q6"] ):
        
        if cfg.frozen_model_for_exploration:
            predicted_criteria_for_policy = cfg.frozen_model_for_exploration(
                replay_buffer_sample.next_observations,
                replay_buffer_sample.next_aspirations,
                noisy=False
            )
            complete_criteria(predicted_criteria_for_policy)
        else:
            predicted_criteria_for_policy = next_criteria

        next_policy = local_policy( cfg.satisfia_agent_params,
                                    predicted_criteria_for_policy,
                                    replay_buffer_sample.next_aspirations )

    for max_or_min in ["max", "min"]:
        criterion_name = f"{max_or_min}AdmissibleQ"
        torch_max_or_min = {"max": max,    "min": min}   [max_or_min]
        torch_argmax_or_argmin = {"max": argmax, "min": argmin}[max_or_min]

        distributions_ns = support.expand_as(next_criteria[criterion_name])
        argmax_indices = distributions_ns.sum(2).argmax(1) 
        probabilities_ns_actions = next_criteria_from_q_network
        for criteria in log_next_criteria_from_q_network.keys():
            probabilities_ns_actions[criteria] = next_criteria_from_q_network[criteria][range(cfg.batch_size), argmax_indices]

        if criterion_name not in criterion_names:
            continue

        if cfg.double_q_learning:
            
            target_from_q_network = next_criteria_from_q_network[criterion_name]
            target_max_or_min = next_criteria[criterion_name].gather(
                -1,
                torch_argmax_or_argmin(target_from_q_network, -1, keepdim=True)
            ).squeeze(-1)

            Tz = replay_buffer_sample.deltas + cfg.discount * support.unsqueeze(0)
            Tz = Tz.clamp(min=cfg.support_min, max=cfg.support_max)

            b = (Tz - support.unsqueeze(0)) / cfg.delta_z
            l = b.floor().long()
            u = b.ceil().long()

            l[(u > 0) * (l == u)] -= 1
            u[(l < (cfg.num_atoms - 1)) * (l == u)] += 1

            m = replay_buffer_sample.next_observations.new_zeros(cfg.batch_size, cfg.num_atoms)
            offset = linspace(0, (cfg.batch_size - 1) * cfg.num_atoms, cfg.batch_size).long().unsqueeze(1)
            m.view(-1).index_add_(0, (l + offset).view(-1), (target_max_or_min * (u.float() - b)).view(-1))
            m.view(-1).index_add_(0, (u + offset).view(-1), (target_max_or_min * (b - l.float())).view(-1))
            criteria[criterion_name] = m * log_next_criteria_from_q_network_a
        else:
            target_max_or_min = torch_max_or_min(next_criteria[criterion_name], dim=-1).values

            target_max_or_min = where( replay_buffer_sample.dones, zeros_like(target_max_or_min), target_max_or_min )
            criteria[criterion_name] = replay_buffer_sample.deltas + cfg.discount * target_max_or_min

    # Make sure that all Q below highest Q are present
    for i in range(2, 7):
        if f"Q{i}" in criterion_names:
            asserted_name = f"Q{i-1}" if i > 2 else "Q"
            assert asserted_name in criterion_names

    if "Q" in criterion_names:
        V = (next_policy.probs * next_criteria["Q"]).sum(-1)
        criteria["Q"] =   replay_buffer_sample.deltas \
                        + where( replay_buffer_sample.dones,
                                 zeros_like(replay_buffer_sample.deltas),
                                 cfg.discount * V )
        
    if "Q2" in criterion_names:
        V2 = (next_policy.probs * next_criteria["Q2"]).sum(-1)
        criteria["Q2"] =   replay_buffer_sample.delats ** 2 \
                         + where( replay_buffer_sample.dones,
                                  zeros_like(replay_buffer_sample.deltas),
                                    2 * cfg.discount * V 
                                  + cfg.discount ** 2 * V2 )
        
    if "Q3" in criterion_names:
        V3 = (next_policy.probs * next_criteria["Q3"]).sum(-1)
        criteria["Q3"] =   replay_buffer_sample.deltas ** 3 \
                         + where( replay_buffer_sample.dones,
                                  zeros_like(replay_buffer_sample.deltas),
                                    3 * cfg.discount * replay_buffer_sample.deltas ** 2 * V
                                  + 3 * cfg.discount ** 2 * replay_buffer_sample.deltas * V2
                                  + cfg.discount ** 3 * V3 )
        
    if "Q4" in criterion_names:
        V4 = (next_policy.probs * next_criteria["Q4"]).sum(-1)
        criteria["Q4"] =   replay_buffer_sample.deltas ** 4 \
                         + where( replay_buffer_sample.dones,
                                  zeros_like(replay_buffer_sample.deltas),
                                    4 * cfg.discount * replay_buffer_sample.deltas ** 3 * V
                                  + 6 * cfg.discount ** 2 * replay_buffer_sample.deltas ** 2 * V2
                                  + 4 * cfg.discount ** 3 * replay_buffer_sample.deltas * V3
                                  + cfg.discount ** 4 * V4 )
        
    if "Q5" in criterion_names:
        V5 = (next_policy.probs * next_criteria["Q5"]).sum(-1)
        criteria["Q5"] =   replay_buffer_sample.deltas ** 5 \
                         + where( replay_buffer_sample.dones,
                                  zeros_like(replay_buffer_sample.deltas),
                                    5 * cfg.discount * replay_buffer_sample.deltas ** 4 * V
                                  + 10 * cfg.discount ** 2 * replay_buffer_sample.deltas ** 3 * V2
                                  + 10 * cfg.discount ** 3 * replay_buffer_sample.deltas ** 2 * V3
                                  + 5 * cfg.discount ** 4 * replay_buffer_sample.deltas * V4
                                  + cfg.discount ** 5 * V5 )
        
    if "Q6" in criterion_names:
        V6 = (next_policy.probs * next_criteria["Q6"]).sum(-1)
        criteria["Q6"] =   replay_buffer_sample.deltas ** 6 \
                         + where( replay_buffer_sample.dones,
                                  zeros_like(replay_buffer_sample.deltas),
                                    6 * cfg.discount * replay_buffer_sample.deltas ** 5 * V
                                  + 15 * cfg.discount ** 2 * replay_buffer_sample.deltas ** 4 * V2
                                  + 20 * cfg.discount ** 3 * replay_buffer_sample.deltas ** 3 * V3
                                  + 15 * cfg.discount ** 4 * replay_buffer_sample.deltas ** 2 * V4
                                  + 6 * cfg.discount ** 5 * replay_buffer_sample.deltas * V5
                                  + cfg.discount ** 6 * V6 )

    return criteria
