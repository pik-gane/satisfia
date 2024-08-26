from test_dqn import *
def agent_setup( env_name: str,
                    gridworld: bool = True,
                    max_achievable_total: float | None = None,
                    min_achievable_total: float | None = None ):

    print(env_name)

    def make_env():
        if gridworld:
            env, _ = make_simple_gridworld(env_name, time=10)
            env = RestrictToPossibleActionsWrapper(env)
        else:
            env = gym.make(env_name)
            env = TimeLimit(env, 1_000)
            env = RescaleDeltaWrapper(env, from_interval=(-500, 100), to_interval=(-5, 1))
        return env

    def make_model(pretrained=None):
        # to do: compute d_observation properly
        d_observation = len(make_env().observation_space) if gridworld else 8
        n_actions = make_env().action_space.n
        model = SatisfiaMLP(
            input_size = d_observation,
            output_not_depending_on_agent_parameters_sizes = { "maxAdmissibleQ": n_actions,
                                                               "minAdmissibleQ": n_actions },
            #output_depending_on_agent_parameters_sizes = dict(), # { "Q": n_actions },
            common_hidden_layer_sizes = [64, 64],
            hidden_layer_not_depending_on_agent_parameters_sizes = [16],
            #hidden_layer_depending_on_agent_parameters_sizes = [], # [64],
            batch_size = cfg.num_envs,
            layer_norms = False,
            dropout = 0
        )
        if pretrained is not None:
            model.load_state_dict(pretrained)
        return model
    


    planning_agent = AgentMDPPlanning(cfg.satisfia_agent_params, make_env()) if gridworld else None

    model = run_or_load( f"dqn-{env_name}-no-discount.pickle",
                         train_dqn,
                         make_env,
                         make_model,
                         dataclasses.replace(
                             cfg,
                             planning_agent_for_plotting_ground_truth=planning_agent
                         ) )
    model = model.to(device)
    learning_agent = AgentMDPDQN( cfg.satisfia_agent_params,
                                        model,
                                        num_actions = make_env().action_space.n,
                                        device = device )
    first_observation, _ = make_env().reset()
    if max_achievable_total is None:
                max_achievable_total = planning_agent.maxAdmissibleV(first_observation)
    if min_achievable_total is None:
                min_achievable_total = planning_agent.minAdmissibleV(first_observation)
    
    learning_agent = AgentMDPDQN( cfg.satisfia_agent_params,
                                            model,
                                            num_actions = make_env().action_space.n,
                                            device = device )
    compute_total(learning_agent, model, [0, 10])
agent_setup( 'LunarLander-v2',
                    gridworld = False,
                    min_achievable_total = -5,
                    max_achievable_total = 5 )