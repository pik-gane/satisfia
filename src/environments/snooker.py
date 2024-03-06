from ..world_model import MDPWorldModel

class Snooker(MDPWorldModel):
    """
    A simplified version of a two-player snooker game. In order to make this an infinite-horizon MDP with a rather small state space, we introduce three main differences from the real game of snooker: red balls are returned to the table after being potted, the game has no end, and we ignore some possible shots and special situations (like "free ball" etc.). This way the state space is small enough that we can enumerate all the states and transitions.

    There are ten states, grouped in 2 sets of 5 states each. In the first set of 5 states, player A is at the table, while in the second set of 5 states, player B is at the table. In 3 of the 5 states, the player must play a red ball from either a good position, a bad position, or a snookered position. In the other 2 of the 5 states, the player must play a colored (=non-red) ball from either a good or a bad position. The goodness of the position influences the probabilities of potting a ball, ending up in a good position, succeeding in a safety shot, or committing a foul. See below for the detailed probabilities. The states are encoded as follows:

    - 'Arg': player A is at the table, and must play a red ball from a good position.
    - 'Arb': player A is at the table, and must play a red ball from a bad position.
    - 'Ars': player A is at the table, and must play a red ball from a snookered position.
    - 'Acg': player A is at the table, and must play a colored ball from a good position.
    - 'Acb': player A is at the table, and must play a colored ball from a bad position.
    - 'Brg'-'Bcb': similarly for player B.
   
    Depending on the state, a the player at the table has the following possible actions:

    - 'Ph': play a high-risk high-gain pot attempt.
    - 'Pl': play a low-risk low-gain pot attempt.
    - 'Pn': play a "shot to nothing" pot attempt that likely leaves the opponent with a bad position if not successful.
    - 'S': play a safety shot.
    - 'Es': attempt a simple escape from the snooker that likely leaves the opponent with a good position.
    - 'Eh': attempt a harder escape from the snooker that leaves the opponent with a bad position if successful.
    
    The possible actions by state, and transitions and player score deltas by state and action, sorted from most likely to least likely, are as follows:

    - From Arg:
      - action Ph can lead to: 
        - .6 Acg (good pot), A+1
        - .2 Brg (unlucky fail to pot), 0 (no foul, .1) or B+4 (foul, .07) or B+5/6/7 (foul with blue/pink/black, .01/.01/.01)
        - .1 Brb (lucky fail to pot), 0
        - .1 Acb (bad pot), A+1
      - Pl: 
        - .8 Acb (bad pot), A+1
        - .1 Brb (lucky fail to pot), 0
        - .05 Acg (good pot), A+1
        - .05 Brg (unlucky fail to pot), 0 (no foul, .025) or B+4 (foul, .01) or B+5/6/7 (foul with blue/pink/black, .005/.005/.005) 
      - S:
        - .4 Brs (good safety), 0
        - .4 Brb (OK safety), 0
        - .15 Brg (failed safety), 0 (no foul, .1) or B+4 (foul, .02) or B+5/6/7 (foul with blue/pink/black, .01/.01/.01)
        - .05 Acb (fluked pot), A+1
    - Arb:
      - Ph: 
        - .4 Brg (unlucky fail to pot), 0 (no foul, .2) or B+4 (foul, .14) or B+5/6/7 (foul with blue/pink/black, .02/.02/.02)
        - .2 Acg (good pot), A+1
        - .2 Brb (lucky fail to pot), 0
        - .2 Acb (bad pot), A+1
      - Pl: 
        - .6 Acb (bad pot), A+1
        - .2 Brg (unlucky fail to pot), 0 (no foul, .1) or B+4 (foul, .04) or B+5/6/7 (foul with blue/pink/black, .02/.02/.02)
        - .1 Brb (lucky fail to pot), 0
        - .1 Acg (good pot), A+1
      - Pn: 
        - .4 Brb (lucky fail to pot), 0
        - .3 Acb (bad pot), A+1
        - .2 Acg (good pot), A+1
        - .1 Brg (unlucky fail to pot), 0 (no foul, .05) or B+4 (foul, .02) or B+5/6/7 (foul with blue/pink/black, .01/.01/.01)
      - S:
        - .45 Brb (OK safety), 0
        - .3 Brg (failed safety), 0 (no foul, .2) or B+4 (foul, .07) or B+5/6/7 (foul with blue/pink/black, .01/.01/.01)
        - .2 Brs (good safety), 0
        - .05 Acb (fluked pot), A+1
    - Ars:
      - Es:
        - .65 Brg (OK escape), 0 (no foul, .5) or B+4 (foul, .09) or B+5/6/7 (foul with blue/pink/black, .02/.02/.02)
        - .2 Ars (foul and miss), B+4 (no ball or cheap color hit, .14) or B+5/6/7 (hit blue/pink/black, .02/.02/.02)
        - .1 Brb (good escape), 0
        - .05 Acb (fluked pot), A+1
      - Eh:
        - .5 Ars (foul and miss), B+4 (no ball or cheap color hit, .35) or B+5/6/7 (hit blue/pink/black, .05/.05/.05)
        - .3 Brb (good escape), 0
        - .15 Brg (OK escape), 0 (no foul, .1) or B+4 (foul, .02) or B+5/6/7 (foul with blue/pink/black, .01/.01/.01)
        - .05 Acb (fluked pot), A+1

    (TODO: probabilities:)        
            
    - Acg:
      - Ph:
        - Brg (failed to pot black), 0 (no foul) or B+7 (foul)
        - Arg (good pot of black), A+7
        - Arb (bad pot of black), A+7
        - Ars (black potted but snookered self), A+7
      - Pl:
        - Arg (good pot of blue), A+5
        - Arb (bad pot of blue), A+5
        - Brg (failed to pot blue), 0 (no foul) or B+5 (foul)
        - Ars (blue potted but snookered self), A+5
      - S:
        - Brs (good safety), 0
        - Brb (OK safety), 0
        - Brg (failed safety), 0 (no foul) or B+4 (foul) or B+5 (foul with blue) or ... or B+7 (foul with black)
        - Ars (failed to roll up to ball-on and B decides player A should play again),
            B+4 (foul) or B+5 (foul with blue) or ... or B+7 (foul with black)
        - Arb (fluked pot, or good safety with foul and B decides player A should play again),
            A+2 (potted yellow) or A+3 (green) or ... or A+7 (black)
            or B+4 (foul) or B+5 (foul with blue) or ... or B+7 (foul with black)
    - Acb:
      - Ph:
        - Brg (failed to pot brown), 0 (no foul) or B+4 (foul)
        - Arg (good pot of brown), A+4
        - Arb (bad pot of brown), A+4
        - Ars (brown potted but snookered self), A+4
      - Pl:
        - Arb (bad pot of yellow), A+2
        - Brg (failed to pot yellow), 0 (no foul) or B+4 (foul)
        - Arg (good pot of yellow), A+2
        - Ars (yellow potted but snookered self), A+2
      - S:
        - Brb (OK safety), 0
        - Brg (failed safety), 0 (no foul) or B+4 (foul) or B+5 (foul with blue) or ... or B+7 (foul with black)
        - Brs (good safety), 0
        - Ars (failed to roll up to ball-on and B decides player A should play again),
            B+4 (foul) or B+5 (foul with blue) or ... or B+7 (foul with black)
        - Arb (fluked pot, or good safety with foul and B decides player A should play again),
            A+2 (potted yellow) or A+3 (green) or ... or A+7 (black)
            or B+4 (foul) or B+5 (foul with blue) or ... or B+7 (foul with black)
    (and similarly for B**)

    The observation returned by step and reset is a triple (state, score_delta_A, score_delta_B), where state is the state after the action, and score_delta_A and score_delta_B are the changes in the scores of players A and B, respectively. The "reward" returned is the score of the player who was at the table at the beginning of the step. The "done" flag is always False.

    """

    observation_distribution = {
        
    }