from state import NeurumiState


def compute_reward(state: NeurumiState) -> float:
    """
    Computes a scalar reward signal from Neurumi's current drives.

    This is the only feedback the Q-network receives — it has no access
    to ACTION_EFFECTS. It must discover on its own which actions
    produce high reward over time.

    Design:
    - High affection and energy  = good
    - High hunger and fear       = bad (inverted with 1 - value)
    - All drives balanced        = small bonus (emergent stability goal)
    - Critical drive extremes    = sharp penalty
    """
    s = state

    # Weighted wellness score — hunger and fear are inverted because
    # high values mean bad states for those drives
    wellness = (
        (1.0 - s.hunger)  * 0.30 +
        s.affection       * 0.30 +
        s.energy          * 0.20 +
        (1.0 - s.fear)    * 0.15 +
        s.curiosity       * 0.05
    )

    # Sharp penalty for critical states — teaches the agent that
    # letting any drive hit an extreme is especially costly
    penalty = 0.0
    if s.hunger    > 0.90: penalty += 0.30  # starving
    if s.fear      > 0.85: penalty += 0.20  # terrified
    if s.energy    < 0.05: penalty += 0.20  # completely exhausted
    if s.affection < 0.05: penalty += 0.15  # completely alone

    # Balance bonus — small reward when all drives hover near mid-range.
    # Variance across drives: low variance = balanced = small bonus.
    # This encourages the agent to maintain overall stability, not just
    # fix the single worst drive while ignoring the others.
    drives = [s.hunger, s.curiosity, s.affection, s.energy, s.fear]
    variance = sum((d - 0.5) ** 2 for d in drives) / len(drives)
    balance_bonus = max(0.0, 0.1 * (1.0 - variance * 4))

    reward = wellness - penalty + balance_bonus

    # Standard RL practice: clamp reward to [-1, 1]
    return max(-1.0, min(1.0, round(reward, 4)))