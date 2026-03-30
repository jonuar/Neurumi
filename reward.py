from state import NeurumiState


def compute_reward(state: NeurumiState) -> float:
    """
    Computes a scalar reward signal from Neurumi's current state.

    This is the only feedback the Q-network receives, it has no access
    to ACTION_EFFECTS. It must discover on its own which actions
    produce high reward over time.

    Design principles:
    - High affection and energy = good
    - High hunger and fear     = bad (inverted)
    - Curiosity satisfied      = small bonus
    - All drives balanced      = additional bonus (emergent behavior goal)
    """
    s = state

    # Core wellness signal: weighted sum of drive satisfaction
    # Hunger and fear are "bad" drives, we invert them
    wellness = (
        (1.0 - s.hunger)   * 0.30 +   # not being hungry matters most
        s.affection        * 0.30 +   # feeling loved matters equally
        s.energy           * 0.20 +   # having energy is important
        (1.0 - s.fear)     * 0.15 +   # not being scared matters
        s.curiosity        * 0.05     # curiosity is a small bonus
    )

    # Penalty for critical states, sharp drop if any drive hits extremes
    # This teaches the agent that crisis states are especially bad
    penalty = 0.0
    if s.hunger > 0.90:   penalty += 0.3   # starving
    if s.fear > 0.85:     penalty += 0.2   # terrified
    if s.energy < 0.05:   penalty += 0.2   # completely exhausted
    if s.affection < 0.05: penalty += 0.15 # completely alone

    # Balance bonus, reward when all drives are in a healthy mid-range
    # Encourages the agent to maintain stability, not just fix one drive
    drives = [s.hunger, s.curiosity, s.affection, s.energy, s.fear]
    variance = sum((d - 0.5) ** 2 for d in drives) / len(drives)
    # Low variance = drives close to 0.5 = balanced
    balance_bonus = max(0.0, 0.1 * (1.0 - variance * 4))

    reward = wellness - penalty + balance_bonus

    # Clamp to [-1, 1] standard range for RL rewards
    return max(-1.0, min(1.0, round(reward, 4)))