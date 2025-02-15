def generate_action_sequence(num_frames, command):
    """
    Generates a list of action dicts, each of the form:
        {
          "dx": float,
          "dy": float,
          "buttons": list,
          "keys": list
        }

    The total length of the returned list will be exactly num_frames.

    :param num_frames: int - total number of frames across all steps of the command
    :param command: str - the type of command sequence to create.
                         Single-movement commands:
                           - "forward"
                           - "backward"
                           - "left"
                           - "right"

                         Multi-step commands:
                           - "left-right"
                           - "forward-backward"
                           - "forward-turn-forward-turn"
    :return: list of dicts (one per frame)
    """
    # Helper function to subdivide num_frames among n sub-steps
    def subdivide_frames(total_frames, n_sub):
        """
        Evenly distribute total_frames among n_sub steps, returning a list of length n_sub.
        Example: subdivide_frames(10, 2) -> [5, 5]
                 subdivide_frames(11, 2) -> [6, 5]
        """
        base = total_frames // n_sub
        remainder = total_frames % n_sub
        counts = []
        for i in range(n_sub):
            # Give 1 extra frame to the first 'remainder' steps
            step_frames = base + (1 if remainder > 0 else 0)
            remainder -= 1 if remainder > 0 else 0
            counts.append(step_frames)
        return counts

    actions = []

    # -------------------------------------------------------------------------
    # 1) Single-step commands: "forward", "backward", "left", "right"
    #    All frames do the same thing.
    # -------------------------------------------------------------------------
    if command == "forward":
        for _ in range(num_frames):
            actions.append({
                "dx": 0.0,
                "dy": 0.0,
                "buttons": [],
                "keys": ["w"]   # move forward
            })

    elif command == "backward":
        for _ in range(num_frames):
            actions.append({
                "dx": 0.0,
                "dy": 0.0,
                "buttons": [],
                "keys": ["s"]   # move backward
            })

    elif command == "left":
        for _ in range(num_frames):
            actions.append({
                "dx": 0.0,
                "dy": 0.0,
                "buttons": [],
                "keys": ["a"]   # move left
            })

    elif command == "right":
        for _ in range(num_frames):
            actions.append({
                "dx": 0.0,
                "dy": 0.0,
                "buttons": [],
                "keys": ["d"]   # move right
            })

    # -------------------------------------------------------------------------
    # 2) Multi-step commands
    # -------------------------------------------------------------------------
    elif command == "left-right":
        # We have 2 sub-steps: left, then right
        # Subdivide num_frames into two parts
        left_frames, right_frames = subdivide_frames(num_frames, 2)

        # Step 1) Move left
        for _ in range(left_frames):
            actions.append({
                "dx": 0.0,
                "dy": 0.0,
                "buttons": [],
                "keys": ["a"]
            })

        # Step 2) Move right
        for _ in range(right_frames):
            actions.append({
                "dx": 0.0,
                "dy": 0.0,
                "buttons": [],
                "keys": ["d"]
            })

    elif command == "forward-backward":
        # 2 sub-steps: forward, then backward
        forward_frames, backward_frames = subdivide_frames(num_frames, 2)

        # Step 1) Move forward
        for _ in range(forward_frames):
            actions.append({
                "dx": 0.0,
                "dy": 0.0,
                "buttons": [],
                "keys": ["w"]
            })

        # Step 2) Move backward
        for _ in range(backward_frames):
            actions.append({
                "dx": 0.0,
                "dy": 0.0,
                "buttons": [],
                "keys": ["s"]
            })

    elif command == "forward-turn-forward-turn":
        # 4 sub-steps: forward, turn, forward, turn
        # Subdivide num_frames into 4 parts
        # Example: if num_frames = 10 -> [3, 3, 2, 2] or something similar
        forward1_frames, turn1_frames, forward2_frames, turn2_frames = subdivide_frames(num_frames, 4)

        # 1) forward
        for _ in range(forward1_frames):
            actions.append({
                "dx": 0.0,
                "dy": 0.0,
                "buttons": [],
                "keys": ["w"]
            })
        # 2) turn (simulate mouse dx)
        for _ in range(turn1_frames):
            actions.append({
                "dx": 20.0,
                "dy": 0.0,
                "buttons": [],
                "keys": []
            })
        # 3) forward
        for _ in range(forward2_frames):
            actions.append({
                "dx": 0.0,
                "dy": 0.0,
                "buttons": [],
                "keys": ["w"]
            })
        # 4) turn back the other way
        for _ in range(turn2_frames):
            actions.append({
                "dx": -20.0,
                "dy": 0.0,
                "buttons": [],
                "keys": []
            })

    else:
        raise ValueError(f"Unsupported command: {command}")

    return actions