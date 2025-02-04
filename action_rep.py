import torch
from finetune.datasets.utils import load_actions_as_tensors
from pathlib import Path


action_path = Path("/home/ss24m050/Documents/CogVideo/data_test/post/metadata/cheeky-cornflower-setter-6dd8ce374dfa-20220713-17443822_chunk_9.json")
action = load_actions_as_tensors(action_path, num_actions=90)
#formatted_actions = format_action_dict(action)
#print(formatted_actions)
def format_action_string(action_dict):
    num_frames = action_dict['wasd'].shape[1]
    action_names = list(action_dict.keys())
    
    formatted_lines = []
    for frame_idx in range(num_frames):
        active_actions = []
        
        # Process key actions
        if 'wasd' in action_dict:
            wasd_keys = ['w', 'a', 's', 'd']
            active_keys = [wasd_keys[i] for i in range(4) if action_dict['wasd'][0, frame_idx, i] > 0.5]
            active_actions.extend(active_keys)
        
        # Process other actions (space, shift, mouse buttons)
        for key in ['space', 'shift', 'mouse_1', 'mouse_2']:
            if key in action_dict and action_dict[key][0, frame_idx] > 0.5:
                active_actions.append(key.replace('_', ''))
        
        # Process dx and dy
        dx, dy = 0, 0
        if 'dx' in action_dict:
            dx = action_dict['dx'][0, frame_idx].item()
        if 'dy' in action_dict:
            dy = action_dict['dy'][0, frame_idx].item()
        
        if abs(dx) > 0 or abs(dy) > 0:
            active_actions.append(f'dx:{dx:.1f} dy:{dy:.1f}')
        
        # Format the frame output
        formatted_lines.append(f'Frame {frame_idx + 1}: {", ".join(active_actions)} |')
    
    return '\n'.join(formatted_lines)

print(format_action_string(action))


print(0)
