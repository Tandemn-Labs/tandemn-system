import yaml
from typing import Any, Dict

def update_yaml_file(
    yaml_path: str,
    replace_dict: Dict[str, Any],
    output_path: str
) -> Dict[str, Any]:
    """
    Read a YAML file and update specified fields at runtime.
    
    Args:
        yaml_path: Path to the YAML file
        replace_dict: Dict of keys to replace in the YAML and the values
    
    Returns:
        The updated YAML data as a dictionary
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Loop through dict of items to replace
    for key, value in replace_dict.items():
        # Handles nested keys, like "resources.instance_type"
        if '.' in key:
            keys = key.split('.')
            current = data
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value
        else:
            data[key] = value
    
    # Write the updated YAML back
    with open(output_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    return data