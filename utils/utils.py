import yaml
from pathlib import Path
from typing import Any, Dict


def update_yaml_file(
    yaml_path: str, replace_dict: Dict[str, Any], output_path: str
) -> Dict[str, Any]:
    """
    Read a YAML file and update specified fields at runtime.

    Args:
        yaml_path: Path to the YAML file
        replace_dict: Dict of keys to replace in the YAML and the values

    Returns:
        The updated YAML data as a dictionary
    """
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    # Loop through dict of items to replace
    for key, value in replace_dict.items():
        # Handles nested keys, like "resources.instance_type"
        if "." in key:
            keys = key.split(".")
            current = data
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value
        else:
            data[key] = value

    # Write the updated YAML back
    with open(output_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    return data


def update_template(template_file: str, replace_dict: Dict[str, str]) -> str:
    """
    Read a template file and replace placeholders with values from a dictionary.

    Args:
        template_file: Path to the template file containing format placeholders
        replace_dict: Dictionary mapping placeholder names to replacement values.
                      Keys should match the placeholder names in the template
                      (e.g., {"name": "value"} for {name} in template)

    Returns:
        The formatted string with all placeholders replaced by their 
        corresponding values from replace_dict
    """ 
    tmpl = Path(template_file).read_text(encoding="utf-8")
    script = tmpl.format(**replace_dict)

    return script
