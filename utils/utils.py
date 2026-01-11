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


def split_uri(uri: str) -> tuple[str, str]:
    """
    Split a cloud bucket URI into bucket (with scheme) and object path.
    
    Args:
        uri: A URI string like "s3://tandemn/test/input.txt"
    
    Returns:
        A tuple of (bucket_uri, object_path) where:
        - bucket_uri: The scheme and bucket name (e.g., "s3://tandemn")
        - object_path: The path within the bucket (e.g., "test/input.txt")
    
    Examples:
        >>> split_uri("s3://tandemn/test/input.txt")
        ('s3://tandemn', 'test/input.txt')
        >>> split_uri("gs://my-bucket/folder/file.json")
        ('gs://my-bucket', 'folder/file.json')
    """
    # Find the scheme separator "://"
    scheme_end = uri.find("://")
    if scheme_end == -1:
        raise ValueError(f"Invalid URI format: missing scheme (expected '://')")
    
    # Find the first slash after the bucket name
    # Start searching after "://" + bucket name
    bucket_end = uri.find("/", scheme_end + 3)
    if bucket_end == -1:
        # No path after bucket, return bucket and empty string
        return uri, ""
    
    bucket_uri = uri[:bucket_end]
    object_path = uri[bucket_end + 1:]  # +1 to skip the slash
    
    return bucket_uri, object_path