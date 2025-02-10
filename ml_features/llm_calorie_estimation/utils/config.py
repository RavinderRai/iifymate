import yaml

def load_yaml(file_path: str) -> dict:
    """
    Load a YAML file and return its contents.

    :param file_path: Path to the YAML file.
    :return: Contents of the YAML file as a dictionary.
    """
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)
