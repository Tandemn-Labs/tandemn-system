# Form a dict of key value pairs from reading a file
def dict_from_file(filename: str):
    
    output = {}
    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:          # skip empty lines
                continue
            key, value = line.split(":", 1)  # split on the first colon
            output[key.strip()] = value.strip()

    return output