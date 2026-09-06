import os

def has_exactly_two_mkv_files(directory: str) -> bool:
    count = 0
    with os.scandir(directory) as entries:
        for entry in entries:
            # Check if the current entry is a file and ends with ".mkv"
            if entry.is_file() and entry.name.endswith(".mkv"):
                count += 1
                # Exit early if more than two such files are found
                if count > 2:
                    return False
    return count == 2
def list_end_directories(path: str) -> list:
    """
    Recursively traverse a directory and 
    return all paths that are files (end of branch).
    """
    end_files = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.startswith('SoccerNetV2'):                
                pass
            else:
                refined_path = os.path.join(root, file)
                # print(refined_path)
                # if has_exactly_two_mkv_files(refined_path):
                end_files.append(refined_path)
    end_files = ["/".join(file.split("/")[:-1]) for file in end_files]
    end_files = [item for item in end_files if has_exactly_two_mkv_files(item)]
    end_files = sorted(list(set(end_files)))
    return end_files