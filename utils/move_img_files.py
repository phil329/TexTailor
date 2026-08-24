import os
import shutil
from collections import defaultdict
import re

def organize_files_by_prefix(source_folder, target_base_folder="organized_files"):
    """
    Move files with the same prefix into corresponding folders.
    
    Args:
        source_folder (str): source folder path
        target_base_folder (str): target base folder path 
    
    Example:
        # If source_folder contains:
        # file_001.txt, file_002.txt, file_003.txt
        # image_001.jpg, image_002.jpg
        # data_001.csv, data_002.csv
        
        # After running the function, target_base_folder will contain:
        # organized_files/file_/file_001.txt, file_002.txt, file_003.txt
        # organized_files/image_/image_001.jpg, image_002.jpg
        # organized_files/data_/data_001.csv, data_002.csv
    """
    # make sure source folder exists
    if not os.path.exists(source_folder):
        print(f"Source folder {source_folder} does not exist!")
        return
    
    # create target base folder if not exists
    if not os.path.exists(target_base_folder):
        os.makedirs(target_base_folder)
        print(f"create target folder: {target_base_folder}")
    
    # list all files in source folder
    files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
    
    if not files:
        print(f"source folder {source_folder} has no files!")
        return
    
    # Group by the prefix before the last underscore
    prefix_groups = defaultdict(list)
    
    for file in files:
        # Find the prefix in the filename (up to the last underscore)
        prefix = ""
        last_underscore_pos = file.rfind('_')
        if last_underscore_pos != -1:
            prefix = file[:last_underscore_pos]
        else:
            prefix = os.path.splitext(file)[0]
        
        # if no underscore found, use the whole filename as prefix
        if not prefix:
            prefix = os.path.splitext(file)[0]
        
        prefix_groups[prefix].append(file)
    
    # Moving files into corresponding folders
    moved_count = 0
    for prefix, file_list in prefix_groups.items():
        if len(file_list) > 1:  # process only if more than one file shares the prefix
            # create target subfolder
            target_folder = os.path.join(target_base_folder, prefix)
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)
                print(f"create subfolder: {target_folder}")
            
            # moving files
            for file in file_list:
                source_path = os.path.join(source_folder, file)
                target_path = os.path.join(target_folder, file)
                
                try:
                    shutil.move(source_path, target_path)
                    print(f"moving: {file} -> {target_folder}")
                    moved_count += 1
                except Exception as e:
                    print(f"Moving {file} failed: {e}")
    
    print(f"\nFinished! Moved {moved_count} files into {len(prefix_groups)} folders.")

def organize_files_by_pattern(source_folder, target_base_folder="organized_files", pattern_func=None):
    """
    Organize files using a custom pattern function.
    
    Args:
        source_folder (str): source folder path
        target_base_folder (str): target base folder path
        pattern_func (callable): custom pattern function, takes filename and returns group key

    Example:
        # Group by file extension
        def by_extension(filename):
            return os.path.splitext(filename)[1][1:]  # remove dot
        
        organize_files_by_pattern("source", "target", by_extension)
    """
    if not os.path.exists(source_folder):
        print(f"source folder {source_folder} does not exist!")
        return
    
    if not os.path.exists(target_base_folder):
        os.makedirs(target_base_folder)
    
    files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
    
    if not files:
        print(f"source folder {source_folder} has no files!")
        return
    
    if pattern_func:
        groups = defaultdict(list)
        for file in files:
            group_key = pattern_func(file)
            groups[group_key].append(file)
    else:
        groups = defaultdict(list)
        for file in files:
            prefix = ""
            last_underscore_pos = file.rfind('_')
            if last_underscore_pos != -1:
                prefix = file[:last_underscore_pos]
            else:
                prefix = os.path.splitext(file)[0]
            groups[prefix].append(file)
    
    # moving files
    moved_count = 0
    for group_key, file_list in groups.items():
        if len(file_list) > 1:
            target_folder = os.path.join(target_base_folder, str(group_key))
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)
            
            for file in file_list:
                source_path = os.path.join(source_folder, file)
                target_path = os.path.join(target_folder, file)
                
                try:
                    shutil.move(source_path, target_path)
                    print(f"moving: {file} -> {target_folder}")
                    moved_count += 1
                except Exception as e:
                    print(f"Moving {file} failed: {e}")

    print(f"\nFinished! Moved {moved_count} files into {len(groups)} folders.")

def preview_organization(source_folder):
    """
    Preview file organization results without actually moving files
    """
    if not os.path.exists(source_folder):
        print(f"source folder {source_folder} does not exist!")
        return
    
    files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
    
    if not files:
        print(f"source folder {source_folder} has no files!")
        return
    
    prefix_groups = defaultdict(list)
    
    for file in files:
        prefix = ""
        last_underscore_pos = file.rfind('_')
        if last_underscore_pos != -1:
            prefix = file[:last_underscore_pos]
        else:
            prefix = os.path.splitext(file)[0]
        
        prefix_groups[prefix].append(file)
    
    print("show the structure:")
    print("=" * 50)
    
    for prefix, file_list in prefix_groups.items():
        if len(file_list) > 1:
            print(f"\Folders: {prefix}/")
            for file in sorted(file_list):
                print(f"  - {file}")
        else:
            print(f"\Files: {file_list[0]}")
    
    print(f"\Total: {len(prefix_groups)} groups {len(files)} files")

# usage example
if __name__ == "__main__":
    # Example 1: Preview file organization results
    preview_organization("./outputs/strength_conditions_v3")

    # Example 2: Organize files by prefix
    organize_files_by_prefix("./outputs/strength_conditions_v3", "././outputs/strength_conditions_v3/")
    
    # Example 3: Organize files by custom pattern function
    # def by_extension(filename):
    #     return os.path.splitext(filename)[1][1:]  # remove dot
    
    # organize_files_by_pattern("./outputs", "./organized_by_extension", by_extension)

    # Example 4: Organize files by number range
    # def by_number_range(filename):
    #     numbers = re.findall(r'\d+', filename)
    #     if numbers:
    #         num = int(numbers[0])
    #         if num < 20:
    #             return "0-19"
    #         elif num < 40:
    #             return "20-39"
    #         else:
    #             return "40+"
    #     return "no_number"
    
    # organize_files_by_pattern("./outputs", "./organized_by_range", by_number_range)

    print("File organization functions have been loaded!")
    print("Usage:")
    print("1. preview_organization(source_folder) - Preview organization results")
    print("2. organize_files_by_prefix(source_folder, target_folder) - Organize by prefix")
    print("3. organize_files_by_pattern(source_folder, target_folder, pattern_func) - Custom organization")