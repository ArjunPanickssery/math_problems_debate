import os
import json
import shutil
from pathlib import Path
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResultsMerger:
    TARGET_FILES = ['config.json', 'results.jsonl', 'stats.json']
    
    def __init__(self, source_dir, target_dir):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.target_dir.mkdir(parents=True, exist_ok=True)
        
    def count_lines(self, file_path):
        """Count lines in a file."""
        try:
            with open(file_path) as f:
                return sum(1 for _ in f)
        except FileNotFoundError:
            return 0
    
    def get_jsonl_entries(self, file_path):
        """Get all entries from a jsonl file as a dictionary of entries and their line numbers."""
        entries = {}
        try:
            with open(file_path) as f:
                for i, line in enumerate(f, 1):
                    entries[line.strip()] = i
        except FileNotFoundError:
            pass
        return entries
    
    def find_matching_paths(self):
        """Find all unique B/C paths and their corresponding A* sources."""
        path_mapping = defaultdict(list)
        
        # Walk through all A* directories
        for a_dir in self.source_dir.iterdir():
            if not a_dir.is_dir():
                continue
                
            # Walk through B/C structure under each A*
            for root, _, files in os.walk(a_dir):
                root_path = Path(root)
                
                # Check if directory has any of our target files
                if not any(f in files for f in self.TARGET_FILES):
                    continue
                
                # Log which files are present/missing
                missing_files = [f for f in self.TARGET_FILES if f not in files]
                if missing_files:
                    logger.info(f"Directory {root_path} is missing files: {missing_files}")
                
                # Get relative path from A* (B/C structure)
                rel_path = root_path.relative_to(a_dir)
                path_mapping[str(rel_path)].append(root_path)
        
        return path_mapping
    
    def merge_paths(self):
        """Merge matching paths according to the specified rules."""
        path_mapping = self.find_matching_paths()
        
        for rel_path, source_paths in path_mapping.items():
            if not source_paths:
                continue
                
            # Create target directory
            target_path = self.target_dir / rel_path
            target_path.mkdir(parents=True, exist_ok=True)
            
            # Handle results.jsonl if it exists in any source paths
            results_paths = [(p, self.count_lines(p / 'results.jsonl')) 
                           for p in source_paths
                           if (p / 'results.jsonl').exists()]
            
            if results_paths:
                # Sort paths by results.jsonl size in descending order
                sorted_paths = sorted(results_paths, key=lambda x: x[1], reverse=True)
                primary_path, _ = sorted_paths[0]
                
                # Get entries from primary file
                primary_entries = self.get_jsonl_entries(primary_path / 'results.jsonl')
                
                # Verify that the largest results.jsonl contains all entries from others
                for source_path, _ in sorted_paths[1:]:  # Skip the primary path
                    other_results_file = source_path / 'results.jsonl'
                    other_entries = self.get_jsonl_entries(other_results_file)
                    missing_entries = set(other_entries.keys()) - set(primary_entries.keys())
                    
                    if missing_entries:
                        logger.warning(
                            f"\nWarning: {primary_path/'results.jsonl'} is missing entries that exist in "
                            f"{other_results_file}:\n"
                            f"Number of missing entries: {len(missing_entries)}\n"
                            f"Missing entries found at these lines in {other_results_file}:"
                        )
                        for entry in missing_entries:
                            logger.warning(f"Line {other_entries[entry]}: {entry[:100]}...")
                
                # Copy results.jsonl from primary path
                if (primary_path / 'results.jsonl').exists():
                    shutil.copy2(primary_path / 'results.jsonl', target_path / 'results.jsonl')
                
                # For both config.json and stats.json, try paths in order of results.jsonl size
                config_found = False
                stats_found = False
                
                for path, _ in sorted_paths:
                    # Try to find config.json if we haven't yet
                    if not config_found and (path / 'config.json').exists():
                        shutil.copy2(path / 'config.json', target_path / 'config.json')
                        if path != primary_path:
                            logger.info(f"Using config.json from {path} (different from primary results.jsonl path)")
                        config_found = True
                    
                    # Try to find stats.json if we haven't yet
                    if not stats_found and (path / 'stats.json').exists():
                        shutil.copy2(path / 'stats.json', target_path / 'stats.json')
                        if path != primary_path:
                            logger.info(f"Using stats.json from {path} (different from primary results.jsonl path)")
                        stats_found = True
                    
                    # If we found both files, we can stop searching
                    if config_found and stats_found:
                        break
                
                if not stats_found:
                    logger.warning(f"No stats.json found in any of the source paths for {rel_path}")
                
                if not config_found:
                    logger.warning(f"No config.json found in any of the source paths for {rel_path}. Removing target directory.")
                    shutil.rmtree(target_path)
                    continue
            
            else:
                # If no results.jsonl exists, check if at least config.json exists
                config_found = False
                for source_path in source_paths:
                    if (source_path / 'config.json').exists():
                        shutil.copy2(source_path / 'config.json', target_path / 'config.json')
                        # Copy any other existing files from this path
                        for filename in ['results.jsonl', 'stats.json']:
                            source_file = source_path / filename
                            if source_file.exists():
                                shutil.copy2(source_file, target_path / filename)
                        config_found = True
                        break
                
                if not config_found:
                    logger.warning(f"No config.json found in any of the source paths for {rel_path}. Removing target directory.")
                    shutil.rmtree(target_path)
                    continue
            
            logger.info(f"Merged {len(source_paths)} directories into {target_path}")

def main():
    source_dir = "experiments/results"
    target_dir = "experiments/results/MERGED_CLEAN"
    
    merger = ResultsMerger(source_dir, target_dir)
    merger.merge_paths()

if __name__ == "__main__":
    main()