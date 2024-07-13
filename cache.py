from dataclasses import dataclass
import json

from data import Answer


@dataclass(frozen=True)
class Argument:
    debater_id: str
    question: str
    justify_letter: str
    justify_numeric: float
    proof_a: str
    proof_b: str

    # later add things like num rounds, best of N, opponent id, etc.


class Cache:   # map Argument -> prompt
    def __init__(self, path: str):
        self.path = path
        self.reload_cache()

    def reload_cache(self):
        with open(self.path, 'r') as f:
            self.cache = json.load(f)

    def save(self):
        with open(self.path, 'w') as f:
            json.dump(self.cache, f)

    def add_argument(self, argument: Argument, prompt: str, confidence: float):
        # if argument in self.cache:
        #     raise ValueError("Argument already in cache")
        self.cache[argument] = prompt, confidence

    def __getitem__(self, argument: Argument):
        return self.cache.get(argument, None)
    
    def __setitem__(self, argument: Argument, prompt: str, confidence: float):
        self.add_argument(argument, prompt, confidence)
        self.save()