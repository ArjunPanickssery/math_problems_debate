from dataclasses import dataclass
import json


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
        try:
            with open(self.path, 'r') as f:
                self.cache = json.load(f)
        except FileNotFoundError:
            self.cache = {}

    def save(self):
        with open(self.path, 'w') as f:
            json.dump(self.cache, f, indent=4)

    def add_argument(self, argument: Argument, prompt: str):
        # if argument in self.cache:
        #     raise ValueError("Argument already in cache")
        self.cache[str(argument)] = prompt

    def __getitem__(self, argument: Argument):
        return self.cache.get(str(argument), None)
    
    def __setitem__(self, argument: Argument, prompt: str):
        self.add_argument(argument, prompt)
        self.save()