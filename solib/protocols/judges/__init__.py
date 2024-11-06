from .TipOfTongueJudge import TipOfTongueJudge  # noqa
from .JustAskProbabilityJudge import JustAskProbabilityJudge  # noqa

# import os
# import importlib

# # Get the current folder path
# current_folder = os.path.dirname(__file__)

# # Iterate through all .py files in the folder, excluding __init__.py
# for filename in os.listdir(current_folder):
#     if filename.endswith('.py') and filename != '__init__.py':
#         module_name = filename[:-3]  # Strip the .py extension

#         # Dynamically import the module
#         module = importlib.import_module(f'.{module_name}', package=__name__)

#         # Import the object with the same name as the module
#         try:
#             globals()[module_name] = getattr(module, module_name)
#         except AttributeError:
#             raise ImportError(f"Module '{module_name}' does not have an object named '{module_name}'")
