import os
import sys
import importlib.util

# List all Python files in the current directory
print("Python files in current directory:")
for file in os.listdir():
    if file.endswith('.py'):
        print(f"- {file} (size: {os.path.getsize(file)} bytes)")

# Attempt to directly load the token_ledger_db.py file
file_path = os.path.join(os.getcwd(), "token_ledger_db.py")
if os.path.exists(file_path):
    print(f"\nFile exists at: {file_path}")
    # Check if Python can load it
    try:
        spec = importlib.util.spec_from_file_location("token_ledger_db", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print("Successfully loaded module!")
    except Exception as e:
        print(f"Error loading module: {e}")
else:
    print(f"\nFile does NOT exist at: {file_path}")

# Check all directories in the Python path
print("\nChecking all directories in sys.path:")
for path in sys.path:
    possible_file = os.path.join(path, "token_ledger_db.py")
    if os.path.exists(possible_file):
        print(f"Found at: {possible_file}")