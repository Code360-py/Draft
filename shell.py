import os
import subprocess
import readline  # enables arrow key history

while True:
    try:
        user_input = input(" $ ").strip()
        if not user_input:
            continue
        if user_input.lower() == "exit":
            break
        os.system(user_input)
    except EOFError:
        print()
        break
