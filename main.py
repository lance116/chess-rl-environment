#!/usr/bin/env python3
"""
Chess Neural Network Game
Entry point for the chess game application.
"""

import subprocess
import sys
import os

def main():
    """Run the chess game."""
    chess_game_path = os.path.join(os.path.dirname(__file__), 'src', 'chess_game.py')
    
    if not os.path.exists(chess_game_path):
        print("Error: chess_game.py not found in src/ directory")
        return 1
    
    try:
        # Run the chess game as a subprocess
        result = subprocess.run([sys.executable, chess_game_path], cwd=os.path.dirname(__file__))
        return result.returncode
    except KeyboardInterrupt:
        print("\nGame interrupted by user")
        return 0
    except Exception as e:
        print(f"Error running chess game: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())