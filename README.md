#Python Chess AI

This is a console-based chess game developed in Python where a human player competes against an AI opponent. 
The AI uses the Alpha-Beta Pruning optimization of the Minimax algorithm to efficiently determine the best possible moves.

-> Features

- Human vs AI gameplay
- Fully rule-compliant chess engine, including:
  - Castling
  - En passant
  - Pawn promotion
  - Check, checkmate, and draw detection
- AI decision-making with configurable search depth and thinking time
- Command-line interface for clean and distraction-free gameplay
- Move notation using source-to-destination format (e.g., `e2 e4`)

-> How It Works

The AI evaluates the game state using Minimax with Alpha-Beta pruning to avoid unnecessary evaluations and speed up decision-making.
The game prompts the user to set the AI’s maximum depth or time per move before starting. 
Based on these parameters, the AI searches for optimal plays within the defined constraints.
You can set the depth and time taken by AI to think within the code .(would need slight modifications in the Main)

-> Usage

1. Run the program:

  copy the code to an IDE that supports python and run the program.. Enjoy!
