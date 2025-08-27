# Chess Neural Network

A Python chess game with an AI opponent powered by TensorFlow neural networks. Features interactive gameplay, move validation, and a learning AI that improves over time.

## Features

- Interactive chess GUI with move highlighting and animations
- Neural network AI opponent with strategic gameplay
- Complete chess rule validation (castling, en passant, promotion)
- Real-time game analysis and move history
- Cross-platform support (Windows, macOS, Linux)

## Quick Start

1. **Clone and install:**
```bash
git clone https://github.com/lance116/Chess-Neural-Network.git
cd Chess-Neural-Network
pip install -r requirements.txt
```

2. **Run the game:**
```bash
python src/chess_game.py
```

## Requirements

- Python 3.8+
- TensorFlow 2.12.0+
- Pygame
- python-chess
- NumPy

## Game Controls

- **Click** to select pieces and make moves
- **Forfeit Button** to surrender
- **R** to return to menu during game over
- **W/B** to choose white/black in menu

## Project Structure

```
Chess-Neural-Network/
├── src/chess_game.py        # Main game file
├── assets/images/           # Chess piece images
├── models/                  # AI model weights and training data
├── scripts/                 # Setup and run scripts
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## AI Architecture

- **Input**: 8×8×12 board state representation
- **Processing**: Dense neural network layers for position evaluation
- **Output**: Move scoring and selection
- **Training**: Self-play analysis for continuous improvement

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contact

Lance - [@lance116](https://github.com/lance116)