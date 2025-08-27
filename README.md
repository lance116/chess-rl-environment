# Chess Neural Network

A Python-based chess game featuring an AI opponent powered by neural networks. This project combines traditional chess gameplay with modern machine learning techniques, providing an engaging chess experience with a computer opponent that learns and improves over time.

![Chess Game Screenshot](assets/images/screenshot.png)

## Features

- **Interactive Chess Gameplay**: Full-featured chess game with intuitive GUI
- **Neural Network AI**: AI opponent powered by TensorFlow neural networks
- **Visual Feedback**: Highlighted possible moves, check indicators, and smooth animations
- **Move Validation**: Complete chess rule validation including castling, en passant, and pawn promotion
- **Game Analysis**: Move history tracking and game state analysis
- **Model Training**: Continuous learning and improvement of the AI opponent
- **Cross-Platform**: Works on Windows, macOS, and Linux

## Requirements

- Python 3.8 or higher
- TensorFlow 2.12.0 or later
- Pygame for graphics
- python-chess library for chess logic
- NumPy for numerical computations

## Installation

### Quick Setup (Windows)

1. Clone the repository:
```bash
git clone https://github.com/lance116/Chess-Neural-Network.git
cd Chess-Neural-Network
```

2. Run the setup script:
```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_tensorflow.ps1
```

### Manual Installation

1. Create a virtual environment (recommended):
```bash
python -m venv chess_env
source chess_env/bin/activate  # On Windows: chess_env\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Game

After installation, start the chess game:

```bash
python src/chess_game.py
```

Or use the provided batch script:
```bash
scripts/run_chess.bat
```

### Game Controls

- **Mouse Click**: Select pieces and make moves
- **Drag & Drop**: Alternative method to move pieces
- **Forfeit Button**: Surrender the current game
- **ESC**: Exit the game

### Gameplay Features

- Click on a piece to see possible moves highlighted in blue
- Valid moves are automatically calculated based on chess rules
- The AI opponent will think for a few seconds before making its move
- Check situations are highlighted in red
- Captured pieces are displayed at the bottom of the screen

## Project Structure

```
Chess-Neural-Network/
├── src/                     # Source code
│   └── chess_game.py       # Main game file
├── assets/                 # Game assets
│   └── images/            # Chess piece images
├── models/                # Neural network models and training data
│   ├── chess_model.weights.h5     # Trained model weights
│   └── training_progress.json     # Training metrics
├── scripts/               # Utility scripts
│   ├── setup_tensorflow.ps1      # Setup script for Windows
│   └── run_chess.bat            # Game launcher
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── LICENSE               # Project license
```

## AI Architecture

The chess AI uses a neural network architecture designed for strategic gameplay:

- **Input Layer**: Board state representation (8x8x12 for piece positions)
- **Hidden Layers**: Multiple dense layers for position evaluation
- **Output Layer**: Move evaluation and selection
- **Training**: Self-play and position analysis for continuous improvement

## Development

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Style

- Follow PEP 8 Python style guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Include type hints where appropriate

### Testing

Run tests with:
```bash
python -m pytest tests/
```

## Technical Details

### Dependencies

- **TensorFlow**: Neural network framework for AI decision making
- **Pygame**: Graphics and user interface rendering
- **python-chess**: Chess game logic and move validation
- **NumPy**: Numerical computations and array operations

### Performance

- AI thinking time: ~5 seconds per move (configurable)
- Graphics: 30 FPS smooth rendering
- Memory usage: Optimized for efficient gameplay

## Troubleshooting

### Common Issues

1. **TensorFlow Import Error**: Run the setup script or manually install TensorFlow
2. **Graphics Issues**: Ensure Pygame is properly installed
3. **Performance Issues**: Check if GPU acceleration is available

### System Requirements

- **Minimum**: Python 3.8, 4GB RAM, 1GB disk space
- **Recommended**: Python 3.9+, 8GB RAM, GPU support for TensorFlow

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Chess piece images adapted from standard chess iconography
- TensorFlow team for the machine learning framework
- Pygame community for the graphics library
- python-chess library for robust chess logic

## Contact

Lance - [@lance116](https://github.com/lance116)

Project Link: [https://github.com/lance116/Chess-Neural-Network](https://github.com/lance116/Chess-Neural-Network)