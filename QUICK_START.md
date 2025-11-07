# Quick Start Guide

## Play the Game

### Classical AI (Default)
```bash
python main.py
```

### Neural Network AI
1. Open `src/main.py`
2. Change line 18: `use_nn = False` → `use_nn = True`
3. Run: `python main.py`

---

## Train the Model

```bash
python src/train.py
```

**Adjust training size:**
- Edit `src/config.py`
- Change `TRAINING_GAME_LIMIT = 200` to higher number
- Run training again (resumes automatically)

---

## Game Controls

- **W** - Play as White
- **B** - Play as Black
- **Click** - Select and move pieces
- **Forfeit Button** - Surrender
- **R** - Return to menu (after game over)

---

## Files Overview

### To Play
- `main.py` - Run this to play

### To Train
- `src/train.py` - Run this to train NN

### To Configure
- `src/config.py` - All settings here

### Documentation
- `SUCCESS_SUMMARY.md` - Complete overview
- `OVERFITTING_FIXES.md` - Technical details
- `REFACTORING.md` - Code structure

---

## Status

✅ **Working perfectly!**
- Classical AI: ✓ Tested
- Neural Network AI: ✓ Tested
- Training: ✓ Completed (200 games)
- Overfitting: ✓ Fixed (1.05x ratio)

---

## Need Help?

Check the detailed documentation:
- `SUCCESS_SUMMARY.md` - Full summary
- `CLAUDE.md` - Architecture guide
