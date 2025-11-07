"""
Rendering and UI functions for the chess game.
"""
import pygame
import chess
from typing import Optional, Tuple, List

try:
    from .config import (
        WIDTH, HEIGHT, SQUARE_SIZE, BOARD_SIZE, GAME_AREA_HEIGHT, CAPTURED_AREA_HEIGHT,
        WHITE, BLACK, LIGHT_SQUARE, DARK_SQUARE, HIGHLIGHT_COLOR, POSSIBLE_MOVE_COLOR,
        CHECK_HIGHLIGHT_COLOR, MENU_BG_COLOR, MENU_TEXT_COLOR, CAPTURED_BG_COLOR,
        CAPTURED_TEXT_COLOR, FORFEIT_BUTTON_COLOR, FORFEIT_TEXT_COLOR,
        PROMOTION_BG_COLOR, PROMOTION_BORDER_COLOR, PROMOTION_CHOICE_COLOR, PROMOTION_TEXT_COLOR
    )
except ImportError:
    from config import (
        WIDTH, HEIGHT, SQUARE_SIZE, BOARD_SIZE, GAME_AREA_HEIGHT, CAPTURED_AREA_HEIGHT,
        WHITE, BLACK, LIGHT_SQUARE, DARK_SQUARE, HIGHLIGHT_COLOR, POSSIBLE_MOVE_COLOR,
        CHECK_HIGHLIGHT_COLOR, MENU_BG_COLOR, MENU_TEXT_COLOR, CAPTURED_BG_COLOR,
        CAPTURED_TEXT_COLOR, FORFEIT_BUTTON_COLOR, FORFEIT_TEXT_COLOR,
        PROMOTION_BG_COLOR, PROMOTION_BORDER_COLOR, PROMOTION_CHOICE_COLOR, PROMOTION_TEXT_COLOR
    )


def load_piece_images(image_path: str, piece_set: set) -> dict:
    """
    Load chess piece images.

    Args:
        image_path: Path to images directory
        piece_set: Set of piece identifiers (e.g., 'wP', 'bK')

    Returns:
        Dictionary mapping piece identifiers to pygame.Surface objects
    """
    import os
    piece_images = {}
    missing_files = []

    for piece in piece_set:
        path = os.path.join(image_path, f"{piece}.png")
        try:
            img = pygame.image.load(path).convert_alpha()
            piece_images[piece] = pygame.transform.scale(img, (SQUARE_SIZE, SQUARE_SIZE))
        except (pygame.error, FileNotFoundError) as e:
            missing_files.append(path)

    if missing_files:
        print("Error: Missing piece image files:")
        for f in missing_files:
            print(f"  - {f}")
        raise FileNotFoundError("Required piece images not found")

    return piece_images


def to_display_coords(row: int, col: int, player_color: str) -> Tuple[int, int]:
    """
    Convert internal board coordinates to screen coordinates.

    Args:
        row: Board row (0-7)
        col: Board column (0-7)
        player_color: 'w' or 'b'

    Returns:
        (screen_x, screen_y) tuple
    """
    if player_color == 'b':
        display_r = BOARD_SIZE - 1 - row
    else:
        display_r = row

    return (col * SQUARE_SIZE, display_r * SQUARE_SIZE)


def to_internal_coords(screen_x: int, screen_y: int, player_color: str) -> Optional[Tuple[int, int]]:
    """
    Convert screen coordinates to internal board coordinates.

    Args:
        screen_x: Screen x coordinate
        screen_y: Screen y coordinate
        player_color: 'w' or 'b'

    Returns:
        (row, col) tuple or None if out of bounds
    """
    display_c = screen_x // SQUARE_SIZE
    display_r = screen_y // SQUARE_SIZE

    if not (0 <= display_c < BOARD_SIZE and 0 <= display_r < BOARD_SIZE):
        return None

    if player_color == 'b':
        return (BOARD_SIZE - 1 - display_r, display_c)
    else:
        return (display_r, display_c)


def draw_board(surface: pygame.Surface, player_color: str):
    """Draw the chess board squares."""
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            screen_x, screen_y = to_display_coords(r, c, player_color)
            color = LIGHT_SQUARE if (r + c) % 2 == 0 else DARK_SQUARE
            pygame.draw.rect(surface, color, (screen_x, screen_y, SQUARE_SIZE, SQUARE_SIZE))


def draw_pieces(surface: pygame.Surface, board: chess.Board, piece_images: dict,
               player_color: str, animating_square: Optional[int] = None):
    """
    Draw chess pieces on the board.

    Args:
        surface: Pygame surface to draw on
        board: python-chess Board object
        piece_images: Dictionary of piece images
        player_color: 'w' or 'b'
        animating_square: Square index of piece being animated (don't draw it)
    """
    for square in chess.SQUARES:
        if square == animating_square:
            continue

        piece = board.piece_at(square)
        if piece:
            r = 7 - (square // 8)
            c = square % 8
            piece_key = ('w' if piece.color == chess.WHITE else 'b') + piece.symbol().upper()

            if piece_key in piece_images:
                screen_x, screen_y = to_display_coords(r, c, player_color)
                img = piece_images[piece_key]
                img_rect = img.get_rect(topleft=(screen_x, screen_y))
                surface.blit(img, img_rect)


def highlight_square(surface: pygame.Surface, row: int, col: int,
                    color: Tuple[int, int, int, int], player_color: str):
    """Highlight a square with a transparent color overlay."""
    screen_x, screen_y = to_display_coords(row, col, player_color)
    highlight_surf = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
    highlight_surf.fill(color)
    surface.blit(highlight_surf, (screen_x, screen_y))


def draw_game_state(surface: pygame.Surface, board: chess.Board, piece_images: dict,
                   selected_square: Optional[int], player_color: str,
                   animating_square: Optional[int] = None):
    """
    Draw the complete game state.

    Args:
        surface: Pygame surface
        board: python-chess Board object
        piece_images: Dictionary of piece images
        selected_square: Currently selected square (chess square index)
        player_color: 'w' or 'b'
        animating_square: Square being animated (don't draw piece)
    """
    draw_board(surface, player_color)

    # Highlight king in check
    if board.is_check():
        king_square = board.king(board.turn)
        if king_square is not None:
            king_r = 7 - (king_square // 8)
            king_c = king_square % 8
            highlight_square(surface, king_r, king_c, CHECK_HIGHLIGHT_COLOR, player_color)

    # Highlight selected square and legal moves
    if selected_square is not None and animating_square is None:
        sel_r = 7 - (selected_square // 8)
        sel_c = selected_square % 8
        highlight_square(surface, sel_r, sel_c, HIGHLIGHT_COLOR, player_color)

        # Highlight legal move destinations
        for move in board.legal_moves:
            if move.from_square == selected_square:
                to_r = 7 - (move.to_square // 8)
                to_c = move.to_square % 8
                highlight_square(surface, to_r, to_c, POSSIBLE_MOVE_COLOR, player_color)

    draw_pieces(surface, board, piece_images, player_color, animating_square)


def draw_material_display(surface: pygame.Surface, board: chess.Board,
                         player_color: str, status_message: str,
                         forfeit_button_rect: pygame.Rect, fonts: dict):
    """
    Draw material score and forfeit button in bottom area.

    Args:
        surface: Pygame surface
        board: python-chess Board object
        player_color: 'w' or 'b'
        status_message: Status text to display
        forfeit_button_rect: Rect for forfeit button
        fonts: Dictionary of pygame fonts
    """
    try:
        from .ai import get_material_score
    except ImportError:
        from ai import get_material_score

    area_y = GAME_AREA_HEIGHT
    pygame.draw.rect(surface, CAPTURED_BG_COLOR, (0, area_y, WIDTH, CAPTURED_AREA_HEIGHT))

    text_y_center = area_y + CAPTURED_AREA_HEIGHT // 2

    # Show AI thinking message
    if "thinking" in status_message.lower():
        think_surf = fonts['status'].render(status_message, True, CAPTURED_TEXT_COLOR)
        think_rect = think_surf.get_rect(center=(WIDTH // 2, text_y_center))
        surface.blit(think_surf, think_rect)
        return

    if board is None:
        return

    # Material scores
    player_chess_color = chess.WHITE if player_color == 'w' else chess.BLACK
    ai_chess_color = chess.BLACK if player_color == 'w' else chess.WHITE

    player_score = get_material_score(board, player_chess_color)
    ai_score = get_material_score(board, ai_chess_color)

    player_label = f"Player ({player_color.upper()}): {player_score}"
    ai_label = f"AI ({'W' if ai_chess_color == chess.WHITE else 'B'}): {ai_score}"

    player_surf = fonts['captured'].render(player_label, True, CAPTURED_TEXT_COLOR)
    player_rect = player_surf.get_rect(midleft=(10, text_y_center))
    surface.blit(player_surf, player_rect)

    ai_surf = fonts['captured'].render(ai_label, True, CAPTURED_TEXT_COLOR)
    ai_rect = ai_surf.get_rect(right=(WIDTH - 10 - forfeit_button_rect.width - 10),
                                centery=text_y_center)
    surface.blit(ai_surf, ai_rect)

    # Forfeit button
    pygame.draw.rect(surface, FORFEIT_BUTTON_COLOR, forfeit_button_rect, border_radius=5)
    forfeit_text = fonts['button'].render("Forfeit", True, FORFEIT_TEXT_COLOR)
    forfeit_text_rect = forfeit_text.get_rect(center=forfeit_button_rect.center)
    surface.blit(forfeit_text, forfeit_text_rect)


def draw_promotion_choices(surface: pygame.Surface, fonts: dict) -> List[Tuple[pygame.Rect, str]]:
    """
    Draw promotion piece selection UI.

    Args:
        surface: Pygame surface
        fonts: Dictionary of pygame fonts

    Returns:
        List of (rect, piece_char) tuples for click detection
    """
    overlay = pygame.Surface((WIDTH, GAME_AREA_HEIGHT), pygame.SRCALPHA)
    overlay.fill(PROMOTION_BG_COLOR)
    surface.blit(overlay, (0, 0))

    text_surf = fonts['status'].render("Choose promotion:", True, PROMOTION_TEXT_COLOR)
    text_rect = text_surf.get_rect(center=(WIDTH // 2, GAME_AREA_HEIGHT // 2 - 60))
    surface.blit(text_surf, text_rect)

    promotion_choices = ['Q', 'R', 'B', 'N']
    choice_size = SQUARE_SIZE
    total_width = len(promotion_choices) * choice_size + (len(promotion_choices) - 1) * 10
    start_x = (WIDTH - total_width) // 2
    start_y = GAME_AREA_HEIGHT // 2 - choice_size // 2

    choice_rects = []
    for i, piece_char in enumerate(promotion_choices):
        rect = pygame.Rect(start_x + i * (choice_size + 10), start_y, choice_size, choice_size)
        pygame.draw.rect(surface, PROMOTION_CHOICE_COLOR, rect, border_radius=5)
        pygame.draw.rect(surface, PROMOTION_BORDER_COLOR, rect, 2, border_radius=5)

        piece_surf = fonts['promotion'].render(piece_char, True, PROMOTION_TEXT_COLOR)
        piece_rect = piece_surf.get_rect(center=rect.center)
        surface.blit(piece_surf, piece_rect)
        choice_rects.append((rect, piece_char))

    return choice_rects


def draw_game_over(surface: pygame.Surface, message: str, fonts: dict):
    """
    Draw game over overlay.

    Args:
        surface: Pygame surface
        message: Game over message
        fonts: Dictionary of pygame fonts
    """
    overlay = pygame.Surface((WIDTH, GAME_AREA_HEIGHT), pygame.SRCALPHA)
    overlay.fill((50, 50, 50, 180))
    surface.blit(overlay, (0, 0))

    text_surface = fonts['game_over'].render(message, True, (200, 0, 0))
    text_rect = text_surface.get_rect(center=(WIDTH // 2, GAME_AREA_HEIGHT // 2 - 20))
    surface.blit(text_surface, text_rect)

    restart_text_surf = fonts['restart'].render("Press 'R' to return to Menu", True, WHITE)
    restart_rect = restart_text_surf.get_rect(center=(WIDTH // 2, GAME_AREA_HEIGHT // 2 + 30))
    surface.blit(restart_text_surf, restart_rect)


def draw_menu(surface: pygame.Surface, fonts: dict):
    """
    Draw main menu.

    Args:
        surface: Pygame surface
        fonts: Dictionary of pygame fonts
    """
    surface.fill(MENU_BG_COLOR)

    title_surf = fonts['menu_title'].render("Chess Game", True, MENU_TEXT_COLOR)
    title_rect = title_surf.get_rect(center=(WIDTH // 2, HEIGHT // 4))
    surface.blit(title_surf, title_rect)

    option_w_surf = fonts['menu_option'].render("Play as White (Press W)", True, MENU_TEXT_COLOR)
    option_w_rect = option_w_surf.get_rect(center=(WIDTH // 2, HEIGHT // 2))
    surface.blit(option_w_surf, option_w_rect)

    option_b_surf = fonts['menu_option'].render("Play as Black (Press B)", True, MENU_TEXT_COLOR)
    option_b_rect = option_b_surf.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 60))
    surface.blit(option_b_surf, option_b_rect)
