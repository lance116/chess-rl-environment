"""
Main game controller for Chess Neural Network.
Handles game state, events, and animation.
"""
import pygame
import sys
import chess
from typing import Optional

try:
    from .config import (
        WIDTH, HEIGHT, FPS, ANIMATION_DURATION, AI_THINK_TIME,
        GAME_AREA_HEIGHT, CAPTURED_AREA_HEIGHT, PIECES, IMAGE_PATH,
        BLACK
    )
    from .rendering import (
        load_piece_images, to_internal_coords, to_display_coords,
        draw_game_state, draw_material_display, draw_promotion_choices,
        draw_game_over, draw_menu
    )
    from .ai import get_ai_move
    from .neural_network import load_model
except ImportError:
    from config import (
        WIDTH, HEIGHT, FPS, ANIMATION_DURATION, AI_THINK_TIME,
        GAME_AREA_HEIGHT, CAPTURED_AREA_HEIGHT, PIECES, IMAGE_PATH,
        BLACK
    )
    from rendering import (
        load_piece_images, to_internal_coords, to_display_coords,
        draw_game_state, draw_material_display, draw_promotion_choices,
        draw_game_over, draw_menu
    )
    from ai import get_ai_move
    from neural_network import load_model


class ChessGame:
    """Main chess game controller."""

    def __init__(self, use_neural_network: bool = False):
        """
        Initialize the chess game.

        Args:
            use_neural_network: Whether to use neural network for AI evaluation
        """
        pygame.init()
        pygame.mixer.init()

        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Chess Game")
        self.clock = pygame.time.Clock()

        # Fonts
        self.fonts = {
            'menu_title': pygame.font.SysFont(None, 70),
            'menu_option': pygame.font.SysFont(None, 50),
            'status': pygame.font.SysFont(None, 28),
            'captured': pygame.font.SysFont(None, 22),
            'game_over': pygame.font.SysFont(None, 60),
            'restart': pygame.font.SysFont(None, 30),
            'button': pygame.font.SysFont(None, 26),
            'promotion': pygame.font.SysFont(None, 40)
        }

        # Load piece images
        try:
            self.piece_images = load_piece_images(IMAGE_PATH, PIECES)
        except FileNotFoundError:
            print("Failed to load piece images. Exiting.")
            sys.exit(1)

        # Neural network setup
        self.use_neural_network = use_neural_network
        self.nn_model = None
        if use_neural_network:
            try:
                try:
                    from .config import MODEL_WEIGHTS_FILE
                except ImportError:
                    from config import MODEL_WEIGHTS_FILE
                self.nn_model = load_model(MODEL_WEIGHTS_FILE)
                if self.nn_model is not None:
                    print("Neural network evaluation enabled")
                else:
                    print("Neural network model is None, falling back to classical")
                    self.use_neural_network = False
            except Exception as e:
                print(f"Failed to load neural network: {e}")
                print("Falling back to classical evaluation")
                self.use_neural_network = False

        # Game state
        self.game_state = 'MENU'  # MENU, PLAYING, PROMOTION, GAME_OVER
        self.board: Optional[chess.Board] = None
        self.player_color: Optional[str] = None
        self.selected_square: Optional[int] = None
        self.game_over = False
        self.winner: Optional[str] = None
        self.status_message = ""

        # Animation state
        self.is_animating = False
        self.anim_piece_img: Optional[pygame.Surface] = None
        self.anim_start_pos_screen = (0, 0)
        self.anim_end_pos_screen = (0, 0)
        self.anim_current_pos_screen = (0, 0)
        self.anim_progress = 0
        self.pending_move_uci: Optional[str] = None
        self.anim_from_square: Optional[int] = None

        # Promotion state
        self.promotion_pos: Optional[tuple] = None
        self.promotion_color: Optional[str] = None
        self.promotion_choice_rects = []

        # UI elements
        self.forfeit_button_rect = pygame.Rect(
            WIDTH // 2 - 40,
            GAME_AREA_HEIGHT + 5,
            80,
            CAPTURED_AREA_HEIGHT - 10
        )

        self.running = True

    def reset_game(self):
        """Reset game to initial state."""
        self.board = chess.Board()
        self.selected_square = None
        self.game_over = False
        self.winner = None
        self.status_message = ""
        self.is_animating = False
        self.pending_move_uci = None
        self.promotion_pos = None
        self.promotion_color = None
        self.game_state = 'PLAYING'
        print("Game reset.")

    def start_animation(self, move: chess.Move, piece_str: str):
        """
        Start animating a move.

        Args:
            move: chess.Move to animate
            piece_str: Piece identifier (e.g., 'wP')
        """
        if piece_str not in self.piece_images:
            print(f"Error: Invalid piece string '{piece_str}'")
            return

        self.is_animating = True
        self.pending_move_uci = move.uci()
        self.anim_from_square = move.from_square

        start_r = 7 - (move.from_square // 8)
        start_c = move.from_square % 8
        end_r = 7 - (move.to_square // 8)
        end_c = move.to_square % 8

        self.anim_piece_img = self.piece_images[piece_str]
        self.anim_start_pos_screen = to_display_coords(start_r, start_c, self.player_color)
        self.anim_end_pos_screen = to_display_coords(end_r, end_c, self.player_color)
        self.anim_current_pos_screen = self.anim_start_pos_screen
        self.anim_progress = 0

    def update_animation(self):
        """Update animation state and apply move when complete."""
        self.anim_progress += 1
        t = min(1.0, self.anim_progress / ANIMATION_DURATION)

        start_x, start_y = self.anim_start_pos_screen
        end_x, end_y = self.anim_end_pos_screen
        current_x = start_x + (end_x - start_x) * t
        current_y = start_y + (end_y - start_y) * t
        self.anim_current_pos_screen = (current_x, current_y)

        if self.anim_progress >= ANIMATION_DURATION:
            self.is_animating = False
            move = chess.Move.from_uci(self.pending_move_uci)

            if move in self.board.legal_moves:
                self.board.push(move)
            else:
                print(f"Error: Animated move {self.pending_move_uci} became illegal")

            self.pending_move_uci = None
            self.anim_from_square = None

            # Check game state
            if self.board.is_checkmate():
                self.game_over = True
                self.winner = 'w' if self.board.turn == chess.BLACK else 'b'
                self.status_message = f"Checkmate! {'White' if self.winner == 'w' else 'Black'} wins!"
                self.game_state = 'GAME_OVER'
            elif self.board.is_stalemate() or self.board.is_insufficient_material() or \
                 self.board.is_seventyfive_moves() or self.board.is_fivefold_repetition():
                self.game_over = True
                self.winner = 'stalemate'
                self.status_message = "Draw!"
                self.game_state = 'GAME_OVER'
            elif self.board.is_check():
                self.status_message = f"{'White' if self.board.turn == chess.WHITE else 'Black'} is in Check!"
            else:
                self.status_message = ""

    def handle_player_move(self, clicked_square: int):
        """
        Handle player clicking a square.

        Args:
            clicked_square: Chess square index (0-63)
        """
        player_chess_color = chess.WHITE if self.player_color == 'w' else chess.BLACK

        if self.selected_square is not None:
            # Try to make a move
            move = chess.Move(self.selected_square, clicked_square)
            piece = self.board.piece_at(self.selected_square)

            # Check for pawn promotion
            needs_promotion = False
            if piece and piece.piece_type == chess.PAWN:
                last_rank = 7 if piece.color == chess.WHITE else 0
                if chess.square_rank(clicked_square) == last_rank:
                    needs_promotion = True

            if needs_promotion:
                # Check if move is valid before showing promotion UI
                if move in self.board.legal_moves or any(
                    m.from_square == self.selected_square and m.to_square == clicked_square
                    for m in self.board.legal_moves
                ):
                    clicked_row = 7 - (clicked_square // 8)
                    clicked_col = clicked_square % 8
                    self.promotion_pos = (clicked_row, clicked_col)
                    self.promotion_color = self.player_color
                    self.pending_move_uci = move.uci()
                    self.game_state = 'PROMOTION'
                    self.selected_square = None
                    self.status_message = "Choose promotion..."
                    return
                else:
                    # Not a valid move, try selecting new piece
                    clicked_piece = self.board.piece_at(clicked_square)
                    if clicked_piece and clicked_piece.color == player_chess_color:
                        self.selected_square = clicked_square
                    else:
                        self.selected_square = None
                    return

            # Regular move (not promotion)
            if move in self.board.legal_moves:
                piece_str = ('w' if piece.color == chess.WHITE else 'b') + piece.symbol().upper()
                self.start_animation(move, piece_str)
                self.selected_square = None
                self.status_message = ""
            else:
                # Invalid move - try selecting new piece
                clicked_piece = self.board.piece_at(clicked_square)
                if clicked_piece and clicked_piece.color == player_chess_color:
                    self.selected_square = clicked_square
                else:
                    self.selected_square = None

        else:
            # Select a piece
            clicked_piece = self.board.piece_at(clicked_square)
            if clicked_piece and clicked_piece.color == player_chess_color:
                self.selected_square = clicked_square

    def handle_promotion_choice(self, mouse_pos: tuple):
        """
        Handle player selecting promotion piece.

        Args:
            mouse_pos: (x, y) mouse position
        """
        for rect, piece_char in self.promotion_choice_rects:
            if rect.collidepoint(mouse_pos):
                # Map piece character to chess piece type
                piece_map = {'Q': chess.QUEEN, 'R': chess.ROOK,
                           'B': chess.BISHOP, 'N': chess.KNIGHT}
                promotion_piece_type = piece_map[piece_char]

                # Create move with promotion
                base_move = chess.Move.from_uci(self.pending_move_uci)
                move_with_promotion = chess.Move(
                    base_move.from_square,
                    base_move.to_square,
                    promotion=promotion_piece_type
                )

                if move_with_promotion in self.board.legal_moves:
                    self.board.push(move_with_promotion)
                    print(f"Player promoted to {piece_char}")
                else:
                    print(f"Warning: Promotion move not legal")

                self.game_state = 'PLAYING'
                self.promotion_pos = None
                self.promotion_color = None
                self.pending_move_uci = None

                # Check game state
                if self.board.is_checkmate():
                    self.game_over = True
                    self.winner = 'w' if self.board.turn == chess.BLACK else 'b'
                    self.status_message = f"Checkmate! {'White' if self.winner == 'w' else 'Black'} wins!"
                    self.game_state = 'GAME_OVER'
                elif self.board.is_stalemate() or self.board.is_insufficient_material():
                    self.game_over = True
                    self.winner = 'stalemate'
                    self.status_message = "Draw!"
                    self.game_state = 'GAME_OVER'
                elif self.board.is_check():
                    self.status_message = f"{'White' if self.board.turn == chess.WHITE else 'Black'} is in Check!"
                else:
                    self.status_message = ""
                break

    def handle_ai_turn(self):
        """Execute AI move."""
        if self.is_animating or self.game_over:
            return

        player_chess_color = chess.WHITE if self.player_color == 'w' else chess.BLACK
        ai_chess_color = chess.BLACK if self.player_color == 'w' else chess.WHITE

        if self.board.turn == ai_chess_color:
            self.status_message = f"AI ({'W' if ai_chess_color == chess.WHITE else 'B'}) thinking..."

            # Draw thinking message
            self.screen.fill(BLACK)
            draw_game_state(self.screen, self.board, self.piece_images,
                          None, self.player_color)
            draw_material_display(self.screen, self.board, self.player_color,
                                self.status_message, self.forfeit_button_rect, self.fonts)
            pygame.display.flip()

            # Get AI move
            ai_move = get_ai_move(self.board, AI_THINK_TIME, self.use_neural_network, self.nn_model)
            self.status_message = ""

            if ai_move and ai_move in self.board.legal_moves:
                moved_piece = self.board.piece_at(ai_move.from_square)
                piece_str = ('w' if moved_piece.color == chess.WHITE else 'b') + moved_piece.symbol().upper()
                self.start_animation(ai_move, piece_str)
            else:
                print("AI Error: No valid move found")
                if self.board.is_checkmate():
                    self.game_over = True
                    self.winner = self.player_color
                    self.status_message = f"Checkmate! {'White' if self.winner == 'w' else 'Black'} wins!"
                    self.game_state = 'GAME_OVER'
                elif self.board.is_stalemate() or self.board.is_insufficient_material():
                    self.game_over = True
                    self.winner = 'stalemate'
                    self.status_message = "Draw!"
                    self.game_state = 'GAME_OVER'

    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            # Menu state
            if self.game_state == 'MENU':
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_w:
                        self.player_color = 'w'
                        self.reset_game()
                    elif event.key == pygame.K_b:
                        self.player_color = 'b'
                        self.reset_game()

            # Promotion state
            elif self.game_state == 'PROMOTION':
                if event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_promotion_choice(event.pos)

            # Playing or game over state
            elif self.game_state in ['PLAYING', 'GAME_OVER']:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.game_state = 'MENU'
                        self.player_color = None
                        self.is_animating = False
                        self.board = None

                # Player's turn
                player_chess_color = chess.WHITE if self.player_color == 'w' else chess.BLACK
                if self.game_state == 'PLAYING' and not self.game_over and \
                   self.board.turn == player_chess_color and not self.is_animating:
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        mouse_x, mouse_y = event.pos

                        # Check forfeit button
                        if self.forfeit_button_rect.collidepoint(mouse_x, mouse_y):
                            self.game_over = True
                            self.winner = 'w' if self.player_color == 'b' else 'b'
                            self.status_message = f"{'White' if self.player_color == 'w' else 'Black'} forfeited."
                            self.game_state = 'GAME_OVER'

                        # Board click
                        elif mouse_y < GAME_AREA_HEIGHT:
                            coords = to_internal_coords(mouse_x, mouse_y, self.player_color)
                            if coords:
                                clicked_row, clicked_col = coords
                                clicked_square = chess.square(clicked_col, 7 - clicked_row)
                                self.handle_player_move(clicked_square)

    def update(self):
        """Update game state."""
        if self.is_animating:
            self.update_animation()
        elif self.game_state == 'PLAYING' and not self.game_over:
            self.handle_ai_turn()

    def draw(self):
        """Draw current frame."""
        self.screen.fill(BLACK)

        if self.game_state == 'MENU':
            draw_menu(self.screen, self.fonts)

        elif self.game_state in ['PLAYING', 'PROMOTION', 'GAME_OVER']:
            if self.board:
                animating_square = self.anim_from_square if self.is_animating else None
                draw_game_state(self.screen, self.board, self.piece_images,
                              self.selected_square, self.player_color, animating_square)

                # Draw animating piece
                if self.is_animating and self.anim_piece_img:
                    anim_rect = self.anim_piece_img.get_rect(topleft=self.anim_current_pos_screen)
                    self.screen.blit(self.anim_piece_img, anim_rect)

                draw_material_display(self.screen, self.board, self.player_color,
                                    self.status_message, self.forfeit_button_rect, self.fonts)

                if self.game_state == 'PROMOTION':
                    self.promotion_choice_rects = draw_promotion_choices(self.screen, self.fonts)

                if self.game_over or self.game_state == 'GAME_OVER':
                    if "forfeited" in self.status_message:
                        final_message = self.status_message
                    elif self.winner == 'w':
                        final_message = "Checkmate! White Wins!"
                    elif self.winner == 'b':
                        final_message = "Checkmate! Black Wins!"
                    elif self.winner == 'stalemate':
                        final_message = "Stalemate - Draw!"
                    else:
                        final_message = "Game Over"
                    draw_game_over(self.screen, final_message, self.fonts)

        pygame.display.flip()

    def run(self):
        """Main game loop."""
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(FPS)

        print("Exiting game.")
        pygame.quit()
