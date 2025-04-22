import random
import time
import math

# Global variables for position history and killer moves
positionhistory = {}
killermoves = {}
timelimitexceeded = False  # New flag to track time limit

def resettimelimit():
    global timelimitexceeded
    timelimitexceeded = False

def initialboard():
    board = [
        ['bR', 'bN', 'bB', 'bQ', 'bK', 'bB', 'bN', 'bR'],
        ['bP'] * 8,
        ['*'] * 8,
        ['*'] * 8,
        ['*'] * 8,
        ['*'] * 8,
        ['wP'] * 8,
        ['wR', 'wN', 'wB', 'wQ', 'wK', 'wB', 'wN', 'wR']
    ]
    return board


def printboard(board):
    # Color codes
    WHITE = "\033[97m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"
    
    print('\n')
    for i in range(8):
        rownumber = 8 - i
        row = f"{YELLOW}{rownumber}{RESET}\t"
        for j in range(8):
            piece = board[i][j]
            if piece.startswith('w'):
                coloredpiece = f"{WHITE}{piece}{RESET}"
            elif piece.startswith('b'):
                coloredpiece = f"{RED}{piece}{RESET}"
            else:  # empty square
                coloredpiece = f"{GREEN}{piece}{RESET}"
            row += coloredpiece + "\t"
        print(row + '\n')
    
    # Colorize file letters
    files = "\t" + "\t".join(f"{YELLOW}{chr(97+i)}{RESET}" for i in range(8))
    print(files + "\n\n")


def parsemove(movestr):
    files = {'a': 0, 'b': 1, 'c': 2, 'd': 3,
             'e': 4, 'f': 5, 'g': 6, 'h': 7}
    try:
        src, dst = movestr.strip().split()
        srcrow, srccol = 8 - int(src[1]), files[src[0]]
        dstrow, dstcol = 8 - int(dst[1]), files[dst[0]]
        return (srcrow, srccol), (dstrow, dstcol)
    except:
        return None, None


def formatcoordinate(row, col):
    files = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h'}
    return f"{files[col]}{8-row}"


def Makemove(board, src, dst, promotionpiece=None):
    piece = board[src[0]][src[1]]
    board[dst[0]][dst[1]] = piece
    board[src[0]][src[1]] = '*'
    
    # Handle pawn promotion
    if promotionpiece and piece[1] == 'P':
        if (dst[0] == 0 and piece[0] == 'w') or (dst[0] == 7 and piece[0] == 'b'):
            board[dst[0]][dst[1]] = piece[0] + promotionpiece


def IsValidMove(board, src, dst, playercolor, enpassanttarget=None):
    piece = board[src[0]][src[1]]
    dstpiece = board[dst[0]][dst[1]]
    
    # Check if piece exists
    if piece == '*':
        return False, "No piece at source location."
    
    piecetype = piece[1]
    srcrow, srccol = src
    dstrow, dstcol = dst
    rowdiff = dstrow - srcrow
    coldiff = dstcol - srccol

    if not piece.startswith(playercolor):
        return False, "You can only move your own pieces."

    if dstpiece != '*' and dstpiece.startswith(playercolor):
        return False, "You can't capture your own piece."

    # Handle en passant separately
    if piecetype == 'P' and abs(coldiff) == 1 and dst == enpassanttarget:
        direction = -1 if playercolor == 'w' else 1
        if rowdiff == direction:  # Correct direction for the pawn
            return True, ""

    if piecetype == 'P':  # Pawn
        direction = -1 if playercolor == 'w' else 1
        startrow = 6 if playercolor == 'w' else 1

        # Normal pawn move
        if coldiff == 0:
            if dstpiece != '*':
                return False, "Pawns can't move forward to occupied squares."
            if rowdiff == direction:
                return True, ""
            if rowdiff == 2 * direction and srcrow == startrow and board[srcrow + direction][srccol] == '*':
                return True, ""
            return False, "Invalid pawn move."

        # Pawn capture
        elif abs(coldiff) == 1 and rowdiff == direction:
            # Normal capture
            if dstpiece != '*' and not dstpiece.startswith(playercolor):
                return True, ""
            # En passant capture was already checked above
            return False, "Pawns can only capture diagonally when there's a piece to capture."

    elif piecetype == 'R':
        if srcrow != dstrow and srccol != dstcol:
            return False, "Rook must move in straight lines."
        if not ClearPath(board, src, dst):
            return False, "Rook's path is blocked."
        return True, ""

    elif piecetype == 'N':
        if (abs(rowdiff), abs(coldiff)) in [(2, 1), (1, 2)]:
            return True, ""
        return False, "Invalid knight move."

    elif piecetype == 'B':
        if abs(rowdiff) != abs(coldiff):
            return False, "Bishop must move diagonally."
        if not ClearPath(board, src, dst):
            return False, "Bishop's path is blocked."
        return True, ""

    elif piecetype == 'Q':
        if srcrow == dstrow or srccol == dstcol or abs(rowdiff) == abs(coldiff):
            if not ClearPath(board, src, dst):
                return False, "Queen's path is blocked."
            return True, ""
        return False, "Invalid queen move."

    elif piecetype == 'K':
        # Normal king move
        if max(abs(rowdiff), abs(coldiff)) == 1:
            return True, ""
        # Castling
        if abs(coldiff) == 2 and rowdiff == 0 and srcrow in [0, 7]:
            valid, msg = CanCastle(board, src, dst, playercolor)
            return valid, msg
        return False, "King can only move 1 square."

    return False, "Invalid move."


def ClearPath(board, src, dst):
    rowdiff = dst[0] - src[0]
    coldiff = dst[1] - src[1]

    rowstep = 0 if rowdiff == 0 else (1 if rowdiff > 0 else -1)
    colstep = 0 if coldiff == 0 else (1 if coldiff > 0 else -1)

    r, c = src[0] + rowstep, src[1] + colstep
    while (r, c) != dst:
        if board[r][c] != '*':
            return False
        r += rowstep
        c += colstep
    return True


def IsInCheck(board, playercolor):
    kingpos = FindKing(board, playercolor)
    if not kingpos:
        return False  # Shouldn't happen in valid game

    opponentcolor = 'w' if playercolor == 'b' else 'b'
    return IsSquareUnderAttack(board, kingpos, opponentcolor)


def FindKing(board, playercolor):
    for row in range(8):
        for col in range(8):
            if board[row][col] == f'{playercolor}K':
                return (row, col)
    return None


def IsSquareUnderAttack(board, square, attackercolor):
    for row in range(8):
        for col in range(8):
            piece = board[row][col]
            if piece != '*' and piece.startswith(attackercolor):
                src = (row, col)
                dst = square
                valid, _ = IsValidMove(board, src, dst, attackercolor, None)
                if valid:
                    return True
    return False


def WouldMoveExposeKing(board, src, dst, playercolor):
    tempboard = [row[:] for row in board]
    movingpiece = tempboard[src[0]][src[1]]
    tempboard[dst[0]][dst[1]] = movingpiece
    tempboard[src[0]][src[1]] = '*'
    
    # Handle en passant capture in the temporary board
    if movingpiece[1] == 'P' and src[1] != dst[1] and tempboard[dst[0]][dst[1]] == movingpiece and board[dst[0]][dst[1]] == '*':
        # This might be an en passant, remove the captured pawn
        direction = -1 if playercolor == 'w' else 1
        capturedpawnrow = dst[0] - direction
        capturedpawncol = dst[1]
        if 0 <= capturedpawnrow < 8 and 0 <= capturedpawncol < 8:  # Safety check
            tempboard[capturedpawnrow][capturedpawncol] = '*'
    
    kingpos = FindKing(tempboard, playercolor)
    if not kingpos:
        return False  # Shouldn't happen in valid game
    
    opponentcolor = 'w' if playercolor == 'b' else 'b'
    return IsSquareUnderAttack(tempboard, kingpos, opponentcolor)


def CanCastle(board, src, dst, playercolor):
    row = src[0]
    kingcol = src[1]
    targetcol = dst[1]
    
    # Determine if it's kingside or queenside
    if targetcol > kingcol:  # Kingside
        rookcol = 7
        betweencols = [5, 6]
    else:  # Queenside
        rookcol = 0
        betweencols = [1, 2, 3]
    
    # Check if king and rook are in starting positions
    if board[row][kingcol] != f'{playercolor}K':
        return False, "King is not in the right position for castling."
    if board[row][rookcol] != f'{playercolor}R':
        return False, "Rook is not in the right position for castling."
    
    # Check if squares between are empty
    for col in betweencols:
        if board[row][col] != '*':
            return False, "There are pieces between the king and rook."
    
    # Check if king would move through or into check
    opponentcolor = 'b' if playercolor == 'w' else 'w'
    squarestocheck = [kingcol] + betweencols[:2]  # Only check king's path
    for col in squarestocheck:
        if IsSquareUnderAttack(board, (row, col), opponentcolor):
            return False, "The king cannot castle through or into check."
    
    return True, ""


def HandleCastling(board, src, dst, playercolor):
    row = src[0]
    kingcol = src[1]
    
    # Determine if it's kingside or queenside
    if dst[1] > kingcol:  # Kingside
        newkingcol = 6
        newrookcol = 5
        rookcol = 7
    else:  # Queenside
        newkingcol = 2
        newrookcol = 3
        rookcol = 0
    
    # Move the king
    board[row][newkingcol] = board[row][kingcol]
    board[row][kingcol] = '*'
    
    # Move the rook
    board[row][newrookcol] = board[row][rookcol]
    board[row][rookcol] = '*'


def GetEnpassantTarget(board, src, dst, playercolor):
    if src[0] < 0 or src[0] >= 8 or src[1] < 0 or src[1] >= 8:
        return None  # Invalid source
    
    piece = board[dst[0]][dst[1]]  # Use the piece at destination after the move
    
    if piece == '*' or len(piece) < 2:
        return None  # No piece or invalid piece format
    
    if piece[1] == 'P' and abs(src[0] - dst[0]) == 2:
        direction = -1 if playercolor == 'w' else 1
        return (dst[0] + direction, dst[1])
    
    return None


def HandleEnpassant(board, src, dst, playercolor):
    # Move the pawn to its new position
    Makemove(board, src, dst)
    
    # Remove the captured pawn
    direction = -1 if playercolor == 'w' else 1
    capturedpawnrow = dst[0] - direction
    capturedpawncol = dst[1]
    
    if 0 <= capturedpawnrow < 8 and 0 <= capturedpawncol < 8:  # Safety check
        board[capturedpawnrow][capturedpawncol] = '*'


def CankingEscape(board, kingpos, playercolor):
    row, col = kingpos
    opponentcolor = 'w' if playercolor == 'b' else 'b'
    
    # Check all 8 possible king moves
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue  # skip current position
            newrow, newcol = row + dr, col + dc
            if 0 <= newrow < 8 and 0 <= newcol < 8:
                # Check if square is occupied by own piece
                targetpiece = board[newrow][newcol]
                if targetpiece != '*' and targetpiece.startswith(playercolor):
                    continue
                
                # Check if move would expose king to attack
                tempboard = [r[:] for r in board]
                tempboard[newrow][newcol] = tempboard[row][col]
                tempboard[row][col] = '*'
                
                if not IsSquareUnderAttack(tempboard, (newrow, newcol), opponentcolor):
                    return True
    return False


def CanBlockOrCaptureCheck(board, playercolor, enpassanttarget):
    kingpos = FindKing(board, playercolor)
    if not kingpos:
        return False
    
    opponentcolor = 'w' if playercolor == 'b' else 'b'
    
    # Find all checking pieces
    checkingpieces = []
    for row in range(8):
        for col in range(8):
            piece = board[row][col]
            if piece != '*' and piece.startswith(opponentcolor):
                src = (row, col)
                valid, _ = IsValidMove(board, src, kingpos, opponentcolor, None)
                if valid:
                    checkingpieces.append((row, col))
    
    # If more than one checking piece, only king moves can get out of check
    if len(checkingpieces) > 1:
        return False
    
    # For single check, see if we can block or capture
    if not checkingpieces:
        return False  # No checking pieces found
        
    checkrow, checkcol = checkingpieces[0]
    checkingpiece = board[checkrow][checkcol]
    
    # 1. Can any piece capture the checking piece?
    for row in range(8):
        for col in range(8):
            piece = board[row][col]
            if piece != '*' and piece.startswith(playercolor):
                src = (row, col)
                dst = (checkrow, checkcol)
                valid, _ = IsValidMove(board, src, dst, playercolor, enpassanttarget)
                if valid and not WouldMoveExposeKing(board, src, dst, playercolor):
                    return True
    
    # 2. Can any piece block the attack (only works for sliding pieces)
    if checkingpiece[1] in ['Q', 'R', 'B']:
        path = GetPathBetween((checkrow, checkcol), kingpos)
        for (blockrow, blockcol) in path:
            for row in range(8):
                for col in range(8):
                    piece = board[row][col]
                    if piece != '*' and piece.startswith(playercolor):
                        src = (row, col)
                        dst = (blockrow, blockcol)
                        valid, _ = IsValidMove(board, src, dst, playercolor, enpassanttarget)
                        if valid and not WouldMoveExposeKing(board, src, dst, playercolor):
                            return True
    
    return False


def GetPathBetween(src, dst):
    path = []
    srcrow, srccol = src
    dstrow, dstcol = dst
    
    # Check if movement is along a rank, file, or diagonal
    rowdiff = dstrow - srcrow
    coldiff = dstcol - srccol
    
    if srcrow == dstrow:  # Horizontal
        step = 1 if dstcol > srccol else -1
        for col in range(srccol + step, dstcol, step):
            path.append((srcrow, col))
    elif srccol == dstcol:  # Vertical
        step = 1 if dstrow > srcrow else -1
        for row in range(srcrow + step, dstrow, step):
            path.append((row, srccol))
    elif abs(rowdiff) == abs(coldiff):  # Diagonal
        rowstep = 1 if dstrow > srcrow else -1
        colstep = 1 if dstcol > srccol else -1
        row, col = srcrow + rowstep, srccol + colstep
        while row != dstrow and col != dstcol:
            path.append((row, col))
            row += rowstep
            col += colstep
    
    return path


def HasLegalMoves(board, playercolor, enpassanttarget):
    # First check if king can move
    kingpos = FindKing(board, playercolor)
    if kingpos and CankingEscape(board, kingpos, playercolor):
        return True
    
    # Check all pieces to see if any can make a legal move
    for srcrow in range(8):
        for srccol in range(8):
            piece = board[srcrow][srccol]
            if piece != '*' and piece.startswith(playercolor):
                for dstrow in range(8):
                    for dstcol in range(8):
                        src = (srcrow, srccol)
                        dst = (dstrow, dstcol)
                        valid, _ = IsValidMove(board, src, dst, playercolor, enpassanttarget)
                        if valid:
                            if not WouldMoveExposeKing(board, src, dst, playercolor):
                                return True
    return False


def IsCheckMate(board, playercolor, enpassanttarget):
    if not IsInCheck(board, playercolor):
        return False
    
    kingpos = FindKing(board, playercolor)
    if not kingpos:
        return False
        
    if CankingEscape(board, kingpos, playercolor):
        return False
    
    if CanBlockOrCaptureCheck(board, playercolor, enpassanttarget):
        return False
    
    return True


def IsStalemate(board, playercolor, enpassanttarget):
    if IsInCheck(board, playercolor):
        return False
    return not HasLegalMoves(board, playercolor, enpassanttarget)


def HandlePromotion(board, dst, playercolor):
    promotionpiece = input(f"Promote pawn to (Q, R, B, N) for {playercolor}: ").upper()
    while promotionpiece not in ['Q', 'R', 'B', 'N']:
        promotionpiece = input("Invalid choice. Promote to (Q, R, B, N): ").upper()
    board[dst[0]][dst[1]] = playercolor + promotionpiece


def formatmoveforhistory(board, src, dst, pieceinfo, promotionpiece=None, iscastling=False, iscapture=False):
    if iscastling:
        if dst[1] > src[1]:  # Kingside
            return "O-O"
        else:  # Queenside
            return "O-O-O"
    
    piecetype = pieceinfo[1] if len(pieceinfo) > 1 else '?'
    srccoord = formatcoordinate(src[0], src[1])
    dstcoord = formatcoordinate(dst[0], dst[1])
    
    movestr = ""
    if piecetype != 'P':
        movestr += piecetype
    
    movestr += srccoord
    movestr += " x " if iscapture else " - "
    movestr += dstcoord
    
    if promotionpiece and piecetype == 'P':
        if (dst[0] == 0 and pieceinfo[0] == 'w') or (dst[0] == 7 and pieceinfo[0] == 'b'):
            movestr += f"={promotionpiece}"
    
    return movestr


def printmovehistory(movehistory):
    print("\nMove History:")
    print("White\t\tBlack")
    print("-" * 30)
    
    for i in range(0, len(movehistory), 2):
        movenum = i // 2 + 1
        whitemove = movehistory[i] if i < len(movehistory) else ""
        blackmove = movehistory[i+1] if i+1 < len(movehistory) else ""
        print(f"{movenum}. {whitemove:<15} {blackmove}")
    
    print()


def evaluateboard(board):
    global positionhistory
    # Convert board to a string key for hashing
    boardkey = ''.join(''.join(row) for row in board)
    positioncount = positionhistory.get(boardkey, 0)
    
    # Piece values with slightly better tuning
    piecevalues = {
        'P': 1.0, 'N': 3.2, 'B': 3.3, 'R': 5.0, 'Q': 9.0, 'K': 0.0,
        'p': -1.0, 'n': -3.2, 'b': -3.3, 'r': -5.0, 'q': -9.0, 'k': 0.0
    }
    
    material = 0
    positional = 0
    kingsafety = 0
    pawnstructure = 0
    mobility = 0
    centercontrol = 0
    
    # Check for checkmate first - highest priority
    if IsCheckMate(board, 'w', None):
        return 1000  # Black wins
    elif IsCheckMate(board, 'b', None):
        return -1000  # White wins
    
    # Check for stalemate
    if IsStalemate(board, 'w', None) or IsStalemate(board, 'b', None):
        return 0  # Draw
    
    # Count material and apply positional bonuses
    for row in range(8):
        for col in range(8):
            piece = board[row][col]
            if piece == '*':
                continue
                
            color = piece[0]
            piecetype = piece[1]
            
            # Material evaluation - positive for black pieces, negative for white
            if color == 'b':
                material += piecevalues.get(piecetype, 0)
            else:
                material -= piecevalues.get(piecetype, 0)
            
            # Positional bonuses
            if piecetype == 'P':
                # Passed pawns and advancement bonuses
                if color == 'b':
                    pawnstructure += 0.05 * (7 - row)  # Advance black pawns
                    # Passed pawn check
                    if ispassedpawn(board, row, col, 'b'):
                        pawnstructure += 0.5 + 0.1 * (7 - row)
                else:
                    pawnstructure -= 0.05 * row  # Advance white pawns
                    if ispassedpawn(board, row, col, 'w'):
                        pawnstructure -= 0.5 + 0.1 * row
                
            # Center control
            if row in [3, 4] and col in [3, 4]:
                value = 0.1
                if color == 'b':
                    centercontrol += value
                else:
                    centercontrol -= value
            
            # Knight positioning
            if piecetype == 'N':
                # Knights are better in the center
                distancefromcenter = abs(3.5 - row) + abs(3.5 - col)
                value = 0.05 * (4 - distancefromcenter)
                if color == 'b':
                    positional += value
                else:
                    positional -= value
            
            # Bishop positioning - open diagonals
            if piecetype == 'B':
                diagonalmobility = countdiagonalmobility(board, row, col)
                value = 0.03 * diagonalmobility
                if color == 'b':
                    positional += value
                else:
                    positional -= value
            
            # Rook positioning - open files
            if piecetype == 'R':
                if isopenfile(board, col, color):
                    value = 0.2
                    if color == 'b':
                        positional += value
                    else:
                        positional -= value
            
            # King safety
            if piecetype == 'K':
                if color == 'b':
                    kingsafety += evaluatekingsafety(board, row, col, 'b')
                else:
                    kingsafety -= evaluatekingsafety(board, row, col, 'w')
    
    # Check bonus
    if IsInCheck(board, 'w'):
        kingsafety += 0.5  # Bonus for checking the opponent
    if IsInCheck(board, 'b'):
        kingsafety -= 0.5  # Penalty for being in check
    
    # Mobility calculation
    bmobility = calculatemobility(board, 'b')
    wmobility = calculatemobility(board, 'w')
    mobility = 0.02 * (bmobility - wmobility)
    
    # Combine all factors with weights
    total = (
        material * 1.0 + 
        positional * 0.7 + 
        kingsafety * 1.2 + 
        pawnstructure * 0.8 + 
        mobility * 0.6 +
        centercontrol * 0.5
    )
    
    # Apply repetition penalty to discourage draws
    repetitionpenalty = -0.5 * positioncount
    
    # Small random factor to break ties and add variety
    randomfactor = random.uniform(-0.05, 0.05)
    
    return total + repetitionpenalty + randomfactor

def ispassedpawn(board, row, col, color):
    opponent = 'w' if color == 'b' else 'b'
    direction = 1 if color == 'w' else -1
    
    # Check files ahead of the pawn
    endrow = -1 if color == 'w' else 8
    for r in range(row, endrow, -direction):
        for c in range(max(0, col-1), min(8, col+2)):
            if board[r][c] == f'{opponent}P':
                return False
    return True

def isopenfile(board, col, color):
    for r in range(8):
        if board[r][col] in ['wP', 'bP']:
            return False
    return True

def countdiagonalmobility(board, row, col, maxdistance=3):
    count = 0
    piece = board[row][col]
    color = piece[0]
    
    # Check all four diagonal directions
    directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    for dr, dc in directions:
        for dist in range(1, maxdistance + 1):
            r, c = row + dr * dist, col + dc * dist
            if 0 <= r < 8 and 0 <= c < 8:
                target = board[r][c]
                if target == '*':
                    count += 1
                else:
                    if target[0] != color:  # Can capture opponent's piece
                        count += 1
                    break
            else:
                break
    return count

def calculatemobility(board, color):
    mobility = 0
    for row in range(8):
        for col in range(8):
            piece = board[row][col]
            if piece != '*' and piece[0] == color:
                mobility += countlegalmoves(board, row, col, color)
    return mobility

def countlegalmoves(board, row, col, color):
    count = 0
    for dstrow in range(8):
        for dstcol in range(8):
            src = (row, col)
            dst = (dstrow, dstcol)
            valid, _ = IsValidMove(board, src, dst, color, None)
            if valid and not WouldMoveExposeKing(board, src, dst, color):
                count += 1
    return count

def evaluatepawnstructure(board, row, col, color):
    score = 0
    opponentcolor = 'b' if color == 'w' else 'w'
    file = col
    
    # Doubled pawns
    for r in range(8):
        if r != row and board[r][col] == f'{color}P':
            score -= 0.3
    
    # Isolated pawns
    isolated = True
    for adjfile in [file-1, file+1]:
        if 0 <= adjfile < 8:
            for r in range(8):
                if board[r][adjfile] == f'{color}P':
                    isolated = False
                    break
    if isolated:
        score -= 0.5
    
    # Passed pawns
    passed = True
    direction = -1 if color == 'w' else 1
    for r in range(row + direction, 0 if color == 'w' else 7, direction):
        for adjfile in [file-1, file, file+1]:
            if 0 <= adjfile < 8 and board[r][adjfile] == f'{opponentcolor}P':
                passed = False
                break
        if not passed:
            break
    if passed:
        score += 0.7
    
    return score if color == 'w' else -score


def evaluatekingsafety(board, row, col, color):
    score = 0
    direction = 1 if color == 'w' else -1
    
    # Pawn shield
    for dc in [-1, 0, 1]:
        r, c = row + direction, col + dc
        if 0 <= r < 8 and 0 <= c < 8:
            if board[r][c] != f'{color}P':
                score -= 0.2
    
    # Open files near king
    for dc in [-1, 0, 1]:
        c = col + dc
        if 0 <= c < 8:
            openfile = True
            for r in range(8):
                if board[r][c] == f'{color}P':
                    openfile = False
                    break
            if openfile:
                score -= 0.3
    
    return score if color == 'w' else -score


def evaluatemobility(board, row, col, color):
    count = 0
    piece = board[row][col]
    if piece == '*':
        return 0
    for dstrow in range(8):
        for dstcol in range(8):
            src = (row, col)
            dst = (dstrow, dstcol)
            valid, _ = IsValidMove(board, src, dst, color, None)
            if valid:
                count += 1
    return count * 0.1 if color == 'w' else -count * 0.1


def getallpossiblemoves(board, playercolor, enpassanttarget, depth=0):
    global killermoves
    
    # First generate captures and promotions
    highprioritymoves = []
    regularmoves = []
    
    piecevalues = {'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 0}
    
    # Get killer moves for this depth
    depthkillers = killermoves.get(depth, [])
    
    for srcrow in range(8):
        for srccol in range(8):
            piece = board[srcrow][srccol]
            if piece != '*' and piece.startswith(playercolor):
                piecetype = piece[1]
                
                # Generate only plausible moves for the piece type
                possibledestinations = []
                
                if piecetype == 'P':
                    # Pawn moves: forward, captures, and promotions
                    direction = -1 if playercolor == 'w' else 1
                    startrow = 6 if playercolor == 'w' else 1
                    
                    # Forward moves
                    destrow = srcrow + direction
                    if 0 <= destrow < 8 and board[destrow][srccol] == '*':
                        possibledestinations.append((destrow, srccol))
                        # Two-square move from starting position
                        if srcrow == startrow and board[srcrow + 2*direction][srccol] == '*':
                            possibledestinations.append((srcrow + 2*direction, srccol))
                    
                    # Captures
                    for dc in [-1, 1]:
                        destcol = srccol + dc
                        if 0 <= destcol < 8:
                            # Regular capture
                            if 0 <= destrow < 8:
                                if board[destrow][destcol] != '*' and not board[destrow][destcol].startswith(playercolor):
                                    possibledestinations.append((destrow, destcol))
                            # En passant
                            if enpassanttarget and enpassanttarget == (destrow, destcol):
                                possibledestinations.append((destrow, destcol))
                
                elif piecetype == 'N':
                    # Knight moves - just check all 8 possible knight jumps
                    for dr, dc in [(2,1), (2,-1), (1,2), (1,-2), (-1,2), (-1,-2), (-2,1), (-2,-1)]:
                        destrow, destcol = srcrow + dr, srccol + dc
                        if 0 <= destrow < 8 and 0 <= destcol < 8:
                            target = board[destrow][destcol]
                            if target == '*' or not target.startswith(playercolor):
                                possibledestinations.append((destrow, destcol))
                
                elif piecetype == 'K':
                    # King moves - 8 surrounding squares plus castling
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            destrow, destcol = srcrow + dr, srccol + dc
                            if 0 <= destrow < 8 and 0 <= destcol < 8:
                                target = board[destrow][destcol]
                                if target == '*' or not target.startswith(playercolor):
                                    possibledestinations.append((destrow, destcol))
                    
                    # Castling - kingside
                    if srcrow in [0, 7] and srccol == 4:
                        cancastle, _ = CanCastle(board, (srcrow, srccol), (srcrow, srccol+2), playercolor)
                        if cancastle:
                            possibledestinations.append((srcrow, srccol+2))
                        
                        # Queenside
                        cancastle, _ = CanCastle(board, (srcrow, srccol), (srcrow, srccol-2), playercolor)
                        if cancastle:
                            possibledestinations.append((srcrow, srccol-2))
                
                else:  # Sliding pieces (Rook, Bishop, Queen)
                    directions = []
                    if piecetype in ['R', 'Q']:  # Rook and Queen move horizontally/vertically
                        directions.extend([(0, 1), (1, 0), (0, -1), (-1, 0)])
                    if piecetype in ['B', 'Q']:  # Bishop and Queen move diagonally
                        directions.extend([(1, 1), (1, -1), (-1, 1), (-1, -1)])
                    
                    for dr, dc in directions:
                        for distance in range(1, 8):  # Maximum distance is 7 squares
                            destrow, destcol = srcrow + dr * distance, srccol + dc * distance
                            if not (0 <= destrow < 8 and 0 <= destcol < 8):
                                break  # Off the board
                            target = board[destrow][destcol]
                            if target == '*':
                                possibledestinations.append((destrow, destcol))
                            elif not target.startswith(playercolor):
                                possibledestinations.append((destrow, destcol))
                                break  # Can't move beyond a capture
                            else:
                                break  # Can't move beyond own piece
                
                # For each possible destination, quickly check if it might be valid
                # and categorize as high-priority or regular
                for destrow, destcol in possibledestinations:
                    src = (srcrow, srccol)
                    dst = (destrow, destcol)
                    
                    # Check if this is a killer move
                    if (src, dst) in depthkillers:
                        highprioritymoves.append((src, dst))
                        continue
                    
                    # Check for captures (high priority)
                    dstpiece = board[destrow][destcol]
                    iscapture = dstpiece != '*' and not dstpiece.startswith(playercolor)
                    
                    # Check for promotions (high priority)
                    ispromotion = piecetype == 'P' and ((destrow == 0 and playercolor == 'w') or (destrow == 7 and playercolor == 'b'))
                    
                    # Prioritize central control in opening
                    centralmove = destrow in [3, 4] and destcol in [3, 4]
                    
                    # Prioritize checking moves
                    # (Skip full check calculation for performance)
                    
                    move = (src, dst)
                    
                    if iscapture or ispromotion:
                        # For captures, use MVV-LVA ordering
                        if iscapture:
                            victimvalue = piecevalues.get(dstpiece[1], 0)
                            attackervalue = piecevalues.get(piecetype, 0)
                            # High priority for good captures, lower for even or worse
                            if victimvalue > attackervalue:
                                highprioritymoves.append(move)
                            else:
                                regularmoves.append(move)
                        else:
                            highprioritymoves.append(move)
                    elif centralmove and piecetype in ['P', 'N', 'B']:
                        highprioritymoves.append(move)
                    else:
                        regularmoves.append(move)
    
    # Return moves, with high priority moves first
    return highprioritymoves + regularmoves


def evaluatemove(board, src, dst, playercolor):
    score = 0
    piece = board[src[0]][src[1]]
    dstpiece = board[dst[0]][dst[1]]
    
    # MVV-LVA for captures (Most Valuable Victim - Least Valuable Aggressor)
    if dstpiece != '*':
        piecevalues = {'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 0}
        victimvalue = piecevalues.get(dstpiece[1], 0)
        attackervalue = piecevalues.get(piece[1], 0)
        score += 100 * victimvalue - attackervalue  # Increased weight
    
    # Promotions are very valuable
    if piece[1] == 'P':
        if (dst[0] == 0 and playercolor == 'w') or (dst[0] == 7 and playercolor == 'b'):
            score += 900  # Almost as good as getting a queen
    
    # Check if move gives check (without full calculation)
    if piece[1] == 'Q' or piece[1] == 'R' or piece[1] == 'B':
        opponentcolor = 'w' if playercolor == 'b' else 'b'
        kingpos = FindKing(board, opponentcolor)
        if kingpos:
            # Simple check for direct attacks
            if piece[1] == 'Q' or piece[1] == 'R':
                if dst[0] == kingpos[0] or dst[1] == kingpos[1]:
                    score += 50
            if piece[1] == 'Q' or piece[1] == 'B':
                if abs(dst[0] - kingpos[0]) == abs(dst[1] - kingpos[1]):
                    score += 50
            if piece[1] == 'N':
                if (abs(dst[0] - kingpos[0]), abs(dst[1] - kingpos[1])) in [(1, 2), (2, 1)]:
                    score += 50
    
    # Central control for early-mid game
    centersquares = [(3,3), (3,4), (4,3), (4,4)]
    if dst in centersquares:
        score += 10
    
    # Favor developing pieces in opening
    if piece[1] in ['N', 'B'] and ((playercolor == 'w' and src[0] == 7) or (playercolor == 'b' and src[0] == 0)):
        score += 5
    
    return score


def quiescencesearch(board, alpha, beta, playercolor, enpassanttarget, starttime, maxtime, qdepth=0):
    global timelimitexceeded
    
    # Less frequent time checks - only every 1000 nodes
    if qdepth % 1000 == 0 and time.time() - starttime > maxtime:
        timelimitexceeded = True
        return evaluateboard(board)
    
    # Maximum depth for quiescence to prevent excessive searching
    if qdepth > 5:  # Limit quiescence depth
        return evaluateboard(board)
        
    standpat = evaluateboard(board)
    
    if playercolor == 'b':  # Maximizing player
        if standpat >= beta:
            return beta
        alpha = max(alpha, standpat)
    else:  # Minimizing player
        if standpat <= alpha:
            return alpha
        beta = min(beta, standpat)
    
    # Only consider captures and promotions - skip checks for efficiency
    captures = []
    for srcrow in range(8):
        for srccol in range(8):
            piece = board[srcrow][srccol]
            if piece != '*' and piece.startswith(playercolor):
                for dstrow in range(8):
                    for dstcol in range(8):
                        dst = (dstrow, dstcol)
                        # Only check captures and promotions
                        dstpiece = board[dstrow][dstcol]
                        iscapture = dstpiece != '*' and not dstpiece.startswith(playercolor)
                        ispromotion = piece[1] == 'P' and ((dst[0] == 0 and playercolor == 'w') or (dst[0] == 7 and playercolor == 'b'))
                        
                        if iscapture or ispromotion:
                            src = (srcrow, srccol)
                            valid, _ = IsValidMove(board, src, dst, playercolor, enpassanttarget)
                            if valid and not WouldMoveExposeKing(board, src, dst, playercolor):
                                # Calculate a simple MVV-LVA score for ordering
                                score = 0
                                if iscapture:
                                    piecevalues = {'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 0}
                                    victimvalue = piecevalues.get(dstpiece[1], 0)
                                    attackervalue = piecevalues.get(piece[1], 0)
                                    score = 10 * victimvalue - attackervalue
                                if ispromotion:
                                    score += 9  # Queen promotion value
                                captures.append((src, dst, score))
    
    # Sort captures by score (descending)
    captures.sort(key=lambda x: -x[2])
    
    # Only look at the most promising captures
    prunedcaptures = captures[:7]  # Limit to top 7 captures
    
    for src, dst, _ in prunedcaptures:
        tempboard = [row[:] for row in board]
        piece = tempboard[src[0]][src[1]]
        
        # Handle en passant capture
        if piece[1] == 'P' and src[1] != dst[1] and board[dst[0]][dst[1]] == '*' and dst == enpassanttarget:
            direction = -1 if playercolor == 'w' else 1
            tempboard[dst[0] - direction][dst[1]] = '*'
        
        # Make the move
        tempboard[dst[0]][dst[1]] = piece
        tempboard[src[0]][src[1]] = '*'
        
        # Handle promotion
        if piece[1] == 'P' and ((dst[0] == 0 and playercolor == 'w') or (dst[0] == 7 and playercolor == 'b')):
            tempboard[dst[0]][dst[1]] = playercolor + 'Q'  # Always promote to queen in quiescence
        
        opponentcolor = 'w' if playercolor == 'b' else 'b'
        score = -quiescencesearch(tempboard, -beta, -alpha, opponentcolor, None, starttime, maxtime, qdepth + 1)
        
        if timelimitexceeded:
            return alpha if playercolor == 'b' else beta
        
        if playercolor == 'b':  # Maximizing player
            if score >= beta:
                return beta
            alpha = max(alpha, score)
        else:  # Minimizing player
            if score <= alpha:
                return alpha
            beta = min(beta, score)
    
    return alpha if playercolor == 'b' else beta


def minimax(board, depth, alpha, beta, maximizingplayer, enpassanttarget, starttime, maxtime, nodes=0):
    global killermoves, timelimitexceeded
    
    # Check time less frequently - only every 1000 nodes
    nodes += 1
    if nodes % 1000 == 0 and time.time() - starttime > maxtime:
        timelimitexceeded = True
        return evaluateboard(board), None
    
    playercolor = 'b' if maximizingplayer else 'w'
    
    # Check for game-ending conditions
    if IsCheckMate(board, playercolor, enpassanttarget):
        return (-1000 - depth) if maximizingplayer else (1000 + depth), None
    elif IsStalemate(board, playercolor, enpassanttarget):
        return 0, None
    
    # Use quiescence search at leaf nodes
    if depth == 0:
        return quiescencesearch(board, alpha, beta, playercolor, enpassanttarget, starttime, maxtime), None
    
    moves = getallpossiblemoves(board, playercolor, enpassanttarget, depth)
    
    if not moves:
        return evaluateboard(board), None
    
    bestmove = None
    
    if maximizingplayer:
        maxeval = -math.inf
        for move in moves:
            src, dst = move
            
            # Skip invalid moves quickly
            if board[src[0]][src[1]] != ('b' + board[src[0]][src[1]][1]):  # Not a black piece
                continue
                
            valid, _ = IsValidMove(board, src, dst, playercolor, enpassanttarget)
            if not valid or WouldMoveExposeKing(board, src, dst, playercolor):
                continue
                
            tempboard = [row[:] for row in board]
            piece = tempboard[src[0]][src[1]]
            
            # Handle special moves
            newenpassant = None
            if piece[1] == 'P' and abs(src[0] - dst[0]) == 2:
                direction = -1 if playercolor == 'w' else 1
                newenpassant = (src[0] + direction, src[1])
            
            # Handle en passant capture
            if piece[1] == 'P' and src[1] != dst[1] and tempboard[dst[0]][dst[1]] == '*' and dst == enpassanttarget:
                direction = -1 if playercolor == 'w' else 1
                tempboard[dst[0] - direction][dst[1]] = '*'
            
            # Make the move
            tempboard[dst[0]][dst[1]] = piece
            tempboard[src[0]][src[1]] = '*'
            
            # Handle promotion
            if piece[1] == 'P' and ((dst[0] == 0 and playercolor == 'w') or (dst[0] == 7 and playercolor == 'b')):
                tempboard[dst[0]][dst[1]] = playercolor + 'Q'  # Always promote to queen
            
            # Recursive search
            evaluation, _ = minimax(tempboard, depth-1, alpha, beta, False, newenpassant, starttime, maxtime, nodes)
            
            if timelimitexceeded:
                return maxeval if maxeval != -math.inf else evaluateboard(board), bestmove
            
            if isinstance(evaluation, tuple):  # Handle potential tuple return
                evaluation = evaluation[0]
                
            if evaluation > maxeval:
                maxeval = evaluation
                bestmove = move
            
            alpha = max(alpha, evaluation)
            if beta <= alpha:
                # Store killer move for non-captures
                if tempboard[dst[0]][dst[1]] == '*':
                    if depth not in killermoves:
                        killermoves[depth] = []
                    if move not in killermoves[depth] and len(killermoves[depth]) < 2:
                        killermoves[depth].append(move)
                break
        
        return maxeval, bestmove
    else:
        mineval = math.inf
        for move in moves:
            src, dst = move
            
            # Skip invalid moves quickly
            if board[src[0]][src[1]] != ('w' + board[src[0]][src[1]][1]):  # Not a white piece
                continue
                
            valid, _ = IsValidMove(board, src, dst, playercolor, enpassanttarget)
            if not valid or WouldMoveExposeKing(board, src, dst, playercolor):
                continue
                
            tempboard = [row[:] for row in board]
            piece = tempboard[src[0]][src[1]]
            
            # Handle special moves
            newenpassant = None
            if piece[1] == 'P' and abs(src[0] - dst[0]) == 2:
                direction = -1 if playercolor == 'w' else 1
                newenpassant = (src[0] + direction, src[1])
            
            # Handle en passant capture
            if piece[1] == 'P' and src[1] != dst[1] and tempboard[dst[0]][dst[1]] == '*' and dst == enpassanttarget:
                direction = -1 if playercolor == 'w' else 1
                tempboard[dst[0] - direction][dst[1]] = '*'
            
            # Make the move
            tempboard[dst[0]][dst[1]] = piece
            tempboard[src[0]][src[1]] = '*'
            
            # Handle promotion
            if piece[1] == 'P' and ((dst[0] == 0 and playercolor == 'w') or (dst[0] == 7 and playercolor == 'b')):
                tempboard[dst[0]][dst[1]] = playercolor + 'Q'  # Always promote to queen
            
            # Recursive search
            evaluation, _ = minimax(tempboard, depth-1, alpha, beta, True, newenpassant, starttime, maxtime, nodes)
            
            if timelimitexceeded:
                return mineval if mineval != math.inf else evaluateboard(board), bestmove
            
            if isinstance(evaluation, tuple):  # Handle potential tuple return
                evaluation = evaluation[0]
                
            if evaluation < mineval:
                mineval = evaluation
                bestmove = move
            
            beta = min(beta, evaluation)
            if beta <= alpha:
                # Store killer move for non-captures
                if tempboard[dst[0]][dst[1]] == '*':
                    if depth not in killermoves:
                        killermoves[depth] = []
                    if move not in killermoves[depth] and len(killermoves[depth]) < 2:
                        killermoves[depth].append(move)
                break
        
        return mineval, bestmove

def getaimove(board, enpassanttarget, maxtime=15.0):
    global timelimitexceeded, positionhistory, killermoves
    print("\nAI is thinking...")
    
    starttime = time.time()
    resettimelimit()  # Reset time limit flag
    
    # Clear killer move history for a fresh search
    killermoves = {}
    
    bestmove = None
    maxdepth = 15  # Increased maximum search depth
    
    # Calculate buffer for time management
    timebuffer = 0.1 * maxtime
    adjustedmaxtime = maxtime - timebuffer
    
    # Find any valid moves first as a fallback
    validmoves = []
    for src, dst in getallpossiblemoves(board, 'b', enpassanttarget):
        valid, _ = IsValidMove(board, src, dst, 'b', enpassanttarget)
        if valid and not WouldMoveExposeKing(board, src, dst, 'b'):
            validmoves.append((src, dst))
    
    # Handle no moves or single move case
    if not validmoves:
        print("AI has no valid moves!")
        return None
    
    if len(validmoves) == 1:
        time.sleep(0.5)  # Quick thinking time
        return validmoves[0]
    
    # Look for checkmate in 1 first
    for src, dst in validmoves:
        tempboard = [row[:] for row in board]
        piece = tempboard[src[0]][src[1]]
        tempboard[dst[0]][dst[1]] = piece
        tempboard[src[0]][src[1]] = '*'
        if IsCheckMate(tempboard, 'w', None):
            print("Found instant checkmate!")
            return (src, dst)
    
    # Set up aspiration windows for search
    alpha = -math.inf
    beta = math.inf
    windowsize = 25  # Initial window size
    
    # Initialize stats tracking
    lastdepthcompleted = 0
    besteval = 0
    
    # Start iterative deepening
    for depth in range(1, maxdepth + 1):
        depthstarttime = time.time()
        
        # Use aspiration windows for deeper searches
        if depth >= 4:
            alpha = besteval - windowsize
            beta = besteval + windowsize
        
        # Run minimax search with current window
        evaluation, move = minimax(board, depth, alpha, beta, True, enpassanttarget, starttime, adjustedmaxtime)
        
        # Handle window failures and retry if needed
        if depth >= 4:
            # If evaluation falls outside window, retry with full window
            if (isinstance(evaluation, (int, float)) and 
                (evaluation <= alpha or evaluation >= beta)) and not timelimitexceeded:
                # Window failed, retry with full window
                print(f"Window fail at depth {depth}, retrying with full window")
                windowsize *= 2  # Double window size for next time
                alpha = -math.inf
                beta = math.inf
                evaluation, move = minimax(board, depth, alpha, beta, True, enpassanttarget, 
                                         starttime, adjustedmaxtime)
        
        # Process search results
        depthtime = time.time() - depthstarttime
        
        if timelimitexceeded:
            print(f"Time limit reached during depth {depth} search")
            break
            
        # Store the best move if valid
        if move:
            src, dst = move
            valid, _ = IsValidMove(board, src, dst, 'b', enpassanttarget)
            if valid and not WouldMoveExposeKing(board, src, dst, 'b'):
                bestmove = move
                lastdepthcompleted = depth
                if isinstance(evaluation, (int, float)):
                    besteval = evaluation
                    print(f"Depth {depth} completed in {depthtime:.2f}s, eval: {besteval:.2f}")
                    
                    # Check if we found a checkmate
                    tempboard = [row[:] for row in board]
                    tempboard[dst[0]][dst[1]] = tempboard[src[0]][src[1]]
                    tempboard[src[0]][src[1]] = '*'
                    if IsCheckMate(tempboard, 'w', None):
                        print("Found checkmate!")
                        break
                        
                    # Check if best move leads to losing material - be more cautious
                    if besteval < -3 and depth >= 3:
                        print("Warning: Current best move looks dangerous, searching deeper")
        
        # Decide whether to continue to next depth based on time
        elapsed = time.time() - starttime
        remaining = adjustedmaxtime - elapsed
        
        # Estimate time for next depth (typically 3-5x longer than current depth)
        estnextdepthtime = depthtime * 4  # Conservative estimate
        
        # Stop if we're likely to exceed time limit in next depth
        if depth >= 3 and remaining < estnextdepthtime:
            print(f"Stopping at depth {depth} to stay within time limit")
            break
            
        # For opening positions, don't waste time on very deep searches
        if depth >= 5 and elapsed > adjustedmaxtime * 0.6:
            print(f"Reached sufficient depth {depth}")
            break
    
    # Use minimum thinking time
    elapsed = time.time() - starttime
    mintime = min(2.0, maxtime * 0.2)  # At least 20% of max or 2 seconds
    if elapsed < mintime:
        time.sleep(mintime - elapsed)
    
    endtime = time.time()
    print(f"AI thought for {endtime - starttime:.2f}s, reached depth {lastdepthcompleted}")
    
    # Fallback if no move found
    if not bestmove:
        print("Using fallback move selection")
        # Try to choose a decent move
        validcaptures = [m for m in validmoves if board[m[1][0]][m[1][1]] != '*']
        if validcaptures:
            # Pick highest value capture
            bestcapture = max(validcaptures, key=lambda m: 
                             piecevalues.get(board[m[1][0]][m[1][1]][1], 0))
            bestmove = bestcapture
        else:
            # Just pick a valid move
            bestmove = random.choice(validmoves) if validmoves else None
    
    # Update position history for the chosen move
    if bestmove:
        src, dst = bestmove
        tempboard = [row[:] for row in board] 
        piece = tempboard[src[0]][src[1]]
        tempboard[dst[0]][dst[1]] = piece
        tempboard[src[0]][src[1]] = '*'
        boardkey = ''.join(''.join(row) for row in tempboard)
        positionhistory[boardkey] = positionhistory.get(boardkey, 0) + 1
    
    return bestmove


def main():
    global positionhistory, killermoves
    board = initialboard()
    printboard(board)

    turn = 'w'
    lastmove = None
    enpassanttarget = None
    gameover = False
    movehistory = []
    positionhistory = {}
    killermoves = {}

    while not gameover:
        boardkey = ''.join(''.join(row) for row in board)
        positionhistory[boardkey] = positionhistory.get(boardkey, 0) + 1
        
        print(f"{'White' if turn == 'w' else 'Black'}'s turn.")
        
        if turn == 'w':
            movestr = input("Enter your move (e.g., e2 e4): ").lower()
            
            iscastling = False
            if movestr == 'o-o':
                row = 7 if turn == 'w' else 0
                movestr = f"e{8-row} g{8-row}"
                iscastling = True
            elif movestr == 'o-o-o':
                row = 7 if turn == 'w' else 0
                movestr = f"e{8-row} c{8-row}"
                iscastling = True
            
            src, dst = parsemove(movestr)

            if src is None or dst is None:
                print("Invalid input! Format: e2 e4 (or o-o/o-o-o for castling)\n")
                continue

            if board[src[0]][src[1]] == '*':
                print("No piece at source location!\n")
                continue

            piece = board[src[0]][src[1]]
            
            if not piece.startswith(turn):
                print(f"That's not your piece! It's {'White' if turn == 'w' else 'Black'}'s turn.\n")
                continue
            
            pieceinfo = piece
                
            if piece[1] == 'K' and abs(src[1] - dst[1]) == 2:
                valid, msg = CanCastle(board, src, dst, turn)
                if not valid:
                    print(f"Cannot castle: {msg}\n")
                    continue
                HandleCastling(board, src, dst, turn)
                
                movenotation = "O-O" if dst[1] > src[1] else "O-O-O"
                movehistory.append(movenotation)
                
                printboard(board)
                printmovehistory(movehistory)
                
                lastmove = (src, dst)
                enpassanttarget = None
                
                opponentcolor = 'b' if turn == 'w' else 'w'
                if IsInCheck(board, opponentcolor):
                    print(f"Check! {'Black' if turn == 'w' else 'White'} is in check!")
                    
                turn = 'b' if turn == 'w' else 'w'
                continue

            isenpassantmove = False
            if piece[1] == 'P' and dst == enpassanttarget:
                isenpassantmove = True
                
            valid, msg = IsValidMove(board, src, dst, turn, enpassanttarget)
            if not valid:
                print("Invalid move:", msg, "\n")
                continue

            if WouldMoveExposeKing(board, src, dst, turn):
                print(f"Cannot move {board[src[0]][src[1]]} because it would leave the king in check!\n")
                continue

            iscapture = board[dst[0]][dst[1]] != '*' or isenpassantmove
            
            promotionpiece = None
            if isenpassantmove:
                HandleEnpassant(board, src, dst, turn)
            else:
                if piece[1] == 'P' and ((dst[0] == 0 and turn == 'w') or (dst[0] == 7 and turn == 'b')):
                    promotionpiece = input("Promote to (Q, R, B, N): ").upper()
                    while promotionpiece not in ['Q', 'R', 'B', 'N']:
                        promotionpiece = input("Invalid choice. Promote to (Q, R, B, N): ").upper()
                
                Makemove(board, src, dst, promotionpiece)
            
            movenotation = formatmoveforhistory(
                board, src, dst, 
                pieceinfo=pieceinfo,
                promotionpiece=promotionpiece, 
                iscastling=iscastling, 
                iscapture=iscapture
            )
        else:
            aimove = getaimove(board, enpassanttarget, maxtime=10.0)
            
            if not aimove:
                print("AI couldn't find a valid move!")
                gameover = True
                continue
                
            src, dst = aimove
            piece = board[src[0]][src[1]]
            pieceinfo = piece
            
            iscastling = False
            if piece[1] == 'K' and abs(src[1] - dst[1]) == 2:
                iscastling = True
                HandleCastling(board, src, dst, turn)
                
                movenotation = "O-O" if dst[1] > src[1] else "O-O-O"
                movehistory.append(movenotation)
                
                printboard(board)
                printmovehistory(movehistory)
                
                lastmove = (src, dst)
                enpassanttarget = None
                
                opponentcolor = 'b' if turn == 'w' else 'w'
                if IsInCheck(board, opponentcolor):
                    print(f"Check! {'Black' if turn == 'w' else 'White'} is in check!")
                    
                turn = 'b' if turn == 'w' else 'w'
                continue

            isenpassantmove = False
            if piece[1] == 'P' and dst == enpassanttarget:
                isenpassantmove = True
                HandleEnpassant(board, src, dst, turn)
            else:
                promotionpiece = None
                if piece[1] == 'P' and dst[0] == 0:
                    promotionpiece = 'Q'
                Makemove(board, src, dst, promotionpiece)
            
            iscapture = board[dst[0]][dst[1]] != '*' or isenpassantmove
            
            movenotation = formatmoveforhistory(
                board, src, dst, 
                pieceinfo=pieceinfo,
                promotionpiece=promotionpiece, 
                iscastling=iscastling, 
                iscapture=iscapture
            )
        
        movehistory.append(movenotation)
        
        printboard(board)
        printmovehistory(movehistory)

        lastmove = (src, dst)
        
        if piece[1] == 'P' and abs(src[0] - dst[0]) == 2:
            direction = -1 if turn == 'w' else 1
            enpassanttarget = (src[0] + direction, src[1])
        else:
            enpassanttarget = None

        opponentcolor = 'b' if turn == 'w' else 'w'
        if IsCheckMate(board, opponentcolor, enpassanttarget):
            print(f"Checkmate! {'White' if turn == 'w' else 'Black'} wins!")
            gameover = True
        elif IsStalemate(board, opponentcolor, enpassanttarget):
            print("Stalemate! Game is a draw.")
            gameover = True
        elif IsInCheck(board, opponentcolor):
            print(f"Check! {'Black' if turn == 'w' else 'White'} is in check!")

        turn = 'b' if turn == 'w' else 'w'

    positionhistory = {}

if __name__ == "__main__":
    main()