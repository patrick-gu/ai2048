import torch


class VectorizedGame:
    def __init__(self, num_envs, device):
        self.num_envs = num_envs
        self.device = device
        self.board = torch.zeros((num_envs, 4, 4), dtype=torch.int32, device=device)
        self.reset()

    def set_states(self, env_indices, states):
        if len(env_indices) == 0:
            return
        self.board[env_indices] = states.to(self.device).to(self.board.dtype)

    def reset(self, env_indices=None):
        if env_indices is None:
            self.board.fill_(0)
            env_indices = torch.arange(self.num_envs, device=self.device)

        if len(env_indices) > 0:
            self.board[env_indices] = 0
            self.add_random_tile(env_indices)
            self.add_random_tile(env_indices)

    def add_random_tile(self, env_indices):
        # env_indices: (K,)
        if len(env_indices) == 0:
            return

        # Get boards for these envs
        boards = self.board[env_indices]  # (K, 4, 4)

        # Find empty spots
        flat_boards = boards.view(-1, 16)
        empty_mask = (flat_boards == 0).float()

        # Add small epsilon to avoid all-zero probs if board is full (though it shouldn't be if we check done)
        probs = empty_mask + 1e-6

        # Sample indices
        flat_indices = torch.multinomial(probs, 1).squeeze(-1)  # (K,)

        # Sample values (2 or 4)
        vals = (
            torch.bernoulli(torch.full((len(env_indices),), 0.1, device=self.device))
            * 2
            + 2
        ).int()

        # Set values
        # We use scatter to set the values in the flattened view
        # We need to construct the index for the whole batch view if we were doing it in place on self.board
        # But here we are working on a slice `boards`.

        # Create a mask for the specific positions
        update_mask = torch.nn.functional.one_hot(flat_indices, 16).bool()

        flat_boards[update_mask] = vals
        self.board[env_indices] = flat_boards.view(-1, 4, 4)

    def get_moves(self):
        """
        Returns a tensor of shape (N, 4, 4, 4) representing the next state for each of the 4 actions.
        """
        moves = []
        # 0: Right, 1: Up, 2: Left, 3: Down

        # Right: Rotate 2 times (or flip), move left, rotate back
        # Actually, let's define directions consistent with the original game
        # Original: 0 - Right, 1 - Up, 2 - Left, 3 - Down

        # Move Left (2)
        moves.append(self._move_left_batch(self.board))

        # Move Right (0) -> Rotate 2, Left, Rotate 2
        b_right = torch.rot90(self.board, 2, [1, 2])
        b_right = self._move_left_batch(b_right)
        moves.append(torch.rot90(b_right, -2, [1, 2]))

        # Move Up (1) -> Rotate 1 CCW (Left becomes Up?), No.
        # If we want to move Up, we want tiles to go to top.
        # If we rotate 1 time CCW (90 deg), Top becomes Left.
        # So Rotate 1 CCW, Move Left, Rotate 1 CW.
        b_up = torch.rot90(self.board, 1, [1, 2])
        b_up = self._move_left_batch(b_up)
        moves.append(torch.rot90(b_up, -1, [1, 2]))

        # Move Down (3) -> Rotate 1 CW (or 3 CCW), Move Left, Rotate 1 CCW
        b_down = torch.rot90(self.board, -1, [1, 2])
        b_down = self._move_left_batch(b_down)
        moves.append(torch.rot90(b_down, 1, [1, 2]))

        # Reorder to match 0, 1, 2, 3
        # current list is [Left, Right, Up, Down] -> [2, 0, 1, 3]
        # We want [Right, Up, Left, Down]

        res = torch.stack([moves[1], moves[2], moves[0], moves[3]], dim=1)
        return res

    def _shift_left(self, x):
        # x: (M, 4)
        mask = x != 0
        # stable sort descending puts True (non-zeros) first, preserving order
        _, indices = torch.sort(mask.int(), dim=1, descending=True, stable=True)
        return torch.gather(x, 1, indices)

    def _move_left_batch(self, board):
        # board: (N, 4, 4)
        x = board.reshape(-1, 4)
        x = self._shift_left(x)

        # Merge
        # Column 0-1
        c0 = (x[:, 0] == x[:, 1]) & (x[:, 0] != 0)
        x[c0, 0] *= 2
        x[c0, 1] = 0

        # Column 1-2
        c1 = (x[:, 1] == x[:, 2]) & (x[:, 1] != 0)
        x[c1, 1] *= 2
        x[c1, 2] = 0

        # Column 2-3
        c2 = (x[:, 2] == x[:, 3]) & (x[:, 2] != 0)
        x[c2, 2] *= 2
        x[c2, 3] = 0

        x = self._shift_left(x)
        return x.view(-1, 4, 4)

    def step(self, actions, all_next_states=None):
        # actions: (N,)

        # Get all possible next states
        if all_next_states is None:
            all_next_states = self.get_moves()  # (N, 4, 4, 4)

        # Select the one corresponding to the action
        # actions is (N,), we want (N, 4, 4)
        # gather requires same dims

        # Expand actions to (N, 1, 1)
        # We can use advanced indexing
        batch_indices = torch.arange(self.num_envs, device=self.device)
        next_states = all_next_states[batch_indices, actions]  # (N, 4, 4)

        # Check if the move was valid (state changed)
        # Actually, in 2048, if you try an invalid move, nothing happens, but you don't get a new tile?
        # Or is it illegal?
        # The original code checks `valid` and only allows valid moves.
        # If the policy outputs an invalid move, we should probably punish it or mask it out.
        # The training loop masks out invalid actions. So we assume `actions` are valid.
        # But if they are not (e.g. due to exploration), the state shouldn't change.

        # Let's check validity
        is_valid = (
            next_states.view(self.num_envs, -1) != self.board.view(self.num_envs, -1)
        ).any(dim=1)

        # Update board
        self.board = next_states

        # Add random tile only for valid moves
        valid_indices = torch.nonzero(is_valid).squeeze(-1)
        self.add_random_tile(valid_indices)

        return self.board.clone(), is_valid

    def get_valid_actions(self, all_next_states=None):
        # Returns (N, 4) boolean tensor
        if all_next_states is None:
            all_next_states = self.get_moves()
        # Check if next state != current state
        # current: (N, 4, 4) -> (N, 1, 4, 4)
        current = self.board.unsqueeze(1)

        diff = (all_next_states != current).view(self.num_envs, 4, -1).any(dim=2)
        return diff.float()  # 1.0 if valid, 0.0 if not

    def get_done(self):
        # Done if no valid actions

        # Optimization: If there are empty spots, it's not done.
        flat = self.board.view(self.num_envs, -1)
        has_empty = (flat == 0).any(dim=1)

        # If has_empty is True, done is False.
        # If has_empty is False, we need to check if any move is possible.

        # We only compute valid actions for boards that are full.
        # But to keep it vectorized and simple without dynamic branching,
        # we can just compute for all. The overhead of computing for all vs subset
        # might be similar due to parallelism, but let's see.

        # Actually, if we just return ~has_empty, we are saying "done if full".
        # But 2048 allows merges even if full.

        # So we MUST check merges if full.

        # Let's just run the check. The optimization is that we can skip the check
        # if we know it's not done.

        # But `get_valid_actions` computes for all.
        # If we want to optimize, we need `get_valid_actions` to support masking?
        # Or just accept that we compute for all.

        # However, we can use the fact that `has_empty` implies `valid_actions` has at least one true?
        # Not necessarily, but in 2048, if there is an empty spot, you can usually move?
        # No, you can have empty spots but be stuck? No, if you have empty spots,
        # you can move tiles towards them (unless they are blocked?).
        # Actually, if there is an empty spot, is it always possible to move?
        # If the board is:
        # 2 4 8 16
        # 32 64 128 256
        # 2 4 8 16
        # 0 0 0 0
        # You can move down.
        # It seems if there is an empty spot, you can always move *something* into it?
        # Yes, unless the empty spot is surrounded by walls? But 2048 is a grid.
        # If there is a 0, you can slide tiles into it.
        # So `has_empty` -> `not done` is a safe assumption for 2048.

        # So we only need to check `valid` if `~has_empty`.

        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # We can't easily conditionally execute kernels for a subset in PyTorch without masking overhead.
        # But we can avoid the expensive `get_valid_actions` call if ALL envs have empty spots.

        if has_empty.all():
            return dones

        # Otherwise, we have to compute.
        valid = self.get_valid_actions()
        is_stuck = valid.sum(dim=1) == 0

        return is_stuck & (~has_empty)
