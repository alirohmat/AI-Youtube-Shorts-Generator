class VirtualCamera:
    def __init__(self, frame_width, frame_height, deadzone_percent=0.15,
                 alpha_min=0.02, alpha_max=0.25,
                 score_margin=0.2, lock_frames=30):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.deadzone_x = frame_width * deadzone_percent
        self.deadzone_y = frame_height * deadzone_percent
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.score_margin = score_margin
        self.lock_frames = lock_frames

        self.x = frame_width / 2.0
        self.y = frame_height / 2.0

        self.locked_id = None
        self.locked_score = 0.0
        self.target_x = frame_width / 2.0
        self.target_y = frame_height / 2.0
        self.pending_id = None
        self.pending_count = 0

    def update(self, subject):
        sid, score, tx, ty = subject['id'], subject['score'], subject['x'], subject['y']

        if self.locked_id is None:
            self.locked_id = sid
            self.locked_score = score
            self.target_x, self.target_y = tx, ty

        elif sid != self.locked_id:
            if score > self.locked_score * (1 + self.score_margin):
                if self.pending_id != sid:
                    self.pending_id = sid
                    self.pending_count = 0
                self.pending_count += 1
                if self.pending_count >= self.lock_frames:
                    self.locked_id = sid
                    self.locked_score = score
                    self.target_x, self.target_y = tx, ty
                    self.pending_id = None
                    self.pending_count = 0
            else:
                self.pending_id = None
                self.pending_count = 0

        else:
            self.locked_score = score
            self.target_x, self.target_y = tx, ty
            self.pending_id = None
            self.pending_count = 0

        if abs(self.target_x - self.x) <= self.deadzone_x and abs(self.target_y - self.y) <= self.deadzone_y:
            return self.x, self.y

        dist_x = abs(self.target_x - self.x)
        dist_y = abs(self.target_y - self.y)
        ax = self.alpha_min + (self.alpha_max - self.alpha_min) * min(dist_x / self.frame_width, 1.0)
        self.x = ax * self.target_x + (1 - ax) * self.x
        ay = self.alpha_min + (self.alpha_max - self.alpha_min) * min(dist_y / self.frame_height, 1.0)
        self.y = ay * self.target_y + (1 - ay) * self.y

        return self.x, self.y
