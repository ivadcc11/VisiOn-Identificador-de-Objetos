from time import monotonic


class VehicleRadar:
    def __init__(self):
        self.tracks = {}

    def update(
        self,
        track_id,
        center_x,
        left_line,
        right_line,
        distance_m,
        timestamp=None,
    ):
        if left_line >= right_line:
            raise ValueError("A linha esquerda deve vir antes da linha direita.")
        if distance_m <= 0:
            raise ValueError("A distância entre as linhas deve ser positiva.")

        now = monotonic() if timestamp is None else timestamp
        expired_tracks = [
            identifier
            for identifier, track in self.tracks.items()
            if now - track["last_seen"] > 10
        ]
        for identifier in expired_tracks:
            del self.tracks[identifier]

        track = self.tracks.get(track_id)
        if track is None:
            self.tracks[track_id] = {
                "center_x": center_x,
                "timestamp": now,
                "last_seen": now,
                "first_line": None,
                "first_crossing": None,
                "speed_kmh": None,
            }
            return None

        previous_x = track["center_x"]
        previous_time = track["timestamp"]
        crossed_lines = []

        if center_x > previous_x:
            line_order = (("left", left_line), ("right", right_line))
            for side, line_x in line_order:
                if previous_x < line_x <= center_x:
                    crossing_time = previous_time + (now - previous_time) * (
                        (line_x - previous_x) / (center_x - previous_x)
                    )
                    crossed_lines.append((side, crossing_time))
        elif center_x < previous_x:
            line_order = (("right", right_line), ("left", left_line))
            for side, line_x in line_order:
                if center_x <= line_x < previous_x:
                    crossing_time = previous_time + (now - previous_time) * (
                        (previous_x - line_x) / (previous_x - center_x)
                    )
                    crossed_lines.append((side, crossing_time))

        for side, crossing_time in crossed_lines:
            if track["first_line"] is None or track["first_line"] == side:
                track["first_line"] = side
                track["first_crossing"] = crossing_time
                track["speed_kmh"] = None
            else:
                elapsed = crossing_time - track["first_crossing"]
                if elapsed > 0:
                    track["speed_kmh"] = distance_m / elapsed * 3.6
                track["first_line"] = None
                track["first_crossing"] = None

        track["center_x"] = center_x
        track["timestamp"] = now
        track["last_seen"] = now
        return track["speed_kmh"]