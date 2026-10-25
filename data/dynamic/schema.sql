CREATE TABLE IF NOT EXISTS working_hours (
    weekday    INTEGER PRIMARY KEY CHECK (weekday BETWEEN 0 AND 6),
    open_time  TEXT NOT NULL,
    close_time TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS zones (
    code              TEXT PRIMARY KEY,
    description       TEXT NOT NULL,
    total_spaces      INTEGER NOT NULL CHECK (total_spaces >= 0),
    occupied_spaces   INTEGER NOT NULL CHECK (occupied_spaces >= 0),
    reservable_spaces INTEGER NOT NULL CHECK (reservable_spaces >= 0),
    hourly_price      REAL NOT NULL,
    daily_price       REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS reservations (
    reference          TEXT PRIMARY KEY,
    first_name         TEXT NOT NULL,
    last_name          TEXT NOT NULL,
    car_number         TEXT NOT NULL,
    start_time         TEXT NOT NULL,
    end_time           TEXT NOT NULL,
    status             TEXT NOT NULL DEFAULT 'pending'
                       CHECK (status IN ('pending', 'confirmed', 'rejected')),
    admin_comment      TEXT NOT NULL DEFAULT '',
    request_message_id INTEGER,
    customer_notified  INTEGER NOT NULL DEFAULT 0,
    created_at         TEXT NOT NULL,
    decided_at         TEXT,
    archived           INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_reservations_period ON reservations (start_time, end_time);
