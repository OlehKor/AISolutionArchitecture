INSERT OR REPLACE INTO working_hours (weekday, open_time, close_time) VALUES
    (0, '06:00', '23:00'),
    (1, '06:00', '23:00'),
    (2, '06:00', '23:00'),
    (3, '06:00', '23:00'),
    (4, '06:00', '23:00'),
    (5, '07:00', '23:00'),
    (6, '08:00', '22:00');

INSERT OR REPLACE INTO zones
    (code, description, total_spaces, occupied_spaces, reservable_spaces, hourly_price, daily_price) VALUES
    ('A',  'Level 0, short stay',        120,  86,  0, 3.50, 28.00),
    ('B',  'Level -1, standard',         200, 141, 80, 2.50, 20.00),
    ('C',  'Level -2, long stay',        150,  63, 60, 1.80, 14.00),
    ('EV', 'Level -1, EV charging',       20,  12, 10, 4.00, 32.00);
