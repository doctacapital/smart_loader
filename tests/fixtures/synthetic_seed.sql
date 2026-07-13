-- Synthetic dwh fixture for smart_loader Capa 1 tests (DPM-395 §7.2).
-- Requires the docta-dwh migrations to already be applied (venue XBUE and
-- fx_rate_type MEP/CCL come from db/migrations/20260705000011_seeds.sql).

INSERT INTO dwh.issuer (name, country) VALUES ('Test Issuer SA', 'AR')
ON CONFLICT (lower(name)) DO NOTHING;

INSERT INTO dwh.instrument (kind, name, denom_currency) VALUES
    ('STOCK',   'Test GGAL SA',       'ARS'),
    ('CEDEAR',  'Test NVDA CEDEAR',   'ARS'),
    ('BOND',    'Test AL30 Bond',     'ARS'),
    ('STOCK',   'Test Segmented SA',  'ARS');

INSERT INTO dwh.series (instrument_id, mic, symbol, segment, settlement, trade_currency, is_cable) VALUES
    ((SELECT instrument_id FROM dwh.instrument WHERE name = 'Test GGAL SA'),     'XBUE', 'TSTGGAL', '-', '24H', 'ARS', false),
    ((SELECT instrument_id FROM dwh.instrument WHERE name = 'Test NVDA CEDEAR'), 'XBUE', 'TSTNVDA', '-', '24H', 'ARS', false),
    ((SELECT instrument_id FROM dwh.instrument WHERE name = 'Test AL30 Bond'),   'XBUE', 'TSTAL30', '-', '24H', 'ARS', false),
    -- segment='CT' (not the promoter's standard '-' sentinel) — the pg_reader
    -- resolution filter must NOT pick this up via the standard hist_raw path
    -- (DPM-395 F1: natural key is (mic,symbol,segment,settlement)).
    ((SELECT instrument_id FROM dwh.instrument WHERE name = 'Test Segmented SA'), 'XBUE', 'TSTSEG', 'CT', '24H', 'ARS', false);

-- Alias: an old symbol that used to point at TSTGGAL's series.
INSERT INTO dwh.series_alias (series_id, symbol, valid_from, source) VALUES
    ((SELECT series_id FROM dwh.series WHERE symbol = 'TSTGGAL'), 'TSTOLD', '2020-01-01', 'test');

-- TSTGGAL: 6 bars, one with open NULL (day 3) — tests NULL passthrough.
INSERT INTO dwh.eod_bar (series_id, trade_date, open, high, low, close, volume, turnover, source) VALUES
    ((SELECT series_id FROM dwh.series WHERE symbol = 'TSTGGAL'), '2024-01-02', 100, 105,  99, 103, 1000, 100000, 'TEST'),
    ((SELECT series_id FROM dwh.series WHERE symbol = 'TSTGGAL'), '2024-01-03', 101, 106, 100, 104, 1100, 110000, 'TEST'),
    ((SELECT series_id FROM dwh.series WHERE symbol = 'TSTGGAL'), '2024-01-04', NULL, 107, 101, 105, 1200, 120000, 'TEST'),
    ((SELECT series_id FROM dwh.series WHERE symbol = 'TSTGGAL'), '2024-01-05', 103, 108, 102, 106, 1300, 130000, 'TEST'),
    ((SELECT series_id FROM dwh.series WHERE symbol = 'TSTGGAL'), '2024-01-08', 104, 109, 103, 107, 1400, 140000, 'TEST'),
    ((SELECT series_id FROM dwh.series WHERE symbol = 'TSTGGAL'), '2024-01-09', 105, 110, 104, 108, 1500, 150000, 'TEST');

-- TSTNVDA: 6 bars, no fx_rate at all — tests usd_mep fields all None when MEP missing.
INSERT INTO dwh.eod_bar (series_id, trade_date, open, high, low, close, volume, turnover, source) VALUES
    ((SELECT series_id FROM dwh.series WHERE symbol = 'TSTNVDA'), '2024-01-02', 900, 910, 895, 905, 500, 450000, 'TEST'),
    ((SELECT series_id FROM dwh.series WHERE symbol = 'TSTNVDA'), '2024-01-03', 905, 915, 900, 910, 520, 470000, 'TEST'),
    ((SELECT series_id FROM dwh.series WHERE symbol = 'TSTNVDA'), '2024-01-04', 910, 920, 905, 915, 540, 490000, 'TEST'),
    ((SELECT series_id FROM dwh.series WHERE symbol = 'TSTNVDA'), '2024-01-05', 915, 925, 910, 920, 560, 510000, 'TEST'),
    ((SELECT series_id FROM dwh.series WHERE symbol = 'TSTNVDA'), '2024-01-08', 920, 930, 915, 925, 580, 530000, 'TEST'),
    ((SELECT series_id FROM dwh.series WHERE symbol = 'TSTNVDA'), '2024-01-09', 925, 935, 920, 930, 600, 550000, 'TEST');

-- TSTAL30 (BOND kind): eod_bar exists so we can prove hist_raw/hist_adj's
-- STOCK/CEDEAR kind filter excludes it even though the symbol matches.
INSERT INTO dwh.eod_bar (series_id, trade_date, open, high, low, close, volume, turnover, source) VALUES
    ((SELECT series_id FROM dwh.series WHERE symbol = 'TSTAL30'), '2024-01-02', 50, 51, 49, 50.5, 10, 500, 'TEST'),
    ((SELECT series_id FROM dwh.series WHERE symbol = 'TSTAL30'), '2024-01-03', 50.5, 51.5, 49.5, 51, 12, 600, 'TEST');

-- TSTSEG (segment='CT'): has real bars, but must NOT resolve via the
-- standard hist_raw path (segment != '-').
INSERT INTO dwh.eod_bar (series_id, trade_date, open, high, low, close, volume, turnover, source) VALUES
    ((SELECT series_id FROM dwh.series WHERE symbol = 'TSTSEG'), '2024-01-02', 10, 11, 9, 10.5, 5, 50, 'TEST');

-- MEP fx_rate for TSTGGAL, all dates except the last one (2024-01-09) — tests
-- the "no MEP for this date" -> usd_mep fields None case.
INSERT INTO dwh.fx_rate (base_currency, quote_currency, rate_type, rate_date, value, source) VALUES
    ('USD', 'ARS', 'MEP', '2024-01-02', 1000, 'TEST'),
    ('USD', 'ARS', 'MEP', '2024-01-03', 1010, 'TEST'),
    ('USD', 'ARS', 'MEP', '2024-01-04', 1020, 'TEST'),
    ('USD', 'ARS', 'MEP', '2024-01-05', 1030, 'TEST'),
    ('USD', 'ARS', 'MEP', '2024-01-08', 1040, 'TEST');
