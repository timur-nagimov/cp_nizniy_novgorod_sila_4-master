-- init.sql
CREATE TABLE IF NOT EXISTS images (
    id SERIAL PRIMARY KEY, 
    hash VARCHAR(256) NOT NULL,
    image BYTEA NOT NULL  
);