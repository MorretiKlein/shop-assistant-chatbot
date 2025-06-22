-- Drop database demo;
CREATE DATABASE demo;
USE demo;


CREATE TABLE user (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE
    -- password VARCHAR(255) NOT NULL,
    -- uuid CHAR(36) NOT NULL UNIQUE
);
SELECT * FROM user;
-- DELETE FROM `demo`.`user` WHERE (`id` = '1');
-- SELECT HOST FROM information_schema.processlist WHERE ID = CONNECTION_ID();
SELECT user, host FROM information_schema.processlist;
