CREATE DATABASE IF NOT EXISTS ecommerce;
USE ecommerce;

CREATE TABLE IF NOT EXISTS users (
    user_id INT8 PRIMARY KEY,
    name VARCHAR(100),
    region VARCHAR(20),
    balance DECIMAL(10,2)
);

CREATE TABLE IF NOT EXISTS inventory (
    item_id INT8 PRIMARY KEY,
    item_name VARCHAR(100),
    stock INT8,
    price DECIMAL(10,2),
    home_region VARCHAR(20)
);

CREATE TABLE IF NOT EXISTS orders (
    order_id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id INT8,
    item_id INT8,
    quantity INT,
    total_amount DECIMAL(10,2),
    created_at TIMESTAMPTZ DEFAULT now(),
    gateway_region VARCHAR(20)
);
