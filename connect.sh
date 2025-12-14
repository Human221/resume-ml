#!/bin/bash
# Скрипт для подключения к серверу Cloud.ru

SERVER_IP="176.109.111.108"
USER="root"

echo "Подключение к серверу Cloud.ru..."
echo "IP: $SERVER_IP"
echo "Пользователь: $USER"
echo ""
echo "Если это первый раз, введите пароль от сервера"
echo ""

ssh $USER@$SERVER_IP
