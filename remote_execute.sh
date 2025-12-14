#!/bin/bash
# Скрипт для выполнения команд на удаленном сервере

SERVER_IP="176.109.111.108"
SERVER_USER="root"

echo "Подключение к серверу $SERVER_USER@$SERVER_IP"
echo "Введите пароль от сервера:"
echo ""

# Копируем скрипт установки
scp setup_server.sh $SERVER_USER@$SERVER_IP:~/

# Выполняем установку
ssh $SERVER_USER@$SERVER_IP << 'ENDSSH'
cd ~
chmod +x setup_server.sh
./setup_server.sh
ENDSSH

echo ""
echo "Готово!"
