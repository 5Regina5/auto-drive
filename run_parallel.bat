@echo off
echo 启动5个终端并行处理数据
echo 每个进程处理500条数据

echo 启动进程1 (处理 0-499)
start "Process 1" cmd /k "cd /d f:\auto-drive\code && python get_answer.py --process_id 1"

echo 启动进程2 (处理 500-999)
start "Process 2" cmd /k "cd /d f:\auto-drive\code && python get_answer.py --process_id 2"

echo 启动进程3 (处理 1000-1499)
start "Process 3" cmd /k "cd /d f:\auto-drive\code && python get_answer.py --process_id 3"

echo 启动进程4 (处理 1500-1999)
start "Process 4" cmd /k "cd /d f:\auto-drive\code && python get_answer.py --process_id 4"

echo 启动进程5 (处理 2000-2499)
start "Process 5" cmd /k "cd /d f:\auto-drive\code && python get_answer.py --process_id 5"

echo 所有进程已启动！
pause
