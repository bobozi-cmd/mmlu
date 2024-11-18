用llama.cpp的llama-server启动模型服务，假设IP地址是192.168.1.39，端口是8080，测试命令：

```bash
python mmlu_test.py --host 192.168.1.39 --port 8080 -s 30
```

MMLU一共有57个科目，`-s 30`表示每个科目测前30题。不加`-s`参数就是把整个MMLU数据集都测一遍。

如果用纯CPU跑，比较慢，可以先测`-s 1`。
