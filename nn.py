import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import uuid
import os
import requests
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QComboBox, QLineEdit, QPushButton, QLabel, QFileDialog,
                             QTextEdit, QMessageBox, QStackedWidget)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
try:
    from spikingjelly.activation_based import neuron, functional
except ImportError:
    neuron = None
    functional = None
try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    LoraConfig = None
try:
    from stable_baselines3 import PPO
    import gym
    from gym import spaces
except ImportError:
    PPO = None
    gym = None
    print("警告：未安装 stable-baselines3 或 gym，强化学习功能已禁用。")

# SiliconFlow API 配置（请替换为实际密钥）
SILICONFLOW_API_KEY = "sk-**********************"
SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1/completions"

# 神经网络模型
class MLP(nn.Module):
    def __init__(self, input_size, num_classes, hyperparams):
        super(MLP, self).__init__()
        hidden_sizes = hyperparams.get('hidden_sizes', [128, 64])
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend([nn.Linear(prev_size, hidden_size), nn.ReLU()])
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, num_classes))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x.view(x.size(0), -1))

class CNN(nn.Module):
    def __init__(self, input_shape, num_classes, hyperparams):
        super(CNN, self).__init__()
        kernel_size = hyperparams.get('kernel_size', 3)
        conv_channels = hyperparams.get('conv_channels', [16, 32])
        hidden_size = hyperparams.get('hidden_size', 128)
        
        self.conv_layers = nn.ModuleList()
        in_channels = input_shape[0]
        for out_channels in conv_channels:
            self.conv_layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
            )
            self.conv_layers.append(nn.ReLU())
            self.conv_layers.append(nn.MaxPool2d(2))
            in_channels = out_channels
        
        conv_out_size = input_shape[1] // (2 ** len(conv_channels))
        fc_input_size = conv_channels[-1] * conv_out_size * conv_out_size
        self.fc1 = nn.Linear(fc_input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class RNN(nn.Module):
    def __init__(self, input_size, num_classes, hyperparams):
        super(RNN, self).__init__()
        hidden_size = hyperparams.get('hidden_size', 128)
        num_layers = hyperparams.get('num_layers', 2)
        
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        _, (hn, _) = self.rnn(x)
        out = self.fc(hn[-1])
        return out

class SNN(nn.Module):
    def __init__(self, input_shape, num_classes, hyperparams):
        super(SNN, self).__init__()
        if neuron is None:
            raise ImportError("需要 spikingjelly 支持 SNN。")
        conv_channels = hyperparams.get('conv_channels', [16, 32])
        hidden_size = hyperparams.get('hidden_size', 128)
        T = hyperparams.get('T', 4)
        
        self.T = T
        self.conv_layers = nn.ModuleList()
        in_channels = input_shape[0]
        for out_channels in conv_channels:
            self.conv_layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            )
            self.conv_layers.append(neuron.IFNode())
            self.conv_layers.append(nn.MaxPool2d(2))
            in_channels = out_channels
        
        conv_out_size = input_shape[1] // (2 ** len(conv_channels))
        fc_input_size = conv_channels[-1] * conv_out_size * conv_out_size
        self.fc1 = nn.Linear(fc_input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        for t in range(self.T):
            out = x[t]
            for layer in self.conv_layers:
                if isinstance(layer, neuron.IFNode):
                    out = layer(out)
                else:
                    out = layer(out)
            if t == 0:
                output = out.view(batch_size, -1)
            else:
                output += out.view(batch_size, -1)
        output = torch.relu(self.fc1(output / self.T))
        output = self.fc2(output)
        return output

class Transformer(nn.Module):
    def __init__(self, input_size, num_classes, hyperparams):
        super(Transformer, self).__init__()
        nhead = hyperparams.get('nhead', 4)
        num_layers = hyperparams.get('num_layers', 2)
        dim_feedforward = hyperparams.get('dim_feedforward', 512)
        
        self.embedding = nn.Linear(input_size, dim_feedforward)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_feedforward, nhead=nhead, dim_feedforward=dim_feedforward
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(dim_feedforward, num_classes)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)

# 强化学习环境
if gym:
    class ClassificationEnv(gym.Env):
        def __init__(self, model, data):
            super(ClassificationEnv, self).__init__()
            self.model = model
            self.X, self.y = data
            self.action_space = spaces.Discrete(len(np.unique(self.y)))
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.X.shape[1:], dtype=np.float32)
            self.current_step = 0
        
        def reset(self):
            self.current_step = 0
            return self.X[self.current_step]
        
        def step(self, action):
            reward = 1.0 if action == self.y[self.current_step] else -1.0
            self.current_step += 1
            done = self.current_step >= len(self.X)
            obs = self.X[self.current_step] if not done else self.X[0]
            return obs, reward, done, {}

# 训练函数
def train_network(network_type, hyperparams, train_data, val_data=None, model_path=None, fine_tune=False, lora=False, model=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    X_train, y_train = train_data
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=hyperparams.get('batch_size', 32), shuffle=True)
    
    if val_data:
        X_val, y_val = val_data
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.long)
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=hyperparams.get('batch_size', 32))
    
    input_shape = X_train.shape[1:]
    num_classes = len(torch.unique(y_train))
    
    if not fine_tune:
        if network_type == "MLP":
            model = MLP(np.prod(input_shape), num_classes, hyperparams)
        elif network_type == "CNN":
            model = CNN(input_shape, num_classes, hyperparams)
        elif network_type == "SNN":
            if neuron is None:
                raise ImportError("需要 spikingjelly 支持 SNN。")
            model = SNN(input_shape, num_classes, hyperparams)
        elif network_type == "RNN":
            model = RNN(input_shape[-1], num_classes, hyperparams)
        elif network_type == "Transformer":
            model = Transformer(input_shape[-1], num_classes, hyperparams)
    
    model = model.to(device)
    
    if lora and LoraConfig:
        lora_config = LoraConfig(
            r=hyperparams.get('lora_rank', 8),
            lora_alpha=hyperparams.get('lora_alpha', 16),
            target_modules=["fc1", "fc2", "rnn", "fc"] if network_type != "Transformer" else ["fc", "embedding"]
        )
        model = get_peft_model(model, lora_config)
    
    optimizer = optim.Adam(model.parameters(), lr=hyperparams.get('learning_rate', 0.001))
    criterion = nn.CrossEntropyLoss()
    
    epochs = hyperparams.get('epochs', 10)
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        if val_data:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
            val_losses.append(val_loss / len(val_loader.dataset))
    
    if model_path is None:
        model_path = f"model_{network_type}_{uuid.uuid4().hex}.pth"
    torch.save(model.state_dict(), model_path)
    
    return {'model': model, 'history': {'train_loss': train_losses, 'val_loss': val_losses}, 'model_path': model_path}

# 强化学习训练
def train_rl(model, data, hyperparams):
    if PPO is None or gym is None:
        raise ImportError("强化学习需要 stable-baselines3 和 gym，请先安装。")
    env = ClassificationEnv(model, data)
    rl_model = PPO("MlpPolicy", env, learning_rate=hyperparams.get('rl_lr', 0.0003), n_steps=2048)
    rl_model.learn(total_timesteps=hyperparams.get('rl_steps', 10000))
    return rl_model

# 数据格式化 API 调用
def format_data(input_file, target_format, api_key=SILICONFLOW_API_KEY):
    data = pd.read_csv(input_file, encoding='utf-8')
    prompt = f"将以下数据转换为 {target_format} 格式以用于神经网络训练：\n{data.head().to_string()}\n目标格式：{target_format}（例如，图像数据为 (N,C,H,W)，序列数据为 (N,seq_len,features)）。"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "prompt": prompt,
        "max_tokens": 1000
    }
    response = requests.post(SILICONFLOW_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        result = response.json()['choices'][0]['text']
        output_file = f"formatted_{os.path.basename(input_file)}"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result)
        return output_file
    else:
        raise Exception(f"API 调用失败：{response.text}")

# AI 问答 API 调用
def ask_ai(question, api_key=SILICONFLOW_API_KEY):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "THUDM/glm-4-9b-chat",
        "prompt": f"作为神经网络专家，回答以下问题：{question}",
        "max_tokens": 500
    }
    response = requests.post(SILICONFLOW_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()['choices'][0]['text']
    else:
        raise Exception(f"API 调用失败：{response.text}")

# 图形用户界面
class NeuralNetworkLearner(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("神经网络学习器 - RhymeChime 工作室")
        self.setGeometry(100, 100, 1000, 700)
        self.model = None
        self.hyperparams = {}
        self.train_data = None
        self.val_data = None
        self.model_path = None
        self.param_fields = {}  # 初始化共享超参数字典
        self.current_log = None  # 当前页面的日志输出
        self.init_ui()
    
    def init_ui(self):
        self.central_widget = QStackedWidget()
        self.setCentralWidget(self.central_widget)
    
    # 主页
        home_widget = QWidget()
        home_layout = QVBoxLayout()
        home_widget.setLayout(home_layout)
    
        title = QLabel("神经网络学习器")
        title.setFont(QFont("Segoe UI", 24, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        home_layout.addWidget(title)
    
        subtitle = QLabel("由 RhymeChime 工作室提供支持")
        subtitle.setFont(QFont("Segoe UI", 12))
        subtitle.setAlignment(Qt.AlignCenter)
        home_layout.addWidget(subtitle)
    
        buttons = [
        ("创建新神经网络", self.show_create_page),
        ("微调现有网络", self.show_finetune_page),
        ("强化学习", self.show_rl_page),
        ("使用现有网络", self.show_use_page),
        ("数据格式化", self.show_data_page)
        ]
        for text, func in buttons:
            btn = QPushButton(text)
            btn.clicked.connect(func)
            home_layout.addWidget(btn)
    
    # 日志
        '''
        self.home_log = QTextEdit()
        self.home_log.setReadOnly(True)
        home_layout.addWidget(self.home_log)
        '''    
    # 退出按钮
        exit_btn = QPushButton("退出")
        exit_btn.clicked.connect(self.close)  # 确认绑定到 close
        home_layout.addWidget(exit_btn)
    
        contact = QLabel("联系方式：rhymechime1000@163.com")
        contact.setFont(QFont("Segoe UI", 10))
        contact.setAlignment(Qt.AlignCenter)
        home_layout.addWidget(contact)
    
        self.central_widget.addWidget(home_widget)  # 索引 0
    
    # 其他页面
        self.create_page = self.create_network_page()
        self.central_widget.addWidget(self.create_page)  # 索引 1
    
        self.finetune_page = self.finetune_network_page()
        self.central_widget.addWidget(self.finetune_page)  # 索引 2
    
        self.rl_page = self.rl_network_page()
        self.central_widget.addWidget(self.rl_page)  # 索引 3
    
        self.use_page = self.use_network_page()
        self.central_widget.addWidget(self.use_page)  # 索引 4
    
        self.data_page = self.data_format_page()
        self.central_widget.addWidget(self.data_page)  # 索引 5
    
    # 应用样式表
        try:
            with open("style.qss", "r", encoding='utf-8') as f:
                self.setStyleSheet(f.read())
        except Exception as e:
            print(f"加载样式表失败：{e}")
            self.setStyleSheet("""
            QMainWindow { background: #1E1E2E; color: #E0E0FF; }
            QLabel { color: #E0E0FF; font-family: "Segoe UI"; font-size: 14px; }
            QPushButton { background-color: #3B82F6; color: #FFFFFF; border-radius: 8px; padding: 10px; }
            QComboBox, QLineEdit, QTextEdit { background-color: #2D2D4A; color: #E0E0FF; border: 1px solid #60A5FA; border-radius: 5px; }
            """)
    
    def create_network_page(self):
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
    
        title = QLabel("创建新神经网络")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        layout.addWidget(title)
    
    # 网络类型
        network_layout = QHBoxLayout()
        network_label = QLabel("网络类型：")
        self.network_combo = QComboBox()
        self.network_combo.addItems(["MLP（多层感知机）", "CNN（卷积神经网络）", "RNN（循环神经网络）", "SNN（脉冲神经网络）", "Transformer（变换器）"])
        network_layout.addWidget(network_label)
        network_layout.addWidget(self.network_combo)
        layout.addLayout(network_layout)
    
    # 数据类型
        data_type_layout = QHBoxLayout()
        data_type_label = QLabel("数据类型：")
        self.data_type_combo = QComboBox()
        self.data_type_combo.addItems(["图像(支持28x28或32x32)", "序列", "表格"])
        data_type_layout.addWidget(data_type_label)
        data_type_layout.addWidget(self.data_type_combo)
        layout.addLayout(data_type_layout)
    
    # 数据加载
        data_layout = QHBoxLayout()
        self.train_data_btn = QPushButton("加载训练数据（csv格式）")
        self.val_data_btn = QPushButton("加载验证数据（可选）")
        self.train_data_btn.clicked.connect(self.load_train_data)
        self.val_data_btn.clicked.connect(self.load_val_data)
        data_layout.addWidget(self.train_data_btn)
        data_layout.addWidget(self.val_data_btn)
        layout.addLayout(data_layout)
    
    # 超参数
        params_layout = QVBoxLayout()
        params = [
            ("学习率", "learning_rate", "0.001"),
            ("批量大小", "batch_size", "32"),
            ("训练轮次", "epochs", "10"),
            ("隐藏层大小", "hidden_size", "128"),
            ("隐藏层（MLP）", "hidden_sizes", "128,64"),
            ("卷积通道（CNN/SNN）", "conv_channels", "16,32"),
            ("卷积核大小（CNN/SNN）", "kernel_size", "3"),
            ("RNN/Transformer 层数", "num_layers", "2"),
            ("Transformer 注意力头数", "nhead", "4"),
            ("Transformer 前馈维度", "dim_feedforward", "512"),
            ("SNN 时间步长", "T", "4")
        ]
        for label, key, default in params:
            h_layout = QHBoxLayout()
            lbl = QLabel(label + "：")
            edit = QLineEdit(default)
            self.param_fields[key] = edit
            h_layout.addWidget(lbl)
            h_layout.addWidget(edit)
            params_layout.addLayout(h_layout)
        layout.addLayout(params_layout)
    
    # 训练按钮
        self.train_btn = QPushButton("创建并训练")
        self.train_btn.clicked.connect(self.train_model)
        layout.addWidget(self.train_btn)
    
    # AI 问答
        ai_layout = QHBoxLayout()
        self.create_ai_question = QLineEdit("输入神经网络相关问题")
        self.create_ai_btn = QPushButton("咨询 AI")
        self.create_ai_btn.clicked.connect(lambda: self.ask_ai(self.create_ai_question, self.create_log))
        ai_layout.addWidget(self.create_ai_question)
        ai_layout.addWidget(self.create_ai_btn)
        layout.addLayout(ai_layout)
    
    # 日志和图表
        self.create_log = QTextEdit()
        self.create_log.setReadOnly(True)
        layout.addWidget(self.create_log)
    
    # 设置 matplotlib 支持中文
        import matplotlib
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
        matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
    
    # 返回按钮
        back_btn = QPushButton("返回")
        back_btn.clicked.connect(self.show_home_page)
        layout.addWidget(back_btn)
    
    # 页脚
        footer = QLabel("RhymeChime 工作室 | rhymechime1000@163.com")
        footer.setFont(QFont("Segoe UI", 10))
        footer.setAlignment(Qt.AlignCenter)
        layout.addWidget(footer)
    
        return widget
    
    def finetune_network_page(self):
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        
        title = QLabel("微调现有神经网络")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        layout.addWidget(title)
        
        # 微调类型
        finetune_layout = QHBoxLayout()
        finetune_label = QLabel("微调类型：")
        self.finetune_combo = QComboBox()
        self.finetune_combo.addItems(["完整微调", "LoRA 微调"])
        finetune_layout.addWidget(finetune_label)
        finetune_layout.addWidget(self.finetune_combo)
        layout.addLayout(finetune_layout)
        
        # 模型和数据
        load_layout = QHBoxLayout()
        self.model_load_btn = QPushButton("加载模型")
        self.data_load_btn = QPushButton("加载训练数据")
        self.model_load_btn.clicked.connect(self.load_model)
        self.data_load_btn.clicked.connect(self.load_train_data)
        load_layout.addWidget(self.model_load_btn)
        load_layout.addWidget(self.data_load_btn)
        layout.addLayout(load_layout)
        
        # LoRA 参数
        lora_layout = QVBoxLayout()
        lora_params = [
            ("LoRA 秩", "lora_rank", "8"),
            ("LoRA Alpha", "lora_alpha", "16")
        ]
        for label, key, default in lora_params:
            h_layout = QHBoxLayout()
            lbl = QLabel(label + "：")
            edit = QLineEdit(default)
            self.param_fields[key] = edit
            h_layout.addWidget(lbl)
            h_layout.addWidget(edit)
            lora_layout.addLayout(h_layout)
        layout.addLayout(lora_layout)
        
        # 微调按钮
        self.finetune_btn = QPushButton("开始微调")
        self.finetune_btn.clicked.connect(self.finetune_model)
        layout.addWidget(self.finetune_btn)
        
        # AI 问答
        ai_layout = QHBoxLayout()
        self.finetune_ai_question = QLineEdit("输入神经网络相关问题")
        self.finetune_ai_btn = QPushButton("咨询 AI")
        self.finetune_ai_btn.clicked.connect(lambda: self.ask_ai(self.finetune_ai_question, self.finetune_log))
        ai_layout.addWidget(self.finetune_ai_question)
        ai_layout.addWidget(self.finetune_ai_btn)
        layout.addLayout(ai_layout)
        
        # 日志
        self.finetune_log = QTextEdit()
        self.finetune_log.setReadOnly(True)
        layout.addWidget(self.finetune_log)
        
        # 返回按钮
        back_btn = QPushButton("返回")
        back_btn.clicked.connect(self.show_home_page)
        layout.addWidget(back_btn)
        
        # 页脚
        footer = QLabel("RhymeChime 工作室 | rhymechime1000@163.com")
        footer.setFont(QFont("Segoe UI", 10))
        footer.setAlignment(Qt.AlignCenter)
        layout.addWidget(footer)
        
        return widget
    
    def rl_network_page(self):
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        
        title = QLabel("强化学习")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        layout.addWidget(title)
        
        # 模型和数据
        load_layout = QHBoxLayout()
        self.model_load_btn = QPushButton("加载模型")
        self.data_load_btn = QPushButton("加载训练数据")
        self.model_load_btn.clicked.connect(self.load_model)
        self.data_load_btn.clicked.connect(self.load_train_data)
        load_layout.addWidget(self.model_load_btn)
        load_layout.addWidget(self.data_load_btn)
        layout.addLayout(load_layout)
        
        # RL 参数
        rl_layout = QVBoxLayout()
        rl_params = [
            ("强化学习率", "rl_lr", "0.0003"),
            ("强化学习步数", "rl_steps", "10000")
        ]
        for label, key, default in rl_params:
            h_layout = QHBoxLayout()
            lbl = QLabel(label + "：")
            edit = QLineEdit(default)
            self.param_fields[key] = edit
            h_layout.addWidget(lbl)
            h_layout.addWidget(edit)
            rl_layout.addLayout(h_layout)
        layout.addLayout(rl_layout)
        
        # RL 按钮
        self.rl_btn = QPushButton("开始强化学习")
        self.rl_btn.clicked.connect(self.train_rl)
        layout.addWidget(self.rl_btn)
        
        # AI 问答
        ai_layout = QHBoxLayout()
        self.rl_ai_question = QLineEdit("输入神经网络相关问题")
        self.rl_ai_btn = QPushButton("咨询 AI")
        self.rl_ai_btn.clicked.connect(lambda: self.ask_ai(self.rl_ai_question, self.rl_log))
        ai_layout.addWidget(self.rl_ai_question)
        ai_layout.addWidget(self.rl_ai_btn)
        layout.addLayout(ai_layout)
        
        # 日志
        self.rl_log = QTextEdit()
        self.rl_log.setReadOnly(True)
        layout.addWidget(self.rl_log)
        
        # 返回按钮
        back_btn = QPushButton("返回")
        back_btn.clicked.connect(self.show_home_page)
        layout.addWidget(back_btn)
        
        # 页脚
        footer = QLabel("RhymeChime 工作室 | rhymechime1000@163.com")
        footer.setFont(QFont("Segoe UI", 10))
        footer.setAlignment(Qt.AlignCenter)
        layout.addWidget(footer)
        
        return widget
    
    def use_network_page(self):
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        
        title = QLabel("使用现有网络")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        layout.addWidget(title)
        
        # 模型和数据
        load_layout = QHBoxLayout()
        self.model_load_btn = QPushButton("加载模型")
        self.data_load_btn = QPushButton("加载预测数据")
        self.model_load_btn.clicked.connect(self.load_model)
        self.data_load_btn.clicked.connect(self.load_predict_data)
        load_layout.addWidget(self.model_load_btn)
        load_layout.addWidget(self.data_load_btn)
        layout.addLayout(load_layout)
        
        # 预测按钮
        self.predict_btn = QPushButton("执行预测")
        self.predict_btn.clicked.connect(self.predict)
        layout.addWidget(self.predict_btn)
        
        # AI 问答
        ai_layout = QHBoxLayout()
        self.use_ai_question = QLineEdit("输入神经网络相关问题")
        self.use_ai_btn = QPushButton("咨询 AI")
        self.use_ai_btn.clicked.connect(lambda: self.ask_ai(self.use_ai_question, self.use_log))
        ai_layout.addWidget(self.use_ai_question)
        ai_layout.addWidget(self.use_ai_btn)
        layout.addLayout(ai_layout)
        
        # 日志
        self.use_log = QTextEdit()
        self.use_log.setReadOnly(True)
        layout.addWidget(self.use_log)
        
        # 返回按钮
        back_btn = QPushButton("返回")
        back_btn.clicked.connect(self.show_home_page)
        layout.addWidget(back_btn)
        
        # 页脚
        footer = QLabel("RhymeChime 工作室 | rhymechime1000@163.com")
        footer.setFont(QFont("Segoe UI", 10))
        footer.setAlignment(Qt.AlignCenter)
        layout.addWidget(footer)
        
        return widget
    
    def data_format_page(self):
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        
        title = QLabel("数据格式化")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        layout.addWidget(title)
        
        # 数据和格式
        data_layout = QHBoxLayout()
        self.data_load_btn = QPushButton("加载数据")
        self.format_combo = QComboBox()
        self.format_combo.addItems(["图像 (N,C,H,W)", "序列 (N,seq_len,features)", "表格 (N,features)"])
        self.data_load_btn.clicked.connect(self.load_data_for_format)
        data_layout.addWidget(self.data_load_btn)
        data_layout.addWidget(self.format_combo)
        layout.addLayout(data_layout)
        
        # 格式化按钮
        self.format_btn = QPushButton("格式化数据")
        self.format_btn.clicked.connect(self.format_data)
        layout.addWidget(self.format_btn)
        
        # AI 问答
        ai_layout = QHBoxLayout()
        self.data_ai_question = QLineEdit("输入神经网络相关问题")
        self.data_ai_btn = QPushButton("咨询 AI")
        self.data_ai_btn.clicked.connect(lambda: self.ask_ai(self.data_ai_question, self.data_log))
        ai_layout.addWidget(self.data_ai_question)
        ai_layout.addWidget(self.data_ai_btn)
        layout.addLayout(ai_layout)
        
        # 日志
        self.data_log = QTextEdit()
        self.data_log.setReadOnly(True)
        layout.addWidget(self.data_log)
        
        # 返回按钮
        back_btn = QPushButton("返回")
        back_btn.clicked.connect(self.show_home_page)
        layout.addWidget(back_btn)
        
        # 页脚
        footer = QLabel("RhymeChime 工作室 | rhymechime1000@163.com")
        footer.setFont(QFont("Segoe UI", 10))
        footer.setAlignment(Qt.AlignCenter)
        layout.addWidget(footer)
        
        return widget
    
    def show_home_page(self):
        self.central_widget.setCurrentIndex(0)
        self.current_log = self.home_log
    
    def show_create_page(self):
        self.central_widget.setCurrentWidget(self.create_page)
        self.current_log = self.create_log
    
    def show_finetune_page(self):
        self.central_widget.setCurrentWidget(self.finetune_page)
        self.current_log = self.finetune_log
    
    def show_rl_page(self):
        self.central_widget.setCurrentWidget(self.rl_page)
        self.current_log = self.rl_log
    
    def show_use_page(self):
        self.central_widget.setCurrentWidget(self.use_page)
        self.current_log = self.use_log
    
    def show_data_page(self):
        self.central_widget.setCurrentWidget(self.data_page)
        self.current_log = self.data_log
    
    def load_train_data(self):
        file, _ = QFileDialog.getOpenFileName(self, "选择训练数据 (CSV)")
        if file:
            try:
                data = pd.read_csv(file, encoding='utf-8')
                X = data.iloc[:, :-1].values  # 特征
                y = data.iloc[:, -1].values   # 标签
                data_type = self.data_type_combo.currentText() if hasattr(self, 'data_type_combo') else "表格"
                if data_type == "图像":
                    if X.shape[1] == 3071:  # 32x32x3 RGB 图像
                        X = X.reshape(-1, 3, 32, 32)  # (N, C, H, W)
                    elif X.shape[1] == 784:  # 28x28 灰度图
                        X = X.reshape(-1, 1, 28, 28)
                    else:
                        raise ValueError(f"图像数据形状不正确，特征数为 {X.shape[1]}，预期 784 (28x28x1) 或 3071 (32x32x3)")
                elif data_type == "序列":
                    seq_len = 10  # 可调整
                    if X.shape[1] % seq_len == 0:
                        X = X.reshape(-1, seq_len, X.shape[1] // seq_len)
                    else:
                        raise ValueError(f"序列数据形状不正确，特征数为 {X.shape[1]}")
                elif data_type == "表格" and self.network_combo.currentText().startswith("CNN"):
                    raise ValueError("CNN 要求图像数据，请选择‘图像’数据类型")
                self.train_data = (X, y)
                self.current_log.append(f"训练数据已加载：{file}，特征形状：{X.shape}，标签形状：{y.shape}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载训练数据失败：{str(e)}")
    
    def load_val_data(self):
        file, _ = QFileDialog.getOpenFileName(self, "选择验证数据 (CSV)")
        if file:
            try:
                data = pd.read_csv(file, encoding='utf-8')
                X = data.iloc[:, :-1].values
                y = data.iloc[:, -1].values
                self.val_data = (X, y)
                self.current_log.append(f"验证数据已加载：{file}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载验证数据失败：{str(e)}")
    
    def load_model(self):
        file, _ = QFileDialog.getOpenFileName(self, "选择模型文件 (.pth)")
        if file:
            try:
                network_type = self.network_combo.currentText().split("（")[0] if hasattr(self, 'network_combo') else "MLP"
                X_train, _ = self.train_data if self.train_data else (np.random.rand(1, 3, 32, 32), np.zeros(1))
                input_shape = X_train.shape[1:]
                num_classes = len(np.unique(self.train_data[1])) if self.train_data else 10
                
                if network_type == "MLP":
                    model = MLP(np.prod(input_shape), num_classes, self.hyperparams)
                elif network_type == "CNN":
                    model = CNN(input_shape, num_classes, self.hyperparams)
                elif network_type == "SNN":
                    model = SNN(input_shape, num_classes, self.hyperparams)
                elif network_type == "RNN":
                    model = RNN(input_shape[-1], num_classes, self.hyperparams)
                elif network_type == "Transformer":
                    model = Transformer(input_shape[-1], num_classes, self.hyperparams)
                model.load_state_dict(torch.load(file))
                self.model = model
                self.model_path = file
                self.current_log.append(f"模型已加载：{file}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载模型失败：{str(e)}")
    
    def load_predict_data(self):
        file, _ = QFileDialog.getOpenFileName(self, "选择预测数据 (CSV)")
        if file:
            try:
                data = pd.read_csv(file, encoding='utf-8')
                self.predict_data = torch.tensor(data.values, dtype=torch.float32)
                self.current_log.append(f"预测数据已加载：{file}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载预测数据失败：{str(e)}")
    
    def load_data_for_format(self):
        file, _ = QFileDialog.getOpenFileName(self, "选择要格式化的数据 (CSV)")
        if file:
            self.data_file = file
            self.current_log.append(f"数据已加载：{file}")
    
    def train_model(self):
        if self.train_data is None:
            QMessageBox.critical(self, "错误", "请先加载训练数据")
            return
        
        try:
            self.hyperparams = {
                'learning_rate': float(self.param_fields['learning_rate'].text()),
                'batch_size': int(self.param_fields['batch_size'].text()),
                'epochs': int(self.param_fields['epochs'].text()),
                'hidden_size': int(self.param_fields['hidden_size'].text()),
                'hidden_sizes': [int(x) for x in self.param_fields['hidden_sizes'].text().split(',')],
                'conv_channels': [int(x) for x in self.param_fields['conv_channels'].text().split(',')],
                'kernel_size': int(self.param_fields['kernel_size'].text()),
                'num_layers': int(self.param_fields['num_layers'].text()),
                'nhead': int(self.param_fields['nhead'].text()),
                'dim_feedforward': int(self.param_fields['dim_feedforward'].text()),
                'T': int(self.param_fields['T'].text())
            }
        except Exception as e:
            QMessageBox.critical(self, "错误", f"超参数格式错误：{str(e)}")
            return
        
        network_type = self.network_combo.currentText().split("（")[0]
        self.current_log.append(f"开始训练 {network_type} 网络...")
        
        try:
            result = train_network(network_type, self.hyperparams, self.train_data, self.val_data)
            self.model = result['model']
            self.model_path = result['model_path']
            self.current_log.append(f"训练完成，模型保存至：{self.model_path}")
            
            self.ax.clear()
            self.ax.plot(result['history']['train_loss'], label='训练损失')
            if 'val_loss' in result['history']:
                self.ax.plot(result['history']['val_loss'], label='验证损失')
            self.ax.set_xlabel('轮次')
            self.ax.set_ylabel('损失')
            self.ax.legend()
            self.canvas.draw()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"训练失败：{str(e)}")
    
    def finetune_model(self):
        if self.train_data is None or self.model is None:
            QMessageBox.critical(self, "错误", "请加载模型和训练数据")
            return
        
        try:
            self.hyperparams = {
                'learning_rate': float(self.param_fields['learning_rate'].text()),
                'batch_size': int(self.param_fields['batch_size'].text()),
                'epochs': int(self.param_fields['epochs'].text()),
                'lora_rank': int(self.param_fields['lora_rank'].text()),
                'lora_alpha': int(self.param_fields['lora_alpha'].text())
            }
        except Exception as e:
            QMessageBox.critical(self, "错误", f"超参数格式错误：{str(e)}")
            return
        
        network_type = self.network_combo.currentText().split("（")[0] if hasattr(self, 'network_combo') else "MLP"
        lora = self.finetune_combo.currentText() == "LoRA 微调"
        
        try:
            result = train_network(network_type, self.hyperparams, self.train_data, self.val_data, fine_tune=True, lora=lora, model=self.model)
            self.model = result['model']
            self.model_path = result['model_path']
            self.current_log.append(f"微调完成，模型保存至：{self.model_path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"微调失败：{str(e)}")
    
    def train_rl(self):
        if self.train_data is None or self.model is None:
            QMessageBox.critical(self, "错误", "请加载模型和训练数据")
            return
        
        if PPO is None or gym is None:
            QMessageBox.critical(self, "错误", "强化学习需要 stable-baselines3 和 gym")
            return
        
        try:
            self.hyperparams = {
                'rl_lr': float(self.param_fields['rl_lr'].text()),
                'rl_steps': int(self.param_fields['rl_steps'].text())
            }
        except Exception as e:
            QMessageBox.critical(self, "错误", f"超参数格式错误：{str(e)}")
            return
        
        try:
            rl_model = train_rl(self.model, self.train_data, self.hyperparams)
            self.current_log.append("强化学习完成")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"强化学习失败：{str(e)}")
    
    def predict(self):
        if self.model is None or not hasattr(self, 'predict_data'):
            QMessageBox.critical(self, "错误", "请加载模型和预测数据")
            return
        
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(self.predict_data.to(device))
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            self.current_log.append(f"预测结果：{predictions}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"预测失败：{str(e)}")
    
    def format_data(self):
        if not hasattr(self, 'data_file'):
            QMessageBox.critical(self, "错误", "请加载数据")
            return
        
        target_format = self.format_combo.currentText()
        try:
            output_file = format_data(self.data_file, target_format)
            self.current_log.append(f"数据格式化完成，保存至：{output_file}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"数据格式化失败：{str(e)}")
    
    def ask_ai(self, question_input, log_output):
        question = question_input.text()
        if not question:
            QMessageBox.critical(self, "错误", "请输入问题")
            return
        
        try:
            answer = ask_ai(question)
            log_output.append(f"AI 回答：{answer}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"AI 问答失败：{str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NeuralNetworkLearner()
    window.show()
    sys.exit(app.exec_())
