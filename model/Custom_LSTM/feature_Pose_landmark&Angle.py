import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime

# 디버깅 로그 파일 설정
debug_log_file = "debug_log.txt"
def log_debug(message):
    """디버깅 메시지를 콘솔과 파일에 기록"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}\n"
    print(log_message.strip())
    with open(debug_log_file, "a", encoding="utf-8") as f:
        f.write(log_message)

class CustomLSTM(nn.Module):
    def __init__(self, input_dim=99, hidden_size=64, num_layers=1, num_classes=15, dropout=0.5):
        super(CustomLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_size, num_layers=num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x, attn_mask=None):
        # x: [batch_size, seq_length, input_dim]
        batch_size = x.size(0)
        # 초기 hidden 및 cell 상태
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        
        # LSTM forward
        output, (hn, cn) = self.lstm(x, (h0, c0))  # output: [batch_size, seq_length, hidden_size]
  
        output = output[:, -1, :]  # [batch_size, hidden_size]
        output = self.dropout(output)
        output = self.fc(output)  # [batch_size, num_classes]
        return output

def load_dataset_for_lstm(sequence_dir, csv_path, max_seq_length=130):
    """
    CSV 파일과 시퀀스 파일을 로드하여 LSTM 입력 준비.
    
    Args:
        sequence_dir (str): 시퀀스 파일 디렉토리
        csv_path (str): labels.csv 파일 경로
        max_seq_length (int): 최대 시퀀스 길이
    Returns:
        tuple: (data [num_samples, max_seq_length, 99], mask, labels, lengths)
    """
    if not os.path.exists(csv_path):
        log_debug(f"CSV 파일 {csv_path}가 존재하지 않습니다.")
        return None, None, None, None
    
    df = pd.read_csv(csv_path)
    if not {'filename', 'label', 'type'}.issubset(df.columns):
        log_debug(f"CSV 파일에 필요한 열(filename, label, type)이 없습니다.")
        return None, None, None, None
    
    data = []
    labels = []
    lengths = []
    
    for _, row in df.iterrows():
        file_path = os.path.join(sequence_dir, row['filename'])
        if not os.path.exists(file_path):
            log_debug(f"파일 {file_path}가 존재하지 않습니다.")
            continue
        
        try:
            sequence = np.load(file_path)
            if sequence.shape != (max_seq_length, 99):
                log_debug(f"파일 {file_path}의 형상 {sequence.shape}이 [{max_seq_length}, 99]이 아닙니다.")
                continue
            # 정규화
            sequence = (sequence - sequence.min()) / (sequence.max() - sequence.min() + 1e-8)
        except Exception as e:
            log_debug(f"파일 {file_path} 로드 중 오류: {e}")
            continue
        
        seq_length = np.any(sequence != 0, axis=1).sum()
        if seq_length < 20:  # 최소 길이 필터링
            log_debug(f"파일 {file_path}의 유효 길이 {seq_length}가 너무 짧습니다.")
            continue
        
        data.append(torch.tensor(sequence, dtype=torch.float32))
        labels.append(row['label'])
        lengths.append(seq_length)
    
    if not data:
        log_debug("데이터를 로드하지 못했습니다.")
        return None, None, None, None
    
    data = torch.stack(data)  # [num_samples, max_seq_length, 99]
    labels = torch.tensor(labels, dtype=torch.long)
    lengths = torch.tensor(lengths, dtype=torch.int64)
    mask = torch.arange(max_seq_length).expand(len(lengths), max_seq_length) < lengths.unsqueeze(1)
    mask = mask.to(torch.bool)
    
    log_debug(f"Data shape: {data.shape}, Mask shape: {mask.shape}, Labels shape: {labels.shape}")
    return data, mask, labels, lengths

def main():
    sequence_dir = r"E:\학부연구생 과제_2025\hand_sentence\sentence_dataset_15\pose_label"
    csv_path = r"E:\학부연구생 과제_2025\hand_sentence\sentence_dataset_15\pose_label\labels.csv"
    max_seq_length = 130
    batch_size = 4
    num_epochs = 200
    learning_rate = 0.0005
    hidden_size = 64
    num_layers = 1
    num_classes = 15
    dropout = 0.5
    
    # 디버깅 로그 파일 초기화
    if os.path.exists(debug_log_file):
        os.remove(debug_log_file)
    
    # 데이터 로드
    data, mask, labels, lengths = load_dataset_for_lstm(sequence_dir, csv_path, max_seq_length)
    
    if data is None:
        log_debug("프로그램 종료: 데이터 로드 실패")
        return
    
    # 학습/검증 분할
    train_idx, val_idx = train_test_split(
        np.arange(len(data)), test_size=0.2, stratify=labels, random_state=42
    )
    train_data = data[train_idx]
    train_mask = mask[train_idx]
    train_labels = labels[train_idx]
    val_data = data[val_idx]
    val_mask = mask[val_idx]
    val_labels = labels[val_idx]
    
    log_debug(f"Train data shape: {train_data.shape}, Validation data shape: {val_data.shape}")
    log_debug(f"Train labels shape: {train_labels.shape}, Validation labels shape: {val_labels.shape}")
    
    # DataLoader 생성
    train_dataset = TensorDataset(train_data, train_mask, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(val_data, val_mask, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 모델 정의
    model = CustomLSTM(
        input_dim=99,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout
    )
    
    # 손실 함수 및 최적화
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    # Early Stopping
    best_val_loss = float('inf')
    patience = 10
    trigger_times = 0
    
    # 학습 루프
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_data, batch_mask, batch_labels in train_loader:
            batch_data, batch_mask, batch_labels = batch_data.to(model.device), batch_mask.to(model.device), batch_labels.to(model.device)
            optimizer.zero_grad()
            outputs = model(batch_data, attn_mask=batch_mask) 
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        
        # 검증
        model.eval() 
        val_loss = 0
        val_preds = []
        val_true = []
        with torch.no_grad():
            for batch_data, batch_mask, batch_labels in val_loader:
                batch_data, batch_mask, batch_labels = batch_data.to(model.device), batch_mask.to(model.device), batch_labels.to(model.device)
                outputs = model(batch_data, attn_mask=batch_mask)
                val_loss += criterion(outputs, batch_labels).item()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_true.extend(batch_labels.cpu().numpy())
        val_loss /= len(val_loader)
        val_accuracy = accuracy_score(val_true, val_preds)
        
        scheduler.step()
        
        log_debug(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                log_debug("Early stopping!")
                break
    
    log_debug("Training finished!")
    log_debug(f"Final Validation Accuracy: {val_accuracy:.4f}")

if __name__ == "__main__":
    main()
