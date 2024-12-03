import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
from torchvision import transforms
class TeacherVGG(nn.Module):
    def __init__(self, num_classes=10):
        super(TeacherVGG, self).__init__()
        
        # 입력: 1채널 (흑백)
        self.features = nn.Sequential(
            # 첫 번째 블록
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # 두 번째 블록
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # 세 번째 블록
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        # 분류기
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),  # 가변적인 입력 크기 처리
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # 가중치 초기화
        self._initialize_weights()
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class CompactStudentNet(nn.Module):
    def __init__(self, num_classes=10):
        super(CompactStudentNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 32, 3, padding=1, groups=32),
            nn.Conv2d(32, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(64 * 2 * 2, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class QuickDrawDataset(Dataset):
    def __init__(self, data, labels, transform=None, augment=False):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        if len(image.shape) == 2:
            image = image.unsqueeze(0)
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class QuantizedStudentTrainer:
    def __init__(self, npz_path, teacher_model_path, save_dir='models'):
        self.device = torch.device('cpu')  # Jetson Nano 최적화를 위해 CPU 사용
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.setup_logging()
        
        # 데이터 로드
        data = np.load(npz_path)
        images = data['data'].astype(np.float32) / 255.0
        labels = data['labels']
        self.num_classes = len(np.unique(labels))
        
        # Train/Test 분할
        X_train, X_test, y_train, y_test = train_test_split(
            images, labels, test_size=0.2, random_state=42,
            stratify=labels  # 클래스 비율 유지
        )
        
        # Transform 정의
        train_transform = transforms.Compose([
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomErasing(p=0.2),
            transforms.Normalize((0.5,), (0.5,))
        ])

        test_transform = transforms.Compose([
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        # Dataset 생성
        train_dataset = QuickDrawDataset(
            X_train, y_train,
            transform=train_transform,
            augment=True
        )
        
        test_dataset = QuickDrawDataset(
            X_test, y_test,
            transform=test_transform,
            augment=False
        )
        
        # DataLoader 생성
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=32,  # Jetson Nano를 위해 작은 배치 사이즈 사용
            shuffle=True
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False
        )
        
        # 모델 설정
        self.teacher_model = self.load_teacher_model(teacher_model_path)
        self.student_model = CompactStudentNet(num_classes=self.num_classes).to(self.device)
        
        # 학습 설정
        self.criterion = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.student_model.parameters(), lr=0.001)
        self.temperature = 3.0
        self.alpha = 0.5

    def setup_logging(self):
        logging.basicConfig(
            filename=f'quantized_student_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logging.getLogger('').addHandler(console)
        self.logger = logging.getLogger(__name__)

    def load_teacher_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        model = TeacherVGG(num_classes=self.num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        return model

    def train_epoch(self):
        self.student_model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for inputs, targets in pbar:
            if len(inputs.shape) == 3:
                inputs = inputs.unsqueeze(1)
                
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Teacher prediction
            with torch.no_grad():
                teacher_outputs = self.teacher_model(inputs)
            
            # Student training
            self.optimizer.zero_grad()
            student_outputs = self.student_model(inputs)
            
            # Knowledge Distillation Loss
            soft_targets = torch.nn.functional.softmax(teacher_outputs / self.temperature, dim=1)
            student_log_softmax = torch.nn.functional.log_softmax(student_outputs / self.temperature, dim=1)
            distillation_loss = self.criterion(student_log_softmax, soft_targets)
            
            # Cross Entropy Loss
            student_loss = self.ce_loss(student_outputs, targets)
            
            # Total Loss
            loss = (self.alpha * self.temperature * self.temperature * distillation_loss + 
                   (1 - self.alpha) * student_loss)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = student_outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({'loss': running_loss/total, 'acc': 100.*correct/total})
        
        return running_loss/len(self.train_loader), 100.*correct/total

    def quantize_model(self):
        """Dynamic Quantization 적용"""
        self.student_model.eval()
        self.quantized_model = torch.quantization.quantize_dynamic(
            self.student_model,
            {nn.Linear, nn.Conv2d},  # 양자화할 레이어 타입
            dtype=torch.qint8
        )
        return self.quantized_model

    def evaluate(self, model):
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.test_loader, desc='Evaluating'):
                if len(inputs.shape) == 3:
                    inputs = inputs.unsqueeze(1)
                    
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        accuracy = 100.*correct/total
        return accuracy

    def train(self, epochs=10):
        best_acc = 0
        
        for epoch in range(epochs):
            self.logger.info(f"\nEpoch {epoch+1}/{epochs}")
            
            # 일반 학습
            train_loss, train_acc = self.train_epoch()
            
            # 양자화 및 평가
            quantized_model = self.quantize_model()
            test_acc = self.evaluate(quantized_model)
            
            self.logger.info(f"Train Loss: {train_loss:.4f}")
            self.logger.info(f"Train Accuracy: {train_acc:.2f}%")
            self.logger.info(f"Test Accuracy (Quantized): {test_acc:.2f}%")
            
            # 최고 성능 모델 저장
            if test_acc > best_acc:
                best_acc = test_acc
                model_path = os.path.join(self.save_dir, 'best_quantized_student.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.student_model.state_dict(),
                    'quantized_model_state_dict': self.quantized_model.state_dict(),
                    'accuracy': test_acc,
                    'num_classes': self.num_classes
                }, model_path)
                self.logger.info(f"Saved best model with accuracy: {test_acc:.2f}%")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True,
                      help='Path to the npz dataset file')
    parser.add_argument('--teacher-model', type=str, required=True,
                      help='Path to the teacher model')
    parser.add_argument('--save-dir', type=str, default='models',
                      help='Directory to save models')
    parser.add_argument('--epochs', type=int, default=10,
                      help='Number of epochs to train')
    
    args = parser.parse_args()
    
    trainer = QuantizedStudentTrainer(args.data, args.teacher_model, args.save_dir)
    trainer.train(epochs=args.epochs)

if __name__ == '__main__':
    main()