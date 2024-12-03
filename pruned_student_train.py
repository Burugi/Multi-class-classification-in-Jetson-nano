import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging
from datetime import datetime
import os
from tqdm import tqdm
import torch.nn.functional as F
import random
from torchvision import transforms
from sklearn.model_selection import train_test_split

class QuickDrawDataset(Dataset):
    def __init__(self, data, labels, transform=None, augment=False):
        self.data = data
        self.labels = labels
        self.transform = transform
        self.augment = augment
        
        # 데이터 증강을 위한 transform 정의
        self.rotate_transform = transforms.RandomRotation(degrees=15)
    
    def __len__(self):
        return len(self.data)
    
    def add_noise(self, image, noise_factor=0.1):
        noise = torch.randn_like(image) * noise_factor
        noisy_image = image + noise
        return torch.clamp(noisy_image, 0., 1.)

    def add_occlusion(self, image, block_size=4, num_blocks=4):
        img = image.clone()
        h, w = img.shape[1:]
        
        for _ in range(num_blocks):
            x = random.randint(0, w - block_size)
            y = random.randint(0, h - block_size)
            img[:, y:y+block_size, x:x+block_size] = 0
            
        return img

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        
        image = torch.FloatTensor(image).unsqueeze(0)
        
        if self.augment:
            if random.random() < 0.3:
                image = self.add_noise(image, noise_factor=0.1)
            if random.random() < 0.3:
                image = self.add_occlusion(image, block_size=4, num_blocks=random.randint(1, 3))
            if random.random() < 0.3:
                image = self.rotate_transform(image)

        if self.transform:
            image = self.transform(image)
        
        label = torch.LongTensor([label])[0]
        return image, label
    
class TeacherVGG(nn.Module):
    def __init__(self, num_classes=10):
        super(TeacherVGG, self).__init__()
        
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
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class CompactStudentNet(nn.Module):
    def __init__(self, num_classes=10):
        super(CompactStudentNet, self).__init__()
        
        # 특징 추출기 (더 가벼운 구조)
        self.features = nn.Sequential(
            # 첫 번째 블록
            nn.Conv2d(1, 16, 3, padding=1),  # 채널 수 감소
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # 두 번째 블록
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Depthwise Separable Convolution
            nn.Conv2d(32, 32, 3, padding=1, groups=32),  # depthwise
            nn.Conv2d(32, 64, 1),  # pointwise
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        # 분류기
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),  # 크기 감소
            nn.Flatten(),
            nn.Linear(64 * 2 * 2, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class ImprovedPruning:
    def __init__(self, model):
        self.model = model
        self.masks = {}
        self._initialize_masks()
    
    def _initialize_masks(self):
        """각 레이어의 마스크를 1로 초기화"""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                self.masks[name] = torch.ones_like(module.weight.data)
    
    def calculate_sparsity(self):
        """현재 모델의 스파시티(0인 가중치의 비율) 계산"""
        total_params = 0
        zero_params = 0
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                total_params += module.weight.numel()
                zero_params += (module.weight.data == 0).sum().item()
        return 100.0 * zero_params / total_params if total_params > 0 else 0
    
    def gradual_pruning(self, current_epoch, total_epochs, initial_sparsity=0.0, 
                       final_sparsity=0.5, frequency=2):
        """점진적 프루닝 구현"""
        if current_epoch % frequency != 0:
            return
            
        remaining_epochs = total_epochs - current_epoch
        if remaining_epochs <= 0:
            return
            
        # 현재 목표 스파시티 계산 (선형 증가)
        current_sparsity = initial_sparsity + (
            (final_sparsity - initial_sparsity) *
            (1.0 - remaining_epochs / total_epochs)
        )
        
        # 각 레이어별로 프루닝 적용
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # 현재 레이어의 가중치
                weight = module.weight.data
                
                # 절대값 기준으로 현재 스파시티에 해당하는 임계값 계산
                threshold = torch.quantile(torch.abs(weight), current_sparsity)
                
                # 새로운 마스크 생성 (이전 마스크와 AND 연산)
                new_mask = (torch.abs(weight) > threshold).float()
                self.masks[name] = self.masks[name] * new_mask
                
                # 마스크 적용
                module.weight.data.mul_(self.masks[name])
    
    def apply_masks(self):
        """저장된 마스크를 모델에 적용"""
        with torch.no_grad():
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    if name in self.masks:
                        module.weight.data.mul_(self.masks[name])
    
    def mask_gradients(self):
        """역전파 중에 프루닝된 가중치의 그래디언트를 0으로 만듦"""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if name in self.masks and module.weight.grad is not None:
                    module.weight.grad.mul_(self.masks[name])

class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, temperature=3.0, alpha=0.5):
        super(KnowledgeDistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, student_outputs, teacher_outputs, targets):
        soft_targets = F.softmax(teacher_outputs / self.temperature, dim=1)
        student_log_softmax = F.log_softmax(student_outputs / self.temperature, dim=1)
        
        distillation_loss = F.kl_div(student_log_softmax, soft_targets, reduction='batchmean')
        student_loss = F.cross_entropy(student_outputs, targets)
        
        total_loss = (self.alpha * distillation_loss * (self.temperature ** 2) + 
                     (1 - self.alpha) * student_loss)
        return total_loss

class PrunedStudentTrainer:
    def __init__(self, npz_path, teacher_model_path, save_dir='models'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
            batch_size=64,
            shuffle=True
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=64,
            shuffle=False
        )
        
        # 모델 설정
        self.teacher_model = self.load_teacher_model(teacher_model_path)
        self.student_model = CompactStudentNet(num_classes=self.num_classes).to(self.device)
        self.pruner = ImprovedPruning(self.student_model)
        
        # 학습 설정
        self.criterion = KnowledgeDistillationLoss(temperature=3.0, alpha=0.5)
        self.optimizer = optim.Adam(self.student_model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5)

    def setup_logging(self):
        logging.basicConfig(
            filename=f'pruned_student_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
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
            
            with torch.no_grad():
                teacher_outputs = self.teacher_model(inputs)
            
            self.optimizer.zero_grad()
            student_outputs = self.student_model(inputs)
            loss = self.criterion(student_outputs, teacher_outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = student_outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({'loss': running_loss/total, 'acc': 100.*correct/total})
        
        return running_loss/len(self.train_loader), 100.*correct/total

    def evaluate(self):
        self.student_model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.test_loader, desc='Evaluating'):
                if len(inputs.shape) == 3:
                    inputs = inputs.unsqueeze(1)
                    
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.student_model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        accuracy = 100.*correct/total
        return accuracy

    def train(self, epochs=30):
        self.pruner = ImprovedPruning(self.student_model)
        
        for epoch in range(epochs):
            # 점진적 프루닝 적용
            self.pruner.gradual_pruning(
                current_epoch=epoch,
                total_epochs=epochs,
                initial_sparsity=0.0,
                final_sparsity=0.5,
                frequency=2
            )
            
            # 학습 전에 마스크 적용
            self.pruner.apply_masks()
            
            # 일반적인 학습 과정
            train_loss, train_acc = self.train_epoch()
            
            # 그래디언트 업데이트 후 마스크 재적용
            self.pruner.mask_gradients()
            
            # 평가 및 로깅
            test_acc = self.evaluate()
            current_sparsity = self.pruner.calculate_sparsity()
            
            self.logger.info(f"\nEpoch {epoch+1}/{epochs}")
            self.logger.info(f"Train Loss: {train_loss:.4f}")
            self.logger.info(f"Train Accuracy: {train_acc:.2f}%")
            self.logger.info(f"Test Accuracy: {test_acc:.2f}%")
            self.logger.info(f"Current Sparsity: {current_sparsity:.2f}%")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True,
                      help='Path to the npz dataset file')
    parser.add_argument('--teacher-model', type=str, required=True,
                      help='Path to the teacher model')
    parser.add_argument('--save-dir', type=str, default='models',
                      help='Directory to save models')
    parser.add_argument('--epochs', type=int, default=30,
                      help='Number of epochs to train')
    
    args = parser.parse_args()
    
    trainer = PrunedStudentTrainer(args.data, args.teacher_model, args.save_dir)
    trainer.train(epochs=args.epochs)

if __name__ == '__main__':
    main()