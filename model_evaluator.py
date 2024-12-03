import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import time
import json
import os
from datetime import datetime
import psutil
# 기존 import문 아래에 추가
from functools import partial
import time

def count_conv2d(m, x, y):
    x = x[0]  # 입력 받기
    
    # 커널 연산
    kernel_ops = m.kernel_size[0] * m.kernel_size[1] * m.in_channels * m.out_channels
    if m.bias is not None:
        kernel_ops += m.out_channels
        
    # 각 위치별 연산
    kernel_ops_per_element = kernel_ops / m.groups
    batch_size = x.shape[0]
    output_elements = y.numel()
    
    # 전체 연산 수
    total_ops = batch_size * output_elements * kernel_ops_per_element
    
    return total_ops

def count_linear(m, x, y):
    x = x[0]  # 입력 받기
    total_ops = m.in_features * m.out_features
    if m.bias is not None:
        total_ops += m.out_features
    batch_size = x.shape[0]
    total_ops *= batch_size
    return total_ops

def measure_model_flops(model, input_size=(1, 1, 28, 28)):
    """모델의 FLOPS 측정"""
    module_flops = {}
    total_flops = 0
    
    def conv2d_flops_counter_hook(module, input, output):
        nonlocal total_flops
        # 입력 가져오기
        input = input[0]
        
        # 커널 연산
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels * module.out_channels
        if module.bias is not None:
            kernel_ops += module.out_channels
        
        # 각 위치별 연산
        kernel_ops_per_element = kernel_ops / module.groups
        batch_size = input.shape[0]
        output_elements = output.numel()
        
        # 전체 연산 수
        flops = batch_size * output_elements * kernel_ops_per_element
        total_flops += flops
        module_flops[module] = flops
        
    def linear_flops_counter_hook(module, input, output):
        nonlocal total_flops
        input = input[0]
        flops = module.in_features * module.out_features
        if module.bias is not None:
            flops += module.out_features
        batch_size = input.shape[0]
        flops *= batch_size
        total_flops += flops
        module_flops[module] = flops
    
    # 모든 모듈에 hook 등록
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv2d_flops_counter_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_flops_counter_hook))
    
    # 더미 입력으로 추론 실행
    dummy_input = torch.randn(input_size).to(next(model.parameters()).device)
    with torch.no_grad():
        model(dummy_input)
    
    # 훅 제거
    for hook in hooks:
        hook.remove()
    
    return total_flops

def measure_execution_time(model, input_tensor, num_iterations=100):
    """모델의 실행 시간 측정"""
    model.eval()
    with torch.no_grad():
        # 워밍업
        for _ in range(10):
            _ = model(input_tensor)
        
        # 실제 측정
        start_time = time.perf_counter()
        for _ in range(num_iterations):
            _ = model(input_tensor)
        end_time = time.perf_counter()
        
    avg_time = (end_time - start_time) * 1000 / num_iterations  # ms 단위
    return avg_time

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

class PrunedModelWrapper(nn.Module):
    def __init__(self, model, sparsity_threshold=0.1): # default 0.001
        super(PrunedModelWrapper, self).__init__()
        self.model = model
        self.sparsity_threshold = sparsity_threshold
        self.mask = {}
        self._create_masks()
        
    def _create_masks(self):
        """가중치의 마스크 생성"""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if 'weight' in name:
                    # 절대값이 threshold보다 작은 가중치를 0으로 만드는 마스크
                    mask = (torch.abs(param) > self.sparsity_threshold).float()
                    self.mask[name] = mask
                    param.data.mul_(mask)
    
    def get_sparsity(self):
        """모델의 sparsity 계산"""
        total_params = 0
        zero_params = 0
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                total_params += param.numel()
                zero_params += (param == 0).sum().item()
        return zero_params / total_params if total_params > 0 else 0
    
    def forward(self, x):
        # 추론 전에 마스크 재적용
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.mask:
                    param.data.mul_(self.mask[name])
        return self.model(x)

class QuantizedModelWrapper(nn.Module):
    def __init__(self, model, bits=8): # default 8
        super(QuantizedModelWrapper, self).__init__()
        self.model = model
        self.bits = bits
        self.max_val = 2**bits - 1
        
        # 모델 가중치 양자화
        self.quantize_model_weights()
        
        # 메모리 사용량 계산
        self.original_size = sum(p.numel() * p.element_size() for p in model.parameters())
        self.quantized_size = sum(p.numel() * (bits // 8) for p in model.parameters())
    
    def quantize_model_weights(self):
        """모델의 모든 가중치를 양자화"""
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                min_val = param.min()
                max_val = param.max()
                scale = (max_val - min_val) / self.max_val
                quantized = ((param - min_val) / scale).round()
                param.data = (quantized * scale + min_val).to(param.dtype)
    
    def get_size_reduction(self):
        """모델 크기 감소율 계산"""
        return 1 - (self.quantized_size / self.original_size)
    
    def get_memory_sizes(self):
        """원본과 양자화된 모델의 메모리 사용량 반환"""
        return {
            'original_size': self.original_size / (1024 * 1024),  # MB
            'quantized_size': self.quantized_size / (1024 * 1024)  # MB
        }
    
    def forward(self, x):
        return self.model(x)

def measure_model_complexity(model, input_size=(1, 1, 28, 28)):
    """모델의 복잡도 측정"""
    total_params = 0
    nonzero_params = 0
    total_flops = 0
    
    # 메모리 및 파라미터 계산
    def calculate_memory_size(model):
        if isinstance(model, QuantizedModelWrapper):
            return model.get_memory_sizes()
        else:
            total_size = sum(p.numel() * p.element_size() for p in model.parameters())
            return {
                'original_size': total_size / (1024 * 1024),
                'quantized_size': total_size / (1024 * 1024)
            }
    
    def calculate_params(model):
        total, nonzero = 0, 0
        for name, param in model.named_parameters():
            if 'weight' in name:
                total += param.numel()
                nonzero += (param != 0).sum().item()
        return total, nonzero
    
    def count_conv2d_flops(module, input, output):
        nonlocal total_flops
        input = input[0]
        
        # For pruned models, consider only non-zero weights
        if hasattr(module, 'weight'):
            nonzero_weights = (module.weight != 0).sum().item()
            kernel_ops = nonzero_weights
        else:
            kernel_ops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels * module.out_channels
            
        if module.bias is not None:
            kernel_ops += module.out_channels
        
        kernel_ops_per_element = kernel_ops / module.groups
        batch_size = input.shape[0]
        output_elements = output.numel()
        
        total_flops += batch_size * output_elements * kernel_ops_per_element
    
    def count_linear_flops(module, input, output):
        nonlocal total_flops
        input = input[0]
        
        if hasattr(module, 'weight'):
            nonzero_weights = (module.weight != 0).sum().item()
            flops = nonzero_weights
        else:
            flops = module.in_features * module.out_features
            
        if module.bias is not None:
            flops += module.out_features
        
        total_flops += input.shape[0] * flops
    
    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(count_conv2d_flops))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(count_linear_flops))
    
    # Calculate parameters
    total_params, nonzero_params = calculate_params(model)
    
    # Dummy forward pass
    dummy_input = torch.randn(input_size)
    with torch.no_grad():
        model(dummy_input)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Calculate memory sizes
    memory_sizes = calculate_memory_size(model)
    
    # Calculate reductions
    sparsity = 0 if total_params == 0 else (total_params - nonzero_params) / total_params
    size_reduction = 0
    if isinstance(model, QuantizedModelWrapper):
        size_reduction = model.get_size_reduction()
    
    return {
        'total_params': total_params,
        'nonzero_params': nonzero_params,
        'sparsity': sparsity,
        'total_flops': total_flops,
        'original_size_mb': memory_sizes['original_size'],
        'quantized_size_mb': memory_sizes['quantized_size'],
        'size_reduction': size_reduction
    }


def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=640,
    display_height=480,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        f"width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (flip_method, display_width, display_height)
    )

class ModelEvaluator:
    def __init__(self, model_path, model_type):
        self.device = torch.device('cpu')
        self.model_type = model_type
        self.model = self.load_model(model_path)
        self.model.eval()
        
        print(f"\nMeasuring {model_type} model performance metrics...")
                
        self.classes = [
            'circle', 'triangle', 'square', 'donut', 'house',
            'cloud', 'lightning', 'star', 'diamond', 'banana'
        ]

        complexity_metrics = measure_model_complexity(self.model)
        execution_time = measure_execution_time(self.model, torch.randn(1, 1, 28, 28))
        
        self.metrics = {
            'model_type': model_type,
            'fps': [],
            'inference_times': [],
            'memory_usage': [],
            'cpu_usage': [],
            'total_parameters': complexity_metrics['total_params'],
            'nonzero_parameters': complexity_metrics['nonzero_params'],
            'sparsity': complexity_metrics['sparsity'],
            'total_flops': complexity_metrics['total_flops'],
            'original_size_mb': complexity_metrics['original_size_mb'],
            'quantized_size_mb': complexity_metrics['quantized_size_mb'],
            'size_reduction': complexity_metrics['size_reduction'],
            'execution_time_ms': execution_time
        }
        
        print("\nModel statistics:")
        print(f"Total parameters: {complexity_metrics['total_params']:,}")
        print(f"Non-zero parameters: {complexity_metrics['nonzero_params']:,}")
        print(f"Sparsity: {complexity_metrics['sparsity']*100:.1f}%")
        print(f"Total FLOPS: {complexity_metrics['total_flops']:,}")
        print(f"Original size: {complexity_metrics['original_size_mb']:.2f} MB")
        
        if self.model_type == 'quantized':
            print(f"Quantized size: {complexity_metrics['quantized_size_mb']:.2f} MB")
            print(f"Size reduction: {complexity_metrics['size_reduction']*100:.1f}%")
        
        print(f"Execution time: {execution_time:.2f} ms")
    
    def load_model(self, model_path):
        model = CompactStudentNet(num_classes=10)
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if self.model_type == 'pruned':
            return PrunedModelWrapper(model)
        elif self.model_type == 'quantized':
            return QuantizedModelWrapper(model, bits=4)
        return model
    
    def summarize_metrics(self, frame_count):
        if not self.metrics['fps'] or frame_count == 0:
            print("No valid metrics to summarize.")
            return
            
        summary = {
            'model_type': self.model_type,
            'avg_fps': np.mean(self.metrics['fps']),
            'avg_inference_time': np.mean(self.metrics['inference_times']),
            'avg_cpu_usage': np.mean(self.metrics['cpu_usage']),
            'avg_memory_usage': np.mean(self.metrics['memory_usage']),
            'total_parameters': self.metrics['total_parameters'],
            'nonzero_parameters': self.metrics['nonzero_parameters'],
            'sparsity': self.metrics['sparsity'],
            'total_flops': self.metrics['total_flops'],
            'original_size_mb': self.metrics['original_size_mb'],
            'size_reduction': self.metrics['size_reduction']
        }
        
        if self.model_type == 'quantized':
            summary['quantized_size_mb'] = self.metrics['quantized_size_mb']
        
        print(f"\n{self.model_type} Model Performance Summary:")
        print("-" * 50)
        print(f"Average FPS: {summary['avg_fps']:.2f}")
        print(f"Average Inference Time: {summary['avg_inference_time']:.2f}ms")
        print(f"Total Parameters: {summary['total_parameters']:,}")
        print(f"Non-zero Parameters: {summary['nonzero_parameters']:,}")
        print(f"Sparsity: {summary['sparsity']*100:.1f}%")
        print(f"Total FLOPS: {summary['total_flops']:,}")
        print(f"Original Size: {summary['original_size_mb']:.2f}MB")
        
        if self.model_type == 'quantized':
            print(f"Quantized Size: {summary['quantized_size_mb']:.2f}MB")
            print(f"Size Reduction: {summary['size_reduction']*100:.1f}%")
        
        print(f"Average CPU Usage: {summary['avg_cpu_usage']:.1f}%")
        print(f"Average Memory Usage: {summary['avg_memory_usage']:.1f}%")
        print(f"Total Frames: {frame_count}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"performance_{self.model_type}_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=4)
        print(f"\nResults saved to {filename}")

  
    def get_model_size(self):
        return sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
    
    def preprocess_image(self, frame):
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 11, 2)
            binary = cv2.medianBlur(binary, 3)
            
            contours_output = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours_output[0] if len(contours_output) == 2 else contours_output[1]
            
            if contours:
                main_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(main_contour)
                
                padding = 20
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(binary.shape[1] - x, w + 2*padding)
                h = min(binary.shape[0] - y, h + 2*padding)
                
                roi = binary[y:y+h, x:x+w]
                
                size = max(w, h)
                square = np.zeros((size, size), dtype=np.uint8)
                start_x = (size - w) // 2
                start_y = (size - h) // 2
                square[start_y:start_y+h, start_x:start_x+w] = roi
                
                resized = cv2.resize(square, (28, 28))
            else:
                resized = np.zeros((28, 28), dtype=np.uint8)
            
            normalized = resized.astype(np.float32) / 255.0
            tensor = torch.FloatTensor(normalized).unsqueeze(0).unsqueeze(0)
            return tensor, resized

        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            return torch.zeros((1, 1, 28, 28)), np.zeros((28, 28), dtype=np.uint8)

    def evaluate(self, duration=30, display=False, use_usb=False):
        if use_usb:
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

        if not cap.isOpened():
            if not use_usb:
                print("CSI camera failed, trying USB camera...")
                cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Failed to open any camera")
                return

        start_time = time.time()
        frame_count = 0
        warmup_frames = 30

        print(f"\nStarting evaluation of {self.model_type} model...")
        print(f"Duration: {duration} seconds")
        print("Warming up camera and model...")

        try:
            while (time.time() - start_time) < duration:
                ret, frame = cap.read()
                if not ret:
                    continue

                if frame_count < warmup_frames:
                    frame_count += 1
                    continue

                inference_start = time.time()
                
                tensor, processed_image = self.preprocess_image(frame)
                with torch.no_grad():
                    outputs = self.model(tensor)
                    probs = F.softmax(outputs, dim=1)
                    conf, pred = torch.max(probs, 1)

                inference_time = (time.time() - inference_start) * 1000
                fps = 1000 / inference_time if inference_time > 0 else 0

                self.metrics['fps'].append(fps)
                self.metrics['inference_times'].append(inference_time)
                self.metrics['cpu_usage'].append(psutil.cpu_percent())
                self.metrics['memory_usage'].append(psutil.virtual_memory().percent)

                if display:
                    confidence = conf.item() * 100
                    predicted_class = self.classes[pred.item()]
                    color = (0, 255, 0) if confidence > 50 else (0, 0, 255)

                    input_display = cv2.resize(processed_image, (140, 140))
                    input_display = cv2.cvtColor(input_display, cv2.COLOR_GRAY2BGR)

                    y_offset = 10
                    x_offset = 10
                    
                    cv2.rectangle(frame, (x_offset-5, y_offset-5),
                                (x_offset+145, y_offset+145), (255,255,255), -1)
                    cv2.rectangle(frame, (x_offset-5, y_offset-5),
                                (x_offset+145, y_offset+145), (0,0,0), 1)
                    
                    frame[y_offset:y_offset+140, x_offset:x_offset+140] = input_display

                    text_x = x_offset + 160
                    cv2.putText(frame, f"{self.model_type}: {predicted_class}", (text_x, 35),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    cv2.putText(frame, f"Conf: {confidence:.1f}%", (text_x, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    cv2.putText(frame, f"FPS: {fps:.1f}", (text_x, 105),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                    # Top-3 예측
                    probs_np = probs.squeeze().cpu().numpy()
                    top3_idx = probs_np.argsort()[-3:][::-1]
                    for i, idx in enumerate(top3_idx):
                        text = f"#{i+1}: {self.classes[idx]} ({probs_np[idx]*100:.1f}%)"
                        cv2.putText(frame, text, (text_x, 140 + i*30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    cv2.imshow('Performance Test', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"Processed frames: {frame_count}, Current FPS: {fps:.1f}")

        except KeyboardInterrupt:
            print("\nEvaluation interrupted by user")
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()
            
            if frame_count > warmup_frames:
                self.summarize_metrics(frame_count - warmup_frames)
            else:
                print("Not enough frames processed.")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                      help='Path to the model file')
    parser.add_argument('--type', type=str, required=True,
                      choices=['base', 'pruned', 'quantized'],
                      help='Type of model to evaluate')
    parser.add_argument('--duration', type=int, default=30,
                      help='Duration of evaluation in seconds')
    parser.add_argument('--display', action='store_true',
                      help='Display inference results')
    parser.add_argument('--use-usb', action='store_true',
                      help='Use USB camera instead of CSI')
    
    args = parser.parse_args()
    
    evaluator = ModelEvaluator(args.model, args.type)
    evaluator.evaluate(duration=args.duration, display=args.display, use_usb=args.use_usb)

if __name__ == '__main__':
    main()
